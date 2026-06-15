from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
EMB_DIR = REPO_ROOT / "Embeddings"
if str(EMB_DIR) not in sys.path:
    sys.path.insert(0, str(EMB_DIR))

from emb_gpt2 import GPTModel, TrainConfig, resolve_amp_settings, select_device  # noqa: E402


@dataclass
class InstructionFineTuneConfig:
    train_json: str = str(REPO_ROOT / "FineTuning" / "data" / "instruction_train.json")
    val_json: str = str(REPO_ROOT / "FineTuning" / "data" / "instruction_val.json")
    test_json: str = str(REPO_ROOT / "FineTuning" / "data" / "instruction_test.json")
    tokenizer_path: str = str(REPO_ROOT / "tokenizer-culturax-es-hf.json")
    base_checkpoint_path: str = str(REPO_ROOT / "pequellm_pesado_checkpoint.pth")
    output_root: str = str(REPO_ROOT / "FineTuning" / "artifacts_instruction")
    run_name: str = ""

    batch_size: int = 2
    max_length: int = 128
    max_epochs: int = 5 # 2 og
    eval_interval: int = 50
    eval_batches: int = 10
    lr: float = 5e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "auto"
    precision: str = "auto"

    freeze_base: bool = False
    mask_prompt_tokens: bool = False
    generate_samples: int = 5
    generate_tokens: int = 80

    # Fallback architecture only used when no checkpoint config is available.
    vocab_size: int = 65536
    n_embd: int = 768
    n_head: int = 24
    n_layer: int = 4
    block_size: int = 128


def format_input(entry: Dict[str, str]) -> str:
    has_input = bool(entry.get("input"))
    if has_input:
        intro = (
            "A continuación hay una instrucción que describe una tarea, junto con "
            "una entrada que proporciona más contexto. Escribe una respuesta que "
            "complete adecuadamente lo que se pide."
        )
    else:
        intro = (
            "A continuación hay una instrucción que describe una tarea. Escribe una "
            "respuesta que complete adecuadamente lo que se pide."
        )
    instruction_text = f"{intro}\n\n### Instrucción:\n{entry['instruction']}"
    input_text = f"\n\n### Entrada:\n{entry['input']}" if has_input else ""
    return instruction_text + input_text


def format_response(entry: Dict[str, str]) -> str:
    return f"\n\n### Respuesta:\n{entry['output']}"


def read_instruction_json(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Instruction dataset not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    for idx, entry in enumerate(data):
        for key in ("instruction", "input", "output"):
            if key not in entry:
                raise ValueError(f"{path} entry {idx} is missing key '{key}'")
    return data


class InstructionDataset(Dataset):
    def __init__(self, entries: Sequence[Dict[str, str]], tokenizer: Tokenizer, max_length: int, eos_token_id: int | None):
        self.items: List[Tuple[List[int], int]] = []
        self.entries = list(entries)
        for entry in self.entries:
            prompt_ids = tokenizer.encode(format_input(entry)).ids
            full_ids = tokenizer.encode(format_input(entry) + format_response(entry)).ids
            if eos_token_id is not None:
                full_ids.append(eos_token_id)
            full_ids = full_ids[:max_length]
            if len(full_ids) < 2:
                continue
            self.items.append((full_ids, min(len(prompt_ids), len(full_ids))))
        if not self.items:
            raise ValueError("Instruction dataset has no usable examples after tokenization/truncation")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[List[int], int]:
        return self.items[idx]


def collate_instruction_batch(
    batch: Sequence[Tuple[List[int], int]],
    pad_token_id: int,
    ignore_index: int,
    mask_prompt_tokens: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(token_ids) for token_ids, _ in batch)
    padded = []
    prompt_lengths = []
    for token_ids, prompt_len in batch:
        padded.append(token_ids + [pad_token_id] * (max_len - len(token_ids)))
        prompt_lengths.append(prompt_len)

    batch_tensor = torch.tensor(padded, dtype=torch.long)
    inputs = batch_tensor[:, :-1].contiguous()
    targets = batch_tensor[:, 1:].contiguous()
    targets = targets.masked_fill(targets == pad_token_id, ignore_index)

    if mask_prompt_tokens:
        for row_idx, prompt_len in enumerate(prompt_lengths):
            targets[row_idx, : max(0, prompt_len - 1)] = ignore_index
    return inputs, targets


def make_base_config(cfg: InstructionFineTuneConfig, checkpoint: dict | None) -> TrainConfig:
    base_cfg = TrainConfig()
    if checkpoint and isinstance(checkpoint.get("config"), dict):
        for key, value in checkpoint["config"].items():
            if hasattr(base_cfg, key):
                setattr(base_cfg, key, value)
    else:
        base_cfg.vocab_size = cfg.vocab_size
        base_cfg.n_embd = cfg.n_embd
        base_cfg.n_head = cfg.n_head
        base_cfg.n_layer = cfg.n_layer
        base_cfg.block_size = cfg.block_size

    if cfg.max_length > base_cfg.block_size:
        raise ValueError(
            f"max_length={cfg.max_length} exceeds model block_size={base_cfg.block_size}. "
            "Lower max_length or use a checkpoint trained with a larger context window."
        )
    base_cfg.precision = cfg.precision
    base_cfg.device = cfg.device
    return base_cfg


def load_base_model(cfg: InstructionFineTuneConfig, device: str) -> Tuple[GPTModel, TrainConfig, bool]:
    checkpoint_path = Path(cfg.base_checkpoint_path)
    if checkpoint_path.exists():
        raw = torch.load(checkpoint_path, map_location=device)
        checkpoint = raw if isinstance(raw, dict) else None
        base_cfg = make_base_config(cfg, checkpoint)
        model = GPTModel(base_cfg)
        state_dict = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
        model.load_state_dict(state_dict)
        return model, base_cfg, True

    print(f"[WARN] base checkpoint not found: {checkpoint_path}. Starting from random GPT weights.")
    base_cfg = make_base_config(cfg, None)
    return GPTModel(base_cfg), base_cfg, False


def configure_trainable_params(model: GPTModel, freeze_base: bool) -> None:
    if not freeze_base:
        for param in model.parameters():
            param.requires_grad = True
        return

    for param in model.parameters():
        param.requires_grad = False
    for param in model.lm_head.parameters():
        param.requires_grad = True
    for param in model.ln_f.parameters():
        param.requires_grad = True
    if len(model.blocks) > 0:
        for param in model.blocks[-1].parameters():
            param.requires_grad = True


def configure_optimizer(model: nn.Module, cfg: InstructionFineTuneConfig) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and not name.endswith("bias") and "ln" not in name and "embedding" not in name:
            decay_params.append(param)
        else:
            no_decay_params.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
    )


@torch.no_grad()
def evaluate_loss(model: GPTModel, loader: DataLoader, device: str, amp, max_batches: int) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        with torch.autocast(device_type=amp.device_type, dtype=amp.dtype, enabled=amp.enabled):
            _, loss = model(xb, yb)
        if loss is None:
            raise RuntimeError("Evaluation loss unexpectedly became None")
        total_loss += float(loss.item())
        total_batches += 1
        if 0 < max_batches <= total_batches:
            break
    model.train()
    return total_loss / max(1, total_batches)


def write_metrics_header(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "step", "train_loss", "val_loss", "lr"])


def append_metric(path: Path, epoch: int, step: int, train_loss: float, val_loss: float, lr: float) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([epoch, step, train_loss, val_loss, lr])


def build_run_dir(output_root: Path, run_name: str) -> Path:
    if run_name:
        return output_root / run_name
    return output_root / f"instruction_{time.strftime('%Y%m%d-%H%M%S')}"


@torch.no_grad()
def generate_response(model: GPTModel, tokenizer: Tokenizer, entry: Dict[str, str], device: str, max_new_tokens: int) -> str:
    model.eval()
    prompt = format_input(entry) + "\n\n### Response:\n"
    prompt_ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.block_size :]
        logits, _ = model(idx_cond)
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat((idx, next_id), dim=1)
    text = tokenizer.decode(idx[0].tolist())
    return text[len(prompt) :].strip() if text.startswith(prompt) else text


def save_sample_responses(
    model: GPTModel,
    tokenizer: Tokenizer,
    entries: Sequence[Dict[str, str]],
    run_dir: Path,
    device: str,
    max_samples: int,
    max_new_tokens: int,
) -> None:
    samples = []
    for entry in list(entries)[:max_samples]:
        samples.append(
            {
                "instruction": entry["instruction"],
                "input": entry.get("input", ""),
                "expected_output": entry["output"],
                "model_response": generate_response(model, tokenizer, entry, device, max_new_tokens),
            }
        )
    (run_dir / "sample_responses.json").write_text(json.dumps(samples, ensure_ascii=True, indent=2), encoding="utf-8")


def train_instruction(cfg: InstructionFineTuneConfig) -> Path:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    tokenizer = Tokenizer.from_file(cfg.tokenizer_path)
    pad_token_id = tokenizer.token_to_id("<pad>")
    if pad_token_id is None:
        pad_token_id = tokenizer.token_to_id("</s>")
    if pad_token_id is None:
        pad_token_id = 0
    eos_token_id = tokenizer.token_to_id("</s>")

    train_entries = read_instruction_json(Path(cfg.train_json))
    val_entries = read_instruction_json(Path(cfg.val_json))
    test_entries = read_instruction_json(Path(cfg.test_json))

    device = select_device(cfg.device)
    model, base_cfg, checkpoint_loaded = load_base_model(cfg, device)
    configure_trainable_params(model, cfg.freeze_base)
    model.to(device)
    amp = resolve_amp_settings(base_cfg, device)

    train_ds = InstructionDataset(train_entries, tokenizer, cfg.max_length, eos_token_id)
    val_ds = InstructionDataset(val_entries, tokenizer, cfg.max_length, eos_token_id)

    collate_fn = lambda batch: collate_instruction_batch(
        batch=batch,
        pad_token_id=pad_token_id,
        ignore_index=-100,
        mask_prompt_tokens=cfg.mask_prompt_tokens,
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = configure_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp.use_grad_scaler and device == "cuda"))

    run_dir = build_run_dir(Path(cfg.output_root), cfg.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    write_metrics_header(metrics_path)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "instruction_fine_tune_config": asdict(cfg),
                "base_model_config": asdict(base_cfg),
                "checkpoint_loaded": checkpoint_loaded,
                "pad_token_id": pad_token_id,
                "eos_token_id": eos_token_id,
                "train_examples": len(train_ds),
                "val_examples": len(val_ds),
                "test_examples": len(test_entries),
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    print(f"[INFO] device={device} precision={cfg.precision} amp={amp.enabled} dtype={amp.dtype}")
    print(f"[INFO] checkpoint_loaded={checkpoint_loaded}")
    print(f"[INFO] trainable_params={trainable:,} total_params={total:,}")
    print(f"[INFO] examples train={len(train_ds)} val={len(val_ds)} test={len(test_entries)}")
    print(f"[INFO] run_dir={run_dir}")

    best_val_loss = float("inf")
    global_step = 0
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            global_step += 1
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=amp.device_type, dtype=amp.dtype, enabled=amp.enabled):
                _, loss = model(xb, yb)
            if loss is None:
                raise RuntimeError("Training loss unexpectedly became None")

            if amp.use_grad_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

            if global_step % cfg.eval_interval == 0:
                val_loss = evaluate_loss(model, val_loader, device, amp, cfg.eval_batches)
                append_metric(metrics_path, epoch, global_step, float(loss.item()), val_loss, cfg.lr)
                print(
                    f"[epoch {epoch:02d} step {global_step:05d}] "
                    f"train_loss={float(loss.item()):.4f} val_loss={val_loss:.4f}"
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "base_config": asdict(base_cfg),
                            "instruction_fine_tune_config": asdict(cfg),
                        },
                        run_dir / "best_instruction_checkpoint.pth",
                    )

        val_loss = evaluate_loss(model, val_loader, device, amp, cfg.eval_batches)
        append_metric(metrics_path, epoch, global_step, float("nan"), val_loss, cfg.lr)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "base_config": asdict(base_cfg),
                    "instruction_fine_tune_config": asdict(cfg),
                },
                run_dir / "best_instruction_checkpoint.pth",
            )
        print(f"[epoch {epoch:02d} done] val_loss={val_loss:.4f}")

    (run_dir / "final_metrics.json").write_text(json.dumps({"best_val_loss": best_val_loss}, indent=2), encoding="utf-8")
    if cfg.generate_samples > 0:
        save_sample_responses(model, tokenizer, test_entries, run_dir, device, cfg.generate_samples, cfg.generate_tokens)
    print(f"[INFO] best_val_loss={best_val_loss:.4f}")
    return run_dir


def parse_args() -> InstructionFineTuneConfig:
    cfg = InstructionFineTuneConfig()
    parser = argparse.ArgumentParser(description="Chapter-7 style instruction fine-tuning for PequeLLM.")
    parser.add_argument("--train-json", default=cfg.train_json)
    parser.add_argument("--val-json", default=cfg.val_json)
    parser.add_argument("--test-json", default=cfg.test_json)
    parser.add_argument("--tokenizer-path", default=cfg.tokenizer_path)
    parser.add_argument("--base-checkpoint-path", default=cfg.base_checkpoint_path)
    parser.add_argument("--output-root", default=cfg.output_root)
    parser.add_argument("--run-name", default=cfg.run_name)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--max-length", type=int, default=cfg.max_length)
    parser.add_argument("--max-epochs", type=int, default=cfg.max_epochs)
    parser.add_argument("--eval-interval", type=int, default=cfg.eval_interval)
    parser.add_argument("--eval-batches", type=int, default=cfg.eval_batches)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--weight-decay", type=float, default=cfg.weight_decay)
    parser.add_argument("--grad-clip", type=float, default=cfg.grad_clip)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--device", default=cfg.device)
    parser.add_argument("--precision", default=cfg.precision)
    parser.add_argument("--freeze-base", action="store_true")
    parser.add_argument("--mask-prompt-tokens", action="store_true")
    parser.add_argument("--generate-samples", type=int, default=cfg.generate_samples)
    parser.add_argument("--generate-tokens", type=int, default=cfg.generate_tokens)
    parser.add_argument("--vocab-size", type=int, default=cfg.vocab_size)
    parser.add_argument("--n-embd", type=int, default=cfg.n_embd)
    parser.add_argument("--n-head", type=int, default=cfg.n_head)
    parser.add_argument("--n-layer", type=int, default=cfg.n_layer)
    parser.add_argument("--block-size", type=int, default=cfg.block_size)
    args = parser.parse_args()

    cfg.train_json = args.train_json
    cfg.val_json = args.val_json
    cfg.test_json = args.test_json
    cfg.tokenizer_path = args.tokenizer_path
    cfg.base_checkpoint_path = args.base_checkpoint_path
    cfg.output_root = args.output_root
    cfg.run_name = args.run_name
    cfg.batch_size = args.batch_size
    cfg.max_length = args.max_length
    cfg.max_epochs = args.max_epochs
    cfg.eval_interval = args.eval_interval
    cfg.eval_batches = args.eval_batches
    cfg.lr = args.lr
    cfg.weight_decay = args.weight_decay
    cfg.grad_clip = args.grad_clip
    cfg.seed = args.seed
    cfg.device = args.device
    cfg.precision = args.precision
    cfg.freeze_base = args.freeze_base
    cfg.mask_prompt_tokens = args.mask_prompt_tokens
    cfg.generate_samples = args.generate_samples
    cfg.generate_tokens = args.generate_tokens
    cfg.vocab_size = args.vocab_size
    cfg.n_embd = args.n_embd
    cfg.n_head = args.n_head
    cfg.n_layer = args.n_layer
    cfg.block_size = args.block_size
    return cfg


if __name__ == "__main__":
    train_instruction(parse_args())
