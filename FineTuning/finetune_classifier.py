from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
EMB_DIR = REPO_ROOT / "Embeddings"
if str(EMB_DIR) not in sys.path:
    sys.path.insert(0, str(EMB_DIR))

from emb_gpt2 import GPTModel, TrainConfig, resolve_amp_settings, select_device  # noqa: E402


@dataclass
class FineTuneConfig:
    train_csv: str = str(REPO_ROOT / "FineTuning" / "data" / "classification_demo_train.csv")
    val_csv: str = str(REPO_ROOT / "FineTuning" / "data" / "classification_demo_val.csv")
    test_csv: str = str(REPO_ROOT / "FineTuning" / "data" / "classification_demo_test.csv")
    text_column: str = "text"
    label_column: str = "label"

    tokenizer_path: str = str(REPO_ROOT / "tokenizer-culturax-es-hf.json")
    base_checkpoint_path: str = str(REPO_ROOT / "pequellm_pesado_checkpoint.pth")
    output_root: str = str(REPO_ROOT / "FineTuning" / "artifacts_classifier")
    run_name: str = ""

    batch_size: int = 8
    max_length: int = 128
    max_epochs: int = 5
    eval_interval: int = 50
    lr: float = 5e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 42

    device: str = "auto"
    precision: str = "auto"
    freeze_base: bool = True
    unfreeze_last_block: bool = True
    create_demo_data: bool = False

    # These are used only if there is no usable config inside the base checkpoint.
    # Match GPT-2 Small defaults (Embeddings/emb_gpt2.py TrainConfig).
    vocab_size: int = 65536
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    block_size: int = 128


class ClassificationDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Tuple[str, int]],
        tokenizer: Tokenizer,
        max_length: int,
        pad_token_id: int,
        eos_token_id: int | None,
    ):
        self.rows = list(rows)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text, label_id = self.rows[idx]
        token_ids = self.tokenizer.encode(text).ids
        if self.eos_token_id is not None:
            token_ids.append(self.eos_token_id)
        token_ids = token_ids[: self.max_length]
        if not token_ids:
            token_ids = [self.pad_token_id]
        if len(token_ids) < self.max_length:
            token_ids = token_ids + [self.pad_token_id] * (self.max_length - len(token_ids))
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label_id, dtype=torch.long)


class GPTForClassification(nn.Module):
    def __init__(self, base_model: GPTModel, num_classes: int):
        super().__init__()
        self.base_model = base_model
        self.classification_head = nn.Linear(base_model.cfg.n_embd, num_classes)

    def forward(self, idx: torch.Tensor, labels: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor | None]:
        bsz, seq_len = idx.shape
        pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)
        tok_emb = self.base_model.token_embedding_table(idx)
        pos_emb = self.base_model.position_embedding_table(pos)
        x = tok_emb + pos_emb
        x = self.base_model.blocks(x)
        x = self.base_model.ln_f(x)

        # Raschka's classification fine-tuning uses the final sequence position as
        # the representation that feeds the classifier head.
        logits = self.classification_head(x[:, -1, :])
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return logits, loss


def write_demo_data(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    train_rows = [
        ("hola, puedes revisar el reporte de clase para manana?", "ham"),
        ("recordatorio: junta del equipo a las 2 pm en el laboratorio", "ham"),
        ("tu paquete fue entregado en recepcion, pasa con tu credencial", "ham"),
        ("gana dinero rapido haciendo clic en este enlace exclusivo", "spam"),
        ("felicidades, ganaste un premio, reclama tus datos bancarios", "spam"),
        ("oferta limitada: duplica tu inversion hoy mismo sin riesgo", "spam"),
        ("la lectura del capitulo quedo en la carpeta compartida", "ham"),
        ("urgente, tu cuenta sera cerrada si no verificas ahora", "spam"),
    ]
    val_rows = [
        ("nos vemos en la presentacion del miercoles", "ham"),
        ("reclama tu bono secreto antes de medianoche", "spam"),
    ]
    test_rows = [
        ("puedes subir las diapositivas al drive?", "ham"),
        ("premio inmediato disponible, envia tu contrasena", "spam"),
    ]
    for name, rows in {
        "classification_demo_train.csv": train_rows,
        "classification_demo_val.csv": val_rows,
        "classification_demo_test.csv": test_rows,
    }.items():
        with (data_dir / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "label"])
            writer.writerows(rows)


def read_labeled_csv(path: Path, text_column: str, label_column: str) -> List[Tuple[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    rows: List[Tuple[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if text_column not in (reader.fieldnames or []):
            raise ValueError(f"{path} is missing text column '{text_column}'")
        if label_column not in (reader.fieldnames or []):
            raise ValueError(f"{path} is missing label column '{label_column}'")
        for row_num, row in enumerate(reader, start=2):
            text = (row.get(text_column) or "").strip()
            label = (row.get(label_column) or "").strip()
            if not text or not label:
                raise ValueError(f"{path}:{row_num} has empty text or label")
            rows.append((text, label))
    if not rows:
        raise ValueError(f"{path} has no examples")
    return rows


def build_label_map(train_rows: Sequence[Tuple[str, str]]) -> Dict[str, int]:
    labels = sorted({label for _, label in train_rows})
    if len(labels) < 2:
        raise ValueError("Classification fine-tuning needs at least two labels")
    return {label: idx for idx, label in enumerate(labels)}


def encode_labels(rows: Sequence[Tuple[str, str]], label_to_id: Dict[str, int], split_name: str) -> List[Tuple[str, int]]:
    encoded: List[Tuple[str, int]] = []
    for text, label in rows:
        if label not in label_to_id:
            known = ", ".join(sorted(label_to_id))
            raise ValueError(f"Unknown label '{label}' in {split_name}. Known labels: {known}")
        encoded.append((text, label_to_id[label]))
    return encoded


def make_base_config(cfg: FineTuneConfig, checkpoint: dict | None) -> TrainConfig:
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
            f"max_length={cfg.max_length} cannot exceed model block_size={base_cfg.block_size}. "
            "Use a shorter max_length or a checkpoint trained with larger context."
        )
    base_cfg.block_size = max(base_cfg.block_size, cfg.max_length)
    base_cfg.precision = cfg.precision
    base_cfg.device = cfg.device
    return base_cfg


def load_base_model(cfg: FineTuneConfig, device: str) -> Tuple[GPTModel, TrainConfig, dict | None]:
    checkpoint_path = Path(cfg.base_checkpoint_path)
    checkpoint = None
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        base_cfg = make_base_config(cfg, checkpoint if isinstance(checkpoint, dict) else None)
        model = GPTModel(base_cfg)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Base checkpoint does not match GPTModel. "
                f"missing={missing[:5]} unexpected={unexpected[:5]}"
            )
        return model, base_cfg, checkpoint if isinstance(checkpoint, dict) else None

    print(f"[WARN] base checkpoint not found: {checkpoint_path}. Starting classifier from random GPT weights.")
    base_cfg = make_base_config(cfg, None)
    return GPTModel(base_cfg), base_cfg, None


def configure_trainable_layers(model: GPTForClassification, freeze_base: bool, unfreeze_last_block: bool) -> None:
    if freeze_base:
        for param in model.base_model.parameters():
            param.requires_grad = False
        if unfreeze_last_block and len(model.base_model.blocks) > 0:
            for param in model.base_model.blocks[-1].parameters():
                param.requires_grad = True
        for param in model.base_model.ln_f.parameters():
            param.requires_grad = True

    for param in model.classification_head.parameters():
        param.requires_grad = True


def configure_optimizer(model: nn.Module, cfg: FineTuneConfig) -> torch.optim.Optimizer:
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
def evaluate(model: GPTForClassification, loader: DataLoader, device: str, amp) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        with torch.autocast(device_type=amp.device_type, dtype=amp.dtype, enabled=amp.enabled):
            logits, loss = model(xb, yb)
        if loss is None:
            raise RuntimeError("Evaluation loss unexpectedly became None")
        total_loss += float(loss.item()) * xb.size(0)
        total_correct += int((logits.argmax(dim=-1) == yb).sum().item())
        total_examples += xb.size(0)
    return total_loss / max(1, total_examples), total_correct / max(1, total_examples)


def write_metrics_header(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "step", "train_loss", "val_loss", "val_accuracy", "lr"])


def append_metric(path: Path, epoch: int, step: int, train_loss: float, val_loss: float, val_acc: float, lr: float) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, step, train_loss, val_loss, val_acc, lr])


@torch.no_grad()
def write_predictions(
    model: GPTForClassification,
    rows: Sequence[Tuple[str, int]],
    dataset: ClassificationDataset,
    id_to_label: Dict[int, str],
    path: Path,
    device: str,
) -> None:
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model.eval()
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "true_label", "predicted_label", "confidence"])
        for (text, label_id), (xb, _) in zip(rows, loader):
            xb = xb.to(device)
            logits, _ = model(xb)
            probs = F.softmax(logits, dim=-1).squeeze(0)
            pred_id = int(probs.argmax().item())
            writer.writerow([text, id_to_label[label_id], id_to_label[pred_id], float(probs[pred_id].item())])


def build_run_dir(output_root: Path, run_name: str) -> Path:
    if run_name:
        return output_root / run_name
    return output_root / f"classifier_{time.strftime('%Y%m%d-%H%M%S')}"


def train_classifier(cfg: FineTuneConfig) -> Path:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    if cfg.create_demo_data:
        write_demo_data(Path(cfg.train_csv).parent)

    tokenizer = Tokenizer.from_file(cfg.tokenizer_path)
    pad_token_id = tokenizer.token_to_id("<pad>")
    if pad_token_id is None:
        pad_token_id = tokenizer.token_to_id("</s>")
    if pad_token_id is None:
        pad_token_id = 0
    eos_token_id = tokenizer.token_to_id("</s>")

    train_raw = read_labeled_csv(Path(cfg.train_csv), cfg.text_column, cfg.label_column)
    val_raw = read_labeled_csv(Path(cfg.val_csv), cfg.text_column, cfg.label_column)
    test_raw = read_labeled_csv(Path(cfg.test_csv), cfg.text_column, cfg.label_column)

    label_to_id = build_label_map(train_raw)
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    train_rows = encode_labels(train_raw, label_to_id, "train")
    val_rows = encode_labels(val_raw, label_to_id, "val")
    test_rows = encode_labels(test_raw, label_to_id, "test")

    device = select_device(cfg.device)
    base_model, base_cfg, checkpoint = load_base_model(cfg, device)
    base_model.to(device)
    model = GPTForClassification(base_model, num_classes=len(label_to_id)).to(device)
    configure_trainable_layers(model, cfg.freeze_base, cfg.unfreeze_last_block)
    amp = resolve_amp_settings(base_cfg, device)

    train_ds = ClassificationDataset(train_rows, tokenizer, cfg.max_length, pad_token_id, eos_token_id)
    val_ds = ClassificationDataset(val_rows, tokenizer, cfg.max_length, pad_token_id, eos_token_id)
    test_ds = ClassificationDataset(test_rows, tokenizer, cfg.max_length, pad_token_id, eos_token_id)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    optimizer = configure_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp.use_grad_scaler and device == "cuda"))

    run_dir = build_run_dir(Path(cfg.output_root), cfg.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    write_metrics_header(metrics_path)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "fine_tune_config": asdict(cfg),
                "base_model_config": asdict(base_cfg),
                "label_to_id": label_to_id,
                "pad_token_id": pad_token_id,
                "eos_token_id": eos_token_id,
                "base_checkpoint_loaded": checkpoint is not None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "label_to_id.json").write_text(json.dumps(label_to_id, indent=2), encoding="utf-8")

    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    print(f"[INFO] device={device} precision={cfg.precision} amp={amp.enabled} dtype={amp.dtype}")
    print(f"[INFO] labels={label_to_id}")
    print(f"[INFO] trainable_params={trainable:,} total_params={total:,}")
    print(f"[INFO] run_dir={run_dir}")

    best_val_acc = -1.0
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
                val_loss, val_acc = evaluate(model, val_loader, device, amp)
                append_metric(metrics_path, epoch, global_step, float(loss.item()), val_loss, val_acc, cfg.lr)
                print(
                    f"[epoch {epoch:02d} step {global_step:05d}] "
                    f"train_loss={float(loss.item()):.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
                )
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "base_config": asdict(base_cfg),
                            "fine_tune_config": asdict(cfg),
                            "label_to_id": label_to_id,
                        },
                        run_dir / "best_classifier_checkpoint.pth",
                    )

        val_loss, val_acc = evaluate(model, val_loader, device, amp)
        append_metric(metrics_path, epoch, global_step, float("nan"), val_loss, val_acc, cfg.lr)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "base_config": asdict(base_cfg),
                    "fine_tune_config": asdict(cfg),
                    "label_to_id": label_to_id,
                },
                run_dir / "best_classifier_checkpoint.pth",
            )
        print(f"[epoch {epoch:02d} done] val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loss, test_acc = evaluate(model, test_loader, device, amp)
    (run_dir / "test_metrics.json").write_text(
        json.dumps({"test_loss": test_loss, "test_accuracy": test_acc, "best_val_accuracy": best_val_acc}, indent=2),
        encoding="utf-8",
    )
    write_predictions(model, test_rows, test_ds, id_to_label, run_dir / "test_predictions.csv", device)
    print(f"[INFO] test_loss={test_loss:.4f} test_acc={test_acc:.4f}")
    return run_dir


def parse_args() -> FineTuneConfig:
    cfg = FineTuneConfig()
    parser = argparse.ArgumentParser(description="Chapter-6 style classification fine-tuning for PequeLLM.")
    parser.add_argument("--train-csv", type=str, default=cfg.train_csv)
    parser.add_argument("--val-csv", type=str, default=cfg.val_csv)
    parser.add_argument("--test-csv", type=str, default=cfg.test_csv)
    parser.add_argument("--text-column", type=str, default=cfg.text_column)
    parser.add_argument("--label-column", type=str, default=cfg.label_column)
    parser.add_argument("--tokenizer-path", type=str, default=cfg.tokenizer_path)
    parser.add_argument("--base-checkpoint-path", type=str, default=cfg.base_checkpoint_path)
    parser.add_argument("--output-root", type=str, default=cfg.output_root)
    parser.add_argument("--run-name", type=str, default=cfg.run_name)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--max-length", type=int, default=cfg.max_length)
    parser.add_argument("--max-epochs", type=int, default=cfg.max_epochs)
    parser.add_argument("--eval-interval", type=int, default=cfg.eval_interval)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--weight-decay", type=float, default=cfg.weight_decay)
    parser.add_argument("--grad-clip", type=float, default=cfg.grad_clip)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--device", type=str, default=cfg.device)
    parser.add_argument("--precision", type=str, default=cfg.precision)
    parser.add_argument("--train-full-model", action="store_true")
    parser.add_argument("--freeze-all-base", action="store_true")
    parser.add_argument("--create-demo-data", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=cfg.vocab_size)
    parser.add_argument("--n-embd", type=int, default=cfg.n_embd)
    parser.add_argument("--n-head", type=int, default=cfg.n_head)
    parser.add_argument("--n-layer", type=int, default=cfg.n_layer)
    parser.add_argument("--block-size", type=int, default=cfg.block_size)
    args = parser.parse_args()

    cfg.train_csv = args.train_csv
    cfg.val_csv = args.val_csv
    cfg.test_csv = args.test_csv
    cfg.text_column = args.text_column
    cfg.label_column = args.label_column
    cfg.tokenizer_path = args.tokenizer_path
    cfg.base_checkpoint_path = args.base_checkpoint_path
    cfg.output_root = args.output_root
    cfg.run_name = args.run_name
    cfg.batch_size = args.batch_size
    cfg.max_length = args.max_length
    cfg.max_epochs = args.max_epochs
    cfg.eval_interval = args.eval_interval
    cfg.lr = args.lr
    cfg.weight_decay = args.weight_decay
    cfg.grad_clip = args.grad_clip
    cfg.seed = args.seed
    cfg.device = args.device
    cfg.precision = args.precision
    cfg.freeze_base = not args.train_full_model
    cfg.unfreeze_last_block = not args.freeze_all_base
    cfg.create_demo_data = args.create_demo_data
    cfg.vocab_size = args.vocab_size
    cfg.n_embd = args.n_embd
    cfg.n_head = args.n_head
    cfg.n_layer = args.n_layer
    cfg.block_size = args.block_size
    return cfg


if __name__ == "__main__":
    train_classifier(parse_args())
