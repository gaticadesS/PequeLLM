from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch
from torch.nn import functional as F
from tokenizers import Tokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
EMB_DIR = REPO_ROOT / "Embeddings"
FT_DIR = REPO_ROOT / "FineTuning"
for path in (EMB_DIR, FT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from emb_gpt2 import GPTModel, TrainConfig, select_device  # noqa: E402
from finetune_instruction import format_input  # noqa: E402


def build_train_config(raw: Dict) -> TrainConfig:
    cfg = TrainConfig()
    for key, value in raw.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def load_instruction_model(checkpoint_path: Path, device: str) -> GPTModel:
    raw = torch.load(checkpoint_path, map_location=device)
    base_cfg = build_train_config(raw["base_config"])
    model = GPTModel(base_cfg)
    model.load_state_dict(raw["model"])
    model.to(device)
    model.eval()
    return model


def sample_next_token(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k > 0:
        values, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
        cutoff = values[:, [-1]]
        logits = logits.masked_fill(logits < cutoff, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(
    model: GPTModel,
    tokenizer: Tokenizer,
    instruction: str,
    input_text: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: str,
) -> str:
    entry = {"instruction": instruction, "input": input_text, "output": ""}
    prompt = format_input(entry) + "\n\n### Respuesta:\n"
    prompt_ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.block_size :]
        logits, _ = model(idx_cond)
        next_id = sample_next_token(logits[:, -1, :], temperature=temperature, top_k=top_k)
        idx = torch.cat((idx, next_id), dim=1)

    text = tokenizer.decode(idx[0].tolist())
    return text[len(prompt) :].strip() if text.startswith(prompt) else text


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a response with an instruction-tuned PequeLLM checkpoint.")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--tokenizer-path", default=str(REPO_ROOT / "tokenizer-culturax-es-hf.json"))
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--input", default="")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = select_device(args.device)
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    model = load_instruction_model(Path(args.checkpoint_path), device)
    response = generate(
        model=model,
        tokenizer=tokenizer,
        instruction=args.instruction,
        input_text=args.input,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )
    print(json.dumps({"instruction": args.instruction, "input": args.input, "response": response}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
