"""Prepara datos de instrucciones EN ESPAÑOL para fine-tuning.

Baja un dataset Alpaca en español desde HuggingFace (por defecto
``bertin-project/alpaca-spanish``), lo normaliza al esquema
``{instruction, input, output}`` y genera splits train/val/test en el formato
que ``finetune_instruction.py`` espera.

Escribe ``instruction_es_{train,val,test}.json`` en FineTuning/data/ — con un
prefijo ``_es`` para NO pisar los datos en inglés del libro.

Uso (normalmente vía ``./run.sh prepare-instr-es``):
    python FineTuning/prepare_instruction_data_es.py --limit 0    # dataset completo
    python FineTuning/prepare_instruction_data_es.py --limit 2000 # submuestra rápida
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "bertin-project/alpaca-spanish"


def normalize_rows(dataset) -> List[Dict[str, str]]:
    """Convierte el dataset HF en una lista de {instruction, input, output}.

    Tolera nombres de campo alternativos y descarta filas sin instrucción/salida.
    """
    rows: List[Dict[str, str]] = []
    for entry in dataset:
        instruction = (entry.get("instruction") or "").strip()
        # algunos datasets usan 'input'/'context' y 'output'/'response'
        input_text = (entry.get("input") or entry.get("context") or "").strip()
        output = (entry.get("output") or entry.get("response") or "").strip()
        if not instruction or not output:
            continue
        rows.append({"instruction": instruction, "input": input_text, "output": output})
    return rows


def write_splits(data: List[Dict[str, str]], out_dir: Path, dataset_id: str) -> Dict:
    # Split 85/10/5 preservando el orden (ya barajado antes).
    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    splits = {
        "train": data[:train_portion],
        "test": data[train_portion : train_portion + test_portion],
        "val": data[train_portion + test_portion :],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        (out_dir / f"instruction_es_{split}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    summary = {
        "source": dataset_id,
        "language": "es",
        "split_strategy": "shuffled 85/10/5 split: train/test/val",
        "total_examples": len(data),
        "splits": {
            split: {
                "examples": len(rows),
                "path": str(out_dir / f"instruction_es_{split}.json"),
            }
            for split, rows in splits.items()
        },
        "fields": ["instruction", "input", "output"],
    }
    (out_dir / "instruction_es_dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepara datos de instrucciones Alpaca en español.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="ID del dataset en HuggingFace.")
    parser.add_argument("--split", default="train", help="Split del dataset HF a usar.")
    parser.add_argument("--limit", type=int, default=0, help="Máx. ejemplos (0 = todos).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "FineTuning" / "data"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[INFO] descargando dataset '{args.dataset}' (split={args.split})…")
    dataset = load_dataset(args.dataset, split=args.split)

    rows = normalize_rows(dataset)
    print(f"[INFO] ejemplos válidos tras normalizar: {len(rows)}")

    random.seed(args.seed)
    random.shuffle(rows)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
        print(f"[INFO] submuestreado a {len(rows)} ejemplos (limit={args.limit}).")

    summary = write_splits(rows, Path(args.out_dir), args.dataset)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
