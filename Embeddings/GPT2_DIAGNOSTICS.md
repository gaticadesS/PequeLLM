# GPT2 diagnostics playbook

This document maps your boss requirements to concrete outputs from `Embeddings/emb_gpt2.py`.

## Automatic presentation output

At the end of each run, `emb_gpt2.py` now auto-generates:

- A PDF slide-style report: `<run_dir>/<run_name>_presentation.pdf`
- A short markdown summary: `<run_dir>/presentation_summary.md`

This is created from training logs, gradient logs, UMAP snapshots, config validation, and grokking heuristic outputs.

You can also regenerate the presentation manually from any existing run:

```powershell
python Embeddings/presentation_report.py --run-dir Embeddings/artifacts_gpt2/boss-review-v1
```

## 1) Validation of embedding size

When training starts, the script validates:

- `n_embd % n_head == 0`
- `head_dim = n_embd / n_head` (recommended range check)
- token embedding memory footprint in FP32/FP16

Output files:

- `config_validation.json`
- `parameter_guide.md`

## 2) Understand what each parameter means and why it was chosen

The script writes `parameter_guide.md` with:

- meaning of each hyperparameter
- practical reason for default values
- full raw config in JSON

Key defaults for GPT-2 Small (see Emb_gptMed.py for the Medium defaults):

- `batch_size=16`: compromise between noisy gradients and memory.
- `block_size=128`: longer context than toy models.
- `n_embd=768, n_head=12`: `head_dim=64`, same as the original GPT-2.
- `n_layer=12`: same depth as GPT-2 small.
- `precision=auto`: usa mixed precision en CUDA (bf16/fp16) para acelerar y reducir memoria.
- `lr_max=3e-4, lr_min=3e-5, warmup+cosine`: smoother optimization.
- `weight_decay=0.1`: regularization.
- `grad_clip=1.0`: controls spikes.

## 3) Gradient movement in each layer

The script logs per-layer gradient norm and delta vs previous log step.

Output file:

- `grad_norms_by_layer.csv`

Columns:

- `iter`
- `layer_name`
- `grad_norm`
- `delta_vs_prev`

This directly supports "how gradient moves when comparing layers".

## 4) Parameter optimization

The training loop already includes stronger optimization defaults:

- AdamW with decoupled weight decay groups
- warmup + cosine LR schedule
- gradient clipping

This is a practical upgrade from fixed-LR baseline.

## 5) Gradient norm per layer

Same source as section 3 (`grad_norms_by_layer.csv`), plus global norm:

- `global_grad_norm` column in `train_metrics.csv`.

## 6) UMAP across iterations

At `umap_interval`, token embedding snapshots are projected to 2D.

Output folder:

- `umap_snapshots/`

Output files:

- `umap_tokens_iter_XXXXXX.csv`
- `umap_tokens_iter_XXXXXX.png` (if `matplotlib` is installed)

If `umap-learn` is not installed, it falls back to PCA automatically.

## 7) Grokking signal in embeddings (heuristic)

The script computes:

- train/val history
- probe semantic separation (`probe_gap`)
- delayed val improvement after train loss becomes low

Output file:

- `grokking_heuristic.json`

Important: this is a heuristic, not a proof of grokking. It is useful for tracking signs of "late generalization".

## How to run (example)

```powershell
python Embeddings/emb_gpt2.py `
  --run-name boss-review-v1 `
  --max-iters 25000 `
  --eval-interval 500 `
  --grad-log-interval 20 `
  --umap-interval 2000
```

If you want to disable automatic presentation generation:

```powershell
python Embeddings/emb_gpt2.py --skip-presentation
```

Quick smoke run without touching your real checkpoint:

```powershell
python Embeddings/emb_gpt2.py `
  --run-name smoke `
  --max-iters 5 `
  --eval-interval 1 `
  --eval-batches 2 `
  --precision auto `
  --checkpoint-path Embeddings/artifacts_gpt2/smoke_checkpoint.pth `
  --no-resume
```

Important precision note:

- Token IDs should remain integer (`uint16` in `.bin`, converted to `int64` tensors for `nn.Embedding` lookup).
- Mixed precision (`fp16`/`bf16`) should be used for model weights/activations, not for token IDs.
