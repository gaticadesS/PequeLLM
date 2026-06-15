# syntax=docker/dockerfile:1.6
#
# PequeLLM ROCm/PyTorch container.
#
# Base image: docker.io/rocm/pytorch:latest
#   Verified 2026-04-24 on renna (AMD Strix Halo, gfx1151):
#     torch 2.10.0+rocm7.2.2, HIP 7.2.53211
#     torch.cuda.is_available() == True
#     torch.cuda.get_arch_list() incluye gfx1151 nativamente
#   Por lo tanto NO se requiere HSA_OVERRIDE_GFX_VERSION para Strix Halo
#   con esta imagen. Si en el futuro se baja a ROCm 6.3 o algo previo,
#   probablemente haya que reintroducir:
#       ENV HSA_OVERRIDE_GFX_VERSION=11.0.0
#
# Por que NO instalamos torch nosotros:
#   1. La imagen base ya trae torch compilado contra ROCm. Hacer
#      `pip install torch` desde el indice por defecto bajaria el wheel
#      de CPU desde PyPI y silenciosamente sobrescribiria el de ROCm.
#   2. requirements.txt del repo esta en UTF-16 LE y `pip install -r`
#      no lo puede parsear; ademas pinea torch==2.11.0.
FROM docker.io/rocm/pytorch:latest

WORKDIR /workspace/repo

ENV HF_HOME=/workspace/cache \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Solo dependencias NO-torch que el repo necesita. Si una version ya
# satisface el requerimiento, pip simplemente lo deja como esta.
RUN pip install \
        "numpy<3" \
        "tokenizers>=0.20" \
        "transformers>=4.45" \
        "datasets>=3.0" \
        "huggingface_hub>=0.25" \
        tqdm \
        matplotlib \
        umap-learn \
        scikit-learn \
        streamlit

# El repo se monta en runtime (-v $REPO:/workspace/repo) para que los
# estudiantes editen archivos en el host y ejecuten sin reconstruir la
# imagen. Por eso no hay COPY del codigo aqui.
#
# Sin ENTRYPOINT: dejamos que `podman run image <cmd>` se ejecute
# directamente. El CMD por defecto (sin argumentos) corre smoke + train.
# run.sh sobreescribe el CMD via subcomandos.
CMD ["bash", "-lc", "python /workspace/repo/scripts/check_gpu.py && python /workspace/repo/Embeddings/emb_gpt2.py --train-bin /workspace/data/train.bin --val-bin /workspace/data/val.bin --checkpoint-path /workspace/data/pequellm_pesado_checkpoint.pth --output-root /workspace/data/artifacts_gpt2"]
