#!/usr/bin/env bash
# PequeLLM ROCm container runner.
#
# Uso desde el directorio raiz del repo:
#     ./run.sh                # smoke test + entrenar (default)
#     ./run.sh build          # solo construir la imagen
#     ./run.sh smoke          # solo verificar que la GPU se ve
#     ./run.sh synth          # generar train.bin/val.bin sinteticos
#     ./run.sh prepare-data   # correr Tokenizador/prepare_data/prepare_data.py (CulturaX, lento)
#     ./run.sh train [args]   # entrenar pasando args extras a Emb_gptMed.py
#     ./run.sh prepare-instr-es  # bajar Alpaca en espanol y generar splits (necesita red)
#     ./run.sh finetune-instr [args]  # fine-tuning de instrucciones sobre el modelo Medium
#     ./run.sh dashboard      # chat web (Streamlit) para probar el modelo, puerto 8501
#     ./run.sh shell          # bash interactivo dentro del contenedor
#
# Variables de entorno opcionales:
#     IMAGE      (default: pequellm:rocm)
#     DATA_VOL   (default: pequellm-data)
#     CACHE_VOL  (default: pequellm-cache)
#     RUNTIME    (auto: podman si esta, sino docker)
#     DASHBOARD_PORT (default: 8501)
#     MEDIUM_CKPT    (default: pequellm_medium_checkpoint.pth)  base para finetune-instr

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-pequellm:rocm}"
DATA_VOL="${DATA_VOL:-pequellm-data}"
CACHE_VOL="${CACHE_VOL:-pequellm-cache}"

# --- 1. Elegir runtime: podman > docker -------------------------------------
if [[ -n "${RUNTIME:-}" ]]; then
    :
elif command -v podman >/dev/null 2>&1; then
    RUNTIME="podman"
elif command -v docker >/dev/null 2>&1; then
    RUNTIME="docker"
else
    echo "ERROR: no se encontro podman ni docker en \$PATH." >&2
    echo "       En renna deberia estar instalado podman. Si no, contacta al admin." >&2
    exit 1
fi

# --- 2. Helpers para imagen y volumenes -------------------------------------
image_exists() {
    "$RUNTIME" image exists "$IMAGE" 2>/dev/null
}

ensure_image() {
    if ! image_exists; then
        echo "[run.sh] La imagen '$IMAGE' no existe. Construyendola ahora..." >&2
        "$RUNTIME" build -t "$IMAGE" "$REPO_DIR"
    fi
}

ensure_volume() {
    local name="$1"
    if ! "$RUNTIME" volume exists "$name" 2>/dev/null; then
        echo "[run.sh] Creando volumen persistente '$name'." >&2
        "$RUNTIME" volume create "$name" >/dev/null
    fi
}

ensure_volumes() {
    ensure_volume "$DATA_VOL"
    ensure_volume "$CACHE_VOL"
}

# --- 3. Banderas comunes para `podman run` ----------------------------------
# Pasamos /dev/kfd y /dev/dri para acceso a GPU AMD via ROCm.
# --group-add keep-groups conserva los grupos del host del usuario que invoca.
# --security-opt label=disable evita que SELinux relabele los devices del host.
# --shm-size 8g: PyTorch DataLoader usa /dev/shm para IPC entre workers.
common_run_args=(
    --rm
    --device /dev/kfd
    --device /dev/dri
    --group-add keep-groups
    --security-opt label=disable
    --shm-size 8g
    -v "$DATA_VOL:/workspace/data"
    -v "$CACHE_VOL:/workspace/cache"
    -v "$REPO_DIR:/workspace/repo"
    -w /workspace/repo
    -e HF_HOME=/workspace/cache
    -e PYTHONUNBUFFERED=1
    -e HF_TOKEN="${HF_TOKEN:-}"
)

# Anadimos -it si stdout es un terminal, sino solo -i (ej. en logs/CI).
if [[ -t 1 ]]; then
    common_run_args+=( -it )
else
    common_run_args+=( -i )
fi

# --- 4. Subcomandos ---------------------------------------------------------
cmd="${1:-default}"
[[ $# -gt 0 ]] && shift || true

case "$cmd" in
    build)
        "$RUNTIME" build -t "$IMAGE" "$REPO_DIR"
        ;;

    smoke)
        ensure_image
        ensure_volumes
        "$RUNTIME" run "${common_run_args[@]}" "$IMAGE" \
            python /workspace/repo/scripts/check_gpu.py
        ;;

    synth)
        ensure_image
        ensure_volumes
        "$RUNTIME" run "${common_run_args[@]}" --network none "$IMAGE" \
            python /workspace/repo/scripts/make_synthetic_bins.py "$@"
        ;;

    prepare-data)
        ensure_image
        ensure_volumes
        # prepare_data.py escribe train.bin/val.bin al CWD; lo corremos desde /workspace/data.
        "$RUNTIME" run "${common_run_args[@]}" -w /workspace/data "$IMAGE" \
            python /workspace/repo/Tokenizador/prepare_data/prepare_data.py "$@"
        ;;

    train|default)
        ensure_image
        ensure_volumes
        if [[ "$cmd" == "default" ]]; then
            echo "[run.sh] Paso 1/2: smoke test de GPU..." >&2
            "$RUNTIME" run "${common_run_args[@]}" "$IMAGE" \
                python /workspace/repo/scripts/check_gpu.py
            echo "[run.sh] Paso 2/2: entrenamiento GPT-2 Medium (Embeddings/Emb_gptMed.py)" >&2
        fi
        "$RUNTIME" run "${common_run_args[@]}" "$IMAGE" \
            python /workspace/repo/Embeddings/Emb_gptMed.py \
                --train-bin       /workspace/data/train.bin \
                --val-bin         /workspace/data/val.bin \
                --checkpoint-path /workspace/data/pequellm_medium_checkpoint.pth \
                --output-root     /workspace/data/artifacts_gpt2 \
                "$@"
        ;;
    pruebas)
        ensure_image
        ensure_volumes
        # Ejecuta el script de pruebas interactivo
        "$RUNTIME" run "${common_run_args[@]}" "$IMAGE" \
            python /workspace/repo/Embeddings/pruebas.py
        ;;
    data)
        ensure_image
        ensure_volumes
        # Ejecuta el script de pruebas interactivo
        "$RUNTIME" run "${common_run_args[@]}" "$IMAGE" \
            python /workspace/repo/data.py
        ;;

    prepare-instr-es)
        ensure_image
        ensure_volumes
        # Descarga el dataset Alpaca en español y genera los splits
        # instruction_es_{train,val,test}.json en FineTuning/data/ (necesita red).
        "$RUNTIME" run "${common_run_args[@]}" "$IMAGE" \
            python /workspace/repo/FineTuning/prepare_instruction_data_es.py "$@"
        ;;

    finetune-instr)
        ensure_image
        ensure_volumes
        # Fine-tuning de instrucciones sobre el modelo Medium. Solo-lectura sobre
        # el checkpoint base; escribe el modelo afinado en el volumen de datos.
        # MEDIUM_CKPT permite apuntar a otro checkpoint base.
        # Args extra se pasan tal cual (p. ej. --max-length 256 --batch-size 8 --max-epochs 50).
        "$RUNTIME" run "${common_run_args[@]}" "$IMAGE" \
            python /workspace/repo/FineTuning/finetune_instruction.py \
                --base-checkpoint-path /workspace/data/"${MEDIUM_CKPT:-pequellm_medium_checkpoint.pth}" \
                --train-json /workspace/repo/FineTuning/data/instruction_es_train.json \
                --val-json   /workspace/repo/FineTuning/data/instruction_es_val.json \
                --test-json  /workspace/repo/FineTuning/data/instruction_es_test.json \
                --output-root /workspace/data/artifacts_instruction \
                "$@"
        ;;

    dashboard)
        ensure_image
        ensure_volumes
        # Dashboard de chat (Streamlit). Solo-lectura sobre el checkpoint.
        # Publicamos el puerto para poder exponerlo con cloudflared/ngrok o
        # via SSH port-forwarding. Ver dashboard/README.md.
        # Para que sobreviva a cerrar SSH/laptop, correrlo dentro de tmux.
        PORT="${DASHBOARD_PORT:-8501}"
        echo "[run.sh] Dashboard en http://localhost:${PORT} (Ctrl-C para salir)" >&2
        "$RUNTIME" run "${common_run_args[@]}" -p "${PORT}:8501" "$IMAGE" \
            streamlit run /workspace/repo/dashboard/app.py \
                --server.address 0.0.0.0 --server.port 8501 \
                --server.headless true --browser.gatherUsageStats false
        ;;

    shell)
        ensure_image
        ensure_volumes
        "$RUNTIME" run "${common_run_args[@]}" "$IMAGE" bash
        ;;

    -h|--help|help)
        sed -n '2,22p' "$0" | sed 's/^# *//'
        ;;

    *)
        echo "ERROR: subcomando desconocido: '$cmd'" >&2
        echo "       Usa: $0 [build|smoke|synth|prepare-data|train|prepare-instr-es|finetune-instr|dashboard|shell|help|pruebas|data]" >&2
        exit 2
        ;;
esac