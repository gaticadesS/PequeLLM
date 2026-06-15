"""Dashboard Streamlit para "conversar" con un checkpoint de PequeLLM.

El modelo no conversa de verdad: hace *completación de tokens*. Esta app lo
envuelve en una interfaz tipo chat para poder demostrarlo desde cualquier lado
(exponiendo el puerto 8501 con cloudflared o ngrok).

Reutiliza la lógica de inferencia que ya existe en
``Embeddings/generate_prompt.py`` — en particular ``load_model`` (que infiere la
configuración del checkpoint, así que sirve igual para Small y Medium) y
``sample_next_token``.

IMPORTANTE: esta app es SOLO-LECTURA sobre el checkpoint. Nunca escribe el .pth
ni llama a save_checkpoint; solo carga el modelo en modo eval y genera texto.

Uso (dentro del contenedor ROCm en renna):
    streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501

Normalmente se lanza con ``./run.sh dashboard`` (ver dashboard/README.md).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List

import streamlit as st
import torch
from tokenizers import Tokenizer

# ── Rutas e imports del repo ────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
EMB_DIR = REPO_ROOT / "Embeddings"
for _p in (str(EMB_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from emb_gpt2 import select_device  # noqa: E402
from generate_prompt import load_model, sample_next_token  # noqa: E402
from chat.memory import build_chat_prompt, clean_output  # noqa: E402

# Plantilla de instrucción en español (misma fuente que el fine-tuning, para
# que el prompt en el chat coincida con lo que el modelo afinado aprendió).
sys.path.insert(0, str(REPO_ROOT / "FineTuning"))
from finetune_instruction import format_input as format_instruction_input  # noqa: E402


DEFAULT_TOKENIZER = str(REPO_ROOT / "tokenizer-culturax-es-hf.json")
# Carpeta donde el entrenamiento guarda los checkpoints dentro del contenedor.
DATA_DIR = Path(os.environ.get("PEQUELLM_DATA_DIR", "/workspace/data"))


# ── Descubrimiento de checkpoints ───────────────────────────────────────────
def is_instruction_checkpoint(path: str) -> bool:
    """Heurística: ¿es un modelo afinado para instrucciones?"""
    p = path.lower()
    return "instruction" in p or "instruct" in p


def _label_for(path: Path) -> str:
    name = path.name.lower()
    full = str(path).lower()
    if is_instruction_checkpoint(full):
        # Distinguir por la carpeta de la corrida (instruction_<fecha>).
        return f"Instruct ES (afinado)"
    if "medium" in name or "med" in name:
        return "GPT-2 Medium"
    if "pesado" in name or "small" in name:
        return "GPT-2 Small"
    return path.name


def discover_checkpoints() -> Dict[str, str]:
    """Mapea 'etiqueta visible' -> ruta absoluta de cada .pth encontrado.

    Escanea DATA_DIR de forma recursiva (para detectar automáticamente los
    modelos afinados en artifacts_instruction/<run>/) y la raíz del repo.
    Ordena poniendo primero los modelos afinados (default para la demo).
    """
    paths = set()
    if DATA_DIR.is_dir():
        paths.update(DATA_DIR.rglob("*.pth"))
    if REPO_ROOT.is_dir():
        paths.update(REPO_ROOT.glob("*.pth"))

    # Afinados primero, luego por nombre.
    ordered = sorted(paths, key=lambda p: (not is_instruction_checkpoint(str(p)), str(p)))
    found: Dict[str, str] = {}
    for path in ordered:
        found.setdefault(_label_for(path), str(path))
    return found


# ── Carga (cacheada) de modelo + tokenizer ──────────────────────────────────
@st.cache_resource(show_spinner="Cargando modelo en memoria…")
def get_model_and_tokenizer(checkpoint_path: str, tokenizer_path: str, device: str):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    model = load_model(Path(checkpoint_path), device=device)
    return model, tokenizer


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


# ── Generación con streaming token-a-token ──────────────────────────────────
STOP_SEQUENCES = ("\nUsuario:", "\nAsistente:", "\n###")


@torch.no_grad()
def generate_reply(
    model,
    tokenizer: Tokenizer,
    prompt_ids: List[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: str,
    eos_id: int | None = None,
) -> str:
    """Genera la respuesta completa (sólo la parte nueva, ya limpia).

    Para de generar cuando: alcanza ``max_new_tokens``, el modelo emite ``</s>``
    (eos_id) o aparece el inicio de un nuevo turno (``\\nUsuario:`` etc.).
    """
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated: List[int] = []
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.block_size:]
        logits, _ = model(idx_cond)
        next_token = sample_next_token(logits[:, -1, :], temperature=temperature, top_k=top_k)
        token_id = int(next_token.item())
        if eos_id is not None and token_id == eos_id:
            break
        idx = torch.cat((idx, next_token), dim=1)
        generated.append(token_id)
        text = tokenizer.decode(generated)
        # Parada temprana si el modelo empezó a escribir un nuevo turno.
        if any(stop in text for stop in STOP_SEQUENCES):
            break
    return clean_output(tokenizer.decode(generated))


# ── UI ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="PequeLLM Chat", page_icon="💬", layout="centered")
st.title("💬 PequeLLM")
st.caption(
    "Demo: por dentro el modelo hace *completación de tokens*. Con **memoria "
    "conversacional** se le arman los turnos recientes que caben en su ventana "
    "para simular un chat. No tiene memoria real ni fue afinado para conversar."
)

device = select_device("auto")
checkpoints = discover_checkpoints()

with st.sidebar:
    st.header("⚙️ Configuración")

    if checkpoints:
        etiqueta = st.selectbox("Modelo (checkpoint)", list(checkpoints.keys()))
        checkpoint_path = checkpoints[etiqueta]
    else:
        st.warning(
            f"No se encontraron checkpoints (.pth) en {DATA_DIR} ni en el repo. "
            "Indica la ruta manualmente."
        )
        checkpoint_path = st.text_input(
            "Ruta del checkpoint", value=str(DATA_DIR / "pequellm_medium_checkpoint.pth")
        )

    # Tokenizer fijo del proyecto: se muestra (para que se vea que es el nuestro)
    # pero deshabilitado para que no se pueda cambiar ni romper.
    tokenizer_path = DEFAULT_TOKENIZER
    st.text_input(
        "Tokenizer", value=Path(DEFAULT_TOKENIZER).name, disabled=True,
        help="Tokenizer BPE en español del proyecto (entrenado sobre CulturaX). Fijo.",
    )

    st.divider()
    max_new_tokens = st.slider("Tokens a generar", 16, 400, 120, step=8)
    temperature = st.slider("Temperatura", 0.0, 1.5, 0.9, step=0.05)
    top_k = st.slider("top-k (0 = desactivado)", 0, 200, 50, step=5)
    instruction_mode = st.checkbox(
        "Modo instrucción (modelo afinado)", value=is_instruction_checkpoint(checkpoint_path),
        help="Usa el formato '### Instrucción / ### Respuesta' con el que se afinó "
             "el modelo. Se activa solo al elegir un checkpoint afinado. Cada mensaje "
             "es una instrucción independiente (sin historial).",
    )
    use_memory = st.checkbox(
        "Memoria conversacional", value=True, disabled=instruction_mode,
        help="Arma el prompt con los turnos recientes que quepan en la ventana "
             "(formato Usuario:/Asistente:). Ignorado en modo instrucción.",
    )

    st.divider()
    if st.button("🧹 Limpiar conversación"):
        st.session_state.messages = []
        st.rerun()

# Cargar el modelo seleccionado (cacheado por ruta).
try:
    model, tokenizer = get_model_and_tokenizer(checkpoint_path, tokenizer_path, device)
except FileNotFoundError as exc:
    st.error(f"No se pudo cargar el modelo: {exc}")
    st.stop()
except Exception as exc:  # noqa: BLE001 — mostrar cualquier fallo de carga en la UI
    st.error(f"Error cargando modelo/tokenizer: {exc}")
    st.stop()

with st.sidebar:
    st.divider()
    st.subheader("ℹ️ Modelo cargado")
    cfg = model.cfg
    st.markdown(
        f"- **device**: `{device}`\n"
        f"- **parámetros**: ~{count_parameters(model) / 1e6:.1f} M\n"
        f"- **n_layer**: {cfg.n_layer} · **n_head**: {cfg.n_head}\n"
        f"- **n_embd**: {cfg.n_embd} · **block_size**: {cfg.block_size}\n"
        f"- **vocab_size**: {cfg.vocab_size}"
    )

# ── Estado e historial del chat ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

eos_id = tokenizer.token_to_id("</s>")

prompt = st.chat_input("Escribe un mensaje…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Construcción del prompt según el modo.
    if instruction_mode:
        # Modelo afinado: formato Alpaca en español, una instrucción por turno.
        prompt_text = format_instruction_input({"instruction": prompt, "input": ""}) + "\n\n### Respuesta:\n"
        prompt_ids = tokenizer.encode(prompt_text).ids
        stats = {
            "prompt_tokens": len(prompt_ids),
            "budget": model.cfg.block_size,
            "turns_included": 1,
            "turns_total": len(st.session_state.messages),
        }
    elif use_memory:
        prompt_text, prompt_ids, stats = build_chat_prompt(
            messages=st.session_state.messages,
            tokenizer=tokenizer,
            block_size=model.cfg.block_size,
            reserved_generation_tokens=max_new_tokens,
        )
    else:
        prompt_text = prompt
        prompt_ids = tokenizer.encode(prompt_text).ids
        stats = {
            "prompt_tokens": len(prompt_ids),
            "budget": model.cfg.block_size,
            "turns_included": 1,
            "turns_total": len(st.session_state.messages),
        }

    with st.chat_message("assistant"):
        if not prompt_ids:
            respuesta = "_(El prompt no produjo tokens. Intenta con otro texto.)_"
            st.markdown(respuesta)
        elif max(prompt_ids) >= model.cfg.vocab_size:
            respuesta = (
                f"⚠️ El prompt contiene el token id {max(prompt_ids)} pero el "
                f"vocab_size del modelo es {model.cfg.vocab_size}. "
                "Usa el tokenizer que corresponde al checkpoint."
            )
            st.markdown(respuesta)
        else:
            with st.spinner("PequeLLM está escribiendo…"):
                respuesta = generate_reply(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_ids=prompt_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    device=device,
                    eos_id=eos_id,
                )
            if not respuesta:
                respuesta = "_(El modelo no generó texto útil. Prueba a subir 'Tokens a generar' o cambiar el prompt.)_"
            st.markdown(respuesta)

    # Panel de depuración: muestra el mecanismo de memoria.
    with st.expander("🔍 Depuración (prompt y memoria)"):
        over = stats["prompt_tokens"] > stats["budget"]
        st.markdown(
            f"- **Mensajes en sesión**: {stats['turns_total']}\n"
            f"- **Turnos incluidos en el prompt**: {stats['turns_included']}\n"
            f"- **Tokens del prompt**: {stats['prompt_tokens']} / "
            f"presupuesto {stats['budget']} {'⚠️ excede' if over else '✅'}\n"
            f"- **block_size**: {model.cfg.block_size} · **reservado para respuesta**: {max_new_tokens}"
        )
        if stats["budget"] <= 32:
            st.warning(
                "El presupuesto para historial es muy pequeño. Baja 'Tokens a generar' "
                "para dejar más espacio a la memoria (típico con block_size=128)."
            )
        st.caption("Prompt exacto enviado al modelo:")
        st.code(prompt_text, language="text")

    st.session_state.messages.append({"role": "assistant", "content": respuesta})
