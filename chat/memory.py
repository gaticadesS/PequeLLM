"""Memoria conversacional de corto plazo para PequeLLM.

Es una capa **externa** al modelo: no toca la arquitectura del Transformer ni
requiere reentrenar. El modelo es *stateless* (solo ve `idx[:, -block_size:]`),
así que toda la "memoria" se gestiona aquí, armando el prompt de cada turno con
los mensajes recientes que quepan en la ventana de contexto.

Notas de diseño, condicionadas por nuestro modelo:
- Es un modelo BASE (no instruct), entrenado con texto crudo de CulturaX. Por eso
  usamos un formato PLANO de diálogo (``Usuario:`` / ``Asistente:``) en lugar de
  scaffolding tipo ``### Instruction``: queda más cerca de su distribución.
- ``block_size`` es chico (128/256). El control de presupuesto de tokens es lo
  más importante: seleccionamos turnos del más reciente hacia atrás hasta llenar
  el presupuesto, descartando los viejos.

El módulo no depende de Streamlit: recibe listas de mensajes y devuelve datos,
para que sea testeable y reutilizable.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

USER_PREFIX = "Usuario:"
ASSISTANT_PREFIX = "Asistente:"


@dataclass
class ChatMessage:
    role: str  # "user" | "assistant"
    content: str


# ── Almacén de memoria (seam para futuro largo plazo) ───────────────────────
class MemoryStore(ABC):
    """Interfaz de almacenamiento de memoria por sesión.

    Hoy el dashboard usa ``st.session_state`` directamente, pero esta ABC deja
    el punto de extensión para Fase 2 (persistencia/multi-sesión) y Fase 3
    (``LongTermMemoryStore`` con base vectorial / recuperación semántica).
    """

    @abstractmethod
    def add_message(self, session_id: str, message: ChatMessage) -> None: ...

    @abstractmethod
    def get_messages(self, session_id: str) -> List[ChatMessage]: ...

    @abstractmethod
    def clear(self, session_id: str) -> None: ...


class ShortTermMemoryStore(MemoryStore):
    """Implementación mínima en RAM: dict de session_id -> lista de mensajes."""

    def __init__(self) -> None:
        self._sessions: Dict[str, List[ChatMessage]] = {}

    def add_message(self, session_id: str, message: ChatMessage) -> None:
        self._sessions.setdefault(session_id, []).append(message)

    def get_messages(self, session_id: str) -> List[ChatMessage]:
        return list(self._sessions.get(session_id, []))

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


# ── Utilidades de tokens y prompt ───────────────────────────────────────────
def count_tokens(tokenizer, text: str) -> int:
    """Número de tokens que produce ``text`` con el tokenizer dado."""
    return len(tokenizer.encode(text).ids)


def _format_turn(role: str, content: str) -> str:
    prefix = USER_PREFIX if role == "user" else ASSISTANT_PREFIX
    return f"{prefix} {content.strip()}"


def build_chat_prompt(
    messages: List[Dict[str, str]],
    tokenizer,
    block_size: int,
    reserved_generation_tokens: int,
    safety_margin: int = 8,
    max_recent_turns: Optional[int] = None,
) -> Tuple[str, List[int], Dict[str, int]]:
    """Arma el prompt de chat respetando el presupuesto de tokens.

    Selecciona mensajes del más reciente hacia atrás hasta llenar el presupuesto
    (priorizando contexto reciente), los reconstruye en orden cronológico y
    termina con ``Asistente:`` como pista de generación.

    Args:
        messages: lista de dicts ``{"role": "user"|"assistant", "content": str}``
            en orden cronológico. El último debe ser del usuario (turno actual).
        tokenizer: tokenizer HF (``tokenizers.Tokenizer``).
        block_size: ventana de contexto del modelo.
        reserved_generation_tokens: tokens que se reservan para la respuesta.
        safety_margin: colchón extra para no rozar el límite.
        max_recent_turns: si se indica, tope duro de mensajes a considerar.

    Returns:
        ``(prompt_text, prompt_ids, stats)`` con
        ``stats = {prompt_tokens, budget, turns_included, turns_total}``.
    """
    budget = max(block_size - reserved_generation_tokens - safety_margin, 16)
    cue = ASSISTANT_PREFIX  # el modelo continúa después de esto

    candidates = messages if max_recent_turns is None else messages[-max_recent_turns:]

    # Seleccionar del más reciente hacia atrás mientras quepa.
    selected: List[Dict[str, str]] = []
    running = count_tokens(tokenizer, cue)
    for msg in reversed(candidates):
        turn_text = _format_turn(msg["role"], msg["content"])
        turn_tokens = count_tokens(tokenizer, turn_text + "\n")
        if selected and running + turn_tokens > budget:
            break
        selected.append(msg)
        running += turn_tokens

    selected.reverse()  # de vuelta a orden cronológico

    body = "\n".join(_format_turn(m["role"], m["content"]) for m in selected)
    prompt_text = f"{body}\n{cue}" if body else cue
    prompt_ids = tokenizer.encode(prompt_text).ids

    stats = {
        "prompt_tokens": len(prompt_ids),
        "budget": budget,
        "turns_included": len(selected),
        "turns_total": len(messages),
    }
    return prompt_text, prompt_ids, stats


def clean_output(text: str) -> str:
    """Recorta texto que el modelo genera de más (inicio de un nuevo turno).

    El modelo base tiende a seguir generando ``Usuario:`` u otros marcadores tras
    su respuesta. Cortamos en el primer marcador de nuevo turno y limpiamos.
    """
    markers = [f"\n{USER_PREFIX}", f"\n{ASSISTANT_PREFIX}", "\nUsuario:", "\n###"]
    cut = len(text)
    for marker in markers:
        idx = text.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut].strip()
