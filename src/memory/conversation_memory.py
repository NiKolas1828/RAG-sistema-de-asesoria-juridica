# src/memory/conversation_memory.py
# ============================================================
# Memoria de Conversación Multi-turno
#
# Mantiene el historial de la sesión actual en formato
# [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
# para enviarlo al LLM como contexto de conversación previo.
#
# El historial vive en RAM (por sesión). Al reiniciar el sistema
# se limpia automáticamente — no hay persistencia entre sesiones.
# ============================================================

import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Almacena el historial de la sesión actual como lista de mensajes.

    Cada turno añade un par (user, assistant). El historial se recorta
    automáticamente al superar max_turns para no inflar el contexto
    enviado al LLM.

    Uso básico:
        memory = ConversationMemory(max_turns=5)
        memory.add_turn("¿Cuánto es la multa por no usar casco?", "La multa es de 15 SMLDV...")
        historial = memory.get_history()   # lista de dicts para la API
        memory.clear()                     # reinicia la sesión
    """

    def __init__(self, max_turns: int = 5):
        """
        Args:
            max_turns: Número máximo de turnos (pares user/assistant) a conservar.
                Al superar el límite se elimina el turno más antiguo (FIFO).
                Valor recomendado: 5 (≈ 10 mensajes en el historial).
        """
        self.max_turns = max_turns
        self._history: List[Dict[str, str]] = []

    # ─── API pública ─────────────────────────────────────────

    def add_turn(self, user_message: str, assistant_message: str) -> None:
        """
        Registra un par pregunta/respuesta en el historial.

        Si se supera max_turns, elimina el turno más antiguo (el par
        de mensajes en las posiciones [0] y [1]).

        Args:
            user_message:      Pregunta del ciudadano.
            assistant_message: Respuesta generada por el LLM.
        """
        self._history.append({"role": "user",      "content": user_message})
        self._history.append({"role": "assistant",  "content": assistant_message})

        # Recortar: cada turno son 2 mensajes
        max_messages = self.max_turns * 2
        if len(self._history) > max_messages:
            # Eliminar el turno más antiguo (primeros 2 elementos)
            self._history = self._history[2:]
            logger.debug(
                f"[Memory] Historial recortado a {len(self._history)} mensajes "
                f"(máx {max_messages})."
            )

    def get_history(self) -> List[Dict[str, str]]:
        """
        Retorna el historial completo como lista de mensajes.

        El formato es compatible con la API de Groq/OpenAI y Gemini:
            [
                {"role": "user",      "content": "..."},
                {"role": "assistant", "content": "..."},
                ...
            ]
        """
        return list(self._history)

    def get_history_as_text(self) -> str:
        """
        Retorna el historial formateado como texto plano.
        Útil para incluirlo en un prompt de texto (fallback sin soporte
        de mensajes estructurados).

        Formato:
            Usuario: <pregunta>
            Asistente: <respuesta>
        """
        lines = []
        for msg in self._history:
            prefix = "Usuario" if msg["role"] == "user" else "Asistente"
            lines.append(f"{prefix}: {msg['content']}")
        return "\n".join(lines)

    def is_empty(self) -> bool:
        """Retorna True si no hay historial previo (primera pregunta de la sesión)."""
        return len(self._history) == 0

    def turn_count(self) -> int:
        """Retorna el número de turnos (pares user/assistant) almacenados."""
        return len(self._history) // 2

    def clear(self) -> None:
        """Limpia el historial completo (reinicia la sesión)."""
        self._history = []
        logger.debug("[Memory] Historial de conversación limpiado.")
