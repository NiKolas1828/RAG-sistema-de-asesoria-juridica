# src/memory/persistent_memory.py
# ============================================================
# Memoria Persistente por Usuario (SQLite)
#
# A diferencia de ConversationMemory (solo RAM), esta clase guarda
# el historial de conversación en SQLite por user_id.
# Si el bot se reinicia, el usuario retoma la conversación donde la dejó.
#
# Esquema de la base de datos:
#   conversations(user_id INT, role TEXT, content TEXT, ts REAL)
#   feedback(user_id INT, query TEXT, respuesta TEXT, util INT, ts REAL)
#
# Comandos soportados vía bot:
#   /historial → muestra las últimas 3 preguntas del usuario
# ============================================================

import sqlite3
import time
import logging
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Ruta de la base de datos (relativa al proyecto)
DB_PATH = Path(__file__).parent.parent.parent / "data" / "memory.db"

# Máximo de turnos a guardar por usuario (evita que la BD crezca infinito)
MAX_TURNS_STORED = 20
# Turnos activos enviados al LLM como contexto
MAX_TURNS_ACTIVE = 5


def _get_conn() -> sqlite3.Connection:
    """Abre una conexión a la BD, creando el archivo si no existe."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Crea las tablas si no existen. Llamar una sola vez al arrancar el bot."""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                role    TEXT    NOT NULL,   -- 'user' | 'assistant'
                content TEXT    NOT NULL,
                ts      REAL    NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conv_user
            ON conversations(user_id, ts DESC)
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id   INTEGER NOT NULL,
                query     TEXT    NOT NULL,
                respuesta TEXT    NOT NULL,
                util      INTEGER NOT NULL,  -- 1=útil, 0=no útil
                ts        REAL    NOT NULL
            )
        """)
        conn.commit()
    logger.info(f"[PersistentMemory] BD inicializada en {DB_PATH}")


class PersistentMemory:
    """
    Memoria de conversación persistente en SQLite por usuario de Telegram.

    Uso básico:
        mem = PersistentMemory(user_id=123456)
        mem.add_turn("¿Cuánto es la multa por no usar casco?", "La multa es...")
        historial = mem.get_history()      # últimos MAX_TURNS_ACTIVE turnos
        mem.clear()                        # borra el historial del usuario
    """

    def __init__(self, user_id: int):
        self.user_id = user_id

    def add_turn(self, user_message: str, assistant_message: str) -> None:
        """Guarda un par pregunta/respuesta en la BD."""
        ts = time.time()
        with _get_conn() as conn:
            conn.execute(
                "INSERT INTO conversations(user_id, role, content, ts) VALUES (?,?,?,?)",
                (self.user_id, "user", user_message, ts)
            )
            conn.execute(
                "INSERT INTO conversations(user_id, role, content, ts) VALUES (?,?,?,?)",
                (self.user_id, "assistant", assistant_message, ts + 0.001)
            )
            conn.commit()
        # Limpiar entradas antiguas si hay demasiadas
        self._evict_old()

    def get_history(self, max_turns: int = MAX_TURNS_ACTIVE) -> List[Dict[str, str]]:
        """
        Retorna los últimos N turnos como lista de mensajes compatibles con la API.

        Returns:
            Lista de dicts [{"role": "user"|"assistant", "content": "..."}]
        """
        limit = max_turns * 2  # cada turno = 2 mensajes
        with _get_conn() as conn:
            rows = conn.execute("""
                SELECT role, content FROM conversations
                WHERE user_id = ?
                ORDER BY ts DESC
                LIMIT ?
            """, (self.user_id, limit)).fetchall()
        # Revertir para orden cronológico
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def get_recent_questions(self, n: int = 3) -> List[str]:
        """Retorna las últimas N preguntas del usuario (solo mensajes de role='user')."""
        with _get_conn() as conn:
            rows = conn.execute("""
                SELECT content FROM conversations
                WHERE user_id = ? AND role = 'user'
                ORDER BY ts DESC
                LIMIT ?
            """, (self.user_id, n)).fetchall()
        return [r["content"] for r in rows]

    def is_empty(self) -> bool:
        """Retorna True si el usuario no tiene historial."""
        with _get_conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM conversations WHERE user_id = ?",
                (self.user_id,)
            ).fetchone()[0]
        return count == 0

    def turn_count(self) -> int:
        """Retorna el número de turnos guardados para este usuario."""
        with _get_conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM conversations WHERE user_id = ?",
                (self.user_id,)
            ).fetchone()[0]
        return count // 2

    def clear(self) -> None:
        """Borra todo el historial del usuario."""
        with _get_conn() as conn:
            conn.execute("DELETE FROM conversations WHERE user_id = ?", (self.user_id,))
            conn.commit()
        logger.info(f"[PersistentMemory] Historial del usuario {self.user_id} limpiado.")

    def _evict_old(self) -> None:
        """Elimina los turnos más antiguos si se supera MAX_TURNS_STORED."""
        max_messages = MAX_TURNS_STORED * 2
        with _get_conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM conversations WHERE user_id = ?",
                (self.user_id,)
            ).fetchone()[0]
            if count > max_messages:
                # Eliminar los más antiguos
                conn.execute("""
                    DELETE FROM conversations
                    WHERE user_id = ? AND id IN (
                        SELECT id FROM conversations
                        WHERE user_id = ?
                        ORDER BY ts ASC
                        LIMIT ?
                    )
                """, (self.user_id, self.user_id, count - max_messages))
                conn.commit()


# ─── Funciones de Feedback ────────────────────────────────────

def save_feedback(user_id: int, query: str, respuesta: str, util: bool) -> None:
    """
    Guarda el feedback del usuario (👍/👎) en la tabla feedback.

    Args:
        user_id:   ID de Telegram del usuario.
        query:     Pregunta original del usuario.
        respuesta: Respuesta que recibió.
        util:      True si fue útil (👍), False si no (👎).
    """
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO feedback(user_id, query, respuesta, util, ts) VALUES (?,?,?,?,?)",
            (user_id, query, respuesta, 1 if util else 0, time.time())
        )
        conn.commit()
    logger.info(f"[Feedback] user={user_id} util={'sí' if util else 'no'}")


def get_feedback_stats() -> Dict:
    """Retorna estadísticas globales de feedback."""
    with _get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        utiles = conn.execute("SELECT COUNT(*) FROM feedback WHERE util=1").fetchone()[0]
    return {
        "total": total,
        "utiles": utiles,
        "no_utiles": total - utiles,
        "tasa_utilidad_pct": round(utiles / total * 100, 1) if total > 0 else 0.0,
    }
