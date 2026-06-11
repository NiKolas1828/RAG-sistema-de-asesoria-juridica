# src/generation/response_cache.py
# ============================================================
# Caché de Respuestas en RAM
#
# Evita llamadas innecesarias a la API cuando el usuario
# hace la misma pregunta dentro del período de vida del caché.
#
# key   = SHA-256 de la query en minúsculas
# value = {"respuesta": str, "timestamp": float, "modelo_usado": str}
# ============================================================

import hashlib
import logging
import time
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 1800  # 30 minutos


class ResponseCache:
    """
    Caché de respuestas en memoria con TTL configurable.

    Uso:
        cache = ResponseCache()
        key = cache.make_key("cuánto vale la multa por no usar casco")
        hit = cache.get(key)
        if hit:
            return hit
        respuesta = llm.generate(...)
        cache.set(key, respuesta, modelo="gemini-1.5-flash")
    """

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS):
        self._ttl = ttl_seconds
        self._store: Dict[str, Dict[str, Any]] = {}
        self._hits = 0
        self._misses = 0

    def make_key(self, query: str) -> str:
        """Genera SHA-256 de la query normalizada."""
        normalized = " ".join(query.lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Recupera respuesta si existe y no ha expirado."""
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None
        age = time.time() - entry["timestamp"]
        if age > self._ttl:
            del self._store[key]
            self._misses += 1
            return None
        self._hits += 1
        logger.debug(f"[Cache] HIT (edad: {age:.0f}s, modelo: {entry.get('modelo_usado', '?')})")
        return entry

    def set(self, key: str, respuesta: str, modelo_usado: str = "unknown") -> None:
        """Almacena una respuesta en el caché."""
        self._store[key] = {
            "respuesta": respuesta,
            "modelo_usado": modelo_usado,
            "timestamp": time.time(),
        }
        logger.debug(f"[Cache] Guardado. Entradas: {len(self._store)}")

    def invalidate(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0.0
        return {
            "entradas": len(self._store),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_pct": round(hit_rate, 1),
            "ttl_seconds": self._ttl,
        }


# Instancia global compartida por el proceso
response_cache = ResponseCache()
