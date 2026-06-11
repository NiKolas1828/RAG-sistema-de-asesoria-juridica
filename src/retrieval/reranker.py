# src/retrieval/reranker.py
# ============================================================
# Re-Ranking con Cross-Encoder
#
# Mejora el orden de los chunks recuperados por ChromaDB usando
# un modelo Cross-Encoder que evalúa la relevancia de cada par
# (pregunta, fragmento) de forma más precisa que la similitud
# coseno del embedding.
#
# Activación: variable de entorno RERANKER_ENABLED=true (default)
#             o RERANKER_ENABLED=false para desactivar.
#
# Modelo por defecto: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
#   - Entrenado con mMARCO (incluye español y 12 idiomas más)
#   - Ligero: ~90 MB, funciona en CPU sin GPU
#   - Primera llamada descarga el modelo (se cachea localmente)
# ============================================================

import logging
from typing import List, Dict, Any

from src.config import RERANKER_ENABLED, RERANKER_MODEL

logger = logging.getLogger(__name__)


class Reranker:
    """
    Re-ordena los chunks de búsqueda usando un modelo Cross-Encoder.

    El Cross-Encoder toma cada par (query, documento) y produce un
    puntaje de relevancia más preciso que la similitud coseno, porque
    el modelo procesa ambos textos juntos en lugar de por separado.

    Si RERANKER_ENABLED=false en el entorno, el método rerank()
    devuelve la lista original sin cambios (bypass completo).

    Uso básico:
        reranker = Reranker()
        chunks_reordenados = reranker.rerank(query="...", chunks=[...])
    """

    def __init__(self):
        self._model = None  # Lazy init: descarga el modelo solo al primer uso

    def _get_model(self):
        """Carga el modelo Cross-Encoder (se cachea en disco tras la primera descarga)."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder  # type: ignore
                logger.info(f"[Reranker] Cargando modelo '{RERANKER_MODEL}'...")
                self._model = CrossEncoder(RERANKER_MODEL)
                logger.info("[Reranker] Modelo listo.")
            except ImportError:
                raise ImportError(
                    "sentence-transformers no instalado. "
                    "Ejecuta: pip install sentence-transformers"
                )
        return self._model

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_n: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Re-ordena los chunks por relevancia real frente a la query.

        Si RERANKER_ENABLED=false, retorna los chunks sin modificar.

        Args:
            query:  Pregunta original del usuario.
            chunks: Lista de dicts con al menos la clave 'texto'.
                    (Formato devuelto por SearchHandler.perform_search)
            top_n:  Si se especifica, retorna solo los top_n chunks
                    tras el re-ranking. Si es None, retorna todos.

        Returns:
            Lista de chunks reordenados con la clave 'rerank_score'
            añadida a cada uno para trazabilidad.
        """
        if not RERANKER_ENABLED:
            logger.debug("[Reranker] Desactivado por config. Bypass.")
            return chunks

        if not chunks:
            return chunks

        try:
            model = self._get_model()

            # Pares (query, texto_del_chunk) para el Cross-Encoder
            pairs = [(query, chunk.get("texto", "")) for chunk in chunks]

            # Predecir puntajes de relevancia
            scores = model.predict(pairs, num_workers=0, batch_size=8)

            # Añadir el puntaje de re-ranking a cada chunk
            ranked = []
            for chunk, score in zip(chunks, scores):
                ranked.append({**chunk, "rerank_score": float(score)})

            # Ordenar de mayor a menor puntaje
            ranked.sort(key=lambda x: x["rerank_score"], reverse=True)

            # Reasignar ranks
            for i, chunk in enumerate(ranked):
                chunk["rank"] = i + 1

            result = ranked[:top_n] if top_n else ranked

            logger.debug(
                f"[Reranker] {len(chunks)} chunks → re-rankeados → "
                f"top {len(result)} devueltos."
            )
            return result

        except Exception as e:
            # Si el re-ranker falla por cualquier razón, loguear y continuar
            # sin re-ranking (nunca romper el pipeline principal)
            logger.warning(
                f"[Reranker] Error durante re-ranking: {e}. "
                "Usando orden original de ChromaDB."
            )
            return chunks
