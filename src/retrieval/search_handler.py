# src/retrieval/search_handler.py
# ============================================================
# Búsqueda semántica con Multi-Query + Re-Ranking
#
# Optimizaciones de latencia aplicadas:
#   1. Embeddings en paralelo con ThreadPoolExecutor
#   2. Re-ranking solo sobre top 5 (no 20), pre-filtrados por similitud
#   3. Circuit breaker de Gemini se maneja en response_generator
# ============================================================

import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.retrieval.query_processor import process_query
from src.retrieval.semantic_search import SemanticSearchEngine
from src.retrieval.query_expander import QueryExpander
from src.retrieval.reranker import Reranker

logger = logging.getLogger(__name__)

# Número de chunks a pasar al Cross-Encoder.
# Mantenerlo bajo (5-8) para que el reranker sea rápido en CPU.
RERANK_TOP_N = 5


class SearchHandler:
    def __init__(self):
        self.search_engine = SemanticSearchEngine()
        self.expander  = None  # Lazy init
        self.reranker  = None  # Lazy init

    def perform_search(self, query: str, k: int = 5, verbose: bool = False) -> dict:
        try:
            if verbose:
                print(f"[*] Procesando consulta base: '{query}'...")

            if self.expander is None:
                self.expander = QueryExpander()
            if self.reranker is None:
                self.reranker = Reranker()

            queries = self.expander.expand_query(query)
            if verbose:
                print(f"[*] Multi-Query: Se buscarán {len(queries)} variaciones de la consulta.")

            # ── Detectar código de infracción en cualquiera de las queries ──
            where_doc = None
            for q in queries:
                code_match = re.search(r'\b([A-E]\.\d{2})\b', q, re.IGNORECASE)
                if code_match:
                    code = code_match.group(1).upper()
                    where_doc = {"$contains": code}
                    if verbose:
                        print(f"[*] Filtro estricto aplicado para el código: {code}")
                    break  # Con un código detectado es suficiente

            # ── Generar embeddings en lote (batch) para evitar bloqueos y mejorar latencia ──
            try:
                embeddings = process_query(queries)
            except Exception as e:
                logger.error(f"[SearchHandler] Error generando embeddings en lote: {e}")
                embeddings = []

            all_results = {}
            if embeddings:
                for q, embedding in zip(queries, embeddings):
                    try:
                        results = self.search_engine.search(
                            embedding, k=k, where_document=where_doc
                        )
                        for res in results.get("resultados", []):
                            text_key = res["texto"]
                            # Deduplicar: conservar el mayor score
                            if (text_key not in all_results
                                    or res["similitud"] > all_results[text_key]["similitud"]):
                                all_results[text_key] = res
                    except Exception as e:
                        logger.warning(f"[SearchHandler] Error en búsqueda de variación '{q}': {e}")

            # ── Pre-filtrar por similitud ANTES del Cross-Encoder ──
            # Ordenar y tomar solo los top k para el reranker
            candidates = sorted(
                all_results.values(),
                key=lambda x: x["similitud"],
                reverse=True
            )[:k]

            # ── Re-ranking solo sobre top RERANK_TOP_N (5 chunks) ──
            # El Cross-Encoder en CPU es O(n): de 20→5 es 4x más rápido
            reranked = self.reranker.rerank(
                query=query,
                chunks=candidates,
                top_n=RERANK_TOP_N,
            )

            # Reasignar ranks
            for i, res in enumerate(reranked):
                res["rank"] = i + 1

            final_scores = [res["similitud"] for res in reranked]

            if verbose:
                print(f"[✓] Búsqueda completada: {len(candidates)} candidatos → "
                      f"top {len(reranked)} tras re-ranking.")

            return {
                "query_original": query,
                "query_procesada": queries[0].lower().strip(),
                "queries_expandidas": queries,
                "resultados": reranked,
                "total_encontrados": len(reranked),
                "scores": final_scores,
                "status": "éxito",
                "mensaje": f"Se encontraron {len(reranked)} documentos relevantes",
            }

        except ValueError as e:
            return {"query_original": query, "status": "error",
                    "mensaje": f"Error de validación: {str(e)}", "resultados": []}
        except RuntimeError as e:
            return {"query_original": query, "status": "error",
                    "mensaje": f"Error en sistema: {str(e)}", "resultados": []}
        except Exception as e:
            return {"query_original": query, "status": "error",
                    "mensaje": f"Error inesperado: {str(e)}", "resultados": []}
