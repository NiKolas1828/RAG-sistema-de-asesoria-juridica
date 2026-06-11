from typing import Dict, Any

from src.context_builder.context_builder import ContextBuilder
from src.context_builder.prompt_builder import PromptBuilder
from src.retrieval.search_handler import SearchHandler
from src.retrieval.domain_guard import domain_guard, RESPUESTA_FUERA_DOMINIO


class RAGPipeline:
    """Orquesta búsqueda semántica, construcción de contexto y armado de prompt.

    Integra un DomainGuard de dos capas:
      - Capa léxica (antes del retrieval): rechaza temas claramente ajenos
        a normas de tránsito sin consumir embeddings ni cuota de API.
      - Capa semántica (después del retrieval): rechaza consultas cuyo
        mejor resultado tiene similitud < 0.20 con el corpus indexado.
    """

    def __init__(self, max_context_tokens: int = 5000, min_similarity: float = 0.03, response_buffer: int = 128):
        self.search_handler = SearchHandler()
        self.context_builder = ContextBuilder(max_tokens=max_context_tokens, min_similarity=min_similarity)
        self.prompt_builder = PromptBuilder(response_buffer=response_buffer)
        self.max_context_tokens = max_context_tokens

    def run(self, query: str, k: int = 5, max_documents: int = 3, verbose: bool = False, history: list = None) -> Dict[str, Any]:
        """Ejecuta el flujo completo: consulta → guard → búsqueda → contexto → prompt."""

        # ── Capa 1: Domain Guard léxico (sin costo) ──────────────────────────
        valida, motivo = domain_guard.check_lexical(query)
        if not valida:
            return {
                "query_original": query,
                "status": "fuera_de_dominio",
                "mensaje": motivo,
                "respuesta_directa": RESPUESTA_FUERA_DOMINIO,
                "search_results": None,
                "contexto": None,
                "prompt": None,
            }

        # ── Retrieval ─────────────────────────────────────────────────────────
        search_results = self.search_handler.perform_search(query, k=k, verbose=verbose, history=history)
        if search_results.get("status") != "éxito":
            return {
                "query_original": query,
                "status": search_results.get("status", "error"),
                "mensaje": search_results.get("mensaje", "Error en el flujo RAG"),
                "search_results": search_results,
                "contexto": None,
                "prompt": None,
            }

        # ── Capa 2: Domain Guard semántico (usa scores del retrieval) ─────────
        valida, motivo = domain_guard.check_semantic(search_results)
        if not valida:
            return {
                "query_original": query,
                "status": "fuera_de_dominio",
                "mensaje": motivo,
                "respuesta_directa": RESPUESTA_FUERA_DOMINIO,
                "search_results": search_results,
                "contexto": None,
                "prompt": None,
            }

        # ── Construcción de contexto y prompt ────────────────────────────────
        context_payload = self.context_builder.build_context(search_results, max_documents=max_documents)
        prompt_payload = self.prompt_builder.build_prompt(
            context_payload.get("contexto_formateado", ""),
            query,
            max_prompt_tokens=self.max_context_tokens,
        )

        return {
            "query_original": query,
            "search_results": search_results,
            "contexto": context_payload,
            "prompt": prompt_payload,
            "status": "éxito",
            "mensaje": "Prompt contextualizado generado correctamente",
        }