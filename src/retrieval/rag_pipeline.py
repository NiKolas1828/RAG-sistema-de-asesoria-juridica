from typing import Dict, Any

from src.context_builder.context_builder import ContextBuilder
from src.context_builder.prompt_builder import PromptBuilder
from src.retrieval.search_handler import SearchHandler


class RAGPipeline:
    """Orquesta búsqueda semántica, construcción de contexto y armado de prompt."""

    def __init__(self, max_context_tokens: int = 5000, min_similarity: float = 0.03, response_buffer: int = 128):
        self.search_handler = SearchHandler()
        self.context_builder = ContextBuilder(max_tokens=max_context_tokens, min_similarity=min_similarity)
        self.prompt_builder = PromptBuilder(response_buffer=response_buffer)
        self.max_context_tokens = max_context_tokens

    def run(self, query: str, k: int = 5, max_documents: int = 3, verbose: bool = False) -> Dict[str, Any]:
        """Ejecuta el flujo completo: consulta -> búsqueda -> contexto -> prompt."""
        search_results = self.search_handler.perform_search(query, k=k, verbose=verbose)
        if search_results.get("status") != "éxito":
            return {
                "query_original": query,
                "status": search_results.get("status", "error"),
                "mensaje": search_results.get("mensaje", "Error en el flujo RAG"),
                "search_results": search_results,
                "contexto": None,
                "prompt": None,
            }

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
