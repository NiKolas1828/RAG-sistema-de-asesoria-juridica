# src/generation/response_generator.py
# ============================================================
# Orquestador de generación: Gemini principal → Groq fallback
# Recibe el output de RAGPipeline y produce la respuesta final
# ============================================================

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from src.generation.llm_config import GenerationConfig, generation_config
from src.generation.gemini_client import GeminiClient, GeminiRateLimitError, GeminiClientError
from src.generation.groq_client import GroqClient, GroqClientError

logger = logging.getLogger(__name__)

# Respuesta estándar cuando no hay contexto suficiente
RESPUESTA_SIN_CONTEXTO = (
    "No encontré información suficiente sobre eso en las normas consultadas. "
    "Te recomiendo contactar al organismo de tránsito de tu municipio o "
    "consultar directamente el Código Nacional de Tránsito (Ley 769 de 2002)."
)


class ResponseGenerator:
    """
    Toma el output de RAGPipeline y genera la respuesta final del ciudadano.

    Flujo:
        RAGPipeline.run() → ResponseGenerator.generate() → respuesta en texto

    Ejemplo de uso:
        pipeline = RAGPipeline()
        generator = ResponseGenerator()

        rag_output = pipeline.run("¿Cuál es la multa por no usar casco?")
        resultado = generator.generate(rag_output)
        print(resultado["respuesta"])
    """

    def __init__(self, config: Optional[GenerationConfig] = None):
        self.config = config or generation_config
        self._gemini: Optional[GeminiClient] = None
        self._groq: Optional[GroqClient] = None

    def _get_gemini(self) -> GeminiClient:
        if self._gemini is None:
            self._gemini = GeminiClient(self.config.gemini)
        return self._gemini

    def _get_groq(self) -> GroqClient:
        if self._groq is None:
            self._groq = GroqClient(self.config.groq)
        return self._groq

    def _hay_contexto_suficiente(self, rag_output: Dict[str, Any]) -> bool:
        """Verifica que el retrieval encontró chunks relevantes.

        Usa las claves reales que devuelve ContextBuilder:
          - 'contexto_formateado': texto con los fragmentos normativos
          - 'documentos_usados': cantidad de chunks incluidos
          - 'score_promedio': similitud promedio de los chunks seleccionados

        Nota: la clave 'chunks_seleccionados' no existe en el output de
        ContextBuilder — usar 'score_promedio' y 'documentos_usados'.
        """
        if rag_output.get("status") != "éxito":
            return False

        contexto = rag_output.get("contexto", {})
        if not contexto:
            return False

        # Verificar que hay documentos y que el contexto tiene contenido
        if contexto.get("documentos_usados", 0) == 0:
            return False

        if not contexto.get("contexto_formateado", "").strip():
            return False

        # Verificar similitud promedio contra el umbral configurado
        score = contexto.get("score_promedio", 0.0)
        return score >= self.config.min_similarity_to_generate

    def generate(self, rag_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Genera la respuesta final a partir del output de RAGPipeline.

        Args:
            rag_output: Diccionario retornado por RAGPipeline.run()

        Returns:
            Diccionario con:
                - respuesta (str): Texto final para el ciudadano
                - modelo_usado (str): "gemini" | "groq" | "sin_llm"
                - status (str): "éxito" | "error" | "sin_contexto"
                - tokens_prompt (int): Estimado de tokens del prompt
                - timestamp (str): Momento de la generación
                - error (str | None): Mensaje de error si aplica
        """
        timestamp = datetime.now().isoformat()
        base_result = {
            "query_original": rag_output.get("query_original", ""),
            "timestamp": timestamp,
            "modelo_usado": None,
            "respuesta": None,
            "tokens_prompt": rag_output.get("prompt", {}).get("tokens_prompt", 0),
            "status": "error",
            "error": None,
        }

        # --- Caso 1: RAG no encontró contexto suficiente ---
        if not self._hay_contexto_suficiente(rag_output):
            logger.info("[Generator] Sin contexto suficiente. Respondiendo sin LLM.")
            return {
                **base_result,
                "respuesta": RESPUESTA_SIN_CONTEXTO,
                "modelo_usado": "sin_llm",
                "status": "sin_contexto",
            }

        # --- Extraer el prompt armado por PromptBuilder ---
        prompt = rag_output.get("prompt", {}).get("prompt", "")
        if not prompt:
            return {
                **base_result,
                "respuesta": RESPUESTA_SIN_CONTEXTO,
                "modelo_usado": "sin_llm",
                "status": "error",
                "error": "Prompt vacío recibido de RAGPipeline.",
            }

        if self.config.log_prompts:
            logger.debug(f"[Generator] Prompt enviado:\n{prompt}")

        # --- Caso 2: Intentar con Gemini (principal) ---
        try:
            respuesta = self._get_gemini().generate(prompt)
            if self.config.log_responses:
                logger.debug(f"[Generator] Respuesta Gemini:\n{respuesta}")
            return {
                **base_result,
                "respuesta": respuesta,
                "modelo_usado": "gemini-2.0-flash",
                "status": "éxito",
            }

        except GeminiRateLimitError as e:
            # --- Caso 3: Rate limit → activar fallback Groq ---
            logger.warning(f"[Generator] Gemini rate limit. Activando Groq. ({e})")
            return self._generar_con_groq(prompt, base_result)

        except GeminiClientError as e:
            logger.error(f"[Generator] Error Gemini: {e}")
            return {
                **base_result,
                "respuesta": RESPUESTA_SIN_CONTEXTO,
                "modelo_usado": "gemini-2.0-flash",
                "status": "error",
                "error": str(e),
            }

    def _generar_con_groq(
        self, prompt: str, base_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Intenta generar con Groq como fallback."""
        try:
            respuesta = self._get_groq().generate(prompt)
            if self.config.log_responses:
                logger.debug(f"[Generator] Respuesta Groq:\n{respuesta}")
            return {
                **base_result,
                "respuesta": respuesta,
                "modelo_usado": "groq/llama-3.3-70b",
                "status": "éxito",
            }
        except GroqClientError as e:
            logger.error(f"[Generator] Error Groq: {e}")
            return {
                **base_result,
                "respuesta": RESPUESTA_SIN_CONTEXTO,
                "modelo_usado": "groq/llama-3.3-70b",
                "status": "error",
                "error": str(e),
            }
