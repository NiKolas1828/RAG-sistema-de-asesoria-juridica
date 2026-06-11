# src/generation/response_generator.py
# ============================================================
# Orquestador de generación: Gemini principal → Groq fallback
# Recibe el output de RAGPipeline y produce la respuesta final
# ============================================================

import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.generation.llm_config import GenerationConfig, generation_config
from src.generation.gemini_client import GeminiClient, GeminiRateLimitError, GeminiClientError
from src.generation.groq_client import GroqClient, GroqClientError
from src.generation.llm_config import GroqFallbackConfig
from src.generation.response_cache import response_cache

logger = logging.getLogger(__name__)

# Tiempo de espera del circuit breaker de Gemini (segundos).
# Tras un rate limit, se salta Gemini durante este tiempo.
GEMINI_COOLDOWN_SECONDS = 300  # 5 minutos

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
        self._groq_fallback: Optional[GroqClient] = None  # Tercer nivel: Gemma2 9B
        # Circuit breaker: timestamp de la última vez que Gemini dio rate limit
        self._gemini_rate_limit_at: float = 0.0

    def _get_gemini(self) -> GeminiClient:
        if self._gemini is None:
            self._gemini = GeminiClient(self.config.gemini)
        return self._gemini

    def _get_groq(self) -> GroqClient:
        if self._groq is None:
            self._groq = GroqClient(self.config.groq)
        return self._groq

    def _get_groq_fallback(self) -> GroqClient:
        if self._groq_fallback is None:
            self._groq_fallback = GroqClient(self.config.groq_fallback)
        return self._groq_fallback

    def _hay_contexto_suficiente(self, rag_output: Dict[str, Any]) -> bool:
        """Verifica que el retrieval encontró chunks relevantes.

        Usa las claves reales que devuelve ContextBuilder:
          - 'contexto_formateado': texto con los fragmentos normativos
          - 'documentos_usados': cantidad de chunks incluidos
          - 'chunks_seleccionados': lista de chunks con similitud individual

        Se evalúa si la similitud MÁXIMA de los chunks seleccionados supera
        el umbral configurado, evitando rechazar consultas legítimas donde
        los chunks secundarios disminuyen el promedio.
        """
        if rag_output.get("status") != "éxito":
            return False

        contexto = rag_output.get("contexto", {})
        if not contexto:
            return False

        # Verificar que hay documentos y que el contexto tiene contenido
        chunks = contexto.get("chunks_seleccionados", [])
        if not chunks:
            return False

        if not contexto.get("contexto_formateado", "").strip():
            return False

        # Verificar similitud máxima contra el umbral configurado
        scores = [d.get("similitud", 0.0) for d in chunks]
        max_score = max(scores) if scores else 0.0
        return max_score >= self.config.min_similarity_to_generate

    def generate(
        self,
        rag_output: Dict[str, Any],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Genera la respuesta final a partir del output de RAGPipeline.

        Args:
            rag_output: Diccionario retornado por RAGPipeline.run()
            history:    Historial de conversación previo (multi-turno).
                        Lista de dicts [{"role": "user"|"assistant", "content": "..."}].
                        Si es None, se trata como primera pregunta de la sesión.

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
        prompt_meta = rag_output.get("prompt") or {}
        base_result = {
            "query_original": rag_output.get("query_original", ""),
            "timestamp": timestamp,
            "modelo_usado": None,
            "respuesta": None,
            "tokens_prompt": prompt_meta.get("tokens_prompt", 0),
            "question_type": prompt_meta.get("question_type", "general"),
            "status": "error",
            "error": None,
        }

        # --- Caso 0: Verificar caché antes de cualquier llamada a la API ---
        query_original = rag_output.get("query_original", "")
        cache_key = response_cache.make_key(query_original)
        cached = response_cache.get(cache_key)
        if cached:
            logger.info("[Generator] 🎯 Respondiendo desde caché (0 tokens consumidos).")
            return {
                **base_result,
                "respuesta": cached["respuesta"],
                "modelo_usado": f"cache/{cached['modelo_usado']}",
                "status": "éxito",
            }

        # --- Caso 0.5: Domain Guard rechazó la consulta ---
        if rag_output.get("status") == "fuera_de_dominio":
            logger.info("[Generator] Consulta fuera del dominio de tránsito. Respondiendo sin LLM.")
            return {
                **base_result,
                "respuesta": rag_output.get("respuesta_directa", RESPUESTA_SIN_CONTEXTO),
                "modelo_usado": "domain_guard",
                "status": "fuera_de_dominio",
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
        prompt = prompt_meta.get("prompt", "")
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
        # Circuit breaker: si Gemini dio rate limit hace menos de GEMINI_COOLDOWN_SECONDS,
        # saltar directo a Groq sin hacer la llamada fallida.
        gemini_en_cooldown = (
            time.time() - self._gemini_rate_limit_at < GEMINI_COOLDOWN_SECONDS
        )
        if gemini_en_cooldown:
            logger.debug(
                f"[Generator] Gemini en cooldown, usando Groq directamente."
            )
            return self._generar_con_groq(prompt, base_result, history=history, cache_key=cache_key)

        try:
            respuesta = self._get_gemini().generate(prompt)
            if self.config.log_responses:
                logger.debug(f"[Generator] Respuesta Gemini:\n{respuesta}")
            modelo = f"{self.config.gemini.model_name}"
            response_cache.set(cache_key, respuesta, modelo_usado=modelo)
            return {
                **base_result,
                "respuesta": respuesta,
                "modelo_usado": modelo,
                "status": "éxito",
            }

        except GeminiRateLimitError as e:
            # Registrar el momento del rate limit para el circuit breaker
            self._gemini_rate_limit_at = time.time()
            logger.warning(f"[Generator] Gemini rate limit. Activando Groq. ({e})")
            return self._generar_con_groq(prompt, base_result, history=history, cache_key=cache_key)

        except GeminiClientError as e:
            logger.error(f"[Generator] Error Gemini: {e}")
            return {
                **base_result,
                "respuesta": RESPUESTA_SIN_CONTEXTO,
                "modelo_usado": self.config.gemini.model_name,
                "status": "error",
                "error": str(e),
            }

    def _generar_con_groq(
        self,
        prompt: str,
        base_result: Dict[str, Any],
        history: Optional[List[Dict[str, str]]] = None,
        cache_key: str = "",
    ) -> Dict[str, Any]:
        """Fallback #1: Groq Llama 3.3 70B. Si también falla, activa Fallback #2."""
        try:
            respuesta = self._get_groq().generate(prompt, history=history)
            if self.config.log_responses:
                logger.debug(f"[Generator] Respuesta Groq:\n{respuesta}")
            modelo = f"groq/{self.config.groq.model_name}"
            if cache_key:
                response_cache.set(cache_key, respuesta, modelo_usado=modelo)
            return {
                **base_result,
                "respuesta": respuesta,
                "modelo_usado": modelo,
                "status": "éxito",
            }
        except GroqClientError as e:
            logger.warning(f"[Generator] Groq Llama 70B agotado: {e}. Activando Fallback #2 (Llama 3.1 8B)...")
            return self._generar_con_groq_fallback(prompt, base_result, history=history, cache_key=cache_key)

    def _generar_con_groq_fallback(
        self,
        prompt: str,
        base_result: Dict[str, Any],
        history: Optional[List[Dict[str, str]]] = None,
        cache_key: str = "",
    ) -> Dict[str, Any]:
        """Fallback #2 (tercer nivel): Groq Llama 3.1 8B. Pool de tokens independiente."""
        try:
            respuesta = self._get_groq_fallback().generate(prompt, history=history)
            if self.config.log_responses:
                logger.debug(f"[Generator] Respuesta Groq Llama 8B:\n{respuesta}")
            modelo = f"groq/{self.config.groq_fallback.model_name}"
            if cache_key:
                response_cache.set(cache_key, respuesta, modelo_usado=modelo)
            return {
                **base_result,
                "respuesta": respuesta,
                "modelo_usado": modelo,
                "status": "éxito",
            }
        except GroqClientError as e:
            logger.error(f"[Generator] Error Groq Llama 8B (Fallback #2): {e}")
            return {
                **base_result,
                "respuesta": RESPUESTA_SIN_CONTEXTO,
                "modelo_usado": f"groq/{self.config.groq_fallback.model_name}",
                "status": "error",
                "error": str(e),
            }
