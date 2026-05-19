# src/generation/gemini_client.py
# ============================================================
# Cliente para Gemini 2.0 Flash via google-generativeai SDK
# Maneja autenticación, parámetros, reintentos y errores
# ============================================================

import os
import time
import logging
from typing import Optional

from dotenv import load_dotenv

from src.generation.llm_config import GeminiConfig

load_dotenv()
logger = logging.getLogger(__name__)


class GeminiRateLimitError(Exception):
    """Se lanza cuando Gemini devuelve 429 (rate limit del free tier)."""


class GeminiClientError(Exception):
    """Error general del cliente Gemini."""


class GeminiClient:
    """
    Cliente para Gemini 2.0 Flash.

    Uso básico:
        client = GeminiClient()
        respuesta = client.generate(prompt="...", context="...")
    """

    def __init__(self, config: Optional[GeminiConfig] = None):
        self.config = config or GeminiConfig()
        self._model = None  # inicialización lazy
        self._api_key = os.getenv("GEMINI_API_KEY")

        if not self._api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY no encontrada. "
                "Crea un archivo .env con GEMINI_API_KEY=tu_key. "
                "Obtén tu key gratis en https://aistudio.google.com"
            )

    def _get_model(self):
        """Inicializa el modelo solo cuando se necesita (lazy loading)."""
        if self._model is None:
            try:
                import google.generativeai as genai  # type: ignore
            except ImportError:
                raise ImportError(
                    "Librería google-generativeai no instalada. "
                    "Ejecuta: pip install google-generativeai"
                )

            genai.configure(api_key=self._api_key)

            generation_cfg = genai.types.GenerationConfig(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
            )

            self._model = genai.GenerativeModel(
                model_name=self.config.model_name,
                generation_config=generation_cfg,
                system_instruction=self.config.system_instruction,
            )

        return self._model

    def generate(self, prompt: str) -> str:
        """
        Envía el prompt a Gemini y retorna el texto generado.

        Args:
            prompt: Texto completo del prompt (contexto + pregunta ya armados).

        Returns:
            Texto de la respuesta generada.

        Raises:
            GeminiRateLimitError: Si se supera el límite de 15 RPM.
            GeminiClientError: Para cualquier otro error de la API.
        """
        model = self._get_model()
        last_exception = None

        for intento in range(1, self.config.max_retries + 1):
            try:
                logger.debug(f"[Gemini] Intento {intento}/{self.config.max_retries}")
                response = model.generate_content(prompt)

                # Verificar que la respuesta tiene contenido
                if not response.text:
                    raise GeminiClientError("Gemini retornó respuesta vacía.")

                logger.debug(f"[Gemini] Respuesta recibida ({len(response.text)} chars)")
                return response.text

            except Exception as e:
                error_str = str(e).lower()
                last_exception = e

                # Rate limit (429) → lanzar excepción especial para activar fallback
                if "429" in error_str or "rate" in error_str or "quota" in error_str:
                    logger.warning("[Gemini] Rate limit alcanzado. Activando fallback.")
                    raise GeminiRateLimitError(
                        f"Rate limit de Gemini superado: {e}"
                    ) from e

                # Error recuperable → reintentar con espera
                if intento < self.config.max_retries:
                    wait = self.config.retry_wait_seconds * intento
                    logger.warning(
                        f"[Gemini] Error en intento {intento}: {e}. "
                        f"Reintentando en {wait}s..."
                    )
                    time.sleep(wait)

        raise GeminiClientError(
            f"Gemini falló después de {self.config.max_retries} intentos. "
            f"Último error: {last_exception}"
        ) from last_exception
