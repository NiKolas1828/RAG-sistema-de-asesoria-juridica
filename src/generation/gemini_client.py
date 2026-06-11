# src/generation/gemini_client.py
# ============================================================
# Cliente para Gemini 2.0 Flash via google-genai SDK (nuevo)
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
    Cliente para Gemini 2.0 Flash usando el SDK google-genai (versión actual).

    Uso básico:
        client = GeminiClient()
        respuesta = client.generate(prompt="...")
    """

    def __init__(self, config: Optional[GeminiConfig] = None):
        self.config = config or GeminiConfig()
        self._client = None  # inicialización lazy
        self._api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if not self._api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY o GOOGLE_API_KEY no encontrada. "
                "Crea un archivo .env con GEMINI_API_KEY=tu_key. "
                "Obtén tu key gratis en https://aistudio.google.com"
            )

    def _get_client(self):
        """Inicializa el cliente Gemini solo cuando se necesita (lazy loading)."""
        if self._client is None:
            try:
                from google import genai  # type: ignore
            except ImportError:
                raise ImportError(
                    "Librería google-genai no instalada. "
                    "Ejecuta: pip install google-genai"
                )
            self._client = genai.Client(api_key=self._api_key)
        return self._client

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
        client = self._get_client()
        last_exception = None

        try:
            from google.genai import types  # type: ignore
        except ImportError:
            raise ImportError("Librería google-genai no instalada. Ejecuta: pip install google-genai")

        generation_config = types.GenerateContentConfig(
            system_instruction=self.config.system_instruction,
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_output_tokens,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
        )

        for intento in range(1, self.config.max_retries + 1):
            try:
                logger.debug(f"[Gemini] Intento {intento}/{self.config.max_retries}")

                response = client.models.generate_content(
                    model=self.config.model_name,
                    contents=prompt,
                    config=generation_config,
                )

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
