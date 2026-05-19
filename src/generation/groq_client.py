# src/generation/groq_client.py
# ============================================================
# Cliente para Groq + Llama 3.3 70B (fallback de Gemini)
# API compatible con OpenAI → integración sencilla
# ============================================================

import os
import time
import logging
from typing import Optional

from dotenv import load_dotenv

from src.generation.llm_config import GroqConfig

load_dotenv()
logger = logging.getLogger(__name__)


class GroqClientError(Exception):
    """Error general del cliente Groq."""


class GroqClient:
    """
    Cliente para Groq + Llama 3.3 70B.
    Se activa automáticamente cuando Gemini supera el rate limit.

    Uso básico:
        client = GroqClient()
        respuesta = client.generate(prompt="...")
    """

    def __init__(self, config: Optional[GroqConfig] = None):
        self.config = config or GroqConfig()
        self._client = None  # lazy loading
        self._api_key = os.getenv("GROQ_API_KEY")

        if not self._api_key:
            raise EnvironmentError(
                "GROQ_API_KEY no encontrada. "
                "Crea un archivo .env con GROQ_API_KEY=tu_key. "
                "Obtén tu key gratis en https://console.groq.com"
            )

    def _get_client(self):
        """Inicializa el cliente Groq solo cuando se necesita."""
        if self._client is None:
            try:
                from groq import Groq  # type: ignore
            except ImportError:
                raise ImportError(
                    "Librería groq no instalada. "
                    "Ejecuta: pip install groq"
                )
            self._client = Groq(api_key=self._api_key)
        return self._client

    def generate(self, prompt: str) -> str:
        """
        Envía el prompt a Groq/Llama y retorna el texto generado.

        Args:
            prompt: Texto completo del prompt (contexto + pregunta ya armados).

        Returns:
            Texto de la respuesta generada.

        Raises:
            GroqClientError: Si la llamada falla después de los reintentos.
        """
        client = self._get_client()
        last_exception = None

        for intento in range(1, self.config.max_retries + 1):
            try:
                logger.debug(f"[Groq] Intento {intento}/{self.config.max_retries}")

                response = client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": self.config.system_instruction,
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    top_p=self.config.top_p,
                )

                text = response.choices[0].message.content
                if not text:
                    raise GroqClientError("Groq retornó respuesta vacía.")

                logger.debug(f"[Groq] Respuesta recibida ({len(text)} chars)")
                return text

            except GroqClientError:
                raise
            except Exception as e:
                last_exception = e
                if intento < self.config.max_retries:
                    wait = self.config.retry_wait_seconds * intento
                    logger.warning(
                        f"[Groq] Error en intento {intento}: {e}. "
                        f"Reintentando en {wait}s..."
                    )
                    time.sleep(wait)

        raise GroqClientError(
            f"Groq falló después de {self.config.max_retries} intentos. "
            f"Último error: {last_exception}"
        ) from last_exception
