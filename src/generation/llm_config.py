# src/generation/llm_config.py
# ============================================================
# Configuración centralizada de parámetros para generación LLM
# Ajusta aquí temperature, tokens y umbrales sin tocar el código
# ============================================================

from dataclasses import dataclass, field
from typing import Optional

from src.context_builder.prompt_builder import SYSTEM_PROMPT


@dataclass
class GeminiConfig:
    """Parámetros para Gemini 1.5 Flash (modelo principal).

    gemini-1.5-flash tiene 1500 solicitudes/día en el free tier,
    frente a las ~50 de gemini-2.0-flash. Ideal para producción sin costo.
    """

    model_name: str = "gemini-2.0-flash-lite"

    # --- Control de creatividad vs precisión ---
    # 0.0 = determinista y conservador (ideal para textos legales)
    temperature: float = 0.1

    # --- Longitud máxima de la respuesta ---
    # 1024 permite respuestas exhaustivas con múltiples artículos citados.
    max_output_tokens: int = 1024

    # --- Top-p: diversidad del muestreo de tokens ---
    top_p: float = 0.8

    # --- Top-k: limita el vocabulario candidato por paso ---
    top_k: int = 20

    repetition_penalty: float = 0.0

    # --- Prompt del sistema: importado desde PromptBuilder (fuente única de verdad) ---
    system_instruction: str = SYSTEM_PROMPT

    # --- Timeouts y reintentos ---
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_wait_seconds: float = 2.0


@dataclass
class GroqConfig:
    """Parámetros para Groq + Llama 3.3 70B (Fallback #1 de Gemini).

    Modelo Llama 3.3 70B de Meta servido en Groq. Tiene un pool de tokens
    DIFERENTE al llama-3.1-8b-instant, con mayor calidad y TPM.
    """

    model_name: str = "llama-3.3-70b-versatile"

    temperature: float = 0.1
    max_tokens: int = 1024
    top_p: float = 0.8

    system_instruction: str = SYSTEM_PROMPT

    timeout_seconds: int = 20
    max_retries: int = 2
    retry_wait_seconds: float = 1.0


@dataclass
class GroqFallbackConfig:
    """Parámetros para Groq + Llama 3.1 8B Instant (Fallback #2, tercer nivel).

    Llama 3.1 8B tiene un límite estricto de 6.000 TPM, por lo que usamos max_tokens=512
    para evitar errores 413 (Request too large). Se mantiene como segundo nivel
    por su mayor cuota diaria si se llega a agotar el de 70B.
    """

    model_name: str = "llama-3.1-8b-instant"

    temperature: float = 0.1
    max_tokens: int = 512  # Reducido para evitar 413
    top_p: float = 0.8

    system_instruction: str = SYSTEM_PROMPT

    timeout_seconds: int = 20
    max_retries: int = 2
    retry_wait_seconds: float = 1.0


@dataclass
class GenerationConfig:
    """Configuración global del módulo de generación.

    Cadena de fallback:
        Gemini 2.0 Flash Lite → Groq Llama 3.3 70B → Groq Llama 3.1 8B
    """

    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    groq: GroqConfig = field(default_factory=GroqConfig)
    groq_fallback: GroqFallbackConfig = field(default_factory=GroqFallbackConfig)

    # --- Umbral mínimo de similitud para generar respuesta ---
    # El modelo paraphrase-multilingual-MiniLM-L12-v2 sobre textos legales
    # produce scores máximos de ~0.23 para consultas relevantes.
    # Umbral 0.08 rechaza ruido y acepta consultas jurídicas legítimas.
    min_similarity_to_generate: float = 0.08

    # --- Logging de llamadas al LLM ---
    log_prompts: bool = False        # True en desarrollo, False en producción
    log_responses: bool = False


# Instancia global lista para importar
generation_config = GenerationConfig()
