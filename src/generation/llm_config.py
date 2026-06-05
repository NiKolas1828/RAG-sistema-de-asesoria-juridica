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
    """Parámetros para Gemini 2.0 Flash (modelo principal)."""

    model_name: str = "gemini-2.0-flash"

    # --- Control de creatividad vs precisión ---
    # 0.0 = determinista y conservador (ideal para textos legales)
    # 1.0 = más creativo y variado
    temperature: float = 0.1

    # --- Longitud máxima de la respuesta ---
    # El free tier permite hasta 8192. Para Q&A legal 512 es suficiente.
    max_output_tokens: int = 512

    # --- Top-p: diversidad del muestreo de tokens ---
    # 0.8 = solo considera el 80% de probabilidad acumulada
    top_p: float = 0.8

    # --- Top-k: limita el vocabulario candidato por paso ---
    top_k: int = 20

    # --- Penalización por repetición de frases ---
    # Valores positivos reducen repetición. Rango típico: 0.0 – 2.0
    repetition_penalty: float = 0.0  # Gemini no expone este param directamente

    # --- Prompt del sistema: importado desde PromptBuilder (fuente única de verdad) ---
    # Contiene las 4 reglas anti-alucinación del dominio de tránsito colombiano.
    system_instruction: str = SYSTEM_PROMPT

    # --- Timeouts y reintentos ---
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_wait_seconds: float = 2.0


@dataclass
class GroqConfig:
    """Parámetros para Groq + Llama 3.3 70B (fallback)."""

    model_name: str = "llama-3.3-70b-versatile"

    # Misma lógica conservadora que Gemini para consistencia
    temperature: float = 0.1
    max_tokens: int = 512
    top_p: float = 0.8

    # Misma system instruction que Gemini para respuestas consistentes.
    # Importada desde PromptBuilder (fuente única de verdad).
    system_instruction: str = SYSTEM_PROMPT

    timeout_seconds: int = 20
    max_retries: int = 2
    retry_wait_seconds: float = 1.0


@dataclass
class GenerationConfig:
    """Configuración global del módulo de generación."""

    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    groq: GroqConfig = field(default_factory=GroqConfig)

    # --- Umbral mínimo de similitud para generar respuesta ---
    # Si el retrieval no encuentra chunks con score_promedio >= este valor,
    # se responde con mensaje de "no encontré información" sin llamar al LLM.
    #
    # CALIBRACIÓN EMPÍRICA (corpus normas de tránsito Colombia):
    # El modelo paraphrase-multilingual-MiniLM-L12-v2 sobre textos legales
    # produce scores máximos de ~0.23 para consultas relevantes.
    # Queries sin sentido (typos, texto aleatorio) producen scores < 0.05.
    # Umbral 0.10 rechaza noise y acepta consultas jurídicas legítimas.
    min_similarity_to_generate: float = 0.10

    # --- Logging de llamadas al LLM ---
    log_prompts: bool = False        # True en desarrollo, False en producción
    log_responses: bool = False


# Instancia global lista para importar
generation_config = GenerationConfig()
