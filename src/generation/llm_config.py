# src/generation/llm_config.py
# ============================================================
# Configuración centralizada de parámetros para generación LLM
# Ajusta aquí temperature, tokens y umbrales sin tocar el código
# ============================================================

from dataclasses import dataclass, field
from typing import Optional


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

    # --- Prompt del sistema: estilo jurídico formal ---
    system_instruction: str = (
        "Eres un asistente especializado en normas de tránsito de Colombia. "
        "Tu función es responder preguntas de ciudadanos colombianos de manera "
        "clara, precisa y en lenguaje comprensible para personas sin formación legal.\n\n"
        "REGLAS ESTRICTAS:\n"
        "1. Responde ÚNICAMENTE con base en los artículos y normas proporcionados en el contexto.\n"
        "2. Siempre cita el artículo, ley o resolución específica que respalda tu respuesta. "
        "Ejemplo: 'Según el Artículo 55 del Código Nacional de Tránsito (Ley 769 de 2002)...'\n"
        "3. Si la pregunta no puede responderse con el contexto proporcionado, responde: "
        "'No encontré información sobre eso en las normas consultadas. "
        "Te recomiendo contactar al organismo de tránsito de tu municipio.'\n"
        "4. Nunca inventes multas, valores, plazos o procedimientos que no estén en el contexto.\n"
        "5. Usa un lenguaje cercano y respetuoso, evitando tecnicismos legales innecesarios.\n"
        "6. Estructura tu respuesta así: primero la respuesta directa, luego la explicación, "
        "finalmente las referencias normativas."
    )

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

    # Groq usa la misma system instruction para respuestas coherentes
    system_instruction: str = (
        "Eres un asistente especializado en normas de tránsito de Colombia. "
        "Tu función es responder preguntas de ciudadanos colombianos de manera "
        "clara, precisa y en lenguaje comprensible para personas sin formación legal.\n\n"
        "REGLAS ESTRICTAS:\n"
        "1. Responde ÚNICAMENTE con base en los artículos y normas proporcionados en el contexto.\n"
        "2. Siempre cita el artículo, ley o resolución específica que respalda tu respuesta.\n"
        "3. Si la pregunta no puede responderse con el contexto proporcionado, indícalo claramente.\n"
        "4. Nunca inventes multas, valores, plazos o procedimientos que no estén en el contexto.\n"
        "5. Usa un lenguaje cercano y respetuoso, evitando tecnicismos legales innecesarios."
    )

    timeout_seconds: int = 20
    max_retries: int = 2
    retry_wait_seconds: float = 1.0


@dataclass
class GenerationConfig:
    """Configuración global del módulo de generación."""

    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    groq: GroqConfig = field(default_factory=GroqConfig)

    # --- Umbral mínimo de similitud para generar respuesta ---
    # Si el retrieval no encuentra chunks con similitud >= este valor,
    # se responde con mensaje de "no encontré información" sin llamar al LLM.
    min_similarity_to_generate: float = 0.03

    # --- Logging de llamadas al LLM ---
    log_prompts: bool = False        # True en desarrollo, False en producción
    log_responses: bool = False


# Instancia global lista para importar
generation_config = GenerationConfig()
