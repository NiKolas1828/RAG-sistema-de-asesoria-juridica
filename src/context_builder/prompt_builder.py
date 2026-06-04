# src/context_builder/prompt_builder.py
# ============================================================
# Construcción del prompt RAG para el sistema de asesoría jurídica
#
# El sistema utiliza un único tipo de prompt: el Prompt RAG.
# No existe un "prompt sin contexto" porque responder sin documentos
# recuperados va en contra del propósito del sistema (prevenir
# alucinaciones). Cuando no hay contexto suficiente, el flujo
# se detiene en response_generator.py antes de llegar aquí.
#
# Estructura del prompt ensamblado:
#   [SYSTEM] → [INSTRUCTION] → [CONTEXT] → [QUESTION]
# ============================================================

from typing import Dict


# ─── Constantes del Prompt RAG ───────────────────────────────

SYSTEM_PROMPT = (
    "Eres un asistente jurídico experto especializado en Normas de Tránsito Colombianas.\n"
    "Tu objetivo es brindar respuestas claras, precisas y fundamentadas estrictamente "
    "en el contexto proporcionado.\n"
    "Reglas obligatorias:\n"
    "1. Basa tu respuesta ÚNICAMENTE en el bloque de [CONTEXT]. "
    "Si la información no responde la pregunta, indica explícitamente que no "
    "encontraste información suficiente.\n"
    "2. Cita siempre la norma y el artículo correspondiente utilizando los índices "
    "provistos en el contexto (ej. [1], [2]).\n"
    "3. Mantén un tono formal, objetivo y comprensible para un ciudadano sin "
    "formación legal.\n"
    "4. Bajo ninguna circunstancia inventes, modifiques o asumas artículos o leyes "
    "que no aparezcan en el contexto."
)
"""
System prompt del asistente jurídico.

Define la identidad, el dominio y las restricciones del modelo.
Las reglas están ordenadas por prioridad:
  Regla 1 → fidelidad al contexto RAG (anti-alucinación)
  Regla 2 → trazabilidad normativa (cita obligatoria)
  Regla 3 → accesibilidad ciudadana (tono sin tecnicismos)
  Regla 4 → integridad legal (nunca inventar leyes)
"""

INSTRUCTION_BLOCK = (
    "Estructura tu respuesta de la siguiente manera:\n"
    "- **Respuesta directa:** Responde de manera breve y clara al inicio.\n"
    "- **Detalle normativo:** Explica los detalles basándote en el contexto legal.\n"
    "- **Fuentes:** Lista al final las normas y artículos aplicados, "
    "referenciando los índices (ej. [1])."
)
"""
Instrucción de formato de salida para el LLM.

Fuerza una estructura tripartita en cada respuesta:
  1. Respuesta directa  → respuesta inmediata y sin rodeos
  2. Detalle normativo  → explicación con base en el contexto recuperado
  3. Fuentes            → lista de índices [n] que cita el ciudadano puede verificar

Esta estructura garantiza consistencia entre respuestas y
facilita la trazabilidad de las normas aplicadas.
"""


# ─── Constructor del Prompt RAG ──────────────────────────────

class PromptBuilder:
    """
    Ensambla el prompt RAG completo a partir de los fragmentos recuperados.

    El prompt resultante sigue siempre la misma estructura:
        [SYSTEM]      → rol, dominio y reglas del modelo
        [INSTRUCTION] → formato de salida esperado
        [CONTEXT]     → fragmentos normativos recuperados por el motor RAG
        [QUESTION]    → pregunta original del ciudadano

    Uso básico:
        builder = PromptBuilder()
        resultado = builder.build_prompt(
            contexto_formateado=contexto["contexto_formateado"],
            query="¿Cuál es la multa por no usar casco?"
        )
        prompt_final = resultado["prompt"]

    Uso con instrucciones adicionales:
        builder = PromptBuilder(
            additional_instructions="Responde en menos de 3 oraciones."
        )
    """

    def __init__(
        self,
        system_instructions: str = None,
        response_buffer: int = 128,
    ):
        """
        Args:
            system_instructions: Sobrescribe el SYSTEM_PROMPT por defecto.
                Usar solo si se necesita adaptar el sistema a un dominio diferente.
                Si es None, se usa SYSTEM_PROMPT (recomendado).
            response_buffer: Tokens reservados para la respuesta del modelo.
                Se restan del límite total para evitar truncamientos.
        """
        self.system_instructions = system_instructions or SYSTEM_PROMPT
        self.response_buffer = response_buffer

    def build_prompt(
        self,
        contexto_formateado: str,
        query: str,
        additional_instructions: str = "",
        max_prompt_tokens: int = 2000,
    ) -> Dict:
        """
        Ensambla el prompt RAG completo y estima el uso de tokens.

        No realiza llamadas al LLM; solo concatena bloques y estima tokens.

        Args:
            contexto_formateado: Texto con los fragmentos normativos recuperados,
                formateado por ContextBuilder (incluye fuente, artículo y relevancia).
            query: Pregunta original del ciudadano en lenguaje natural.
            additional_instructions: Instrucciones extra opcionales que se
                añaden al bloque [SYSTEM] (ej. restricciones de longitud).
            max_prompt_tokens: Límite total de tokens del prompt. Se usa para
                calcular cuántos tokens quedan disponibles para la respuesta.

        Returns:
            Diccionario con:
                - prompt (str): Prompt completo listo para enviar al LLM.
                - tokens_prompt (int): Estimado de tokens del prompt.
                - tokens_available_for_response (int): Tokens restantes para la respuesta.
        """
        system_block = f"[SYSTEM]\n{self.system_instructions}\n"
        if additional_instructions:
            system_block += f"{additional_instructions}\n"

        instruction_block = f"[INSTRUCTION]\n{INSTRUCTION_BLOCK}\n"
        context_block = "[CONTEXT]\n" + (contexto_formateado or "") + "\n"
        question_block = f"[QUESTION]\n{query}\n"

        prompt = "\n".join([system_block, instruction_block, context_block, question_block])

        tokens_prompt = self._estimate_tokens(prompt)
        tokens_available_for_response = max(
            0, max_prompt_tokens - tokens_prompt - self.response_buffer
        )

        return {
            "prompt": prompt,
            "tokens_prompt": tokens_prompt,
            "tokens_available_for_response": tokens_available_for_response,
        }

    def _estimate_tokens(self, text: str) -> int:
        """
        Estima el número de tokens del texto.

        Intenta usar tiktoken (cl100k_base) para mayor precisión.
        Si no está disponible, usa la aproximación: palabras / 0.75.
        """
        try:
            import tiktoken  # type: ignore[import-not-found]

            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            words = len(text.split())
            return int(words / 0.75)
