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
    "Tu objetivo es brindar respuestas claras, precisas, completas y fundamentadas estrictamente "
    "en el contexto proporcionado.\n"
    "Reglas obligatorias:\n"
    "1. Basa tu respuesta ÚNICAMENTE en el bloque de [CONTEXT]. "
    "Si la información no responde la pregunta, indica explícitamente que no "
    "encontraste información suficiente.\n"
    "2. Cita siempre la norma y el artículo correspondiente utilizando los índices "
    "provistos en el contexto (ej. [1], [2]).\n"
    "3. Mantén un tono formal, objetivo y comprensible para un ciudadano sin "
    "formación legal, pero sin omitir detalles técnicos o requisitos importantes mencionados en la norma.\n"
    "4. Bajo ninguna circunstancia inventes, modifiques o asumas artículos o leyes "
    "que no aparezcan en el contexto.\n"
    "5. CÁLCULO DE MULTAS (usa esta tabla siempre que el contexto mencione una infracción):\n"
    "   Valores vigentes en 2025:\n"
    "   - SMMLV 2025 = $1.423.500 COP (Salario Mínimo Mensual Legal Vigente)\n"
    "   - SMDLV 2025 = $47.450 COP (Salario Mínimo Diario Legal Vigente)\n"
    "   - UVT 2025   = $49.799 COP (Unidad de Valor Tributario — Ley 2294 de 2023)\n"
    "   Categorías de infracción y su valor:\n"
    "   ┌──────────────┬─────────┬────────────────────┬───────────────┐\n"
    "   │ Categoría    │ SMDLV   │ Valor en COP       │ Aprox. UVT    │\n"
    "   ├──────────────┼─────────┼────────────────────┼───────────────┤\n"
    "   │ A            │ 4       │ $189.800           │ ~3.8 UVT      │\n"
    "   │ B            │ 8       │ $379.600           │ ~7.6 UVT      │\n"
    "   │ C            │ 15      │ $711.750           │ ~14.3 UVT     │\n"
    "   │ D            │ 30      │ $1.423.500         │ ~28.6 UVT     │\n"
    "   │ E            │ 45      │ $2.135.250         │ ~42.9 UVT     │\n"
    "   └──────────────┴─────────┴────────────────────┴───────────────┘\n"
    "   TABLA DE INFRACCIONES COMUNES (Art. 131 Ley 769 de 2002):\n"
    "   ┌──────────┬───────────────────────────────────────────────┬──────────┐\n"
    "   │ Código   │ Infracción                                    │ Categ.   │\n"
    "   ├──────────┼───────────────────────────────────────────────┼──────────┤\n"
    "   │ C.24     │ Conducir moto sin casco (conductor/acompañ.)  │ C (15 S.)│\n"
    "   │ D.02     │ Conducir sin SOAT vigente                     │ D (30 S.)│\n"
    "   │ D.01     │ Conducir sin licencia de conducción           │ D (30 S.)│\n"
    "   │ D.04     │ No respetar semáforo en rojo / señal de PARE  │ D (30 S.)│\n"
    "   │ C.6      │ No usar cinturón de seguridad                 │ C (15 S.)│\n"
    "   │ C.14     │ Circular por sitios o horas restringidos      │ C (15 S.)│\n"
    "   │ D.08     │ Conducir sin luces reglamentarias             │ D (30 S.)│\n"
    "   │ E.1      │ Conducir bajo efectos del alcohol (1ª vez)    │ E (45 S.)│\n"
    "   │ B.7      │ Exceso de velocidad en vía urbana             │ B (8 S.) │\n"
    "   │ B.11     │ Propaganda en vidrios que obstruya visib.     │ B (8 S.) │\n"
    "   │ B.14     │ Vidrios polarizados sin permiso               │ B (8 S.) │\n"
    "   └──────────┴───────────────────────────────────────────────┴──────────┘\n"
    "   (S. = SMDLV 2025 = $47.450 COP c/u)\n"
    "   REGLA: Si el contexto menciona un código de infracción (ej. C.24, D.02, B.07), "
    "SIEMPRE muestra la tabla anterior con la fila correspondiente, indicando que "
    "según la Ley 2294 de 2023 las multas se cobran en UVT y advirtiendo que el valor "
    "puede reducirse un 50% si se paga dentro de los 5 días hábiles siguientes.\n"
    "6. REGLA ANTI-CONFUSIÓN DE MULTAS: Cuando calcules una multa, asegúrate que el artículo "
    "citado corresponda EXACTAMENTE a la infracción preguntada. "
    "NUNCA apliques la multa de un artículo sobre vidrios para una pregunta sobre cascos, "
    "ni mezcles infracciones distintas. Si el contexto no contiene el código exacto de la "
    "infracción preguntada, usa la TABLA DE INFRACCIONES COMUNES de arriba si aplica, "
    "o indica explícitamente que no encontraste el monto exacto."
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

# ─── Plantillas de Instrucción por Tipo de Pregunta ──────────────────────────
# Cada plantilla dicta exactamente el formato que debe usar el LLM según
# la intención detectada. Esto genera respuestas que se sienten hechas a medida.

_INSTRUCCION_MULTA = """\
Estructura tu respuesta EXACTAMENTE así:

**💰 Multa aplicable**
[Responde en 1-2 oraciones: qué infracción es, qué categoría y por qué.]

**📊 Tabla de sanción**
| Concepto | Valor |
|---|---|
| Categoría de infracción | [Letra] |
| SMDLV equivalente | [N] SMDLV |
| **Valor en pesos (2025)** | **$[X.XXX.XXX] COP** |
| Equivalente en UVT | ~[N] UVT |

**📋 Detalle normativo**
[Explica la norma exacta que aplica: qué artículo, qué resolución, cuáles son las condiciones exactas de la infracción según el texto de ley. Incluye excepciones si las hay.]

**⚡ Descuento por pronto pago**
Según el Código Nacional de Tránsito, el valor se reduce un **50%** si se paga dentro de los **5 días hábiles** siguientes a la imposición del comparendo.

**📌 Fuentes**
[Lista las normas y artículos citados, referenciando los índices del contexto, ej: [1], [2].]
"""

_INSTRUCCION_REQUISITOS = """\
Estructura tu respuesta EXACTAMENTE así:

**📋 Requisitos para [tema]**
[1 oración resumiendo el trámite o exigencia.]

**✅ Lista de documentos / condiciones requeridas**
[Usa una lista numerada. Para cada ítem: nombre del documento + descripción breve de qué debe decir o contener según la norma. Ejemplo:]
1. **Documento de identidad** — Cédula de ciudadanía original o tarjeta de identidad.
2. **Licencia de conducción** — Vigente y para la categoría del vehículo.

**⚠️ Condiciones adicionales**
[Si hay requisitos técnicos, de edad, de tiempo, o excepciones importantes, listarlas aquí.]

**📌 Fuentes**
[Lista las normas y artículos citados con sus índices, ej: [1], [2].]
"""

_INSTRUCCION_USO_CORRECTO = """\
Estructura tu respuesta EXACTAMENTE así:

**📖 Cómo [tema] correctamente según la ley**
[1-2 oraciones con la respuesta directa.]

**✅ Pasos / condiciones obligatorias**
[Lista numerada de cada condición o paso que exige la norma. Sé específico con los detalles técnicos (medidas, materiales, posición, etc.) tal como aparecen en el texto legal.]
1. [Condición 1 con detalle técnico]
2. [Condición 2 con detalle técnico]
...

**❌ Lo que NO está permitido**
[Lista de prohibiciones o condiciones que invalidan el uso correcto, si las menciona la norma.]

**📌 Fuentes**
[Lista las normas y artículos citados con sus índices, ej: [1], [2].]
"""

_INSTRUCCION_COMPARATIVO = """\
Estructura tu respuesta EXACTAMENTE así:

**⚖️ Comparación: [Elemento A] vs [Elemento B]**
[1 oración introductoria.]

**📊 Tabla comparativa**
| Aspecto | [Elemento A] | [Elemento B] |
|---|---|---|
| [Aspecto 1] | [Dato A] | [Dato B] |
| [Aspecto 2] | [Dato A] | [Dato B] |
[Completa la tabla con los aspectos relevantes encontrados en el contexto: multa, documentos, requisitos, restricciones, etc.]

**📝 Análisis**
[2-3 oraciones explicando las diferencias clave y cuándo aplica cada uno.]

**📌 Fuentes**
[Lista las normas y artículos citados con sus índices, ej: [1], [2].]
"""

_INSTRUCCION_INFRACCION = """\
Estructura tu respuesta EXACTAMENTE así:

**🚨 Situación: Comparendo impuesto**
[1 oración describiendo qué infracción se cometió y su categoría según la norma.]

**💰 Valor de la multa**
| Concepto | Valor |
|---|---|
| Categoría | [Letra] |
| Valor normal (2025) | $[X.XXX.XXX] COP |
| **Con descuento (pago en 5 días hábiles)** | **$[X.XXX.XXX] COP (-50%)** |

**📋 Pasos a seguir**

**Opción 1 — Pagar con descuento (recomendada)**
1. Tienes **5 días hábiles** desde la fecha del comparendo.
2. Paga el 50% del valor en el organismo de tránsito de tu ciudad o en línea.
3. Guarda el comprobante de pago.

**Opción 2 — Impugnar el comparendo**
1. Tienes **5 días hábiles** para presentar **Recurso de Reposición** ante el organismo de tránsito.
2. Debes argumentar por qué la infracción fue incorrecta (error en los datos, mal identificado, etc.).
3. Si el recurso es negado, puedes apelar en **Apelación** ante el superior jerárquico.

**⚠️ Consecuencias de no pagar ni impugnar**
La deuda se incrementa y puede quedar registrada en tu historial de conducción, afectando la renovación de tu licencia.

**📌 Fuentes**
[Lista las normas y artículos citados con sus índices, ej: [1], [2].]
"""

_INSTRUCCION_PROCEDIMIENTO = """\
Estructura tu respuesta EXACTAMENTE así:

**🗂️ Procedimiento: [Trámite/Proceso]**
[1 oración describiendo de qué trata el trámite.]

**📋 Pasos del procedimiento**
[Lista numerada clara y ordenada. Para cada paso: qué hacer, dónde hacerlo y qué documentos llevar si aplica.]
1. **Paso 1** — [Descripción]
2. **Paso 2** — [Descripción]
...

**⏱️ Tiempos y plazos**
[Si la norma menciona plazos, fechas límite o tiempos de respuesta, lístelos aquí.]

**📌 Fuentes**
[Lista las normas y artículos citados con sus índices, ej: [1], [2].]
"""

_INSTRUCCION_GENERAL = """\
Estructura tu respuesta de la siguiente manera:

**Respuesta directa:** Responde de manera breve y clara al inicio.

**Detalle normativo:** Explica de forma exhaustiva los detalles basándote en el contexto legal. \
Incluye TODAS las condiciones, excepciones, descripciones y especificaciones técnicas que mencione \
la norma. No omitas detalles aunque parezcan menores. Si la norma especifica medidas, materiales, \
posiciones o condiciones técnicas, inclúyelas textualmente.

**Fuentes:** Lista al final las normas y artículos aplicados, referenciando los índices (ej. [1]).
"""

# Mapa de tipo → plantilla
from src.context_builder.question_classifier import QuestionType

INSTRUCTION_TEMPLATES = {
    QuestionType.MULTA:        _INSTRUCCION_MULTA,
    QuestionType.REQUISITOS:   _INSTRUCCION_REQUISITOS,
    QuestionType.USO_CORRECTO: _INSTRUCCION_USO_CORRECTO,
    QuestionType.COMPARATIVO:  _INSTRUCCION_COMPARATIVO,
    QuestionType.INFRACCION:   _INSTRUCCION_INFRACCION,
    QuestionType.PROCEDIMIENTO: _INSTRUCCION_PROCEDIMIENTO,
    QuestionType.GENERAL:      _INSTRUCCION_GENERAL,
}

# Alias para compatibilidad con código existente
INSTRUCTION_BLOCK = _INSTRUCCION_GENERAL


# ─── Constructor del Prompt RAG ──────────────────────────────

class PromptBuilder:
    """
    Ensambla el prompt RAG completo a partir de los fragmentos recuperados.

    Detecta automáticamente el tipo de pregunta (multa, requisitos, uso correcto,
    comparativo, infracción, procedimiento, general) y aplica la plantilla de
    instrucción más adecuada para ese formato.

    El prompt resultante sigue siempre la estructura:
        [SYSTEM]      → rol, dominio y reglas del modelo
        [INSTRUCTION] → formato de salida esperado (adaptado al tipo)
        [CONTEXT]     → fragmentos normativos recuperados por el motor RAG
        [QUESTION]    → pregunta original del ciudadano

    Uso básico:
        builder = PromptBuilder()
        resultado = builder.build_prompt(
            contexto_formateado=contexto["contexto_formateado"],
            query="¿Cuál es la multa por no usar casco?"
        )
        prompt_final = resultado["prompt"]
    """

    def __init__(
        self,
        system_instructions: str = None,
        response_buffer: int = 128,
    ):
        """
        Args:
            system_instructions: Sobrescribe el SYSTEM_PROMPT por defecto.
            response_buffer: Tokens reservados para la respuesta del modelo.
        """
        from src.context_builder.question_classifier import QuestionClassifier
        self.system_instructions = system_instructions or SYSTEM_PROMPT
        self.response_buffer = response_buffer
        self._classifier = QuestionClassifier()

    def build_prompt(
        self,
        contexto_formateado: str,
        query: str,
        additional_instructions: str = "",
        max_prompt_tokens: int = 2000,
        question_type=None,
    ) -> Dict:
        """
        Ensambla el prompt RAG completo con el formato adecuado al tipo de pregunta.

        Args:
            contexto_formateado: Texto con los fragmentos normativos recuperados.
            query: Pregunta original del ciudadano en lenguaje natural.
            additional_instructions: Instrucciones extra opcionales para el [SYSTEM].
            max_prompt_tokens: Límite total de tokens del prompt.
            question_type: Tipo detectado externamente (QuestionType). Si es None,
                se detecta automáticamente con el QuestionClassifier.

        Returns:
            Diccionario con:
                - prompt (str): Prompt completo listo para enviar al LLM.
                - tokens_prompt (int): Estimado de tokens del prompt.
                - tokens_available_for_response (int): Tokens restantes para la respuesta.
                - question_type (str): Tipo de pregunta detectado.
        """
        # Detectar tipo de pregunta si no viene dado
        if question_type is None:
            question_type = self._classifier.classify(query)

        # Seleccionar la plantilla de instrucción correcta
        instruction = INSTRUCTION_TEMPLATES.get(question_type, _INSTRUCCION_GENERAL)

        system_block = f"[SYSTEM]\n{self.system_instructions}\n"
        if additional_instructions:
            system_block += f"{additional_instructions}\n"

        instruction_block = f"[INSTRUCTION]\n{instruction}\n"
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
            "question_type": question_type.value,
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

