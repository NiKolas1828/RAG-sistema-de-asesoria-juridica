# Diseño de Prompts — Sistema RAG de Asesoría Jurídica

> **Historia de usuario:** Como desarrollador del sistema RAG, quiero diseñar y documentar los prompts para garantizar respuestas coherentes y alineadas con el contexto jurídico.

| Atributo    | Valor                                          |
|-------------|------------------------------------------------|
| Versión     | 2.0                                            |
| Estado      | ✅ Verificado con tests                         |
| Archivo     | `src/context_builder/prompt_builder.py`        |
| Tests       | `test_prompt_builder.py` — 8/8 pasaron         |

---

## 1. Principio de Diseño

El sistema utiliza **un único tipo de prompt: el Prompt RAG**.

No existe un "prompt base sin contexto" porque responder sin documentos normativos recuperados iría en contra del propósito del sistema: **prevenir que el LLM invente artículos o multas que no existen**. El caso en que no hay contexto suficiente ya está manejado en `response_generator.py`, que retorna el mensaje estándar sin llamar al LLM.

```
Usuario pregunta
     ↓
RAG recupera fragmentos normativos (k=5)
     ↓
¿Similitud suficiente? ──NO──→ "No encontré información suficiente..."
     ↓ SÍ
PromptBuilder ensambla el Prompt RAG
     ↓
LLM genera la respuesta (Gemini → Groq fallback)
```

---

## 2. Estructura del Prompt RAG

El prompt se ensambla dinámicamente en `PromptBuilder.build_prompt()` concatenando 4 bloques en orden fijo.

### Bloque 1 — `[SYSTEM]`
**Constante:** `SYSTEM_PROMPT`

Define la identidad, el dominio y las restricciones del modelo. Contiene 4 reglas obligatorias ordenadas por prioridad:

```text
[SYSTEM]
Eres un asistente jurídico experto especializado en Normas de Tránsito Colombianas.
Tu objetivo es brindar respuestas claras, precisas y fundamentadas estrictamente
en el contexto proporcionado.
Reglas obligatorias:
1. Basa tu respuesta ÚNICAMENTE en el bloque de [CONTEXT]. Si la información no
   responde la pregunta, indica explícitamente que no encontraste información suficiente.
2. Cita siempre la norma y el artículo correspondiente utilizando los índices
   provistos en el contexto (ej. [1], [2]).
3. Mantén un tono formal, objetivo y comprensible para un ciudadano sin formación legal.
4. Bajo ninguna circunstancia inventes, modifiques o asumas artículos o leyes
   que no aparezcan en el contexto.
```

| Regla | Propósito |
|-------|-----------|
| 1     | Fidelidad al contexto RAG — anti-alucinación |
| 2     | Trazabilidad normativa — cita obligatoria |
| 3     | Accesibilidad ciudadana — sin tecnicismos |
| 4     | Integridad legal — prohibición explícita de inventar |

---

### Bloque 2 — `[INSTRUCTION]`
**Constante:** `INSTRUCTION_BLOCK`

Fuerza al modelo a estructurar su salida de forma consistente en 3 partes:

```text
[INSTRUCTION]
Estructura tu respuesta de la siguiente manera:
- **Respuesta directa:** Responde de manera breve y clara al inicio.
- **Detalle normativo:** Explica los detalles basándote en el contexto legal.
- **Fuentes:** Lista al final las normas y artículos aplicados, referenciando
  los índices (ej. [1]).
```

| Sección | Propósito |
|---------|-----------|
| Respuesta directa | Respuesta inmediata y sin rodeos para el ciudadano |
| Detalle normativo | Explicación fundamentada en los fragmentos recuperados |
| Fuentes           | Trazabilidad: el ciudadano puede verificar la norma |

---

### Bloque 3 — `[CONTEXT]`
Insertado dinámicamente por `ContextBuilder`. Contiene los fragmentos normativos recuperados de ChromaDB, ordenados por similitud semántica.

```text
[CONTEXT]
NORMATIVAS JURÍDICAS RELEVANTES
════════════════════════════════════════════════════════════

[1] Código Nacional de Tránsito (Ley 769 de 2002) | Art. 106 | Relevancia: 88.5%
------------------------------------------------------------
Todo conductor de motocicleta, motoneta y bicicleta motorizada debe portar
casco de seguridad homologado. El incumplimiento acarrea multa de 15 salarios
mínimos legales diarios vigentes.
```

---

### Bloque 4 — `[QUESTION]`
La pregunta original del ciudadano sin modificaciones.

```text
[QUESTION]
¿Cuál es la multa por no usar casco en moto?
```

---

## 3. Ejemplos de Entrada y Salida

### Ejemplo A — Consulta con contexto suficiente

**Prompt ensamblado (resumido):**
```text
[SYSTEM]
Eres un asistente jurídico experto especializado en Normas de Tránsito Colombianas...
Reglas obligatorias:
1. Basa tu respuesta ÚNICAMENTE en el bloque de [CONTEXT]...
4. Bajo ninguna circunstancia inventes...

[INSTRUCTION]
Estructura tu respuesta de la siguiente manera:
- **Respuesta directa:**...
- **Detalle normativo:**...
- **Fuentes:**...

[CONTEXT]
[1] Código Nacional de Tránsito (Ley 769 de 2002) | Art. 106 | Relevancia: 88.5%
Todo conductor de motocicleta debe portar casco de seguridad homologado.
El incumplimiento acarrea multa de 15 salarios mínimos legales diarios vigentes.

[QUESTION]
¿Cuál es la multa por no usar casco en moto?
```

**Respuesta del LLM:**
```markdown
**Respuesta directa:**
La multa por no usar casco en motocicleta equivale a 15 salarios mínimos legales
diarios vigentes (SMLDV).

**Detalle normativo:**
Según el Artículo 106 del Código Nacional de Tránsito (Ley 769 de 2002), todo
conductor de motocicleta, motoneta y bicicleta motorizada está obligado a portar
casco de seguridad homologado mientras conduce. El incumplimiento de esta norma
genera la sanción económica indicada [1].

**Fuentes:**
- [1] Código Nacional de Tránsito — Ley 769 de 2002, Artículo 106.
```

---

### Ejemplo B — Consulta fuera del dominio de tránsito

**Contexto recuperado** (similitud baja, fragmento no relacionado):
```text
[1] Ley 769 de 2002 | Art. 30 | Relevancia: 42.1%
Equipos de prevención. Ningún vehículo podrá transitar sin portar...
```

**Pregunta:** `¿Cuánto tiempo tengo para pagar mis impuestos prediales?`

**Respuesta del LLM (Regla 1 activa):**
```markdown
**Respuesta directa:**
No encontré información suficiente en las normas consultadas para responder
esta pregunta.

**Detalle normativo:**
El contexto proporcionado no contiene normativas relacionadas con impuestos
prediales ni plazos fiscales. Este sistema está especializado exclusivamente
en Normas de Tránsito Colombianas.

**Fuentes:**
- Ninguna fuente del contexto aplica a esta consulta.
```

> [!NOTE]
> Si la similitud es menor al umbral configurado (`min_similarity_to_generate`), el sistema retorna `RESPUESTA_SIN_CONTEXTO` directamente desde `response_generator.py` sin llegar a construir el prompt.

---

## 4. Variaciones Evaluadas Durante el Diseño

Se probaron tres variantes del system prompt. Los resultados están verificados mediante los tests en `test_prompt_builder.py` y las pruebas de generación en `test_generation.py`.

| Versión | System prompt | Problema encontrado |
|---------|---------------|---------------------|
| **V1** | `"Eres un asistente jurídico. Responde usando el contexto."` | El LLM usaba conocimiento general cuando el contexto era parcial. No citaba fuentes. |
| **V2** | `"Responde solo con el contexto. No uses conocimiento previo."` | Respuestas robóticas. No infería conclusiones lógicas de los artículos. Sin estructura de salida. |
| **V3 (actual)** | Reglas 1–4 + `INSTRUCTION_BLOCK` con estructura tripartita | Cita correctamente `[1]`, responde con claridad ciudadana y rechaza preguntas fuera del dominio. |

**Criterio de selección:** V3 equilibra fidelidad legal (no alucina) con legibilidad ciudadana (tono accesible y estructura clara).

---

## 5. Guía de Uso y Reutilización

### Uso básico

```python
from src.context_builder.prompt_builder import PromptBuilder

builder = PromptBuilder()
resultado = builder.build_prompt(
    contexto_formateado=contexto["contexto_formateado"],  # desde ContextBuilder
    query="¿Cuál es la multa por no usar casco?"
)

prompt_final  = resultado["prompt"]
tokens_usados = resultado["tokens_prompt"]
tokens_libres = resultado["tokens_available_for_response"]
```

### Agregar instrucciones adicionales puntuales

```python
resultado = builder.build_prompt(
    contexto_formateado=contexto["contexto_formateado"],
    query="¿Cuál es la multa por no usar casco?",
    additional_instructions="Responde en máximo 3 oraciones."
)
```

### Sobrescribir el system prompt (casos especiales)

```python
# Solo si se adapta el sistema a un dominio diferente al tránsito colombiano
builder = PromptBuilder(
    system_instructions="Eres un experto en normativas ambientales..."
)
```

> [!WARNING]
> Sobrescribir `system_instructions` elimina las 4 reglas anti-alucinación del dominio de tránsito. Úsalo solo si cambias de dominio completamente.

### Acceder a las constantes directamente

```python
from src.context_builder.prompt_builder import SYSTEM_PROMPT, INSTRUCTION_BLOCK

# Para inspeccionarlas, documentarlas o usarlas en otros contextos
print(SYSTEM_PROMPT)
print(INSTRUCTION_BLOCK)
```

---

## 6. Criterios de Aceptación

| # | Criterio | Estado |
|---|----------|--------|
| 1 | Se definen prompts base para consultas jurídicas | ✅ `SYSTEM_PROMPT` como constante nombrada con 4 reglas |
| 2 | Se incluye manejo de contexto (normas, artículos) | ✅ Bloque `[CONTEXT]` con fragmentos de ChromaDB |
| 3 | Se documentan ejemplos de entrada/salida | ✅ Ejemplos A y B — con contexto y fuera de dominio |
| 4 | El documento es entendible y reutilizable | ✅ Sección 5 con guía de uso y reutilización |
