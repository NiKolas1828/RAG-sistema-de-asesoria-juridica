# Selección de LLM para sistema RAG — Normas de Tránsito

> **Estado:** Decisión tomada · **Versión:** 1.0 · **Fecha:** 2026-04-27

## Tabla de contenidos

- [Contexto del proyecto](#contexto-del-proyecto)
- [Restricciones y requisitos](#restricciones-y-requisitos)
- [Criterios de evaluación](#criterios-de-evaluación)
- [Modelos candidatos](#modelos-candidatos)
- [Matriz de evaluación](#matriz-de-evaluación)
- [Modelo recomendado](#modelo-recomendado)
- [Segunda opción — fallback](#segunda-opción--fallback)
- [Arquitectura RAG propuesta](#arquitectura-rag-propuesta)
- [Consideraciones de escalabilidad](#consideraciones-de-escalabilidad)
- [Decisiones descartadas](#decisiones-descartadas)

---

## Contexto del proyecto

Sistema de preguntas y respuestas sobre normas de tránsito colombianas basado en arquitectura RAG (*Retrieval-Augmented Generation*). El corpus documental incluye el Código Nacional de Tránsito (Ley 769 de 2002), resoluciones del Ministerio de Transporte y decretos reglamentarios.

| Atributo | Valor |
|---|---|
| Tipo de sistema | RAG sobre corpus documental legal |
| Idioma principal | Español colombiano |
| Usuarios finales | Ciudadanos en general (app pública) |
| Despliegue | Cloud vía API de terceros |
| Presupuesto | Muy limitado — se priorizan opciones gratuitas |

---

## Restricciones y requisitos

### Requisitos funcionales

- Respuestas en español natural, claras y comprensibles para ciudadanos sin formación legal.
- Las respuestas deben citar explícitamente el artículo o resolución de origen.
- El modelo no debe generar información fuera del corpus recuperado (no alucinaciones).
- Latencia aceptable para app pública: objetivo < 3 segundos por respuesta.

### Restricciones técnicas y económicas

- Sin infraestructura propia de GPU. Despliegue 100% vía API REST.
- Presupuesto inicial de $0 USD/mes en costos de LLM. Se acepta escalar a costo mínimo cuando la carga lo justifique.
- La solución debe poder operar en el free tier de al menos un proveedor en la fase MVP.

---

## Criterios de evaluación

Se definieron cinco criterios ponderados según su impacto en el caso de uso específico:

| # | Criterio | Peso | Justificación |
|---|---|---|---|
| 1 | **Soporte en español** | 25% | El corpus y los usuarios son 100% hispanohablantes. Calidad lingüística directamente perceptible por el ciudadano. |
| 2 | **Tier gratuito** | 25% | Restricción de presupuesto dura en fase MVP. Determina viabilidad inicial del proyecto. |
| 3 | **Ventana de contexto** | 20% | Un chunk de norma legal puede ser extenso. Con k=5 chunks recuperados se necesita espacio suficiente sin truncar. |
| 4 | **Seguimiento de instrucciones** | 20% | Capacidad de mantenerse anclado al contexto RAG. Crítico en dominio legal para evitar alucinaciones. |
| 5 | **Latencia** | 10% | Experiencia de usuario aceptable, pero no crítica frente a los anteriores. |

---

## Modelos candidatos

Se evaluaron los modelos más relevantes del mercado para el perfil del proyecto: bajo costo, buena cobertura en español y disponibilidad vía API cloud.

### Gemini 2.0 Flash — Google AI Studio

Modelo de generación rápida de Google con una de las ventanas de contexto más grandes del mercado (1M tokens). Disponible gratuitamente vía Google AI Studio con límites de 15 RPM y 1.500 RPD.

### Llama 3.3 70B — Groq

Modelo open-source de Meta (70B parámetros) con inferencia ultrarrápida vía hardware LPU de Groq. API gratuita con rate limits. Excelente rendimiento en español por el tamaño del modelo.

### Mistral Small 3.1 — Mistral AI

Modelo europeo eficiente con buen manejo del español (entrenado con corpus multilingüe europeo). Solo disponible en tier de pago salvo créditos de prueba iniciales.

### GPT-4o mini — OpenAI

Versión económica de GPT-4o. Excelente español y seguimiento de instrucciones. No tiene tier gratuito; costo de $0.15/M tokens de entrada.

### Claude Haiku 3.5 — Anthropic

Modelo de bajo costo de Anthropic con excelente instrucción-following y ventana de 200K tokens. Sin tier gratuito; costo de $0.80/M tokens de entrada.

---

## Matriz de evaluación

Escala de puntaje por criterio: 1 (deficiente) a 5 (excelente). El score ponderado aplica los pesos definidos en la sección anterior.

| Modelo | Español (25%) | Tier gratuito (25%) | Contexto (20%) | Instrucciones (20%) | Latencia (10%) | **Score total** |
|---|---|---|---|---|---|---|
| **Gemini 2.0 Flash** | 5 | 5 | 5 | 5 | 5 | **5.00** |
| Llama 3.3 70B (Groq) | 4 | 4 | 3 | 4 | 5 | **3.95** |
| Mistral Small 3.1 | 4 | 2 | 3 | 4 | 4 | **3.35** |
| GPT-4o mini | 5 | 1 | 3 | 4 | 4 | **3.30** |
| Claude Haiku 3.5 | 5 | 1 | 4 | 5 | 3 | **3.25** |

> **Nota metodológica:** Los scores son cualitativos y comparativos, calibrados para el perfil específico del proyecto. No representan benchmarks formales. Si alguna restricción cambia (ej. datos confidenciales que no pueden salir del país), el análisis favorecería un modelo local como Mistral 7B via Ollama.

---

## Modelo recomendado

### Gemini 2.0 Flash

**Proveedor:** Google AI Studio / Google Cloud Vertex AI  
**Versión evaluada:** `gemini-2.0-flash` (estable)  
**Endpoint:** `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent`

#### Límites del tier gratuito

| Métrica | Límite gratuito |
|---|---|
| Solicitudes por minuto (RPM) | 15 |
| Solicitudes por día (RPD) | 1.500 |
| Tokens de entrada por minuto | 1.000.000 |
| Ventana de contexto máxima | 1.048.576 tokens |
| Tokens de salida máximos | 8.192 tokens |

#### Fortalezas para este caso de uso

- **Español de primer nivel:** Entrenado con corpus masivo en español. Genera texto natural, fluido y con vocabulario apropiado para ciudadanos.
- **Ventana de 1M tokens:** Permite ingerir el Código Nacional de Tránsito completo en un solo contexto si se requiere, o chunks muy generosos sin riesgo de truncamiento.
- **Instrucción-following robusto:** Responde de forma consistente a prompts que restringen la respuesta al contexto recuperado y exigen citas de artículos.
- **Latencia competitiva:** Tiempos de respuesta típicos de 500ms–1s para respuestas de longitud normal.
- **Multimodalidad:** Soporte nativo de imágenes, útil si en el futuro se incorporan señales de tránsito o infografías al corpus.
- **Costo de escala razonable:** Al superar el free tier, el costo es de $0.10/M tokens de entrada y $0.40/M tokens de salida — entre los más bajos del mercado para su nivel de capacidad.

#### Limitaciones a considerar

- El límite de 15 RPM puede ser un cuello de botella con tráfico concurrente alto. Mitigación: implementar cola de solicitudes y/o fallback a Groq.
- Las consultas de usuarios son enviadas a servidores de Google. Si en el futuro el proyecto maneja datos sensibles, evaluar Vertex AI con VPC Service Controls.
- Requiere gestión segura de API keys (nunca en frontend público).

#### Ejemplo de configuración mínima

```python
import google.generativeai as genai

genai.configure(api_key="TU_API_KEY")

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    system_instruction=(
        "Eres un asistente especializado en normas de tránsito colombianas. "
        "Responde ÚNICAMENTE basándote en los artículos recuperados que se te proporcionan. "
        "Siempre cita el artículo o resolución específica que respalda tu respuesta. "
        "Si la información no está en el contexto proporcionado, indica que no puedes responder esa pregunta."
    )
)

def query_rag(question: str, retrieved_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(retrieved_chunks)
    prompt = f"Contexto normativo recuperado:\n\n{context}\n\nPregunta del ciudadano: {question}"
    response = model.generate_content(prompt)
    return response.text
```

---

## Segunda opción — Fallback

### Llama 3.3 70B vía Groq

**Proveedor:** Groq Cloud  
**Endpoint:** `https://api.groq.com/openai/v1/chat/completions`  
**Compatibilidad:** OpenAI-compatible (drop-in replacement en la mayoría de frameworks)

Se recomienda implementar Groq + Llama 3.3 70B como **fallback automático** cuando Gemini supere el rate limit de 15 RPM. La lógica es sencilla:

```python
import httpx

async def query_with_fallback(question: str, chunks: list[str]) -> str:
    try:
        return await query_gemini(question, chunks)
    except RateLimitError:
        return await query_groq(question, chunks)
```

**Ventajas del fallback con Groq:**
- Inferencia LPU: latencias de ~200ms, más rápida que Gemini en muchos casos.
- API gratuita con límites propios (rate limits diferentes, complementarios).
- Llama 3.3 70B tiene buen rendimiento en español para tareas de Q&A sobre documentos.

**Limitación:** Contexto de 128K tokens (vs 1M de Gemini). Suficiente para RAG estándar con k=5 chunks de ~512 tokens.

---

## Arquitectura RAG propuesta

```
┌─────────────────────────────────────────────────────────────────┐
│                        FASE DE INGESTA                          │
│                                                                 │
│  PDFs / Docs  →  Parser  →  Chunker  →  Embeddings  →  Vector  │
│  (Código de              (512 tok,    (text-embed-    Store     │
│   Tránsito,              50 overlap)  004 / Google)  (Chroma / │
│   resoluciones)                                      pgvector)  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       FASE DE CONSULTA                          │
│                                                                 │
│  Pregunta  →  Embedding  →  Retrieval  →  Reranking  →  LLM   │
│  ciudadano    (mismo         (Top-K=5     (opcional,    Gemini  │
│               modelo)        cosine sim)  Cohere free)  2.0    │
│                                                         Flash   │
│                                              ↓ (rate limit)     │
│                                           Groq + Llama 3.3     │
└─────────────────────────────────────────────────────────────────┘
```

### Componentes del stack

| Capa | Herramienta recomendada | Alternativa | Costo |
|---|---|---|---|
| Framework RAG | LangChain / LlamaIndex | Haystack | Gratuito |
| Embeddings | `text-embedding-004` (Google) | `multilingual-e5-large` (HuggingFace, local) | Gratuito |
| Vector store (dev) | ChromaDB (local) | FAISS | Gratuito |
| Vector store (prod) | Supabase `pgvector` | Pinecone free tier | Gratuito |
| LLM principal | Gemini 2.0 Flash | — | Gratuito (free tier) |
| LLM fallback | Groq + Llama 3.3 70B | Groq + Llama 3.1 8B | Gratuito (free tier) |
| Reranking (opcional) | Cohere Rerank free tier | Cross-encoder local | Gratuito |
| Backend API | FastAPI (Python) | Express (Node.js) | Gratuito |

### Estrategia de chunking para documentos legales

Los documentos legales tienen estructura jerárquica (Libro > Título > Capítulo > Artículo). Se recomienda:

1. **Chunking por artículo:** Cada artículo es una unidad semántica autónoma. Preservar el número de artículo como metadata.
2. **Tamaño:** ~512 tokens por chunk con overlap de 50 tokens hacia el artículo anterior/siguiente para mantener contexto de sección.
3. **Metadata obligatoria por chunk:**
   - `fuente`: nombre del documento (ej. `Ley_769_2002`)
   - `articulo`: número de artículo (ej. `Art. 55`)
   - `capitulo`: capítulo o título de la sección
   - `fecha_vigencia`: fecha de la versión del documento

```python
chunk_metadata = {
    "fuente": "Código Nacional de Tránsito — Ley 769 de 2002",
    "articulo": "Art. 55",
    "capitulo": "Título IV — Normas de Comportamiento",
    "fecha_vigencia": "2024-01-01"
}
```

### Prompt system recomendado

```
Eres un asistente especializado en normas de tránsito de Colombia.
Tu función es responder preguntas de ciudadanos colombianos de manera
clara, precisa y en lenguaje comprensible para personas sin formación legal.

REGLAS ESTRICTAS:
1. Responde ÚNICAMENTE con base en los artículos y normas proporcionados en el contexto.
2. Siempre cita el artículo, ley o resolución específica que respalda tu respuesta.
   Ejemplo: "Según el Artículo 55 del Código Nacional de Tránsito (Ley 769 de 2002)..."
3. Si la pregunta no puede responderse con el contexto proporcionado, responde:
   "No encontré información sobre eso en las normas consultadas. Te recomiendo
   contactar al organismo de tránsito de tu municipio."
4. Nunca inventes multas, valores, plazos o procedimientos que no estén en el contexto.
5. Usa un lenguaje cercano y respetuoso, evitando tecnicismos legales innecesarios.
```

---

## Consideraciones de escalabilidad

Cuando el proyecto supere el free tier de Gemini (>1.500 consultas/día), la transición es directa:

| Volumen diario | Costo estimado Gemini Flash | Recomendación |
|---|---|---|
| < 1.500 consultas | $0 | Free tier |
| 1.500 – 10.000 consultas | ~$0.50 – $3/mes | Pago por uso, sin cambios de código |
| 10.000 – 100.000 consultas | $3 – $30/mes | Considerar caché de respuestas frecuentes |
| > 100.000 consultas | > $30/mes | Evaluar modelo local (Mistral 7B + vLLM) |

**Optimizaciones de costo para escala:**
- **Caché semántica:** Almacenar respuestas a preguntas frecuentes (ej. "¿Cuánto es la multa por no usar cinturón?"). Reducción estimada del 40–60% en llamadas al LLM.
- **Embedding caché:** Las queries de usuarios se repiten. Cachear embeddings con Redis o similar.
- **Reducción de chunks:** Ajustar k=3 en vez de k=5 si el reranker es bueno. Menos tokens de entrada = menor costo.

---

## Decisiones descartadas

### Modelos locales (Ollama + Mistral 7B)

**Razón de descarte:** Requiere servidor dedicado con GPU o CPU potente para latencias aceptables. Contradice la restricción de despliegue cloud sin infraestructura propia. Se mantiene como opción de largo plazo si el proyecto crece y los costos de API justifican el CAPEX de un servidor.

### GPT-4o mini

**Razón de descarte:** Sin tier gratuito. A igualdad de costo (pagando), Gemini 2.0 Flash ofrece mayor contexto (1M vs 128K tokens) a precio similar. No hay ventaja funcional que justifique el costo extra en este caso de uso.

### Claude Haiku 3.5

**Razón de descarte:** Sin tier gratuito y costo más alto que alternativas equivalentes ($0.80/M vs $0.10/M tokens). Excelente modelo, pero no es competitivo bajo la restricción de presupuesto de este proyecto.

### Mistral Small 3.1

**Razón de descarte:** El tier gratuito es solo para pruebas (créditos iniciales limitados), no sostenible para un MVP en producción. Al requerir pago, Gemini resulta superior en contexto y español.

---

*Documento generado como parte del proceso de arquitectura del sistema RAG de normas de tránsito. Revisar ante cambios en los precios o planes de los proveedores.*
