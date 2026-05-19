# Alcance del Sistema — RAG Normas de Tránsito Colombia

> **Historia de usuario:** Como líder del proyecto, quiero delimitar el alcance del sistema para evitar desviaciones durante el desarrollo.

| Atributo | Valor |
|---|---|
| Versión | 1.0 |
| Estado | En revisión |
| Responsable | Líder del Proyecto |

---

## 1. Descripción general

Sistema de preguntas y respuestas en lenguaje natural sobre normas de tránsito colombianas, basado en arquitectura RAG (*Retrieval-Augmented Generation*). Las respuestas se generan **ancladas exclusivamente al corpus documental legal**, con cita obligatoria del artículo o resolución de origen.

| Componente | Detalle |
|---|---|
| Tipo de sistema | RAG sobre corpus documental legal |
| Idioma | Español colombiano |
| Usuarios finales | Ciudadanos en general — app pública |
| Despliegue | 100% cloud vía API REST |
| Presupuesto LLM | $0 USD/mes en fase MVP (free tier obligatorio) |
| Latencia objetivo | < 3 segundos por respuesta |

---

## 2. Corpus documental incluido

El sistema opera **exclusivamente** sobre los siguientes documentos legales colombianos:

- Ley 769 de 2002 — Código Nacional de Tránsito Terrestre y sus modificaciones vigentes
- Resoluciones del Ministerio de Transporte relacionadas con circulación, señalización y sanciones
- Decretos reglamentarios que desarrollan el Código Nacional de Tránsito

> El corpus se indexa con chunking semántico. Se recuperan **k=5 fragmentos** por consulta para construir el contexto del LLM.

---

## 3. Funcionalidades incluidas

### 3.1 Consulta en lenguaje natural
- Recepción de preguntas escritas en español colombiano informal
- Recuperación de fragmentos relevantes del corpus mediante búsqueda semántica (embedding + vector store)
- Generación de respuesta clara, sin tecnicismos innecesarios para el ciudadano
- **Cita obligatoria** del artículo, resolución o decreto de origen en cada respuesta

### 3.2 Control de alucinaciones
- El LLM recibe instrucciones explícitas de responder solo con base en los fragmentos recuperados
- Si la información no está en el corpus, el sistema lo informa sin inventar datos
- Seguimiento estricto del system prompt para mantenerse anclado al contexto RAG

### 3.3 API REST pública
- Endpoints consumibles por una app web o móvil
- Respuestas estructuradas en JSON con campos: `respuesta`, `citas`, `fragmentos_fuente`
- Latencia objetivo < 3 segundos en condiciones normales de carga

### 3.4 Escalabilidad controlada
- Fase MVP operativa en el free tier de al menos un proveedor de LLM
- Arquitectura preparada para escalar a costo mínimo sin rediseño

---

## 4. Exclusiones explícitas

| Exclusión | Justificación |
|---|---|
| Asesoría jurídica personalizada | El sistema informa sobre normas; no interpreta casos individuales ni reemplaza a un abogado |
| Procesamiento de multas o trámites RUNT | Requiere integración con sistemas gubernamentales transaccionales |
| Normas de tránsito de otros países | El corpus es 100% colombiano |
| Normas municipales o departamentales específicas | Solo normativa nacional en el corpus inicial |
| Información en tiempo real (congestión, accidentes) | El sistema es documental y estático |
| Generación de documentos o certificados oficiales | El sistema solo responde preguntas textuales |
| Soporte multilingüe | Fase MVP solo en español colombiano |
| Autenticación o perfiles de usuario | Aplicación pública anónima en esta fase |
| Infraestructura GPU propia o modelos locales | Restricción presupuestaria — solo API REST de terceros |

---

## 5. Limitaciones técnicas conocidas

- Las respuestas son tan precisas como el corpus indexado — normas no incluidas no pueden ser respondidas.
- Sin memoria conversacional entre sesiones; cada consulta es independiente.
- La calidad de recuperación depende del chunking y los embeddings.
- En alta concurrencia, la latencia puede superar 3 s si el proveedor aplica throttling en el free tier.
- El corpus requiere **mantenimiento manual periódico** para reflejar actualizaciones normativas.

---

## 6. Entregables

| Entregable | Fase | Criterio de completitud |
|---|---|---|
| Documento de alcance (este archivo) | Inicio | Aprobado por el líder |
| Corpus documental indexado (vector store) | MVP | k=5 chunks con precisión ≥ 80% |
| Pipeline RAG (embedding + retrieval + LLM) | MVP | Cita presente en 100% de las respuestas |
| API REST documentada | MVP | OpenAPI publicado y probado |
| Interfaz ciudadana básica (web o móvil) | MVP | Pruebas de usabilidad con 5 usuarios |
| Informe de evaluación de modelos LLM | MVP | Matriz de criterios ponderados completada |
| Plan de mantenimiento del corpus | Post-MVP | Procedimiento de actualización documentado |

---

## 7. Criterios de aceptación

| # | Criterio | Estado |
|---|---|---|
| 1 | Se define qué incluye el sistema (RAG sobre normas de tránsito) | ✅ Cubierto — Secciones 2 y 3 |
| 2 | Se definen exclusiones explícitas | ✅ Cubierto — Sección 4 |
| 3 | Se documentan los entregables | ✅ Cubierto — Sección 6 |
| 4 | El alcance es validado por el líder del proyecto | ⏳ Pendiente de aprobación |

---

## 8. Aprobación

| Rol | Firma | Fecha |
|---|---|---|
| Líder del Proyecto | | |
| Responsable Técnico | | |