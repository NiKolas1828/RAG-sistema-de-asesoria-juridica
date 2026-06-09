# Especificación de Requisitos Funcionales y No Funcionales

## 1. Introducción y Contexto

Este documento establece la especificación detallada de los requisitos del **Sistema de Asesoría Jurídica RAG** para las normas de tránsito de Colombia. El objetivo del sistema es democratizar el acceso a la normativa vial, permitiendo que cualquier ciudadano realice preguntas en lenguaje natural (incluyendo jerga colombiana) y reciba respuestas jurídicamente fundamentadas, libres de alucinaciones y con citas normativas explícitas.

El sistema se basa en una arquitectura **RAG (Retrieval-Augmented Generation)** que combina almacenamiento relacional (SQLite), indexación vectorial local (ChromaDB) y generación de lenguaje con modelos LLM externos (Gemini 2.0 Flash como motor principal y Llama 3.3 70B vía Groq como fallback).

---

## 2. Identificación de Usuarios y Personas

Para garantizar que el sistema sea de utilidad, se identifican cuatro perfiles de usuario clave con necesidades y expectativas específicas:

### 2.1 Conductor / Ciudadano Común
*   **Perfil**: Propietario de vehículos, motociclistas, ciclistas o peatones sin formación jurídica.
*   **Necesidades**: 
    *   Entender si una sanción es justa o no.
    *   Saber el costo y los plazos de pago para multas y comparendos.
    *   Conocer los requisitos de trámites comunes (traspasos, licencias, SOAT, revisión técnico-mecánica).
*   **Expectativas**: Respuestas rápidas, en un lenguaje sencillo y cercano, libre de excesivos tecnicismos legales pero que a la vez le dé tranquilidad de estar respaldado por la norma.

### 2.2 Abogado / Asesor Jurídico
*   **Perfil**: Profesional del derecho, consultor o gestor de trámites de tránsito.
*   **Necesidades**:
    *   Encontrar la base legal exacta para redactar derechos de petición o recursos de apelación.
    *   Verificar la vigencia de resoluciones y decretos complementarios al Código Nacional de Tránsito.
*   **Expectativas**: Citas textuales y exactas de los artículos, indicación clara del documento del que proviene (ley, decreto, resolución) y precisión técnica rigurosa en el sustento de la respuesta.

### 2.3 Agente de Tránsito / Autoridad Vial
*   **Perfil**: Personal en campo encargado de la regulación del tráfico y la imposición de comparendos.
*   **Necesidades**:
    *   Corroborar rápidamente los códigos de infracción asociados a comportamientos viales (por ejemplo, los códigos de la Resolución 3027 de 2010).
    *   Consultar los límites de velocidad permitidos o las especificaciones técnicas requeridas para ciertos vehículos (como luces o llantas).
*   **Expectativas**: Consulta instantánea por consola o dispositivo móvil, con respuestas concisas que apunten directamente a la infracción y su sanción correspondiente.

### 2.4 Administrador de Datos del Sistema (IT)
*   **Perfil**: Desarrollador o analista de datos encargado de la operatividad del sistema.
*   **Necesidades**:
    *   Indexar nuevos documentos cuando se promulguen reformas a las leyes de tránsito.
    *   Verificar la salud del pipeline RAG y la correcta generación de embeddings locales.
*   **Expectativas**: Procesos de ingesta automatizados, logs comprensibles de errores, modularidad en el código y facilidad de actualización del corpus sin alterar el núcleo de búsqueda.

---

## 3. Requisitos Funcionales (RF)

### 3.1 Procesamiento e Ingesta de Documentos
*   **RF-01: Ingesta Multiformato**: El sistema debe cargar y extraer texto de documentos en formatos `.pdf`, `.docx`, `.doc`, `.html` e `.htm`.
*   **RF-02: Extracción con OCR**: Para archivos PDF que consistan en imágenes escaneadas (con menos de 100 caracteres extraíbles directamente), el sistema debe procesar el archivo mediante un motor OCR local (Tesseract) en idioma español.
*   **RF-03: Estandarización y Limpieza NLP**: El sistema debe realizar un proceso de limpieza del texto extraído, eliminando cabeceras institucionales repetitivas, pies de página, caracteres decorativos y espaciados redundantes. Adicionalmente, guardará una versión normalizada en minúsculas y sin caracteres especiales para optimizar la generación de embeddings.
*   **RF-04: Persistencia en SQLite**: Todos los documentos originales y procesados deben persistir en una base de datos relacional SQLite (`data/normas.db`) con tablas para `documentos`, `articulos` y `chunks` para control y trazabilidad.

### 3.2 Segmentación y Vectorización (Chunking & Embeddings)
*   **RF-05: Segmentación Jerárquica por Artículo**: En lugar de dividir el texto ciegamente por tamaño de caracteres, el sistema debe usar una máquina de estados para detectar la estructura legal (Títulos, Capítulos y Artículos). Cada fragmento de texto (*chunk*) debe conservar esta metadata de origen.
*   **RF-06: Tamaño de Fragmento Controlado**: El tamaño del fragmento por artículo se limitará a un estimado de 512 tokens con un solapamiento (*overlap*) de 50 tokens con el fragmento adyacente para conservar el hilo del contexto normativo.
*   **RF-07: Indexación Local de Embeddings**: Los fragmentos deben ser vectorizados localmente mediante el modelo multilingüe `paraphrase-multilingual-MiniLM-L12-v2` y almacenarse en una base de datos vectorial persistente local (ChromaDB).

### 3.3 Búsqueda y Recuperación (Retrieval)
*   **RF-08: Normalización de Consultas**: Toda consulta del usuario debe normalizarse (limpieza de espacios, conversión a minúsculas) antes de generar su representación vectorial.
*   **RF-09: Búsqueda Semántica**: El sistema debe recuperar los $k$ fragmentos más similares a la consulta utilizando similitud de coseno en ChromaDB.
*   **RF-10: Filtrado por Umbral de Similitud**: Se debe establecer un umbral mínimo de similitud para los fragmentos recuperados. Si el fragmento con mayor relevancia no supera este umbral, el sistema debe detener el flujo y evitar el consumo de APIs externas del LLM.

### 3.4 Construcción del Contexto y Prompting
*   **RF-11: Construcción Dinámica del Contexto**: El sistema debe estructurar un bloque de contexto delimitado, ordenando los fragmentos por similitud decreciente e indicando su fuente, artículo y porcentaje de relevancia.
*   **RF-12: Control de Límites de Tokens**: Si el volumen de texto de los fragmentos a enviar supera la ventana máxima asignada al contexto, el sistema debe recortar proporcionalmente los fragmentos o eliminar de forma iterativa el fragmento de menor relevancia.
*   **RF-13: Inyección de Directrices del Sistema (System Instructions)**: El prompt final debe inyectar instrucciones estrictas al LLM que le exijan anclarse 100% al contexto provisto, prohibir la invención de normativas o valores de multas, y forzar la estructura de la respuesta.

### 3.5 Generación y Tolerancia a Fallos
*   **RF-14: Generación Principal (Gemini)**: El sistema debe enviar el prompt al endpoint de Gemini 2.0 Flash de forma prioritaria.
*   **RF-15: Fallback de Generación (Groq)**: Si el cliente de Gemini detecta un error de Rate Limit (HTTP 429) por la saturación de solicitudes permitidas en el tier gratuito, debe redirigir la consulta de forma transparente a Groq consumiendo Llama 3.3 70B.
*   **RF-16: Respuestas por Defecto**: Si no hay contexto que supere el umbral de relevancia, o si ambos servicios de LLM fallan tras sus respectivos reintentos, el sistema debe retornar un mensaje cortés predefinido indicando la ausencia de información en la base de datos local y recomendando acudir a las entidades oficiales de tránsito.

---

## 4. Requisitos No Funcionales (RNF)

### 4.1 Rendimiento y Latencia
*   **RNF-01: Latencia del LLM**: El tiempo total transcurrido desde que se ingresa la consulta hasta que se obtiene la respuesta del LLM no debe superar los **3 segundos** en condiciones estándar de red.
*   **RNF-02: Eficiencia del Chunker y Vectorizador**: El proceso de segmentación y vectorización debe ser capaz de procesar lotes de texto en memoria RAM sin bloqueos, utilizando el procesamiento por lotes (*batch size* configurable de 100 fragmentos).

### 4.2 Precisión Jurídica y Fiabilidad
*   **RNF-03: Trazabilidad y Citas Obligatorias**: El 100% de las respuestas que posean sustento legal deben citar explícitamente la fuente de donde se extrajo la información. Ejemplo: *"Artículo 55 de la Ley 769 de 2002..."*
*   **RNF-04: Cero Alucinaciones (Fidelidad de Información)**: El LLM debe rechazar preguntas de carácter especulativo o que requieran información que no se encuentre explícitamente contenida en el bloque de contexto inyectado.

### 4.3 Seguridad y Privacidad
*   **RNF-05: Gestión Segura de Credenciales**: Las claves de API (`GEMINI_API_KEY`, `GROQ_API_KEY`) bajo ninguna circunstancia deben estar hardcodeadas en el código fuente. Se debe utilizar un archivo `.env` gestionado por `python-dotenv`.
*   **RNF-06: Sanitización de Entradas**: Las consultas del usuario deben ser limpiadas de secuencias de comandos y prompts maliciosos que intenten eludir las directrices del sistema (*prompt injection*).

### 4.4 Disponibilidad, Escalabilidad y Costos
*   **RNF-07: Costo Operativo Cero (Fase MVP)**: La arquitectura RAG debe operar en el tier gratuito del proveedor de LLM (Gemini 2.0 Flash permite hasta 15 RPM y 1.500 RPD).
*   **RNF-08: Alta Disponibilidad por Redundancia**: El sistema de fallback (Gemini -> Groq) debe garantizar que el servicio continúe disponible aun si uno de los endpoints experimenta caídas temporales o agotamiento de cuotas.

### 4.5 Portabilidad y Mantenibilidad
*   **RNF-09: Almacenamiento Local Autónomo**: Las bases de datos SQLite y ChromaDB deben residir localmente en directorios configurables del proyecto para facilitar su respaldo y portabilidad sin necesidad de configurar servidores de bases de datos remotos en la fase de MVP.
*   **RNF-10: Modularidad de Código**: El acoplamiento entre los componentes del RAG (Procesamiento de datos, Búsqueda, Contexto y Generación) debe ser mínimo, exponiendo interfaces claras para facilitar el reemplazo o actualización de cualquier módulo en fases futuras.
