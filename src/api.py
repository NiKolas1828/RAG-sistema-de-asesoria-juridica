from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.retrieval.rag_pipeline import RAGPipeline
from src.generation.response_generator import ResponseGenerator
import logging
import sqlite3
from src.config import DB_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("rag_api")

app = FastAPI(
    title="Sistema RAG - Asesoría Jurídica de Tránsito",
    description="API REST para consultar las normas de tránsito colombianas utilizando RAG.",
    version="1.0"
)

# Inicialización diferida/única del pipeline y generador al arrancar la app
pipeline = None
generator = None

try:
    pipeline = RAGPipeline()
    generator = ResponseGenerator()
    logger.info("Componentes RAG (Pipeline y Generator) inicializados con éxito.")
except Exception as e:
    logger.error(f"Error crítico al inicializar componentes RAG: {e}", exc_info=True)


class QueryRequest(BaseModel):
    pregunta: str


@app.get("/")
def read_root():
    return {
        "app": "RAG Sistema de Asesoría Jurídica de Tránsito Colombiano",
        "docs_url": "/docs",
        "status": "running"
    }


@app.get("/health")
def health_check():
    health_status = {
        "status": "healthy",
        "chromadb_connection": "unknown",
        "sqlite_db": "unknown"
    }
    
    # 1. Verificar conexión a ChromaDB
    if pipeline and pipeline.search_handler and pipeline.search_handler.search_engine:
        try:
            count = pipeline.search_handler.search_engine.collection.count()
            health_status["chromadb_connection"] = f"connected (embeddings count: {count})"
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["chromadb_connection"] = f"failed: {str(e)}"
    else:
        health_status["status"] = "unhealthy"
        health_status["chromadb_connection"] = "not_initialized"
        
    # 2. Verificar acceso a SQLite
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM documentos")
        doc_count = c.fetchone()[0]
        health_status["sqlite_db"] = f"accessible (documentos count: {doc_count})"
        conn.close()
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["sqlite_db"] = f"failed: {str(e)}"
        
    return health_status


@app.post("/query")
def post_query(request: QueryRequest):
    if not pipeline or not generator:
        raise HTTPException(
            status_code=503,
            detail="El servicio RAG no está inicializado correctamente. Verifica los logs."
        )
        
    if not request.pregunta.strip():
        raise HTTPException(
            status_code=400,
            detail="La pregunta no puede estar vacía."
        )
        
    try:
        logger.info(f"Procesando consulta de la API: '{request.pregunta}'")
        
        # Paso 1: Retrieval + prompt
        rag_output = pipeline.run(request.pregunta, k=15, max_documents=10)
        
        # Paso 2: Generación de respuesta con LLM (Gemini -> Groq fallback)
        resultado = generator.generate(rag_output)
        
        # Estructurar fragmentos fuente y citas para el cliente
        chunks = rag_output.get("contexto", {}).get("chunks_seleccionados", []) if rag_output.get("contexto") else []
        
        citas = []
        fragmentos = []
        for c in chunks:
            meta = c.get("metadata", {})
            fuente = meta.get("fuente", "Desconocida")
            articulo = meta.get("articulo", "")
            
            # Cita formateada
            cita = f"{articulo} - {fuente}" if articulo else fuente
            if cita not in citas:
                citas.append(cita)
                
            # Fragmento completo
            fragmentos.append({
                "texto": c.get("texto", ""),
                "similitud": c.get("similitud", 0.0),
                "metadata": meta
            })
            
        return {
            "query_original": request.pregunta,
            "respuesta": resultado.get("respuesta") or resultado.get("error") or "No se pudo generar una respuesta.",
            "modelo_usado": resultado.get("modelo_usado") or "sin_llm",
            "tokens_prompt": resultado.get("tokens_prompt", 0),
            "status": resultado.get("status", "error"),
            "citas": citas,
            "fragmentos_fuente": fragmentos
        }
    except Exception as e:
        logger.error(f"Error al procesar la query: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error interno del servidor RAG: {str(e)}"
        )
