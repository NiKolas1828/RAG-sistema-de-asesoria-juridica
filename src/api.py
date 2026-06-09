"""FastAPI entrypoint for RAG legal consultas.
    ejecuta: python -m uvicorn src.api:app --reload
    pruebas: curl -X POST "http://localhost:8000/consulta" -H "Content-Type: application/json" -d '{"question": "¿Cuál es la multa por no usar casco en moto?"}'
"""

from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from main import run_consulta


app = FastAPI(
    title="RAG Normas de Transito",
    description="API REST para consultas juridicas sobre normas de transito.",
    version="1.0.0",
)


class ConsultaRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Pregunta juridica")
    verbose: bool = Field(default=False, description="Mostrar detalles internos")


class ConsultaResponse(BaseModel):
    answer: str
    status: str
    model: Optional[str] = None
    prompt_tokens: Optional[int] = None
    original_question: str
    error: Optional[str] = None



class HealthResponse(BaseModel):
    status: str


class VersionResponse(BaseModel):
    service: str
    version: str


@app.post("/consulta", response_model=ConsultaResponse)
def consulta(payload: ConsultaRequest) -> ConsultaResponse:
    resultado = run_consulta(payload.question, verbose=payload.verbose)

    if not isinstance(resultado, dict):
        raise HTTPException(status_code=500, detail="Respuesta invalida del motor")

    if resultado.get("error"):
        status = "error"
    else:
        status = resultado.get("status", "ok")

    return ConsultaResponse(
        answer=resultado.get("respuesta", ""),
        status=status,
        model=resultado.get("modelo_usado"),
        prompt_tokens=resultado.get("tokens_prompt"),
        original_question=resultado.get("query_original", payload.question),
        error=resultado.get("error"),
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/version", response_model=VersionResponse)
def version() -> VersionResponse:
    return VersionResponse(service="rag-transito", version=app.version)

