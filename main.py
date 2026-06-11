# main.py
# ============================================================
# Punto de entrada principal del sistema RAG
# Modo 1 — ingesta:  python main.py --ingest
# Modo 2 — consulta: python main.py --query "¿Cuál es la multa por no usar casco?"
# Modo 3 — interactivo: python main.py
# ============================================================

import sys
import argparse
import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ─── Pipeline de ingesta (existente) ────────────────────────
from src.processors.document_loader import process_documents, standardize
from src.processors.text_segmenter import segment_documents_for_article
from src.processors.embedding_processor import run_embedding_pipeline

# ─── Pipeline de consulta (nuevo) ────────────────────────────
from src.retrieval.rag_pipeline import RAGPipeline
from src.generation.response_generator import ResponseGenerator
from src.memory.conversation_memory import ConversationMemory


def run_ingesta():
    """Pipeline de carga y procesamiento de documentos (sin cambios)."""
    print("=== SISTEMA RAG: INICIO DE PIPELINE DE DATOS ===")

    print("\n[1/3] Cargando documentos...")
    process_documents()

    print("\n[2/3] Estandarizando contenido...")
    standardize()

    print("\n[3/3] Generando chunks para RAG...")
    segment_documents_for_article()
    run_embedding_pipeline()

    print("\n=== PIPELINE FINALIZADO EXITOSAMENTE ===")


def run_consulta(pregunta: str, verbose: bool = False) -> dict:
    """
    Pipeline completo de consulta: retrieval → contexto → generación.

    Args:
        pregunta: Pregunta del ciudadano en lenguaje natural.
        verbose:  Si True, muestra detalles internos del retrieval.

    Returns:
        Diccionario con la respuesta y metadatos.
    """
    pipeline   = RAGPipeline()
    generator  = ResponseGenerator()

    # Paso 1: Retrieval + construcción de contexto + prompt
    # k=10: candidatos para ChromaDB (el reranker reduce a top 5)
    # max_documents=8: fragmentos que llegan al LLM
    rag_output = pipeline.run(pregunta, k=10, max_documents=8, verbose=verbose)

    # Paso 2: Generación de respuesta con LLM
    resultado  = generator.generate(rag_output)

    return resultado


def imprimir_resultado(resultado: dict):
    """Muestra el resultado de forma legible en consola."""
    print("\n" + "=" * 60)
    print("  RESPUESTA DEL SISTEMA")
    print("=" * 60)
    print(f"\n📋 Pregunta: {resultado.get('query_original', '')}\n")
    print(f"💬 Respuesta:\n\n{resultado.get('respuesta', '')}")
    print(f"\n🤖 Modelo usado : {resultado.get('modelo_usado', 'N/A')}")
    print(f"📊 Tokens prompt: {resultado.get('tokens_prompt', 0)}")
    print(f"✅ Estado       : {resultado.get('status', 'N/A')}")

    if resultado.get("error"):
        print(f"⚠️  Error        : {resultado['error']}")

    print("=" * 60 + "\n")


def run_interactivo():
    """Modo interactivo con memoria multi-turno: el ciudadano hace preguntas en un loop."""
    print("\n" + "=" * 60)
    print("  SISTEMA RAG — NORMAS DE TRÁNSITO COLOMBIANAS")
    print("  Escribe 'salir' para terminar | 'limpiar' para reiniciar la memoria")
    print("=" * 60 + "\n")

    pipeline  = RAGPipeline()
    generator = ResponseGenerator()
    memory    = ConversationMemory(max_turns=5)  # recuerda los últimos 5 turnos

    while True:
        try:
            pregunta = input("\U0001f6a6 Tu pregunta: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nHasta luego.")
            break

        if not pregunta:
            continue

        if pregunta.lower() in {"salir", "exit", "quit"}:
            print("Hasta luego.")
            break

        if pregunta.lower() in {"limpiar", "clear", "reset"}:
            memory.clear()
            print("\U0001f9f9 Memoria limpiada. Nueva sesión iniciada.\n")
            continue

        print("\n⏳ Consultando normas...\n")
        rag_output = pipeline.run(pregunta, k=10, max_documents=8)

        # Pasar el historial previo al generador para contexto multi-turno
        historial  = memory.get_history() if not memory.is_empty() else None
        resultado  = generator.generate(rag_output, history=historial)

        # Guardar el turno en memoria solo si fue exitoso
        if resultado.get("status") == "éxito" and resultado.get("respuesta"):
            memory.add_turn(
                user_message=pregunta,
                assistant_message=resultado["respuesta"],
            )
            if memory.turn_count() > 1:
                print(f"\U0001f9e0 Memoria activa: {memory.turn_count()} turnos previos en contexto\n")

        imprimir_resultado(resultado)


def main():
    parser = argparse.ArgumentParser(
        description="Sistema RAG — Normas de Tránsito Colombianas"
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Ejecutar pipeline de ingesta de documentos",
    )
    parser.add_argument(
        "--query",
        type=str,
        metavar="PREGUNTA",
        help='Hacer una consulta directa. Ej: --query "¿Qué es la licencia C1?"',
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mostrar detalles internos del retrieval",
    )

    args = parser.parse_args()

    if args.ingest:
        run_ingesta()

    elif args.query:
        resultado = run_consulta(args.query, verbose=args.verbose)
        imprimir_resultado(resultado)

    else:
        # Sin argumentos → modo interactivo
        run_interactivo()


if __name__ == "__main__":
    main()
