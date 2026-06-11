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


def run_reindex():
    """
    Re-indexa todos los documentos desde cero usando el chunking inteligente.

    Pasos:
      1. Limpia la tabla chunks en SQLite (resetea el segmentador).
      2. Limpia la colección ChromaDB (borra todos los vectores).
      3. Vuelve a segmentar con _merge_and_split_articles (nuevo chunking).
      4. Re-genera todos los embeddings y los sube a ChromaDB.

    Los PDFs y documentos originales NO se tocan.
    """
    import sqlite3
    from src.config import DB_PATH

    print("=" * 60)
    print("  RE-INDEXACIÓN COMPLETA — Chunking Inteligente")
    print("=" * 60)
    print("⚠️  Se borrarán todos los chunks y vectores actuales.")
    print("   Los PDFs originales NO se modifican.")
    print()

    # ── Paso 1: Limpiar tabla chunks en SQLite ────────────────
    print("[1/3] Limpiando chunks en SQLite...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chunks")
    cursor.execute("DELETE FROM sqlite_sequence WHERE name='chunks'")
    conn.commit()
    conn.close()
    print("      ✅ Tabla chunks vaciada.")

    # ── Paso 2: Limpiar ChromaDB ──────────────────────────────
    print("[2/3] Limpiando colección ChromaDB...")
    try:
        import chromadb
        from src.config import CHROMA_PATH
        COLLECTION_NAME = "embeddings"   # definido en src/vector_store/vector_manager.py
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"      ✅ Colección '{COLLECTION_NAME}' eliminada.")
        except Exception:
            print(f"      ℹ️  Colección '{COLLECTION_NAME}' no existía aún.")
    except Exception as e:
        print(f"      ⚠️  Error limpiando ChromaDB: {e}")

    # ── Paso 3: Re-segmentar con chunking inteligente ─────────
    print("[3/3] Re-segmentando documentos con chunking inteligente...")
    segment_documents_for_article()

    # ── Paso 4: Re-generar embeddings ────────────────────────
    print("[4/4] Generando embeddings y subiendo a ChromaDB...")
    run_embedding_pipeline()

    print()
    print("=" * 60)
    print("  ✅ RE-INDEXACIÓN COMPLETADA")
    print("=" * 60)


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
    # k=20: candidatos para ChromaDB (el reranker reduce a top 8)
    # max_documents=8: fragmentos que llegan al LLM
    rag_output = pipeline.run(pregunta, k=20, max_documents=8, verbose=verbose)

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
    # Mostrar el tipo de pregunta detectado para trazabilidad
    q_type = resultado.get("question_type", "")
    if q_type:
        tipo_emoji = {
            "multa": "💰", "requisitos": "📋", "uso_correcto": "📖",
            "comparativo": "⚖️", "infraccion": "🚨", "procedimiento": "🗂️",
            "general": "💬",
        }.get(q_type, "💬")
        print(f"🎯 Tipo de consulta: {tipo_emoji} {q_type}")
    print(f"✅ Estado       : {resultado.get('status', 'N/A')}")

    if resultado.get("status") == "fuera_de_dominio":
        print("🚫 Consulta detectada fuera del dominio de tránsito colombiano.")

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
        # Obtener historial previo para contexto multi-turno
        historial  = memory.get_history() if not memory.is_empty() else None
        
        rag_output = pipeline.run(pregunta, k=20, max_documents=8, history=historial)

        # Pasar el historial al generador
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

    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Re-indexar todos los documentos desde cero con el nuevo chunking inteligente",
    )

    args = parser.parse_args()

    if args.reindex:
        run_reindex()

    elif args.ingest:
        run_ingesta()

    elif args.query:
        resultado = run_consulta(args.query, verbose=args.verbose)
        imprimir_resultado(resultado)

    else:
        # Sin argumentos → modo interactivo
        run_interactivo()


if __name__ == "__main__":
    main()
