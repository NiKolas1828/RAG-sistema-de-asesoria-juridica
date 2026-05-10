from src.processors.document_loader import process_documents, standardize
from src.processors.text_segmenter import segment_documents_for_article


def main():
    print("=== SISTEMA RAG: INICIO DE PIPELINE DE DATOS ===")

    print("\n[1/3] Cargando documentos...")
    process_documents()

    print("\n[2/3] Estandarizando contenido...")
    standardize()

    print("\n[3/3] Generando chunks para RAG...")
    segment_documents_for_article()

    print("\n=== PIPELINE FINALIZADO EXITOSAMENTE ===")


if __name__ == "__main__":
    main()
