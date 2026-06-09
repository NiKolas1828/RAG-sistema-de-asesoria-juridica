"""
Script de prueba para validar el servicio de búsqueda semántica.
Realiza pruebas de los 3 componentes: query processing, búsqueda, y orchestración.
Ejecutar: python -m pytest tests/integration/test_search.py
"""

from src.retrieval.search_handler import SearchHandler
from src.retrieval.query_processor import process_query
from src.retrieval.semantic_search import SemanticSearchEngine


def test_query_processor():
    print("\n=== PRUEBA 1: Query Processor ===")
    try:
        queries = [
            "infracciones de tránsito en transporte",
            "normas sobre documentación de conductores",
            "sanciones por exceso de velocidad"
        ]
        
        for query in queries:
            print(f"\nConsulta: '{query}'")
            embedding = process_query(query)
            print(f"✓ Embedding generado - Dimensión: {len(embedding)}")
            print(f"  Primeros 5 valores: {embedding[:5]}")
        
        print("\nPrueba de Query Processor EXITOSA")
        return True
    
    except Exception as e:
        print(f"\nError en Query Processor: {e}")
        return False


def test_semantic_search():
    """Prueba la búsqueda en ChromaDB."""
    print("\n=== PRUEBA 2: Semantic Search Engine ===")
    try:
        engine = SemanticSearchEngine()
        
        # Generar embedding de prueba
        test_query = "requisitos para conductores de vehículos de transporte"
        embedding = process_query(test_query)
        
        print(f"Buscando: '{test_query}'")
        
        results = engine.search(embedding, k=3)
        
        print(f"\n✓ Búsqueda completada")
        print(f"  Total de resultados: {results['total_encontrados']}")
        
        if results['total_encontrados'] > 0:
            print(f"\n  Top resultado:")
            top = results['resultados'][0]
            print(f"    Similitud: {top['similitud']}")
            print(f"    Artículo: {top['metadata'].get('articulo', 'N/A')}")
            print(f"    Fuente: {top['metadata'].get('fuente', 'N/A')}")
            print(f"    Texto (primeros 200 chars): {top['texto'][:200]}...")
        
        print("\nPrueba de Semantic Search EXITOSA")
        return True
    
    except Exception as e:
        print(f"\nError en Semantic Search: {e}")
        print("\nNota: Asegúrate de que:")
        print("  1. El pipeline de embedding se ejecutó (python src/main.py)")
        print("  2. ChromaDB tiene datos indexados en data/chroma_db")
        return False


def test_search_handler():
    """Prueba el flujo completo de búsqueda."""
    print("\n=== PRUEBA 3: Search Handler (Flujo Completo) ===")
    try:
        handler = SearchHandler()
        
        test_queries = [
            "¿Cuál es la velocidad máxima permitida en carretera?",
            "Requisitos para transportistas",
            "Artículos sobre sanciones y multas"
        ]
        
        for query in test_queries:
            print(f"\n--- Consulta: '{query}' ---")
            
            result = handler.perform_search(query, k=3, verbose=True)
            
            print(f"\nResultado:")
            print(f"  Status: {result['status']}")
            print(f"  Total encontrados: {result['total_encontrados']}")
            
            if result['total_encontrados'] > 0:
                print(f"\n  Top 3 resultados:")
                for i, item in enumerate(result['resultados'][:3], 1):
                    print(f"    {i}. [Similitud: {item['similitud']}]")
                    print(f"       {item['metadata'].get('articulo', 'N/A')} - {item['metadata'].get('fuente', 'N/A')[:30]}")
                    print(f"       {item['texto'][:200]}...")
            print()
        
        print("Prueba de Search Handler EXITOSA")
        return True
    
    except Exception as e:
        print(f"\nError en Search Handler: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 70)
    print("PRUEBAS DEL SERVICIO DE BÚSQUEDA SEMÁNTICA")
    print("=" * 70)
    
    results = []

    results.append(("Query Processor", test_query_processor()))
    results.append(("Semantic Search", test_semantic_search()))
    results.append(("Search Handler", test_search_handler()))
    
    print("\n" + "=" * 70)
    print("RESUMEN DE PRUEBAS")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "Pasó" if passed else "Falló"
        print(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "=" * 70)
    if all_passed:
        print("TODAS LAS PRUEBAS PASARON")
        print("Ejemplo de uso:")
        print("  from src.retrieval.search_handler import SearchHandler")
        print("  handler = SearchHandler()")
        print("  resultado = handler.perform_search('tu consulta aquí')")
    
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
