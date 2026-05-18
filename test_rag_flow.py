#!/usr/bin/env python3
"""
Script interactivo para probar el flujo RAG completo.
Ejecuta: python test_rag_flow.py
"""

from src.retrieval.rag_pipeline import RAGPipeline
import json


def print_section(title):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")


def print_result(result):
    """Imprime el resultado de forma legible."""
    status = result.get("status", "error")
    
    print_section("RESULTADO GENERAL")
    print(f"Status: {status}")
    print(f"Mensaje: {result.get('mensaje', 'N/A')}")
    
    if status != "éxito":
        return
    
    # Información de búsqueda
    search_results = result.get("search_results", {})
    print_section("BÚSQUEDA SEMÁNTICA")
    print(f"Total encontrados: {search_results.get('total_encontrados', 0)}")
    
    if search_results.get('total_encontrados', 0) > 0:
        print(f"\nTop resultados:")
        for i, item in enumerate(search_results.get('resultados', [])[:3], 1):
            print(f"\n  [{i}] Similitud: {item.get('similitud', 0)}")
            meta = item.get('metadata', {})
            print(f"      Artículo: {meta.get('articulo', 'N/A')}")
            print(f"      Fuente: {meta.get('fuente', 'N/A')}")
            print(f"      Texto: {item.get('texto', '')[:150]}...")
    
    # Información de contexto
    contexto = result.get("contexto", {})
    print_section("CONSTRUCCIÓN DE CONTEXTO")
    print(f"Documentos usados: {contexto.get('documentos_usados', 0)}")
    print(f"Tokens contexto: {contexto.get('tokens_contexto', 0)}")
    print(f"Score promedio: {contexto.get('score_promedio', 0)}")
    print(f"\nContexto formateado (primeros 500 chars):")
    print(contexto.get('contexto_formateado', '')[:500])
    
    # Información del prompt
    prompt = result.get("prompt", {})
    print_section("PROMPT GENERADO")
    print(f"Tokens prompt: {prompt.get('tokens_prompt', 0)}")
    print(f"Tokens disponibles para respuesta: {prompt.get('tokens_available_for_response', 0)}")
    print(f"\nPrompt completo:")
    print(prompt.get('prompt', ''))


def test_mode():
    """Modo de prueba con consultas predefinidas."""
    print_section("MODO PRUEBA - Flujo RAG Completo")
    
    pipeline = RAGPipeline(max_context_tokens=2000, min_similarity=0.7, response_buffer=128)
    
    test_queries = [
        "¿Cuáles son los requisitos para conductores de transporte?",
        "Infracciones de tránsito",
        "Normas sobre documentación vehicular"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n\n{'#' * 80}")
        print(f"CONSULTA {i}: {query}")
        print(f"{'#' * 80}")
        
        result = pipeline.run(query, k=5, max_documents=3, verbose=True)
        print_result(result)


def interactive_mode():
    """Modo interactivo para ingreso manual de consultas."""
    print_section("MODO INTERACTIVO - Flujo RAG Completo")
    print("Escribe 'salir' para terminar\n")
    
    pipeline = RAGPipeline(max_context_tokens=2000, min_similarity=0.7, response_buffer=128)
    
    while True:
        try:
            query = input("\n📝 Ingresa tu consulta: ").strip()
            
            if query.lower() == 'salir':
                print("\n👋 ¡Hasta luego!\n")
                break
            
            if not query:
                print("⚠️  Por favor, ingresa una consulta válida.")
                continue
            
            result = pipeline.run(query, k=5, max_documents=3, verbose=True)
            print_result(result)
        
        except KeyboardInterrupt:
            print("\n\n👋 Flujo interrumpido.\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def main():
    print("\n" + "=" * 80)
    print("  PRUEBA DEL FLUJO RAG COMPLETO")
    print("  consulta -> búsqueda -> contexto -> prompt")
    print("=" * 80)
    print("\nOpciones:")
    print("  1. Modo prueba (consultas predefinidas)")
    print("  2. Modo interactivo (ingresa consultas)")
    
    choice = input("\nSelecciona opción (1 o 2): ").strip()
    
    if choice == "1":
        test_mode()
    elif choice == "2":
        interactive_mode()
    else:
        print("Opción no válida. Ejecutando modo prueba por defecto...")
        test_mode()


if __name__ == "__main__":
    main()
