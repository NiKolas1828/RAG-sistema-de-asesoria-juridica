from src.retrieval.query_processor import process_query
from src.retrieval.semantic_search import SemanticSearchEngine


class SearchHandler:
    def __init__(self):
        self.search_engine = SemanticSearchEngine()
    
    def perform_search(self, query: str, k: int = 5, verbose: bool = False) -> dict:
        try:
            if verbose:
                print(f"[*] Procesando consulta: '{query}'...")
            
            embedding = process_query(query)
            if verbose:
                print(f"[✓] Embedding generado (dimensión: {len(embedding)})")
            
            if verbose:
                print(f"[*] Buscando en vector DB (top-{k})...")
            
            search_results = self.search_engine.search(embedding, k=k)
            
            if verbose:
                print(f"[✓] Búsqueda completada. {search_results['total_encontrados']} resultados encontrados.")

            response = {
                "query_original": query,
                "query_procesada": query.lower().strip(),
                "resultados": search_results["resultados"],
                "total_encontrados": search_results["total_encontrados"],
                "scores": search_results["scores"],
                "status": "éxito",
                "mensaje": f"Se encontraron {search_results['total_encontrados']} documentos relevantes"
            }
            
            return response
        
        except ValueError as e:
            return {
                "query_original": query,
                "status": "error",
                "mensaje": f"Error de validación: {str(e)}",
                "resultados": []
            }
        except RuntimeError as e:
            return {
                "query_original": query,
                "status": "error",
                "mensaje": f"Error en sistema: {str(e)}",
                "resultados": []
            }
        except Exception as e:
            return {
                "query_original": query,
                "status": "error",
                "mensaje": f"Error inesperado: {str(e)}",
                "resultados": []
            }
