from src.retrieval.query_processor import process_query
from src.retrieval.semantic_search import SemanticSearchEngine
from src.retrieval.query_expander import QueryExpander


class SearchHandler:
    def __init__(self):
        self.search_engine = SemanticSearchEngine()
        self.expander = None  # Lazy init
    
    def perform_search(self, query: str, k: int = 5, verbose: bool = False) -> dict:
        try:
            if verbose:
                print(f"[*] Procesando consulta base: '{query}'...")
            
            if self.expander is None:
                self.expander = QueryExpander()
                
            queries = self.expander.expand_query(query)
            if verbose:
                print(f"[*] Multi-Query: Se buscarán {len(queries)} variaciones de la consulta.")
            
            import re
            
            all_results = {}
            for q in queries:
                try:
                    embedding = process_query(q)
                    
                    # Detectar si la query contiene un código de infracción (ej. C.24, D.02)
                    where_doc = None
                    code_match = re.search(r'\b([A-E]\.\d{2})\b', q, re.IGNORECASE)
                    if code_match:
                        code = code_match.group(1).upper()
                        where_doc = {"$contains": code}
                        if verbose:
                            print(f"[*] Filtro estricto aplicado para el código: {code}")
                    
                    search_results = self.search_engine.search(embedding, k=k, where_document=where_doc)
                    
                    # Deduplicar combinando (mantener el mayor score)
                    for res in search_results["resultados"]:
                        text_key = res["texto"]
                        if text_key not in all_results or res["similitud"] > all_results[text_key]["similitud"]:
                            all_results[text_key] = res
                except Exception as e:
                    if verbose:
                        print(f"[!] Error buscando variación '{q}': {e}")
            
            # Ordenar por similitud descendente y tomar top K
            final_results = sorted(all_results.values(), key=lambda x: x["similitud"], reverse=True)[:k]
            
            # Reasignar ranks
            for i, res in enumerate(final_results):
                res["rank"] = i + 1
                
            final_scores = [res["similitud"] for res in final_results]
            
            if verbose:
                print(f"[✓] Búsqueda combinada completada. {len(final_results)} resultados únicos encontrados.")

            response = {
                "query_original": query,
                "query_procesada": queries[0].lower().strip(),
                "queries_expandidas": queries,
                "resultados": final_results,
                "total_encontrados": len(final_results),
                "scores": final_scores,
                "status": "éxito",
                "mensaje": f"Se encontraron {len(final_results)} documentos relevantes tras expansión"
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
