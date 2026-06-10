from src.vector_store.vector_manager import VectorManager
import json


class SemanticSearchEngine:
    
    def __init__(self):
        self.vector_manager = VectorManager()
        self.collection = self.vector_manager.collection
    
    def search(self, embedding: list, k: int = 5, where_document: dict = None) -> dict:
        try:
            if not embedding or len(embedding) != 384:
                raise ValueError("El embedding debe ser una lista de 384 elementos")
            
            if k <= 0:
                raise ValueError("k debe ser mayor a 0")
            
            query_args = {
                "query_embeddings": [embedding],
                "n_results": k,
                "include": ["documents", "metadatas", "distances"]
            }
            if where_document:
                query_args["where_document"] = where_document

            # Realizar búsqueda en ChromaDB
            results = self.collection.query(**query_args)

            processed_results = self._process_results(results)
            
            return processed_results
        
        except ValueError as e:
            raise ValueError(f"Error de validación en búsqueda: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error al buscar en ChromaDB: {str(e)}")
    
    def _process_results(self, chroma_results: dict) -> dict:
        documents = chroma_results.get("documents", [[]])[0]
        metadatas = chroma_results.get("metadatas", [[]])[0]
        distances = chroma_results.get("distances", [[]])[0]
        
        max_dist = max(distances) if distances else 1.0
        similarities = [round(1 - (d / max_dist), 4) for d in distances]
        
        results = []
        for i, (doc, metadata, sim) in enumerate(zip(documents, metadatas, similarities)):
            try:
                meta = json.loads(metadata) if isinstance(metadata, str) else metadata
            except:
                meta = metadata
            
            results.append({
                "rank": i + 1,
                "texto": doc,
                "metadata": meta,
                "similitud": round(sim, 4)
            })
        
        return {
            "resultados": results,
            "total_encontrados": len(results),
            "scores": [round(s, 4) for s in similarities]
        }
