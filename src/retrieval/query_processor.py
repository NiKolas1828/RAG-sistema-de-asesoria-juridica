from src.vector_store.local_embedder import model


def normalize_query(query: str) -> str:
    if not query:
        raise ValueError("La consulta no puede estar vacía")
    
    query = query.strip()
    query = query.lower()
    
    return query


from typing import Union, List

def process_query(query: Union[str, List[str]]) -> Union[list, List[list]]:
    try:
        if isinstance(query, str):
            normalized_query = normalize_query(query)
            embedding = model.encode([normalized_query], show_progress_bar=False)
            return embedding.tolist()[0]
        else:
            normalized_queries = [normalize_query(q) for q in query]
            embeddings = model.encode(normalized_queries, show_progress_bar=False)
            return embeddings.tolist()
    
    except ValueError as e:
        raise ValueError(f"Error al procesar la consulta: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Error inesperado al generar embedding: {str(e)}")
