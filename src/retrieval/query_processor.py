from src.vector_store.local_embedder import model


def normalize_query(query: str) -> str:
    if not query:
        raise ValueError("La consulta no puede estar vacía")
    
    query = query.strip()
    query = query.lower()
    
    return query


def process_query(query: str) -> list:
    try:
        normalized_query = normalize_query(query)
        embedding = model.encode([normalized_query], show_progress_bar=False)
        return embedding.tolist()[0]
    
    except ValueError as e:
        raise ValueError(f"Error al procesar la consulta: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Error inesperado al generar embedding: {str(e)}")
