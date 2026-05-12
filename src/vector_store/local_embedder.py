from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SentenceTransformer(EMBEDDING_MODEL, device=device)


def generate_local_embeddings(texts: list[str]):
    """
    Genera embeddings de forma local sin límites de cuota.
    Dimensión resultante: 384.
    """
    try:
        # Realiza la inferencia localmente
        embeddings = model.encode(texts, show_progress_bar=True)

        return embeddings.tolist()
    except Exception as e:
        print(f"[!] Error generando embeddings locales: {e}")
        return []
