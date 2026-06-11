from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cpu":
    torch.set_num_threads(1)

model = SentenceTransformer(EMBEDDING_MODEL, device=device)

if device == "cpu":
    # Reducir consumo de memoria de ~470MB a ~120MB mediante cuantización dinámica en CPU
    # Configurar backend a qnnpack para evitar error de instrucción ilegal (SIGILL)
    # en procesadores sin soporte para AVX2 (fbgemm lo requiere).
    torch.backends.quantized.engine = 'qnnpack'
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )


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
