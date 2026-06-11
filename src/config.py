import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno del archivo .env
load_dotenv()

# Ubicación de este archivo (src/config.py)
CURRENT_FILE = Path(__file__).resolve()

# La raíz del proyecto (un nivel arriba de src/)
BASE_DIR = CURRENT_FILE.parent.parent

# Rutas centralizadas
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "normas.db"
CSV_REVISION_PATH = DATA_DIR / "chunks_revision.csv"
CHROMA_PATH = DATA_DIR / "chroma_db"

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Configuración de ChromaDB (Docker / Local)
CHROMA_MODE = os.getenv("CHROMA_MODE", "local").lower().strip()  # "local" o "http"
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost").strip()
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

# ─── Re-Ranking (Cross-Encoder) ────────────────────────────
# Desactivado por defecto: el modelo de 471MB es demasiado lento en CPU.
# Para activar: RERANKER_ENABLED=true en el archivo .env
# Modelos disponibles:
#   Ligero (~85MB, solo inglés): cross-encoder/ms-marco-MiniLM-L-6-v2
#   Multilingüe (~471MB):        cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower().strip() == "true"
RERANKER_MODEL   = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"  # Multilingüe para alta precisión en español
)
