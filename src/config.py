from pathlib import Path

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
