import os
from pathlib import Path
from dotenv import load_dotenv

# Carga el .env
# load_dotenv()

# Ubicación de este archivo (src/config.py)
CURRENT_FILE = Path(__file__).resolve()

# La raíz del proyecto (un nivel arriba de src/)
BASE_DIR = CURRENT_FILE.parent.parent

# Rutas centralizadas
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "normas.db"
CSV_REVISION_PATH = DATA_DIR / "chunks_revision.csv"

# Configuraciones de IA
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
