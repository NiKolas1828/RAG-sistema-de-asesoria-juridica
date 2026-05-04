import glob
import os
import sqlite3
import re
from pathlib import Path

try:
    import docx
    import pytesseract
    from bs4 import BeautifulSoup
    from pdf2image import convert_from_path
    from pypdf import PdfReader
except ImportError:
    print(
        "Faltan librerías. Por favor ejecuta: pip install pypdf python-docx beautifulsoup4 pdf2image pytesseract Pillow"
    )
    exit(1)

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
DB_PATH = os.path.join(BASE_DIR, "data", "normas.db")


def init_db():
    """Inicializa la base de datos SQLite con la tabla necesaria."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()
    # Forzamos la creación de la tabla con las columnas exactas que necesitamos,
    # ignorando schema.sql para evitar conflictos.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documentos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            titulo TEXT,
            tipo TEXT,
            contenido TEXT,
            procesado INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    print("[-] Base de datos inicializada correctamente.")
    return conn


def extract_text_with_ocr(pdf_path):
    """Extrae texto de un PDF escaneado usando OCR (Tesseract)."""
    text = ""
    try:
        # Convierte las páginas del PDF a imágenes
        images = convert_from_path(pdf_path)
        for img in images:
            # Usa OCR en español (lang='spa') para entender mejor tildes y eñes
            page_text = pytesseract.image_to_string(img, lang="spa")
            text += page_text + "\n"
    except Exception as e:
        print(f"[!] Error en OCR para {pdf_path}: {e}")
        print("    Asegúrate de tener instalados Tesseract y Poppler en tu sistema.")
        print(
            "    - Fedora: sudo dnf install tesseract tesseract-langpack-spa poppler-utils"
        )
        print(
            "    - Ubuntu/Debian: sudo apt install tesseract-ocr tesseract-ocr-spa poppler-utils"
        )
    return text.strip()


def extract_text_from_pdf(pdf_path):
    """Extrae texto de un archivo PDF usando pypdf, con fallback a OCR si es necesario."""
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"

        # Si el texto extraído es muy corto, probablemente sea un documento escaneado (imagen)
        if len(text.strip()) < 100:
            print(
                f"   -> [!] El PDF parece estar escaneado. Procesando con OCR (esto puede tardar unos segundos)..."
            )
            text = extract_text_with_ocr(pdf_path)

    except Exception as e:
        print(f"[!] Error leyendo PDF {pdf_path}: {e}")
    return text.strip()


def extract_text_from_docx(docx_path):
    """Extrae texto de un archivo DOCX usando python-docx."""
    text = ""
    try:
        doc = docx.Document(docx_path)
        for para in doc.paragraphs:
            if para.text.strip():
                text += para.text + "\n"
    except Exception as e:
        print(f"[!] Error leyendo DOCX {docx_path}: {e}")
    return text.strip()


def extract_text_from_html(html_path):
    """Extrae texto limpio de un archivo HTML usando BeautifulSoup."""
    text = ""
    try:
        # Algunos HTML viejos del gobierno usan otras codificaciones, probamos utf-8 primero
        with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f, "html.parser")

            # Eliminar etiquetas no deseadas que ensucian el texto (scripts, estilos, navegación)
            for elemento in soup(["script", "style", "nav", "header", "footer"]):
                elemento.decompose()

            # Extraer el texto separando por saltos de línea
            text = soup.get_text(separator="\n", strip=True)
    except Exception as e:
        print(f"[!] Error leyendo HTML {html_path}: {e}")
    return text


def process_documents():
    """Recorre la carpeta raw/, extrae el texto y lo guarda en la DB."""
    conn = init_db()
    cursor = conn.cursor()

    # Buscar todos los PDFs, DOCX y HTML
    files = glob.glob(os.path.join(RAW_DATA_DIR, "**", "*.*"), recursive=True)
    target_files = [
        f for f in files if f.endswith((".pdf", ".docx", ".doc", ".html", ".htm"))
    ]

    if not target_files:
        print(f"[!] No se encontraron documentos en {RAW_DATA_DIR}")
        return

    print(f"[*] Encontrados {len(target_files)} documentos para procesar.")

    for file_path in target_files:
        filename = os.path.basename(file_path)
        ext = filename.split(".")[-1].lower()

        print(f"Procesando: {filename}...")

        # Verificar si ya existe en la DB
        cursor.execute("SELECT id FROM documentos WHERE titulo = ?", (filename,))
        if cursor.fetchone():
            print(f"   -> Ya existe en la base de datos. Saltando.")
            continue

        content = ""
        if ext == "pdf":
            content = extract_text_from_pdf(file_path)
        elif ext in ["docx", "doc"]:
            content = extract_text_from_docx(file_path)
        elif ext in ["html", "htm"]:
            content = extract_text_from_html(file_path)

        if content:
            try:
                # Se asume que el schema.sql tiene una tabla 'documentos'
                cursor.execute(
                    "INSERT INTO documentos (titulo, tipo, contenido) VALUES (?, ?, ?)",
                    (filename, ext, content),
                )
                conn.commit()
                print(f"   -> Éxito. {len(content)} caracteres extraídos.")
            except sqlite3.OperationalError as e:
                print(f"   -> [!] Error de base de datos (verifica tu schema.sql): {e}")
        else:
            print(
                f"   -> [!] No se pudo extraer texto o el archivo está vacío/es una imagen escaneada."
            )

    conn.close()
    print("\n[*] Proceso de recolección completado.")

DB_PATH = os.path.join(BASE_DIR, "data", "normas.db")

def limpiar_contenido(texto: str) -> str:
    texto = re.sub(r'Ministerio de\s+\w+.*?Página\s+\d+\s+de\s+\d+', '', texto, flags=re.IGNORECASE|re.DOTALL)
    texto = re.sub(r'[-─═_]{3,}', '', texto)
    texto = re.sub(r'\bPágina\s+\d+\s*(de\s+\d+)?\b', '', texto, flags=re.IGNORECASE)
    texto = re.sub(r'\n{3,}', '\n\n', texto)
    texto = re.sub(r'[ \t]+', ' ', texto)
    return texto.strip()

def extraer_metadata(titulo: str) -> dict:
    """Extrae año y tipo desde el título del documento."""
    anio = re.search(r'\b(19|20)\d{2}\b', titulo)
    tipo = "ley" if "ley" in titulo.lower() else \
           "resolucion" if "resolución" in titulo.lower() or "resolucion" in titulo.lower() else \
           "decreto" if "decreto" in titulo.lower() else "otro"
    return {
        "anio": anio.group() if anio else None,
        "tipo_detectado": tipo
    }

def normalize_for_embeddings(texto: str) -> str:
    """
    Normalización optimizada para embeddings de documentos legales.
    Mantiene estructura semántica pero limpia ruido.
    """
    if not texto:
        return ""
    texto = limpiar_contenido(texto)
    texto = texto.lower()
    texto = re.sub(r'http\S+|www\.\S+', '', texto)
    texto = re.sub(r'\[\d+\]', '', texto)
    texto = re.sub(r'[^\w\s.,;:()áéíóúüñ¿¡\-]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto)
    texto = re.sub(r'\s+([.,;:()])', r'\1', texto)
    
    return texto.strip()

def standardize():
    """Aplica limpieza Y normalización NLP a todos los documentos"""
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute("ALTER TABLE documentos ADD COLUMN contenido_nlp TEXT")
        cursor.execute("ALTER TABLE documentos ADD COLUMN normalizado INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    
    cursor.execute("""
        SELECT id, titulo, tipo, contenido 
        FROM documentos 
        WHERE normalizado = 0 OR normalizado IS NULL
    """)
    docs = cursor.fetchall()
    
    print(f"Documentos por normalizar: {len(docs)}")
    
    for doc_id, titulo, contenido in docs:
        contenido_limpio = limpiar_contenido(contenido or "")
        contenido_nlp = normalize_for_embeddings(contenido_limpio)
        metadata = extraer_metadata(titulo or "")
        
        cursor.execute("""
            UPDATE documentos 
            SET contenido = ?, 
                tipo = ?, 
                contenido_nlp = ?,
                procesado = 1,
                normalizado = 1
            WHERE id = ?
        """, (contenido_limpio, metadata["tipo_detectado"], contenido_nlp, doc_id))
        
        print(f"  ✓ [{doc_id}] {titulo[:50]}... ({len(contenido_nlp)} caracteres NLP)")
    
    conn.commit()
    conn.close()
    print("Normalización completada")

if __name__ == "__main__":
    process_documents()
    standardize()