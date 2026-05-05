import glob
import os
import sqlite3
import re
import json
import csv
from pathlib import Path

try:
    import docx
    import pytesseract
    from bs4 import BeautifulSoup
    from pdf2image import convert_from_path
    from pypdf import PdfReader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    print(
        "Faltan librerías. Por favor ejecuta: pip install pypdf python-docx beautifulsoup4 pdf2image pytesseract Pillow langchain-text-splitters"
    )
    exit(1)

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
DB_PATH = os.path.join(BASE_DIR, "data", "normas.db")

# Configuración RAG (Documento de Arquitectura)
TOKENS_SIZE = 512
TOKENS_OVERLAP = 50
CHARS_PER_TOKEN = 4


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
        SELECT id, titulo, contenido 
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

# FASE 3: SEGMENTACIÓN RAG (CHUNKING)

def init_chunks_table(cursor):
    """Crea la tabla chunks para el RAG."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER NOT NULL REFERENCES documentos(id),
            texto TEXT NOT NULL,
            metadata TEXT,
            tokens_estimados INTEGER
        )
    """)
    
    # Limpiamos los chunks anteriores para evitar duplicados si corremos el pipeline completo
    cursor.execute("DELETE FROM chunks")
    cursor.execute("DELETE FROM sqlite_sequence WHERE name='chunks'")

def extract_articles_with_context(content):
    """Máquina de estados para extraer artículos rastreando su Título y Capítulo."""
    articles = []

    # MEMORIA GLOBAL
    current_title = "Sin Título"
    current_chapter = "Sin Capítulo"
    
    # MEMORIA DEL ARTÍCULO ACTIVO
    active_art_num = "Introducción / Preámbulo"
    active_art_context = "Sin Título > Sin Capítulo" # Foto inicial
    active_art_text = []

    # Buffer temporal para los textos de jerarquía
    pending_text_buffer = []

    for line in content.split('\n'):
        line_str = line.strip()
        
        if not line_str:
            if active_art_text:
                active_art_text.append("")
            continue

        # Actualizamos la Memoria Global si encontramos jerarquías
        titulo_match = re.search(r'\bT[ÍI]TULO\s+([IVXLCDM]+|\d+(?:\.\d+)*)', line_str)
        if titulo_match:
            current_title = line_str
            pending_text_buffer.append(line_str)
            continue
        
        capitulo_match = re.search(r'\bCAP[ÍI]TULO\s+([IVXLCDM]+|\d+(?:\.\d+)*)', line_str)
        if capitulo_match:
            current_chapter = line_str
            pending_text_buffer.append(line_str)
            continue

        # Evaluamos los Artículos
        art_match = re.match(r'(?i)^(?:ART[ÍI]CULO|ART\.?)\s+([0-9]+(?:\.[0-9]+)*(?:-[0-9]+)?(?:\s*BIS)?\s*[A-Z]?)', line_str)
        
        if art_match:
            # Usamos el 'active_art_context', protegiéndolo de los cambios globales
            if active_art_text:
                articles.append({
                    'numero': active_art_num,
                    'capitulo': active_art_context.replace('Sin Título > ', ''),
                    'texto': "\n".join(active_art_text).strip()
                })
            
            active_art_num = f"Art. {art_match.group(1)}"
            active_art_context = f"{current_title} > {current_chapter}"
            
            active_art_text = pending_text_buffer.copy()
            active_art_text.append(line_str)
            
            # Limpiar el buffer temporal
            pending_text_buffer = []
        else:
            # Si es texto normal y ya pasamos un Título, pertenece al nuevo bloque
            if pending_text_buffer:
                pending_text_buffer.append(line_str)
            else:
                active_art_text.append(line_str)

    # Guardamos el último artículo que quedó en el aire al terminar el documento
    if active_art_text:
        articles.append({
            'numero': active_art_num,
            'capitulo': active_art_context.replace('Sin Título > ', ''),
            # Unir cualquier texto residual del buffer si el documento termina abruptamente
            'texto': "\n".join(active_art_text + pending_text_buffer).strip()
        })
        
    return articles

def extract_title_date(doc_title):
    match = re.search(r'\b(19|20)\d{2}\b', doc_title)
    return f"{match.group(0)}-01-01" if match else "Desconocida"

def export_chunks_to_csv(cursor):
    """Exporta los resultados para control de calidad."""
    cursor.execute("SELECT id, doc_id, texto, tokens_estimados, metadata FROM chunks")
    rows = cursor.fetchall()
    
    csv_path = os.path.join(BASE_DIR, "data", "chunks_revision.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['chunk_id', 'doc_id', 'texto_chunk', 'tokens', 'metadata'])
        writer.writerows(rows)

    print(f"[*] Archivo de revisión de chunks generado en: {csv_path}")

def segment_documents_for_article():
    """Divide el contenido en chunks optimizando la memoria RAM y DB."""
    print("\n--- INICIANDO FASE 3: SEGMENTACIÓN RAG ---")

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        init_chunks_table(cursor)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = TOKENS_SIZE * CHARS_PER_TOKEN,       
            chunk_overlap = TOKENS_OVERLAP * CHARS_PER_TOKEN, 
            length_function = len,
            separators=["\n\n", "\n", ".", " ", ""] 
        )

        total_chunks = 0
        
        # Ejecutamos la consulta y leemos línea por línea.
        cursor.execute("SELECT id, titulo, contenido FROM documentos WHERE procesado = 1")
        
        # Iteramos directamente sobre el cursor (Streaming)
        for doc_id, title, content in cursor:
            if not content: continue

            print(f"Segmentando documento ID: {doc_id} - {title[:30]}...")

            effective_date = extract_title_date(title)
            articles_data = extract_articles_with_context(content)

            for art_data in articles_data:
                art_number = art_data['numero']
                chapter_context = art_data['capitulo']
                art_text = art_data['texto']

                if len(art_text) < 15: continue

                metadata = {
                    "fuente": title.replace(".pdf", "").replace(".docx", ""),
                    "articulo": art_number,
                    "capitulo": chapter_context, 
                    "fecha_vigencia": effective_date
                }

                langchain_chunks = text_splitter.split_text(art_text)

                for i, chunk_text in enumerate(langchain_chunks):
                    if i > 0:
                        chunk_text = f"[{art_number} - Continuación] {chunk_text}"
                    
                    estimated_tokens = len(chunk_text) // CHARS_PER_TOKEN
                    
                    # Usamos un segundo cursor para insertar, para no chocar con el cursor de lectura
                    insert_cursor = conn.cursor()
                    insert_cursor.execute("""
                        INSERT INTO chunks (doc_id, texto, metadata, tokens_estimados)
                        VALUES (?, ?, ?, ?)
                    """, (doc_id, chunk_text, json.dumps(metadata, ensure_ascii=False), estimated_tokens))
                    
                    total_chunks += 1
                    
                    # Liberar memoria cada 500 chunks
                    if total_chunks % 500 == 0:
                        conn.commit()
                        print(f"   -> [Guardado intermedio] {total_chunks} chunks procesados en disco...")

        # Commit final de lo que haya quedado suelto
        conn.commit()
        print(f"[*] Segmentación exitosa. Se generaron {total_chunks} chunks en total.")
        
        export_chunks_to_csv(conn.cursor())
        conn.close()

        print("[*] Fase 3 completada.")

    except sqlite3.Error as e:
        raise Exception(f"DatabaseTransactionException: Segment generation failed. Details: {str(e)}")

if __name__ == "__main__":
    process_documents()
    standardize()
    segment_documents_for_article()