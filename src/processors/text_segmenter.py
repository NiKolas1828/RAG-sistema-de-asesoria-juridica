import sqlite3
import re
import json
import csv
from src.config import DB_PATH, CSV_REVISION_PATH
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuración RAG
TOKENS_SIZE = 512
TOKENS_OVERLAP = 50
CHARS_PER_TOKEN = 4


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

    cursor.execute("DELETE FROM chunks")
    cursor.execute("DELETE FROM sqlite_sequence WHERE name='chunks'")


def extract_articles_with_context(content):
    """Máquina de estados para extraer artículos rastreando su Título y Capítulo."""
    articles = []

    current_title = "Sin Título"
    current_chapter = "Sin Capítulo"

    active_art_num = "Introducción / Preámbulo"
    active_art_context = "Sin Título > Sin Capítulo"
    active_art_text = []

    pending_text_buffer = []

    for line in content.split("\n"):
        line_str = line.strip()

        if not line_str:
            if active_art_text:
                active_art_text.append("")
            continue

        titulo_match = re.search(r"\bT[ÍI]TULO\s+([IVXLCDM]+|\d+(?:\.\d+)*)", line_str)
        if titulo_match:
            current_title = line_str
            pending_text_buffer.append(line_str)
            continue

        capitulo_match = re.search(
            r"\bCAP[ÍI]TULO\s+([IVXLCDM]+|\d+(?:\.\d+)*)", line_str
        )
        if capitulo_match:
            current_chapter = line_str
            pending_text_buffer.append(line_str)
            continue

        art_match = re.match(
            r"(?i)^(?:ART[ÍI]CULO|ART\.?)\s+([0-9]+(?:\.[0-9]+)*(?:-[0-9]+)?(?:\s*BIS)?\s*[A-Z]?)",
            line_str,
        )

        if art_match:
            if active_art_text:
                articles.append(
                    {
                        "numero": active_art_num,
                        "capitulo": active_art_context.replace("Sin Título > ", ""),
                        "texto": "\n".join(active_art_text).strip(),
                    }
                )

            active_art_num = f"Art. {art_match.group(1)}"
            active_art_context = f"{current_title} > {current_chapter}"

            active_art_text = pending_text_buffer.copy()
            active_art_text.append(line_str)

            pending_text_buffer = []
        else:
            if pending_text_buffer:
                pending_text_buffer.append(line_str)
            else:
                active_art_text.append(line_str)

    if active_art_text:
        articles.append(
            {
                "numero": active_art_num,
                "capitulo": active_art_context.replace("Sin Título > ", ""),
                "texto": "\n".join(active_art_text + pending_text_buffer).strip(),
            }
        )

    return articles


def extract_title_date(doc_title):
    match = re.search(r"\b(19|20)\d{2}\b", doc_title)
    return f"{match.group(0)}-01-01" if match else "Desconocida"


def export_chunks_to_csv(cursor):
    """Exporta los resultados para control de calidad."""
    cursor.execute("SELECT id, doc_id, texto, tokens_estimados, metadata FROM chunks")
    rows = cursor.fetchall()

    with open(CSV_REVISION_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["chunk_id", "doc_id", "texto_chunk", "tokens", "metadata"])
        writer.writerows(rows)

    print(f"[*] Archivo de revisión de chunks generado en: {CSV_REVISION_PATH}")


def segment_documents_for_article():
    """Divide el contenido en chunks optimizando la memoria RAM y DB."""
    print("\n--- INICIANDO FASE 3: SEGMENTACIÓN RAG ---")

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        init_chunks_table(cursor)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=TOKENS_SIZE * CHARS_PER_TOKEN,
            chunk_overlap=TOKENS_OVERLAP * CHARS_PER_TOKEN,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""],
        )

        total_chunks = 0

        cursor.execute(
            "SELECT id, titulo, contenido FROM documentos WHERE procesado = 1"
        )

        for doc_id, title, content in cursor:
            if not content:
                continue

            print(f"Segmentando documento ID: {doc_id} - {title[:30]}...")

            effective_date = extract_title_date(title)
            articles_data = extract_articles_with_context(content)

            for art_data in articles_data:
                art_number = art_data["numero"]
                chapter_context = art_data["capitulo"]
                art_text = art_data["texto"]

                if len(art_text) < 15:
                    continue

                metadata = {
                    "fuente": title.replace(".pdf", "").replace(".docx", ""),
                    "articulo": art_number,
                    "capitulo": chapter_context,
                    "fecha_vigencia": effective_date,
                }

                langchain_chunks = text_splitter.split_text(art_text)

                for i, chunk_text in enumerate(langchain_chunks):
                    if i > 0:
                        chunk_text = f"[{art_number} - Continuación] {chunk_text}"

                    estimated_tokens = len(chunk_text) // CHARS_PER_TOKEN

                    insert_cursor = conn.cursor()
                    insert_cursor.execute(
                        """
                        INSERT INTO chunks (doc_id, texto, metadata, tokens_estimados)
                        VALUES (?, ?, ?, ?)
                    """,
                        (
                            doc_id,
                            chunk_text,
                            json.dumps(metadata, ensure_ascii=False),
                            estimated_tokens,
                        ),
                    )

                    total_chunks += 1

                    if total_chunks % 500 == 0:
                        conn.commit()
                        print(
                            f"   -> [Guardado intermedio] {total_chunks} chunks procesados en disco..."
                        )

        conn.commit()
        print(f"[*] Segmentación exitosa. Se generaron {total_chunks} chunks en total.")

        export_chunks_to_csv(conn.cursor())
        conn.close()

        print("[*] Fase 3 completada.")

    except sqlite3.Error as e:
        raise Exception(
            f"DatabaseTransactionException: Segment generation failed. Details: {str(e)}"
        )
