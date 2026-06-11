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


def init_chunks_table(cursor, force_reset=False):
    """
    Crea la tabla chunks.
    Solo borra todo si force_reset es True.
    """
    if force_reset:
        print("[!] Reseteando tabla chunks por completo...")
        cursor.execute("DROP TABLE IF EXISTS chunks")
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='chunks'")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER NOT NULL REFERENCES documentos(id),
            texto TEXT NOT NULL,
            metadata TEXT,
            tokens_estimados INTEGER,
            procesado INTEGER DEFAULT 0
        )
    """)


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


def _merge_and_split_articles(articles_data: list, title: str, effective_date: str) -> list:
    """
    Transforma los artículos extraídos en chunks óptimos para el RAG.

    Reglas:
      1. Artículo < MIN_CHUNK_TOKENS  → merge con el siguiente artículo.
         (Evita chunks triviales de 1-2 líneas que no aportan contexto.)
      2. Artículo > MAX_CHUNK_TOKENS  → split semántico en puntos naturales:
         primero párrafos (doble salto de línea), luego oraciones, luego espacio.
         Cada parte lleva el prefijo "[Art.X - Continuación (N/M)]" para mantener
         trazabilidad y mejorar la similitud con queries que mencionen el artículo.
      3. Artículo en rango óptimo     → chunk único, sin modificaciones.
    """
    CHARS_PER_TOKEN = 4
    MIN_CHUNK_TOKENS = 300   # artículos más cortos se fusionan
    MAX_CHUNK_TOKENS = 700   # artículos más largos se parten

    base_meta = {
        "fuente": title.replace(".pdf", "").replace(".docx", ""),
        "fecha_vigencia": effective_date,
    }

    chunks_out = []
    # Buffer de artículos pequeños que todavía no se han emitido
    merge_buffer_texts: list = []
    merge_buffer_meta: dict = {}

    def _flush_merge_buffer():
        """Emite el buffer de fusión como un chunk."""
        if not merge_buffer_texts:
            return
        text = "\n\n".join(merge_buffer_texts).strip()
        if len(text) >= 30:
            chunks_out.append({
                "texto": text,
                "metadata": {**base_meta, **merge_buffer_meta},
            })
        merge_buffer_texts.clear()
        merge_buffer_meta.clear()

    for art in articles_data:
        art_text   = art["texto"].strip()
        art_number = art["numero"]
        chapter    = art["capitulo"]

        if len(art_text) < 15:
            continue  # demasiado corto incluso para el buffer

        tokens = len(art_text) // CHARS_PER_TOKEN
        meta = {**base_meta, "articulo": art_number, "capitulo": chapter}

        if tokens < MIN_CHUNK_TOKENS:
            # Artículo pequeño → acumular en buffer de fusión
            merge_buffer_texts.append(art_text)
            # Actualizar metadata: usa el último artículo del grupo
            merge_buffer_meta = {"articulo": art_number, "capitulo": chapter}

            # Si el buffer ya supera el mínimo, emitir
            total_buffer_tokens = sum(len(t) for t in merge_buffer_texts) // CHARS_PER_TOKEN
            if total_buffer_tokens >= MIN_CHUNK_TOKENS:
                _flush_merge_buffer()

        elif tokens <= MAX_CHUNK_TOKENS:
            # Rango óptimo: emitir el buffer previo y este artículo como chunk único
            _flush_merge_buffer()
            chunks_out.append({"texto": art_text, "metadata": meta})

        else:
            # Artículo grande: emitir buffer previo, luego split semántico
            _flush_merge_buffer()

            # Split en puntos naturales: párrafos > oraciones > espacios
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=MAX_CHUNK_TOKENS * CHARS_PER_TOKEN,
                chunk_overlap=60 * CHARS_PER_TOKEN,    # 60 tokens de solapamiento
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            parts = splitter.split_text(art_text)
            total = len(parts)
            for i, part_text in enumerate(parts):
                if i == 0:
                    texto = part_text
                else:
                    # Prefijo rico: artículo + número de parte para trazabilidad
                    texto = f"[{art_number} — Continuación {i+1}/{total}]\n{part_text}"
                chunks_out.append({"texto": texto, "metadata": meta})

    # Emitir lo que quede en el buffer al final
    _flush_merge_buffer()

    return chunks_out


def segment_documents_for_article():
    """Divide el contenido en chunks optimizados por artículo con fusión y split inteligente."""
    print("\n--- INICIANDO FASE 3: SEGMENTACIÓN RAG (Chunking Inteligente) ---")

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        init_chunks_table(cursor)

        total_chunks = 0
        main_cursor   = conn.cursor()
        action_cursor = conn.cursor()

        main_cursor.execute(
            "SELECT id, titulo, contenido FROM documentos WHERE procesado = 1"
        )

        for doc_id, title, content in main_cursor:
            if not content:
                continue

            # Verificamos si este documento ya tiene chunks
            action_cursor.execute(
                "SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,)
            )
            is_segmented = action_cursor.fetchone()[0]

            if is_segmented > 0:
                print(
                    f"[-] Saltando: '{title[:40]}...' ya tiene {is_segmented} fragmentos."
                )
                continue

            print(f"[*] Segmentando: {title[:50]}...")

            effective_date = extract_title_date(title)
            articles_data  = extract_articles_with_context(content)

            # Aplicar chunking inteligente (merge + split)
            chunks = _merge_and_split_articles(articles_data, title, effective_date)

            doc_chunks = 0
            for chunk in chunks:
                chunk_text = chunk["texto"]
                metadata   = chunk["metadata"]
                estimated_tokens = len(chunk_text) // 4

                action_cursor.execute(
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
                doc_chunks   += 1

                if total_chunks % 500 == 0:
                    conn.commit()
                    print(f"   → [Guardado intermedio] {total_chunks} chunks procesados...")

            print(f"   → {doc_chunks} chunks generados para este documento.")

        conn.commit()
        print(f"\n[✓] Segmentación exitosa. Total: {total_chunks} chunks generados.")

        export_chunks_to_csv(conn.cursor())
        conn.close()

        print("[*] Fase 3 completada.")

    except sqlite3.Error as e:
        raise Exception(
            f"DatabaseTransactionException: Segment generation failed. Details: {str(e)}"
        )


