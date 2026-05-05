import glob
import hashlib
import os
import re
import sqlite3

try:
    import docx
    import pdfplumber
    import pytesseract
    from bs4 import BeautifulSoup
    from pdf2image import convert_from_path
    from pypdf import PdfReader
except ImportError:
    print(
        "Faltan librerías. Por favor ejecuta: pip install pdfplumber pypdf python-docx beautifulsoup4 pdf2image pytesseract Pillow"
    )
    exit(1)

# Artefactos residuales típicos de PDFs con membrete de entidades .gov.co
# (sello de fecha, número de resolución del timbre, NIT del membrete)
_ARTEFACTOS_PDF = re.compile(
    r"(?:^\d{1,2}\s+[A-Z]{2,3}\s+\d{4}\s*$"  # "22 AR 2019" — sello de fecha
    r"|^n\d{5,}\s*$"  # "n000718" — número del timbre
    r"|^NIT\.\d[\d.\-]+\s*$"  # "NIT.899.999.055-4"
    r"|^-?\s*\d{1,3}\s*-?\s*$"  # números de página sueltos
    r"|^\s*[~`|\\#*^]{2,}\s*$"  # líneas de caracteres basura
    r")",
    re.MULTILINE,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
DB_PATH = os.path.join(BASE_DIR, "data", "normas.db")
LOG_PATH = os.path.join(BASE_DIR, "data", "quality_report.txt")

MIN_CHARS = 300
MIN_PALABRAS = 50
MAX_RATIO_BASURA = 0.25
MIN_RATIO_ESPANOL = 0.35  # documentos legales tienen muchos números y siglas

PALABRAS_ESPANOL = {
    # Palabras funcionales frecuentes
    "de",
    "la",
    "el",
    "en",
    "y",
    "a",
    "los",
    "del",
    "las",
    "un",
    "una",
    "por",
    "con",
    "se",
    "que",
    "es",
    "al",
    "su",
    "para",
    "como",
    "no",
    "lo",
    "mas",
    "o",
    "sus",
    "le",
    "si",
    "sobre",
    "tambien",
    "este",
    "esta",
    "esto",
    "son",
    "sera",
    "han",
    "hay",
    "fue",
    "ser",
    "cuando",
    "donde",
    "quien",
    "cual",
    "cuyo",
    "cuya",
    # Vocabulario normativo colombiano de tránsito (sin tildes para comparar en .lower())
    "articulo",
    "ley",
    "decreto",
    "resolucion",
    "transito",
    "vehiculo",
    "conductor",
    "multa",
    "infraccion",
    "norma",
    "colombiano",
    "paragrafo",
    "capitulo",
    "titulo",
    "vigencia",
    "dispone",
    "establece",
    "mediante",
    "conforme",
    "dispuesto",
    "previsto",
    "nacional",
    "ministerio",
    "transporte",
    "autoridad",
    "registro",
    "licencia",
    "comparendo",
    "sancion",
    "servicio",
    "publico",
    "persona",
    "debera",
    "podra",
    "siguiente",
    "presente",
    "dicha",
    "dicho",
    "mismo",
    "misma",
}

PATRON_ARTICULO = re.compile(r"(artículo|art\.?)\s+\d+", re.IGNORECASE)


def validar_longitud(texto):
    if len(texto) < MIN_CHARS:
        return False, f"Texto muy corto: {len(texto)} chars (mínimo {MIN_CHARS})"
    palabras = texto.split()
    if len(palabras) < MIN_PALABRAS:
        return False, f"Pocas palabras: {len(palabras)} (mínimo {MIN_PALABRAS})"
    return True, "ok"


def validar_ratio_basura(texto):
    """Detecta OCR fallido o encoding roto: demasiados caracteres no legibles."""
    total = len(texto)
    if total == 0:
        return False, "Texto vacío"
    basura = len(re.findall(r"[^\w\sáéíóúüñÁÉÍÓÚÜÑ.,;:()\-\"'/]", texto))
    ratio = basura / total
    if ratio > MAX_RATIO_BASURA:
        return (
            False,
            f"Alto ratio de caracteres basura: {ratio:.1%} (máximo {MAX_RATIO_BASURA:.0%})",
        )
    return True, "ok"


def _normalizar(texto: str) -> str:
    """Quita tildes para comparación robusta (OCR y texto nativo)."""
    import unicodedata

    return unicodedata.normalize("NFD", texto).encode("ascii", "ignore").decode()


def validar_idioma_espanol(texto):
    palabras = re.findall(r"\b\w+\b", _normalizar(texto).lower())
    if not palabras:
        return False, "Sin palabras detectadas"
    ratio = sum(1 for p in palabras if p in PALABRAS_ESPANOL) / len(palabras)
    if ratio < MIN_RATIO_ESPANOL:
        return (
            False,
            f"Posible idioma incorrecto: ratio español {ratio:.1%} (mínimo {MIN_RATIO_ESPANOL:.0%})",
        )
    return True, "ok"


def validar_estructura_legal(texto):
    """Verifica que el documento contiene referencias a artículos."""
    matches = PATRON_ARTICULO.findall(texto)
    if len(matches) == 0:
        return (
            False,
            "No se encontraron referencias a artículos (¿es un documento normativo?)",
        )
    if len(matches) < 3:
        return (
            False,
            f"Muy pocas referencias a artículos: {len(matches)} (mínimo esperado: 3)",
        )
    return True, f"{len(matches)} referencias a artículos encontradas"


def validar_no_duplicado(texto, hashes_existentes):
    """Deduplicación por SHA-256 del contenido normalizado."""
    normalizado = re.sub(r"\s+", " ", texto.strip().lower())
    sha = hashlib.sha256(normalizado.encode()).hexdigest()
    if sha in hashes_existentes:
        return False, f"Documento duplicado (hash: {sha[:12]}...)"
    hashes_existentes.add(sha)
    return True, sha


def validar_documento(titulo, texto, hashes_existentes):
    resultado = {
        "titulo": titulo,
        "aprobado": True,
        "errores": [],
        "advertencias": [],
        "hash": None,
    }

    for nombre, (ok, msg) in [
        ("longitud", validar_longitud(texto)),
        ("caracteres_basura", validar_ratio_basura(texto)),
        ("idioma", validar_idioma_espanol(texto)),
        ("estructura_legal", validar_estructura_legal(texto)),
    ]:
        if not ok:
            resultado["errores"].append(f"{nombre}: {msg}")
            resultado["aprobado"] = False
        elif nombre == "estructura_legal":
            resultado["advertencias"].append(f"estructura_legal: {msg}")

    ok_dup, msg_dup = validar_no_duplicado(texto, hashes_existentes)
    if not ok_dup:
        resultado["errores"].append(msg_dup)
        resultado["aprobado"] = False
    else:
        resultado["hash"] = msg_dup

    return resultado


def guardar_log(log, aprobados, rechazados):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("REPORTE DE CALIDAD — NORMAS DE TRÁNSITO\n")
        f.write("=" * 60 + "\n")
        f.write(f"Aprobados : {aprobados}\n")
        f.write(f"Rechazados: {rechazados}\n")
        f.write("=" * 60)
        f.write("\n".join(log))
    print(f"\n[*] Reporte guardado en: {LOG_PATH}")


def extract_text_with_ocr(pdf_path):
    text = ""
    try:
        images = convert_from_path(pdf_path)
        for img in images:
            text += pytesseract.image_to_string(img, lang="spa") + "\n"
    except Exception as e:
        print(f"[!] Error en OCR para {pdf_path}: {e}")
        print(
            "    Fedora: sudo dnf install tesseract tesseract-langpack-spa poppler-utils"
        )
        print(
            "    Ubuntu: sudo apt install tesseract-ocr tesseract-ocr-spa poppler-utils"
        )
    return text.strip()


def extract_text_from_pdf(pdf_path):
    """
    Extrae texto de un PDF usando pdfplumber con recorte de membrete.

    Estrategia por tipo de página:
      - Página 1 : recorta el 25% superior (logos, sellos de certificación, NIT)
      - Páginas 2+: recorta el 10% superior (encabezado repetido de resolución)

    Casos especiales manejados:
      - HTML guardado con extensión .pdf → procesado como HTML
      - Bounding box fuera de límites → clip a dimensiones reales de la página
      - PDF sin capa de texto (imagen pura) → fallback a OCR
    """
    # Fix 4: detectar por magic bytes si es realmente un PDF o un HTML disfrazado
    with open(pdf_path, "rb") as f:
        cabecera = f.read(10)
    if not cabecera.startswith(b"%PDF"):
        if cabecera[:1] in (b"<", b"\xef"):  # HTML o UTF-8 BOM + HTML
            print(
                "   -> [!] Archivo con extensión .pdf contiene HTML. Procesando como HTML..."
            )
            return extract_text_from_html(pdf_path)
        print(f"   -> [!] Archivo no reconocido (cabecera: {cabecera[:6]})")
        return ""

    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            paginas = []
            for i, pagina in enumerate(pdf.pages):
                recorte = 0.25 if i == 0 else 0.10
                y_top = pagina.height * recorte

                # Fix 3: clip bbox a las dimensiones reales de la página
                x0 = max(0, pagina.bbox[0])
                y0 = max(pagina.bbox[1], y_top)
                x1 = min(pagina.width, pagina.bbox[2])
                y1 = min(pagina.height, pagina.bbox[3])

                try:
                    zona = pagina.crop((x0, y0, x1, y1))
                    extraido = zona.extract_text(x_tolerance=2, y_tolerance=3) or ""
                except Exception:
                    extraido = pagina.extract_text(x_tolerance=2, y_tolerance=3) or ""

                extraido = _ARTEFACTOS_PDF.sub("", extraido)
                extraido = re.sub(r"\n{3,}", "\n\n", extraido).strip()
                if extraido:
                    paginas.append(extraido)
            text = "\n\n".join(paginas)
    except Exception as e:
        print(f"[!] Error leyendo PDF con pdfplumber {pdf_path}: {e}")

    # Fallback a OCR solo si no se extrajo texto (PDF de imagen pura)
    if len(text.strip()) < 100:
        print("   -> [!] PDF sin capa de texto. Usando OCR (puede tardar)...")
        text = extract_text_with_ocr(pdf_path)

    return text.strip()


def extract_text_from_docx(docx_path):
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
    """
    Extrae texto de un HTML respetando el charset declarado en el meta tag.

    Abre el archivo en modo bytes y deja que BeautifulSoup/lxml detecte
    el encoding desde <meta charset> o <meta http-equiv="Content-Type">.
    Esto evita perder tildes y ñ en documentos con ISO-8859-1 o windows-1252.
    """
    text = ""
    try:
        with open(html_path, "rb") as f:
            raw = f.read()
        # lxml detecta charset desde meta tags; si falla, intenta utf-8 y latin-1
        soup = BeautifulSoup(raw, "lxml")
        for elemento in soup(["script", "style", "nav", "header", "footer"]):
            elemento.decompose()
        text = soup.get_text(separator="\n", strip=True)
        # Normalizar espacios y saltos excesivos
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
    except Exception as e:
        print(f"[!] Error leyendo HTML {html_path}: {e}")
    return text


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documentos (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            titulo    TEXT,
            tipo      TEXT,
            contenido TEXT,
            hash      TEXT,
            procesado INTEGER DEFAULT 0,
            calidad   TEXT DEFAULT 'pendiente'
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documentos_rechazados (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            titulo  TEXT,
            tipo    TEXT,
            motivo  TEXT,
            fecha   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    print("[-] Base de datos inicializada.")
    return conn


def process_documents():
    conn = init_db()
    cursor = conn.cursor()
    log = []
    hashes_existentes = set()

    for (h,) in cursor.execute("SELECT hash FROM documentos WHERE hash IS NOT NULL"):
        hashes_existentes.add(h)

    files = glob.glob(os.path.join(RAW_DATA_DIR, "**", "*.*"), recursive=True)
    target_files = [
        f for f in files if f.endswith((".pdf", ".docx", ".doc", ".html", ".htm"))
    ]

    if not target_files:
        print(f"[!] No se encontraron documentos en {RAW_DATA_DIR}")
        return

    print(f"[*] {len(target_files)} documentos encontrados.\n")
    aprobados = rechazados = 0

    for file_path in target_files:
        filename = os.path.basename(file_path)
        ext = filename.rsplit(".", 1)[-1].lower()
        print(f"Procesando: {filename}...")

        cursor.execute("SELECT id FROM documentos WHERE titulo = ?", (filename,))
        if cursor.fetchone():
            print("   -> Ya existe en BD. Saltando.")
            continue

        content = ""
        if ext == "pdf":
            content = extract_text_from_pdf(file_path)
        elif ext in ("docx", "doc"):
            content = extract_text_from_docx(file_path)
        elif ext in ("html", "htm"):
            content = extract_text_from_html(file_path)

        resultado = validar_documento(filename, content, hashes_existentes)

        estado = "✓ APROBADO" if resultado["aprobado"] else "✗ RECHAZADO"
        lineas = [f"\n[{estado}] {resultado['titulo']}"]
        for e in resultado["errores"]:
            lineas.append(f"  ERROR: {e}")
        for a in resultado["advertencias"]:
            lineas.append(f"  INFO:  {a}")
        log.extend(lineas)

        if resultado["aprobado"]:
            cursor.execute(
                "INSERT INTO documentos (titulo, tipo, contenido, hash, calidad) VALUES (?, ?, ?, ?, ?)",
                (filename, ext, content, resultado["hash"], "aprobado"),
            )
            conn.commit()
            aprobados += 1
            print(f"   -> ✓ Aprobado. {len(content):,} caracteres guardados.")
        else:
            motivo = " | ".join(resultado["errores"])
            cursor.execute(
                "INSERT INTO documentos_rechazados (titulo, tipo, motivo) VALUES (?, ?, ?)",
                (filename, ext, motivo),
            )
            conn.commit()
            rechazados += 1
            print(f"   -> ✗ Rechazado: {motivo}")

    guardar_log(log, aprobados, rechazados)
    conn.close()

    print(f"\n{'─' * 50}")
    print(f"  Aprobados : {aprobados}")
    print(f"  Rechazados: {rechazados}")
    print(f"{'─' * 50}")
    if rechazados > 0:
        print(f"  Revisa {LOG_PATH} para el detalle de rechazados.")
    print("\n[*] Siguiente paso: python process.py")


if __name__ == "__main__":
    process_documents()
