-- ============================================================
-- normas.db — esquema SQLite para el corpus de normas de tránsito
-- ============================================================

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ------------------------------------------------------------
-- Tabla: documentos
-- Registro de cada fuente documental (ley, resolución, decreto)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documentos (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre           TEXT    NOT NULL,
    tipo             TEXT    NOT NULL CHECK (tipo IN ('ley', 'resolucion', 'decreto', 'manual', 'otro')),
    url              TEXT,
    fecha_vigencia   DATE,
    fecha_descarga   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    activo           INTEGER DEFAULT 1,   -- 1 = vigente, 0 = derogado
    notas            TEXT
);

-- ------------------------------------------------------------
-- Tabla: articulos
-- Cada artículo extraído de un documento
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS articulos (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id           INTEGER NOT NULL REFERENCES documentos(id) ON DELETE CASCADE,
    numero           TEXT,                -- ej. "55", "55A"
    titulo           TEXT,                -- título del artículo si existe
    capitulo         TEXT,                -- sección/capítulo de pertenencia
    texto            TEXT    NOT NULL,
    orden            INTEGER,             -- posición dentro del documento
    tokens_estimados INTEGER GENERATED ALWAYS AS (
                         CAST(LENGTH(texto) / 4.0 AS INTEGER)
                     ) VIRTUAL
);

-- ------------------------------------------------------------
-- Tabla: chunks
-- Fragmentos procesados listos para embedding
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    art_id           INTEGER REFERENCES articulos(id) ON DELETE CASCADE,
    doc_id           INTEGER NOT NULL REFERENCES documentos(id),
    texto            TEXT    NOT NULL,
    tokens_estimados INTEGER,
    metadata         TEXT,               -- JSON: fuente, articulo, capitulo, fecha
    embedding_ok     INTEGER DEFAULT 0   -- 0 = pendiente, 1 = ya embebido
);

-- ------------------------------------------------------------
-- Índices para consultas frecuentes
-- ------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_articulos_doc     ON articulos (doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc        ON chunks    (doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding  ON chunks    (embedding_ok);
CREATE INDEX IF NOT EXISTS idx_documentos_tipo   ON documentos (tipo);
