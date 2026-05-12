import sqlite3
import json
from src.config import DB_PATH
from src.vector_store.local_embedder import generate_local_embeddings
from src.vector_store.vector_manager import VectorManager


def run_embedding_pipeline():
    """
    Orquestador para generar embeddings locales y marcar el progreso en SQLite.
    """
    print("\n--- [INICIANDO PIPELINE DE EMBEDDINGS LOCALES] ---")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Inicializamos el gestor de ChromaDB
    v_manager = VectorManager()

    # 1. Obtener solo los chunks que no han sido indexados
    cursor.execute("SELECT id, texto, metadata FROM chunks WHERE procesado = 0")
    rows = cursor.fetchall()
    total_pendientes = len(rows)

    if total_pendientes == 0:
        print("✅ No hay fragmentos pendientes. La base de datos está al día.")
        conn.close()
        return

    print(f"[*] Se encontraron {total_pendientes} fragmentos para procesar.")

    # 2. Procesar en bloques para manejar mejor la memoria RAM
    batch_size = 100
    procesados_exitosamente = 0

    for i in range(0, total_pendientes, batch_size):
        batch_rows = rows[i : i + batch_size]
        texts = [r["texto"] for r in batch_rows]
        ids = [r["id"] for r in batch_rows]
        metadata = [json.loads(r["metadata"]) for r in batch_rows]

        embeddings = generate_local_embeddings(texts)

        if embeddings and len(embeddings) == len(ids):
            v_manager.upsert_batch(ids, embeddings, texts, metadata)

            # 3. Actualizar el flag en la base de datos
            cursor.executemany(
                "UPDATE chunks SET procesado = 1 WHERE id = ?", [(idx,) for idx in ids]
            )
            conn.commit()  # Guardamos progreso tras cada lote

            procesados_exitosamente += len(ids)
            print(
                f"    -> Progreso: {procesados_exitosamente}/{total_pendientes} completados...",
                end="\r",
            )
        else:
            print(f"\n[!] Error procesando el lote que inicia en ID {ids[0]}.")
            break

    print(f"\n✅ Sincronización completada. {len(rows)} fragmentos indexados.")
    conn.close()
