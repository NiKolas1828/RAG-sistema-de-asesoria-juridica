import chromadb
from src.config import CHROMA_PATH, CHROMA_MODE, CHROMA_HOST, CHROMA_PORT


class VectorManager:
    def __init__(self):
        if CHROMA_MODE == "http":
            # Conexión remota/Docker mediante HTTP
            self.client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        else:
            # Cliente local persistente en disco
            self.client = chromadb.PersistentClient(path=str(CHROMA_PATH))
            
        # Creamos o cargamos la colección. No asignamos modelo aquí porque ya los generamos localmente
        self.collection = self.client.get_or_create_collection(name="embeddings")

    def upsert_batch(self, ids, embeddings, documents, metadatas):
        """
        Inserta o actualiza fragmentos en la base vectorial.
        """
        self.collection.upsert(
            ids=[str(i) for i in ids],
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
