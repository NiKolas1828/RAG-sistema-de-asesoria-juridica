import chromadb
from src.config import CHROMA_PATH


class VectorManager:
    def __init__(self):
        # PersistentClient asegura que los datos se guarden en tu disco duro
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        # Creamos la colección. No asignamos modelo aquí porque ya los generamos localmente
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
