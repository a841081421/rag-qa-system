"""ChromaDB vector store with cosine distance."""

import chromadb
from config import CHROMA_PERSIST_DIR


class VectorStore:
    def __init__(self, persist_dir: str | None = None, collection_name: str = "knowledge_base"):
        path = persist_dir or CHROMA_PERSIST_DIR
        self.client = chromadb.PersistentClient(path=path)
        self.collection_name = collection_name
        self._collection = None

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add_documents(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
    ) -> None:
        self.collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    def search(self, query_embedding: list[float], top_k: int = 10) -> dict:
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        return {
            "ids": results["ids"][0],
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
        }

    def count(self) -> int:
        return self.collection.count()

    def clear(self) -> None:
        self.client.delete_collection(name=self.collection_name)
        self._collection = None
