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
            self._collection = self.client.get_or_create_collection(name=self.collection_name)
        return self._collection

    def add_documents(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
    ) -> None:
        """批量存入文档、向量和元数据。"""
        self.collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    def search(self, query_embedding: list[float], top_k: int = 3) -> dict:
        """检索最相似的 top_k 个文档片段。"""
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        return {
            "ids": results["ids"][0],
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
        }

    def count(self) -> int:
        """返回已存储的文档数量。"""
        return self.collection.count()

    def clear(self) -> None:
        """清空集合。"""
        self.client.delete_collection(name=self.collection_name)
        self._collection = None
