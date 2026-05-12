from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL_NAME


class Embedder:
    def __init__(self, model_name: str | None = None):
        name = model_name or EMBED_MODEL_NAME
        self.model = SentenceTransformer(name)

    def embed(self, text: str) -> list[float]:
        """将单段文本转换为向量。"""
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量将文本转换为向量。"""
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        """返回向量维度。"""
        return self.model.get_sentence_embedding_dimension()
