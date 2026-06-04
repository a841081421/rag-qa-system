"""BGE Reranker: cross-encoder model for result re-scoring."""

from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL_NAME


class Reranker:
    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or RERANKER_MODEL_NAME
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(
        self, query: str, documents: list[str], top_k: int = 3
    ) -> list[dict]:
        if not documents:
            return []
        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs).tolist()
        ranked = sorted(
            zip(documents, scores), key=lambda x: x[1], reverse=True
        )
        return [
            {"document": doc, "score": float(s)}
            for doc, s in ranked[:top_k]
        ]

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
