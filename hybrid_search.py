"""Hybrid search: vector (ChromaDB) 0.6 + keyword (BM25) 0.4."""

from __future__ import annotations

from store import VectorStore
from bm25_search import BM25Search
from embedder import Embedder
from config import VECTOR_WEIGHT, BM25_WEIGHT, EXPAND_K


def _normalize_scores(scored: list[dict], key: str = "score") -> list[dict]:
    """Min-max normalize scores to [0, 1]. Single result defaults to 1.0."""
    if not scored:
        return scored
    values = [r[key] for r in scored]
    lo, hi = min(values), max(values)
    if lo == hi:
        for r in scored:
            r["norm_score"] = 1.0
        return scored
    rng = hi - lo
    for r in scored:
        r["norm_score"] = (r[key] - lo) / rng
    return scored


class HybridSearch:
    def __init__(self):
        self.vector_store = VectorStore()
        self.bm25_search = BM25Search()
        self.embedder = Embedder()

    def search(self, query: str, top_k: int = 3, expand_k: int | None = None) -> list[dict]:
        """Run hybrid search and return merged results.

        Each result dict: {id, document, source, score, vector_score, bm25_score}
        """
        k = expand_k or EXPAND_K
        vector_results = self._vector_search(query, k)
        bm25_results = self._bm25_search(query, k)
        return self._merge(vector_results, bm25_results, top_k)

    def _vector_search(self, query: str, top_k: int) -> list[dict]:
        if self.vector_store.count() == 0:
            return []
        emb = self.embedder.embed(query)
        res = self.vector_store.search(emb, top_k=top_k)
        results = []
        for i in range(len(res["ids"])):
            distance = res["distances"][i]
            results.append({
                "id": res["ids"][i],
                "document": res["documents"][i],
                "source": res["metadatas"][i].get("source", "") if res["metadatas"] else "",
                "score": 1.0 - distance,
            })
        return results

    def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        raw = self.bm25_search.search(query, top_k=top_k)
        return [{"id": r["id"], "document": r["document"], "source": "", "score": r["score"]} for r in raw]

    @staticmethod
    def _merge(vector_results: list[dict], bm25_results: list[dict], top_k: int) -> list[dict]:
        vector_results = _normalize_scores(vector_results)
        bm25_results = _normalize_scores(bm25_results)

        merged: dict[str, dict] = {}

        for r in vector_results:
            merged[r["id"]] = {
                "id": r["id"],
                "document": r["document"],
                "source": r.get("source", ""),
                "vector_score": r["norm_score"],
                "bm25_score": 0.0,
            }

        for r in bm25_results:
            if r["id"] in merged:
                merged[r["id"]]["bm25_score"] = r["norm_score"]
            else:
                merged[r["id"]] = {
                    "id": r["id"],
                    "document": r["document"],
                    "source": "",
                    "vector_score": 0.0,
                    "bm25_score": r["norm_score"],
                }

        for item in merged.values():
            item["score"] = VECTOR_WEIGHT * item["vector_score"] + BM25_WEIGHT * item["bm25_score"]

        ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]
