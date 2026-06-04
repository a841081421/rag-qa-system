"""BM25 keyword search engine with Chinese tokenization."""

import json
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

from config import BM25_INDEX_FILE


class BM25Search:
    def __init__(self, index_file: str | None = None):
        self._path = Path(index_file or BM25_INDEX_FILE)
        self.ids: list[str] = []
        self.corpus: list[str] = []
        self.bm25: BM25Okapi | None = None
        self._load()

    # ── persistence ──

    def _load(self) -> None:
        if not self._path.exists():
            return
        data = json.loads(self._path.read_text(encoding="utf-8"))
        self.ids = data.get("ids", [])
        self.corpus = data.get("documents", [])
        if self.corpus:
            tokenized = [self._tokenize(d) for d in self.corpus]
            self.bm25 = BM25Okapi(tokenized)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {"ids": self.ids, "documents": self.corpus}
        self._path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── tokenization ──

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [w for w in jieba.cut(text) if w.strip()]

    # ── public API ──

    def add_documents(self, ids: list[str], documents: list[str]) -> None:
        existing = set(self.ids)
        new_ids, new_docs = [], []
        for id_, doc in zip(ids, documents):
            if id_ not in existing:
                new_ids.append(id_)
                new_docs.append(doc)
                existing.add(id_)
        if not new_ids:
            return
        self.ids.extend(new_ids)
        self.corpus.extend(new_docs)
        tokenized = [self._tokenize(d) for d in self.corpus]
        self.bm25 = BM25Okapi(tokenized)
        self._save()

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        if not self.bm25 or not self.corpus:
            return []
        scores = self.bm25.get_scores(self._tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for idx, score in ranked:
            if score <= 0:
                continue
            results.append({
                "id": self.ids[idx],
                "document": self.corpus[idx],
                "score": float(score),
            })
        return results

    def count(self) -> int:
        return len(self.corpus)

    def clear(self) -> None:
        self.ids.clear()
        self.corpus.clear()
        self.bm25 = None
        if self._path.exists():
            self._path.write_text('{"ids":[],"documents":[]}', encoding="utf-8")
