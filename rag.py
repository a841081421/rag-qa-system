"""RAG pipeline: hybrid search → rerank → generate."""

from __future__ import annotations

from pathlib import Path
from typing import Generator

from embedder import Embedder
from hybrid_search import HybridSearch
from reranker import Reranker
from generator import Generator
from config import TRANSLATED_DIR, RERANK_TOP_K, EXPAND_K


def chunk_text(text: str, min_length: int = 50) -> list[str]:
    paragraphs = text.split("\n\n")
    return [p.strip() for p in paragraphs if len(p.strip()) >= min_length]


class RAG:
    def __init__(self):
        self.hybrid = HybridSearch()
        self.embedder = self.hybrid.embedder
        self.reranker = Reranker()
        self.generator = Generator()

    @property
    def store(self):
        return self.hybrid.vector_store

    # ── Ingestion ──

    def ingest_file(self, file_path: str | Path) -> int:
        path = Path(file_path)
        content = path.read_text(encoding="utf-8")
        chunks = chunk_text(content)
        if not chunks:
            return 0

        embeddings = self.embedder.embed_batch(chunks)
        ids = [f"{path.stem}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": str(path), "chunk_index": i} for i in range(len(chunks))]

        self.hybrid.vector_store.add_documents(
            ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas
        )
        self.hybrid.bm25_search.add_documents(ids=ids, documents=chunks)
        return len(chunks)

    def ingest_directory(self, directory: str | None = None, glob_pattern: str = "*.md") -> int:
        base = Path(directory or TRANSLATED_DIR)
        if not base.exists():
            raise FileNotFoundError(f"目录不存在: {base}")
        files = list(base.rglob(glob_pattern))
        total = 0
        for f in files:
            try:
                n = self.ingest_file(f)
                print(f"  入库: {f.name} -> {n} 个片段")
                total += n
            except Exception as e:
                print(f"  入库失败: {f.name}: {e}")
        return total

    # ── Retrieval ──

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        use_reranker: bool = True,
    ) -> list[dict]:
        expand = max(top_k * 3, EXPAND_K)
        results = self.hybrid.search(query, top_k=expand, expand_k=expand)
        if not results:
            return []

        if use_reranker:
            documents = [r["document"] for r in results]
            reranked = self.reranker.rerank(query, documents, top_k=top_k)
            doc_score = {r["document"]: r["score"] for r in reranked}
            id_map = {r["document"]: r for r in results}
            final = []
            for rr in reranked:
                orig = id_map.get(rr["document"], {})
                final.append({
                    "document": rr["document"],
                    "source": orig.get("source", ""),
                    "score": rr["score"],
                })
            return final

        return results[:top_k]

    # ── Generation ──

    def ask(
        self,
        question: str,
        top_k: int = 3,
        history: list[dict] | None = None,
        use_reranker: bool = True,
    ) -> dict:
        results = self.retrieve(question, top_k=top_k, use_reranker=use_reranker)

        if not results:
            return {
                "answer": "知识库中暂无相关内容，请先导入文档后再提问。",
                "contexts": [],
                "sources": [],
                "scores": [],
                "retrieval_mode": "hybrid",
            }

        contexts = [r["document"] for r in results]
        answer = self.generator.generate(question, contexts, history=history)

        return {
            "answer": answer,
            "contexts": contexts,
            "sources": [r["source"] for r in results],
            "scores": [r["score"] for r in results],
            "retrieval_mode": "hybrid",
        }

    def ask_stream(
        self,
        question: str,
        top_k: int = 3,
        history: list[dict] | None = None,
        use_reranker: bool = True,
    ) -> Generator[str, None, None]:
        results = self.retrieve(question, top_k=top_k, use_reranker=use_reranker)
        if not results:
            yield "知识库中暂无相关内容，请先导入文档后再提问。"
            return
        contexts = [r["document"] for r in results]
        yield from self.generator.generate_stream(question, contexts, history=history)

    def ask_json(
        self,
        question: str,
        top_k: int = 3,
        history: list[dict] | None = None,
        use_reranker: bool = True,
    ) -> dict:
        results = self.retrieve(question, top_k=top_k, use_reranker=use_reranker)
        if not results:
            return {
                "answer": "知识库中暂无相关内容，请先导入文档后再提问。",
                "confidence": 0.0,
                "key_points": [],
                "sources": [],
            }
        contexts = [r["document"] for r in results]
        result = self.generator.generate_json(question, contexts, history=history)
        result["sources"] = [r["source"] for r in results]
        return result
