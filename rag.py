from pathlib import Path
from embedder import Embedder
from store import VectorStore
from generator import Generator
from config import TRANSLATED_DIR


def chunk_text(text: str, min_length: int = 50) -> list[str]:
    """按空行切分文本为段落，过滤掉过短的段落。"""
    paragraphs = text.split("\n\n")
    chunks = []
    for p in paragraphs:
        p = p.strip()
        if len(p) >= min_length:
            chunks.append(p)
    return chunks


class RAG:
    def __init__(self):
        self.embedder = Embedder()
        self.store = VectorStore()
        self.generator = Generator()

    def ingest_file(self, file_path: str | Path) -> int:
        """将单个 Markdown 文件分块、向量化、存入向量库。返回存入的 chunk 数量。"""
        path = Path(file_path)
        content = path.read_text(encoding="utf-8")
        chunks = chunk_text(content)

        if not chunks:
            return 0

        embeddings = self.embedder.embed_batch(chunks)
        ids = [f"{path.stem}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": str(path), "chunk_index": i} for i in range(len(chunks))]

        self.store.add_documents(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
        return len(chunks)

    def ingest_directory(self, directory: str | None = None, glob_pattern: str = "*.md") -> int:
        """批量导入目录下所有 Markdown 文件。返回总 chunk 数。"""
        base = Path(directory or TRANSLATED_DIR)
        if not base.exists():
            raise FileNotFoundError(f"目录不存在: {base}")

        files = list(base.rglob(glob_pattern))
        total = 0

        for f in files:
            n = self.ingest_file(f)
            print(f"  入库: {f.name} -> {n} 个片段")
            total += n

        return total

    def ask(self, question: str, top_k: int = 3) -> dict:
        """提问并获取回答，同时返回检索到的来源片段。"""
        query_embedding = self.embedder.embed(question)
        search_results = self.store.search(query_embedding, top_k=top_k)

        contexts = search_results["documents"]
        answer = self.generator.generate(question, contexts)

        return {
            "answer": answer,
            "contexts": contexts,
            "sources": [m.get("source", "") for m in search_results["metadatas"]],
            "distances": search_results["distances"],
        }
