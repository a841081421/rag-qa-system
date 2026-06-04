import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# ── API ──
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# ── Models ──
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-small-zh-v1.5")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")

# ── Paths ──
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
BM25_INDEX_FILE = os.getenv("BM25_INDEX_FILE", "./bm25_index.json")
KNOWLEDGE_BASE_DIR = os.getenv("KNOWLEDGE_BASE_DIR", "./knowledge_base")
TRANSLATED_DIR = os.getenv("TRANSLATED_DIR", "./translated_docs")

# ── Hybrid Search ──
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.6"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.4"))
EXPAND_K = int(os.getenv("EXPAND_K", "10"))

# ── Reranker ──
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))

# ── Server ──
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
