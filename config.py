import os
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-small-zh-v1.5")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
KNOWLEDGE_BASE_DIR = os.getenv("KNOWLEDGE_BASE_DIR", "./knowledge_base")
TRANSLATED_DIR = os.getenv("TRANSLATED_DIR", "./translated_docs")
