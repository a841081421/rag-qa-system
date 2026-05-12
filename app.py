import streamlit as st
from pathlib import Path
from rag import RAG
from fetcher import clone_repos
from translator import translate_directory
from config import KNOWLEDGE_BASE_DIR, TRANSLATED_DIR

st.set_page_config(page_title="RAG 知识库问答", layout="wide")
st.title("AI Agent 知识库问答系统")

# Initialize RAG system (cached, loads once)
@st.cache_resource
def get_rag():
    return RAG()

rag = get_rag()

# === Sidebar: Document Management ===
with st.sidebar:
    st.header("📄 文档管理")

    if st.button("🔄 拉取 Anthropic 文档", use_container_width=True):
        with st.spinner("正在从 GitHub 拉取..."):
            try:
                files = clone_repos()
                st.success(f"已拉取 {len(files)} 个文档文件")
            except Exception as e:
                st.error(f"拉取失败: {e}")

    if st.button("🌐 翻译文档", use_container_width=True):
        with st.spinner("正在翻译（可能需要几分钟）..."):
            try:
                results = translate_directory(KNOWLEDGE_BASE_DIR, TRANSLATED_DIR)
                st.success(f"已翻译 {len(results)} 个文档")
            except Exception as e:
                st.error(f"翻译失败: {e}")

    if st.button("📥 入库", use_container_width=True):
        with st.spinner("正在入库..."):
            try:
                if Path(TRANSLATED_DIR).exists():
                    n = rag.ingest_directory(TRANSLATED_DIR)
                    st.success(f"已入库 {n} 个文档片段")
                else:
                    st.warning("请先翻译文档再入库")
            except Exception as e:
                st.error(f"入库失败: {e}")

    st.divider()
    st.metric("已入库片段数", rag.store.count())

    st.divider()
    st.caption("知识来源：Anthropic Cookbook & Courses")
    st.caption("Embedding：BAAI/bge-small-zh-v1.5")
    st.caption("LLM：DeepSeek")

# === Main Area: Q&A ===
st.header("💬 问答")

query = st.text_input("请输入你的 AI Agent 相关问题", placeholder="例如：什么是 Tool Use？如何实现一个 Agent？")

if query:
    with st.spinner("检索中..."):
        try:
            result = rag.ask(query)

            st.markdown("### 📝 回答")
            st.markdown(result["answer"])

            with st.expander("🔍 参考来源（检索到的原文片段）"):
                for i, (ctx, src, dist) in enumerate(
                    zip(result["contexts"], result["sources"], result["distances"])
                ):
                    similarity = 1 - dist if dist else 0
                    st.markdown(f"**片段 {i+1}**（相似度: {similarity:.2%} | 来源: {src}）")
                    st.text(ctx[:500])
                    st.divider()
        except Exception as e:
            st.error(f"查询失败: {e}")
