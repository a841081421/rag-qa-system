"""FastAPI application: end-to-end RAG system."""

import json
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from rag import RAG
from schemas import (
    AskRequest,
    AskResponse,
    IngestRequest,
    IngestResponse,
    StatsResponse,
    HealthResponse,
    JsonAskResponse,
)
from config import (
    EMBED_MODEL_NAME,
    RERANKER_MODEL_NAME,
    VECTOR_WEIGHT,
    BM25_WEIGHT,
    HOST,
    PORT,
)

# ── Lifespan ──

rag: RAG | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    print("Loading RAG system …")
    rag = RAG()
    print(f"  Embedding model : {EMBED_MODEL_NAME}")
    print(f"  Reranker model  : {RERANKER_MODEL_NAME}")
    print(f"  Hybrid weights  : vector={VECTOR_WEIGHT} bm25={BM25_WEIGHT}")
    print(f"  Stored chunks   : {rag.store.count()}")
    print("RAG system ready.")
    yield


app = FastAPI(
    title="RAG 知识库问答系统",
    description="基于 FastAPI 的端到端 RAG 系统：文档摄入 → 混合搜索 → 重排序 → 智能问答",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_rag() -> RAG:
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    return rag


# ── Static frontend ──

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def index():
    html = STATIC_DIR / "index.html"
    if html.exists():
        return FileResponse(str(html))
    return {"message": "RAG API is running. Visit /docs for Swagger UI."}


# ── Health & Stats ──

@app.get("/api/health", response_model=HealthResponse)
async def health():
    r = _get_rag()
    return HealthResponse(
        status="ok",
        embed_model_loaded=True,
        reranker_loaded=r.reranker.is_loaded,
    )


@app.get("/api/stats", response_model=StatsResponse)
async def stats():
    r = _get_rag()
    return StatsResponse(
        total_chunks=r.store.count(),
        embed_model=EMBED_MODEL_NAME,
        reranker_model=RERANKER_MODEL_NAME,
        vector_weight=VECTOR_WEIGHT,
        bm25_weight=BM25_WEIGHT,
    )


# ── Document Ingestion ──

@app.post("/api/documents/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    r = _get_rag()
    try:
        n = r.ingest_directory(req.directory, req.glob_pattern)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return IngestResponse(chunk_count=n, message=f"成功入库 {n} 个文档片段")


@app.delete("/api/documents/clear")
async def clear_documents():
    r = _get_rag()
    r.store.clear()
    r.hybrid.bm25_search.clear()
    return {"message": "已清空所有文档"}


# ── Q&A: Normal ──

@app.post("/api/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    r = _get_rag()
    history = [h.model_dump() for h in req.history] if req.history else None
    result = r.ask(
        question=req.question,
        top_k=req.top_k,
        history=history,
        use_reranker=req.use_reranker,
    )
    return AskResponse(**result)


# ── Q&A: SSE Streaming ──

@app.post("/api/ask/stream")
async def ask_stream(req: AskRequest):
    r = _get_rag()
    history = [h.model_dump() for h in req.history] if req.history else None

    def event_stream():
        try:
            for token in r.ask_stream(
                question=req.question,
                top_k=req.top_k,
                history=history,
                use_reranker=req.use_reranker,
            ):
                data = json.dumps({"token": token}, ensure_ascii=False)
                yield f"data: {data}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Q&A: JSON Structured ──

@app.post("/api/ask/json", response_model=JsonAskResponse)
async def ask_json(req: AskRequest):
    r = _get_rag()
    history = [h.model_dump() for h in req.history] if req.history else None
    result = r.ask_json(
        question=req.question,
        top_k=req.top_k,
        history=history,
        use_reranker=req.use_reranker,
    )
    return JsonAskResponse(**result)


# ── CLI ──

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=HOST, port=PORT, reload="--reload" in sys.argv)
