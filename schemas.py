"""FastAPI request / response models."""

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=20)
    history: list[Message] = Field(default_factory=list)
    use_reranker: bool = Field(default=True)


class AskResponse(BaseModel):
    answer: str
    contexts: list[str]
    sources: list[str]
    scores: list[float]
    retrieval_mode: str = "hybrid"


class IngestRequest(BaseModel):
    directory: str | None = None
    glob_pattern: str = "*.md"


class IngestResponse(BaseModel):
    chunk_count: int
    message: str


class StatsResponse(BaseModel):
    total_chunks: int
    embed_model: str
    reranker_model: str
    vector_weight: float
    bm25_weight: float


class HealthResponse(BaseModel):
    status: str
    embed_model_loaded: bool
    reranker_loaded: bool


class JsonAskResponse(BaseModel):
    answer: str
    confidence: float
    key_points: list[str]
    sources: list[str]
