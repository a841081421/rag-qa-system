# Multimodal Image Q&A Design

## Goal

Add image question-answering to the existing FastAPI RAG system. Users can upload images alongside text questions; the system analyzes images with DeepSeek VL and returns answers. The text retrieval pipeline (hybrid search + reranker) remains unchanged.

## Approach

Base64 inline: images are sent as base64 strings in the request body. No file storage, no extra infrastructure.

## Data Flow

```
User: text question + images (base64, optional)
  → FastAPI /api/ask
    → RAG.retrieve(): text-only hybrid search (unchanged)
    → Generator: if images present, build multimodal messages → DeepSeek VL
    → Return answer
```

## File Changes

### config.py (+2 lines)

Add `VL_MODEL_NAME` config variable. Defaults to `deepseek-chat` (DeepSeek's vision-capable model).

### schemas.py (+3 lines)

Add optional field to `AskRequest`:

```python
images: list[str] = Field(default_factory=list)  # base64 encoded images
```

### generator.py (~25 lines changed)

- `_build_messages()`: when `images` is non-empty, construct OpenAI multimodal message format:
  ```json
  {"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
    {"type": "text", "text": "..."}
  ]}
  ```
- `generate()`, `generate_stream()`, `generate_json()`: add `images: list[str] | None = None` parameter
- When images present, use `VL_MODEL_NAME`; otherwise keep `deepseek-chat`

### rag.py (~6 lines changed)

- `ask()`, `ask_stream()`, `ask_json()`: add `images` parameter, pass through to generator

### main.py (~6 lines changed)

- `ask()`, `ask_stream()`, `ask_json()` endpoints: pass `req.images` to RAG methods
- Validation: reject if `len(images) > 3`

### static/index.html (~30 lines)

- Image upload button (📷) next to the input bar
- File picker → FileReader → base64 → store in variable
- Thumbnail preview with ✕ clear button
- Include base64 list in request body `images` field

## Files NOT Changed

store.py, bm25_search.py, hybrid_search.py, reranker.py, embedder.py — the retrieval chain is unaffected.

## Constraints

- Max 3 images per request
- Single image max 4MB (base64 ~5.3MB)
- Supported formats: JPEG, PNG, WebP
- Images only used as LLM input, not indexed into knowledge base
