# Multimodal Image Q&A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add image question-answering to the RAG system — users send base64 images alongside text questions, DeepSeek VL analyzes them.

**Architecture:** Images are passed as base64 strings in the existing `AskRequest`. The generator detects images and constructs multimodal OpenAI-format messages. Retrieval pipeline is untouched. Frontend gets an upload button.

**Tech Stack:** DeepSeek VL (via existing OpenAI SDK), base64 encoding, FastAPI multipart-aware JSON body.

---

### Task 1: Add VL model config to config.py

**Files:**
- Modify: `config.py:10-12`

- [ ] **Step 1: Add VL_MODEL_NAME after existing model configs**

Open `config.py` and add one line after line 12 (`RERANKER_MODEL_NAME`):

```python
VL_MODEL_NAME = os.getenv("VL_MODEL_NAME", "deepseek-chat")
```

The full Models section becomes:

```python
# ── Models ──
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-small-zh-v1.5")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
VL_MODEL_NAME = os.getenv("VL_MODEL_NAME", "deepseek-chat")
```

- [ ] **Step 2: Verify config loads**

Run: `python -c "from config import VL_MODEL_NAME; print(VL_MODEL_NAME)"`
Expected: `deepseek-chat`

- [ ] **Step 3: Commit**

```bash
git add config.py
git commit -m "feat: add VL_MODEL_NAME config for vision model"
```

---

### Task 2: Add images field to AskRequest schema

**Files:**
- Modify: `schemas.py:11-15`

- [ ] **Step 1: Add images field to AskRequest**

Open `schemas.py` and add one field to `AskRequest` after `use_reranker` (line 15):

```python
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=20)
    history: list[Message] = Field(default_factory=list)
    use_reranker: bool = Field(default=True)
    images: list[str] = Field(default_factory=list)
```

- [ ] **Step 2: Verify schema loads and defaults correctly**

Run: `python -c "from schemas import AskRequest; r = AskRequest(question='test'); print('images:', r.images)"`
Expected: `images: []`

- [ ] **Step 3: Commit**

```bash
git add schemas.py
git commit -m "feat: add optional images field to AskRequest"
```

---

### Task 3: Add multimodal message building to generator.py

**Files:**
- Modify: `generator.py:9` (import)
- Modify: `generator.py:39-57` (`_build_messages`)
- Modify: `generator.py:60-134` (`Generator` class)

- [ ] **Step 1: Add VL_MODEL_NAME import**

Change line 9 from:

```python
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
```

to:

```python
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, VL_MODEL_NAME
```

- [ ] **Step 2: Update _build_messages to accept and handle images**

Replace the entire `_build_messages` function (lines 39-57) with:

```python
def _build_messages(
    query: str,
    contexts: list[str],
    history: list[dict] | None = None,
    json_mode: bool = False,
    images: list[str] | None = None,
) -> list[dict]:
    system = JSON_SYSTEM_PROMPT if json_mode else SYSTEM_PROMPT
    messages = [{"role": "system", "content": system}]

    if history:
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})

    context_text = "\n\n---\n\n".join(
        f"[来源 {i + 1}]\n{ctx}" for i, ctx in enumerate(contexts)
    )
    user_text = CONTEXT_PROMPT_TEMPLATE.format(context=context_text, query=query)

    if images:
        content = [{"type": "text", "text": user_text}]
        for img in images:
            content.insert(0, {"type": "image_url", "image_url": {"url": img}})
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": user_text})
    return messages
```

Key change: when `images` is non-empty, the user message `content` becomes an array with image_url objects + text, instead of a plain string. Images are inserted before the text so the model sees them first.

- [ ] **Step 3: Update generate() to accept images**

Replace the `generate` method (lines 64-83) with:

```python
    def generate(
        self,
        query: str,
        contexts: list[str],
        history: list[dict] | None = None,
        images: list[str] | None = None,
    ) -> str:
        messages = _build_messages(query, contexts, history, images=images)
        model = VL_MODEL_NAME if images else "deepseek-chat"
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=2048,
            )
        except APIError as e:
            raise RuntimeError(f"LLM API 调用失败: {e}") from e
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("模型返回了空内容")
        return content
```

- [ ] **Step 4: Update generate_stream() to accept images**

Replace the `generate_stream` method (lines 85-105) with:

```python
    def generate_stream(
        self,
        query: str,
        contexts: list[str],
        history: list[dict] | None = None,
        images: list[str] | None = None,
    ) -> Generator[str, None, None]:
        messages = _build_messages(query, contexts, history, images=images)
        model = VL_MODEL_NAME if images else "deepseek-chat"
        try:
            stream = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=2048,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except APIError as e:
            raise RuntimeError(f"LLM API 调用失败: {e}") from e
```

- [ ] **Step 5: Update generate_json() to accept images**

Replace the `generate_json` method (lines 107-134) with:

```python
    def generate_json(
        self,
        query: str,
        contexts: list[str],
        history: list[dict] | None = None,
        images: list[str] | None = None,
    ) -> dict:
        messages = _build_messages(query, contexts, history, json_mode=True, images=images)
        model = VL_MODEL_NAME if images else "deepseek-chat"
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=2048,
            )
        except APIError as e:
            raise RuntimeError(f"LLM API 调用失败: {e}") from e
        content = response.choices[0].message.content or ""
        content = re.sub(r'^```\w*\n?', '', content.strip())
        content = content.rstrip('`').strip()
        try:
            return json.loads(content.strip())
        except json.JSONDecodeError:
            return {
                "answer": content,
                "confidence": 0.0,
                "key_points": [],
                "sources": [],
            }
```

- [ ] **Step 6: Verify generator loads**

Run: `python -c "from generator import Generator, _build_messages; print('OK')"`
Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add generator.py
git commit -m "feat: add multimodal message building and image support to generator"
```

---

### Task 4: Thread images through rag.py

**Files:**
- Modify: `rag.py:97-158`

- [ ] **Step 1: Add images parameter to ask()**

Replace the `ask` method (lines 97-124) with:

```python
    def ask(
        self,
        question: str,
        top_k: int = 3,
        history: list[dict] | None = None,
        use_reranker: bool = True,
        images: list[str] | None = None,
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
        answer = self.generator.generate(question, contexts, history=history, images=images)

        return {
            "answer": answer,
            "contexts": contexts,
            "sources": [r["source"] for r in results],
            "scores": [r["score"] for r in results],
            "retrieval_mode": "hybrid",
        }
```

- [ ] **Step 2: Add images parameter to ask_stream()**

Replace the `ask_stream` method (lines 126-138) with:

```python
    def ask_stream(
        self,
        question: str,
        top_k: int = 3,
        history: list[dict] | None = None,
        use_reranker: bool = True,
        images: list[str] | None = None,
    ) -> Generator[str, None, None]:
        results = self.retrieve(question, top_k=top_k, use_reranker=use_reranker)
        if not results:
            yield "知识库中暂无相关内容，请先导入文档后再提问。"
            return
        contexts = [r["document"] for r in results]
        yield from self.generator.generate_stream(question, contexts, history=history, images=images)
```

- [ ] **Step 3: Add images parameter to ask_json()**

Replace the `ask_json` method (lines 140-158) with:

```python
    def ask_json(
        self,
        question: str,
        top_k: int = 3,
        history: list[dict] | None = None,
        use_reranker: bool = True,
        images: list[str] | None = None,
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
        result = self.generator.generate_json(question, contexts, history=history, images=images)
        result["sources"] = [r["source"] for r in results]
        return result
```

- [ ] **Step 4: Verify rag.py loads**

Run: `python -c "from rag import RAG; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add rag.py
git commit -m "feat: thread images parameter through RAG pipeline"
```

---

### Task 5: Add validation and image passthrough to main.py

**Files:**
- Modify: `main.py:130-181`

- [ ] **Step 1: Add validation helper and update ask endpoint**

Add a constant after `_get_rag()` (after line 68) and update the three ask endpoints. Replace lines 130-181 with:

```python
MAX_IMAGES = 3


# ── Q&A: Normal ──

@app.post("/api/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    if len(req.images) > MAX_IMAGES:
        raise HTTPException(status_code=422, detail=f"最多支持 {MAX_IMAGES} 张图片")
    r = _get_rag()
    history = [h.model_dump() for h in req.history] if req.history else None
    images = req.images or None
    result = r.ask(
        question=req.question,
        top_k=req.top_k,
        history=history,
        use_reranker=req.use_reranker,
        images=images,
    )
    return AskResponse(**result)


# ── Q&A: SSE Streaming ──

@app.post("/api/ask/stream")
async def ask_stream(req: AskRequest):
    if len(req.images) > MAX_IMAGES:
        raise HTTPException(status_code=422, detail=f"最多支持 {MAX_IMAGES} 张图片")
    r = _get_rag()
    history = [h.model_dump() for h in req.history] if req.history else None
    images = req.images or None

    def event_stream():
        try:
            for token in r.ask_stream(
                question=req.question,
                top_k=req.top_k,
                history=history,
                use_reranker=req.use_reranker,
                images=images,
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
    if len(req.images) > MAX_IMAGES:
        raise HTTPException(status_code=422, detail=f"最多支持 {MAX_IMAGES} 张图片")
    r = _get_rag()
    history = [h.model_dump() for h in req.history] if req.history else None
    images = req.images or None
    result = r.ask_json(
        question=req.question,
        top_k=req.top_k,
        history=history,
        use_reranker=req.use_reranker,
        images=images,
    )
    return JsonAskResponse(**result)
```

- [ ] **Step 2: Verify app loads with all routes**

Run: `python -c "from main import app; print([r.path for r in app.routes if hasattr(r, 'methods')])"`
Expected: list containing `/api/ask`, `/api/ask/stream`, `/api/ask/json`

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat: add image validation and passthrough to API endpoints"
```

---

### Task 6: Add image upload UI to frontend

**Files:**
- Modify: `static/index.html`

- [ ] **Step 1: Add CSS for image preview area**

After the `.input-bar button:disabled` rule (line 35), add:

```css
  .preview-bar { display: flex; gap: 8px; padding: 8px 24px 0; flex-wrap: wrap; }
  .preview-item { position: relative; width: 64px; height: 64px; }
  .preview-item img { width: 64px; height: 64px; object-fit: cover; border-radius: 6px; border: 1px solid var(--border); }
  .preview-item .remove { position: absolute; top: -4px; right: -4px; background: var(--red); color: #fff; border: none; border-radius: 50%; width: 18px; height: 18px; font-size: 11px; cursor: pointer; display: flex; align-items: center; justify-content: center; }
  .upload-btn { background: var(--surface); color: var(--text); border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; font-size: 18px; cursor: pointer; }
  .upload-btn:hover { border-color: var(--accent); }
```

- [ ] **Step 2: Add upload button and preview bar to HTML**

Replace the input-bar div (lines 59-62) with:

```html
<div id="preview-bar" class="preview-bar"></div>
<div class="input-bar">
  <button class="upload-btn" onclick="document.getElementById('fileInput').click()" title="上传图片">&#128247;</button>
  <input type="file" id="fileInput" accept="image/jpeg,image/png,image/webp" multiple hidden />
  <input id="query" placeholder="输入你的问题…" autofocus />
  <button id="send" onclick="send()">发送</button>
</div>
```

- [ ] **Step 3: Add image handling JavaScript**

After the `let history = [];` line (line 70), add:

```javascript
let pendingImages = [];  // array of {base64: "data:image/...", name: "file.jpg"}

const fileInput = document.getElementById('fileInput');
fileInput.addEventListener('change', e => {
  for (const file of e.target.files) {
    if (pendingImages.length >= 3) { alert('最多上传 3 张图片'); break; }
    if (file.size > 4 * 1024 * 1024) { alert(file.name + ' 超过 4MB 限制'); continue; }
    const reader = new FileReader();
    reader.onload = ev => {
      pendingImages.push({ base64: ev.target.result, name: file.name });
      renderPreviews();
    };
    reader.readAsDataURL(file);
  }
  fileInput.value = '';
});

function renderPreviews() {
  const bar = document.getElementById('preview-bar');
  bar.innerHTML = pendingImages.map((img, i) =>
    '<div class="preview-item"><img src="' + img.base64 + '" /><button class="remove" onclick="removeImage(' + i + ')">×</button></div>'
  ).join('');
}

function removeImage(idx) {
  pendingImages.splice(idx, 1);
  renderPreviews();
}
```

- [ ] **Step 4: Include images in request body**

In the `send()` function, change the `body` construction (around line 101) from:

```javascript
  const body = { question: q, top_k: 3, history, use_reranker: true };
```

to:

```javascript
  const images = pendingImages.map(img => img.base64);
  const body = { question: q, top_k: 3, history, use_reranker: true, images };
  pendingImages = [];
  renderPreviews();
```

- [ ] **Step 5: Verify frontend loads**

Start the server with `python main.py`, open `http://localhost:8000` in a browser, confirm the camera button (📷) appears in the input bar.

- [ ] **Step 6: Commit**

```bash
git add static/index.html
git commit -m "feat: add image upload UI with preview to frontend"
```

---

### Task 7: Final integration commit

**Files:** none

- [ ] **Step 1: Verify full system loads**

Run: `python -c "from main import app; from rag import RAG; r = RAG(); print('System OK, routes:', len([x for x in app.routes if hasattr(x,'methods')]))"`
Expected: `System OK, routes: 9`

- [ ] **Step 2: Start server and test with curl**

```bash
python main.py &
sleep 8
curl -s http://127.0.0.1:8000/api/health
```

Expected: `{"status":"ok","embed_model_loaded":true,"reranker_loaded":false}`

- [ ] **Step 3: Push to GitHub**

```bash
git push origin master
```
