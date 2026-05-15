# PDF Chat — Agentic RAG over EN/AR PDFs

A 100% open-source agentic RAG application. Ingests English and Arabic PDFs, retrieves with reranking, answers with citations, traces every step, and is evaluated with RAGAs.

The Python package is `rag_exp` (internal); the product is **PDF Chat**.

## Stack and design choices

| Concern | Choice | Why |
| --- | --- | --- |
| PDF text and layout | **Docling** (OCR disabled, pypdf fallback) | Fast layout and table-aware extraction for text-native PDFs. |
| OCR | **PaddleOCR-VL-1.5** served by **vLLM** | 0.9B VL OCR model. Strong on Arabic incl. diacritics and ligatures. GPU-batched, OpenAI-compatible API. Replaced EasyOCR. |
| Chunking | Boilerplate-stripped + semantic (Article/clause) + recursive char fallback, EN+AR | Removes PDF page furniture (headers, page numbers, rule lines) and splits on Article/clause boundaries so each chunk is topically tight. Lifted context precision from a 0.47 ceiling to 0.60. |
| Embeddings | **BGE-M3** via Ollama (1024-dim) | Strong multilingual model, covers EN and AR with a single index. |
| Vector DB | **Qdrant** | One container, server-side BM25 sparse vectors (IDF modifier), payload indexes. |
| Retrieval | **Hybrid (dense + BM25) with RRF fusion** | Catches both semantic and exact-term hits. BM25 implemented as a hash-tokenized sparse vector with `Modifier.IDF`. |
| Re-ranking | **bge-reranker-v2-m3** with task-instruction prompt + score floor | Cross-encoder on top of fused candidates; drops chunks under the configured floor. |
| LLM hosting | **Ollama** (Qwen2.5-7B-Instruct) | Open-weight, strong multilingual chat. |
| RAG / agent framework | **LangChain + LangGraph** | Explicit, debuggable state machine. Reusable node factories. |
| Conversation memory | SQLite via SQLModel | Per-session turn log and running summary. Pluggable via `MemoryBackend`. |
| Citations | Pluggable `citation_builder`; default shape `{index, source, page, score, snippet}` | UI renders as expandable source cards. |
| Prompts | Filesystem `prompts/*.md` synced to **Langfuse** | Filesystem is the source-of-truth and offline default; Langfuse for versioning and live overrides. |
| Observability | **Langfuse** | Auto-trace every LangChain call via the callback handler. Session-scoped. |
| Evaluation | **RAGAs** with Ollama judge LLM | faithfulness, answer relevancy, context precision, context recall. |
| Backend API | **FastAPI** + uvicorn | Pythonic, async, auto OpenAPI docs at `/docs`. Single REST surface for any UI. |
| API contract | REST + OpenAI-compatible shim | Native REST for Streamlit; `/v1/chat/completions` for OpenWebUI/LangChain. |
| Frontend (primary) | **Streamlit** | Pure HTTP client of the API. In-UI PDF upload, citation cards, session UX. |
| Frontend (alternative) | **OpenWebUI** | Talks to the same API via the OpenAI-compatible shim. Demoable side-by-side. |
| Orchestration | docker-compose | Reproducible local stack on a single GPU host. |

## Repo layout

```
rag_exp/
├── docker-compose.yml
├── docker/
│   ├── Dockerfile                   # CUDA base, preloads reranker + docling models
│   └── preload_models.py
├── pyproject.toml
├── prompts/                         # router, query_rewriter, answerer, summarizer,
│                                    # direct_chitchat, direct_clarify
├── src/rag_exp/
│   ├── config/settings.py           # typed env-backed settings
│   ├── ingestion/                   # parser, chunker, embedder, vector_store, paddleocr_vl_client
│   ├── retrieval/                   # vector search + cross-encoder rerank
│   ├── memory/                      # SQLite turn log + summary + session documents
│   ├── observability/               # Langfuse client + prompt registry
│   ├── agent/                       # graph.py (composer), nodes.py (factories), llm.py
│   ├── evaluation/                  # RAGAs harness
│   ├── app/
│   │   ├── api.py                   # FastAPI backend (REST + OpenAI shim)
│   │   └── streamlit_app.py         # Streamlit chatbot UI (HTTP client of api)
│   └── scripts/                     # rag-ingest, rag-evaluate CLIs
├── scripts/seed_prompts.py
├── scripts/fetch_sample_pdfs.py
├── data/pdfs/                       # uploaded / dropped PDFs
└── tests/
```

## Services and ports

| Service | Host port | Purpose |
| --- | --- | --- |
| `api` (FastAPI) | 8000 | REST backend, OpenAPI docs at `/docs` |
| `app` (Streamlit) | 8501 | Chatbot UI (calls `api` over HTTP) |
| `openwebui` | 3001 | Alternative chatbot UI (OpenAI-compatible against `api`) |
| `qdrant` | 6333, 6334 | Vector DB HTTP + gRPC |
| `ollama` | 11435 | LLM and embedding server (compose). Host's own Ollama keeps 11434. |
| `paddleocr-vl` | 8118 | VL OCR via vLLM (OpenAI-compatible at /v1) |
| `langfuse` | 3000 | Trace + prompt UI |
| `postgres` | internal | Langfuse storage |

## API contract

The FastAPI backend lives at `http://localhost:8000` (or `http://api:8000` inside the compose network). OpenAPI/Swagger docs auto-render at `/docs`.

| Method | Path | Purpose | Body / Form | Response |
| --- | --- | --- | --- | --- |
| `GET`  | `/healthz` | Liveness probe | — | `{"status":"ok"}` |
| `GET`  | `/v1/health/services` | Dependency health | — | `{qdrant, ollama, paddleocr_vl, langfuse}` (bools) |
| `POST` | `/v1/sessions` | Create a new conversation session | `{}` | `{"session_id": "<uuid>"}` |
| `POST` | `/v1/chat` | Synchronous chat turn | `{"session_id","question"}` | `{"answer","citations":[…],"route"}` |
| `POST` | `/v1/chat/stream` | Same as above, SSE | same | events: `delta`, `done` |
| `POST` | `/v1/ingest` | Upload + ingest PDFs | multipart: `session_id`, `files[]` | `{"session_id","files":[{name,pages,chunks}]}` |
| `GET`  | `/v1/sessions/{id}/documents` | Session-scoped docs | — | `[{source_path, display_name}]` |
| `GET`  | `/v1/corpus` | All indexed docs | — | `{"documents":[paths]}` |
| `POST` | `/v1/chat/completions` | **OpenAI-compatible** shim used by OpenWebUI | `{model, messages, stream, user}` | OpenAI chat-completion or SSE chunks |
| `GET`  | `/v1/models` | OpenAI-compatible `models` list | — | `{object: "list", data: [{id:"pdf-chat"}]}` |

`citations` shape (default): `{index:int, source:str, page:int, score:float, snippet:str}`.

### Why FastAPI and this contract

- **FastAPI**: async, Pythonic, free OpenAPI docs, native Pydantic validation, fits this stack better than Flask/Starlette-bare.
- **Native REST for our own UI**: explicit endpoints for chat, ingest, session, corpus and health — purpose-built, no protocol overloading.
- **OpenAI-compatible shim**: lets OpenWebUI (and any LangChain / OpenAI SDK client) talk to the agent unchanged. Citations are folded into the assistant message as a markdown footer because the OpenAI spec doesn't have a citations field.
- **Process-isolated frontends**: Streamlit and OpenWebUI both run as separate containers and reach the API by service name. No in-process imports of `rag_exp.agent`.

## Prerequisites

- Linux host with NVIDIA GPU (tested on A100).
- Docker 24+ with the **Compose v2** plugin.
- **nvidia-container-toolkit** installed and configured on the host (`docker info | grep -i runtime` should mention `nvidia`).
- ~40 GB free disk for images and model weights.
- Outbound access to: Docker Hub, HuggingFace Hub, Baidu container registry (`ccr-2vdh3abv-pub.cnc.bj.baidubce.com`).

## One-shot runbook (end-to-end demo)

If you want the assignment evidence (50-page ingest, traces, evaluation) running with no manual UI clicks, this is the path:

```bash
cd ~/rag_exp
cp .env.example .env                                # ships with default Langfuse keys

# 1. Wipe any previous Langfuse state so the init env vars take effect on first boot.
#    SAFE on a fresh demo; SKIP if you already have Langfuse data you want to keep.
sudo docker compose down -v

# 2. Bring up infra and wait for paddleocr-vl to be ready (~5 min on first boot).
sudo docker compose up -d qdrant ollama postgres langfuse paddleocr-vl
sudo docker compose logs -f paddleocr-vl            # ctrl-C when you see "Uvicorn running on http://0.0.0.0:8118"

# 3. Pull LLM + embedding models.
sudo docker compose up ollama-bootstrap

# 4. Build the shared image (used by both api and app), then start everything.
sudo docker compose build api app
sudo docker compose up -d api app openwebui

# 5. Fetch sample EN+AR PDFs (~50+ pages total from public UN documents).
sudo docker compose exec api python /app/scripts/fetch_sample_pdfs.py

# 6. Ingest them via the API (multipart upload also works).
sudo docker compose exec api rag-ingest

# 7. Seed prompts into Langfuse (uses keys baked into .env).
sudo docker compose exec api python /app/scripts/seed_prompts.py

# 8. Run RAGAs evaluation against data/eval_cases.jsonl.
sudo docker compose exec api rag-evaluate /app/data/eval_cases.jsonl

# 9. Open any of the three UIs and ask a few questions.
echo "Streamlit:    http://localhost:8501"
echo "OpenWebUI:    http://localhost:3001"
echo "API docs:     http://localhost:8000/docs"
echo "Langfuse:     http://localhost:3000   admin@rag-exp.local / changeme123"
```

After step 9 you have: ≥50 pages ingested, two frontends running against the same REST API, Langfuse traces and managed prompts, RAGAs scores written to `data/eval_results.json`.

## First-time bootstrap

```bash
cd ~/rag_exp
cp .env.example .env                    # Langfuse keys go here after step 4

# 1. Pull infra images (paddleocr-vl image is large; first pull is slow)
sudo docker compose pull qdrant ollama postgres langfuse paddleocr-vl openwebui

# 2. Start infra. paddleocr-vl downloads ~2 GB of model weights on first boot.
sudo docker compose up -d qdrant ollama postgres langfuse paddleocr-vl
sudo docker compose logs -f paddleocr-vl    # wait for: Uvicorn running on http://0.0.0.0:8118
                                             # ctrl-C when you see it

# 3. Pull Ollama models (Qwen2.5-7B + BGE-M3 -- ~6 GB total). Runs and exits.
sudo docker compose up ollama-bootstrap

# 4. Build the shared app image (api + app reuse it).
sudo docker compose build api

# 5. Start the API, both frontends, and verify.
sudo docker compose up -d api app openwebui
curl http://localhost:8000/healthz                        # {"status":"ok"}
curl http://localhost:8000/v1/health/services             # per-dependency health
echo "Streamlit: http://localhost:8501"
echo "OpenWebUI: http://localhost:3001"
echo "API docs:  http://localhost:8000/docs"

# 6. (Optional) Seed prompts into Langfuse so they're editable in the UI:
sudo docker compose exec api python /app/scripts/seed_prompts.py
```

Langfuse default login (created on first boot via the init env vars):
`admin@rag-exp.local` / `changeme123`.

### Smoke tests after bootstrap

```bash
# All services running?
sudo docker compose ps

# API healthy?
curl -s http://localhost:8000/healthz
curl -s http://localhost:8000/v1/health/services | python -m json.tool

# Qdrant
curl -s http://localhost:6333/collections | python -m json.tool

# Ollama (compose) reachable + models present
curl -s http://localhost:11435/api/tags | python -m json.tool

# PaddleOCR-VL reachable
curl -s http://localhost:8118/v1/models | python -m json.tool

# GPU visible inside api container (reranker)
sudo docker compose exec api python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

## Ingesting PDFs

**Option A: upload from a UI (Streamlit or OpenWebUI).**
- Streamlit (http://localhost:8501): sidebar -> **Upload PDFs (EN/AR)** -> **Ingest uploaded**. Posts to `POST /v1/ingest`.
- OpenWebUI (http://localhost:3001): chat-side document upload also routes through the API.

**Option B: REST call.**
```bash
curl -X POST http://localhost:8000/v1/ingest \
  -F "session_id=$(curl -s -X POST http://localhost:8000/v1/sessions | jq -r .session_id)" \
  -F "files=@/path/to/your.pdf"
```

**Option C: bulk corpus via CLI.**
```bash
cp /path/to/your-corpus/*.pdf ~/rag_exp/data/pdfs/
sudo docker compose exec api rag-ingest
```

Idempotent in both routes: chunk IDs are content-hashed (UUID5), so re-uploading or re-running upserts in place.

### Session scoping (which documents a question searches)

All chunks live in **one shared Qdrant collection**, but retrieval is scoped per session:

- Every PDF uploaded **in a session** is registered to that `session_id` (SQLite `SessionDocument`).
- When a session has registered documents, retrieval is **hard-filtered to exactly those documents** (Qdrant `source_path` filter) — every document in the session is searched together, and nothing else.
- When a session has **no** registered documents (fresh session, or chat-only without upload), retrieval **falls back to the entire corpus** — every file ever ingested, including other sessions and CLI bulk loads. This is why "old files also get queried" in a doc-less session.
- **Start a new session to reset scope.** In Streamlit, the **"New session (resets document scope)"** button issues a fresh `session_id` (via `POST /v1/sessions`); subsequent uploads form a new, isolated scope. The sidebar shows exactly which documents the current session searches, and warns when a session is unscoped.

To scope answers to just your documents: start a session, upload your PDFs, then ask. To search everything on purpose: ask without uploading.

## Retrieval pipeline

Each user turn runs four steps inside the agent:

1. **Encode** the query into a dense BGE-M3 vector AND a BM25 sparse vector (hash-tokenized, see [retrieval/bm25.py](src/rag_exp/retrieval/bm25.py)).
2. **Hybrid prefetch in Qdrant** -- two prefetches against the same collection (one per vector field), each returning the top `SPARSE_PREFETCH_K`/`DENSE_PREFETCH_K` candidates. Qdrant fuses them server-side with **Reciprocal Rank Fusion** (`Fusion.RRF`) and returns `TOP_K` unified candidates.
3. **Cross-encoder rerank** with `bge-reranker-v2-m3`, called with a task-instruction prompt so the cross-encoder scores in retrieval mode.
4. **Score floor + reranker-collapse fallback** -- chunks below `RERANK_SCORE_FLOOR` (default 0.10) are dropped. If fewer than `RERANK_MIN_CHUNKS` (default 3) clear the floor, the cross-encoder is deemed unreliable for this query (typical for vague, contentless prompts like *"what is this doc about?"* — a cross-encoder has no signal to rank them and collapses every score toward 0). Instead of surfacing misleading ~0.000 reranker scores, the system **falls back to the upstream hybrid/dense retrieval order and scores**, which *do* carry signal. So vague queries still get a grounded answer with meaningful citation scores. Top-N (default 7) returned.

   The query rewriter also resolves "this doc / this paper / it" to the in-scope document name(s) from the session, turning a contentless query into a content-bearing one before retrieval even runs.

### Tunable hyperparameters

All retrieval, re-ranking, chunking, and OCR-trigger knobs live in [configs/retrieval.toml](configs/retrieval.toml) -- one file, versioned, the source of truth. Edit the file, recreate the `api` container, done.

```toml
[retrieval]
top_k = 30
rerank_top_n = 7
hybrid_enabled = true
dense_prefetch_k = 60
sparse_prefetch_k = 60
rerank_score_floor = 0.10
rerank_min_chunks = 3                       # floor-bypass safety net for vague queries
rerank_instruction = "Given a question, retrieve passages that contain the answer."

[chunking]
chunk_size = 1200
chunk_overlap = 150

[ingestion]
embed_batch = 256
ocr_min_chars_per_page = 40
ocr_max_concurrency = 8
```

**Resolution order**: environment variable (e.g. `TOP_K=15`) wins, then this file, then a built-in default in [settings.py](src/rag_exp/config/settings.py). So one-off experiments stay convenient:

```bash
# Try a tighter score floor for one eval run only:
sudo docker compose exec -e RERANK_SCORE_FLOOR=0.25 api rag-evaluate /app/data/eval_cases.jsonl
```

Point at a different config file by setting `RAG_RETRIEVAL_CONFIG=/path/to/profile.toml`. Useful for A/B comparing tuned profiles.

### Schema migration

The collection schema went from a single anonymous dense vector to a named `{dense, sparse}` pair with an IDF modifier on `sparse`. The `ensure_collection()` function detects old-schema collections, drops them, and recreates with the new schema. Re-ingest is required after upgrading; ingestion is idempotent so just re-run.

## OCR strategy

Each PDF page is parsed in phases:

1. **pypdf** fast text pass; **Docling** runs only when pypdf is sparse (layout/table-aware).
2. **PaddleOCR-VL** per page when text is still under `OCR_MIN_CHARS_PER_PAGE` (default 40). Pages are rasterized via pypdfium2 at 200 DPI **single-threaded** (pypdfium2 is not thread-safe — concurrent access corrupts its global state and breaks later PDFs), then OCR'd **concurrently** over the in-memory PNGs (the HTTP call is the slow, thread-safe part).
3. **Resilient embedding**: degenerate chunks (empty / whitespace / non-text) are dropped before embedding, and a batch that makes Ollama emit `NaN` falls back to per-chunk embedding so one bad page can't skip a whole document.

Text-native PDFs cost no GPU OCR cycles. Scanned PDFs get a state-of-the-art Arabic-capable VL OCR. Force OCR everywhere with `OCR_MIN_CHARS_PER_PAGE=999999`.

Other task prompts available via [paddleocr_vl_client.ocr_page](src/rag_exp/ingestion/paddleocr_vl_client.py): `OCR`, `Table Recognition`, `Formula Recognition`, `Chart Recognition`.

## Evaluation (RAGAs)

```bash
# Edit data/eval_cases.jsonl with {"question": "...", "ground_truth": "..."} per line.
# Use bash -lc (or -it) so rich-formatted output isn't swallowed by non-TTY exec:
sudo docker compose exec api bash -lc "rag-evaluate /app/data/eval_cases.jsonl"
# Scores print and are written to data/eval_results.json.
```

Metrics: faithfulness, answer relevancy, context precision, context recall. Judge LLM and embeddings come from the same Ollama instance.

### Latest measured results

On the bundled 6-case EN/AR UDHR eval set (`data/eval_cases.jsonl`), Qwen 2.5 7B agent + judge, hybrid retrieval, semantic chunking:

| Metric | Score |
| --- | --- |
| faithfulness | ~0.72 |
| answer_relevancy | ~0.84 |
| context_precision | ~0.60 |
| context_recall | ~0.70 |

Notes:
- Scores carry ±0.10 run-to-run jitter on N=6 (RAGAs LLM-judge is non-deterministic on Ollama). Expand the eval set to ~25 cases for stable numbers.
- `context_precision` was pinned at ~0.47 by PDF boilerplate in chunks; the semantic/cleaned chunker broke that ceiling to ~0.60. It's the structural lever for this metric — no retrieval knob (top_k, floor, top_n) moved it.
- Precision ↔ recall trade by design; tune `rerank_top_n` / `rerank_score_floor` in `configs/retrieval.toml` to shift the balance for your corpus.

## Applying changes after edits

`src/`, `scripts/`, `prompts/`, and `data/` are bind-mounted into both the `api` and `app` containers. Most logic edits don't need an image rebuild, but in-process caches mean a `--force-recreate` is needed to clear them.

| You edited | What to run | Why |
| --- | --- | --- |
| `src/rag_exp/app/streamlit_app.py` only | nothing | Streamlit hot-reloads the entry script. |
| `src/rag_exp/app/api.py` (FastAPI routes) | `sudo docker compose up -d --force-recreate api` | uvicorn doesn't hot-reload in this setup. |
| Any agent / ingestion / retrieval / memory module | `sudo docker compose up -d --force-recreate api` | Both Streamlit (chat) and the backend route through `api`. Recreating it picks up the change for everyone. |
| `prompts/*.md` (filesystem prompts) | `sudo docker compose up -d --force-recreate api` | `get_prompt()` is `lru_cache`d; recreate clears it. Bind mount means no rebuild. |
| Langfuse prompt edits in the UI | nothing or `--force-recreate api` | New traces use the new prompt; in-flight `lru_cache` may serve the previous version. |
| `pyproject.toml` (new dep) | `sudo docker compose build --no-cache api && sudo docker compose up -d api app` | Deps install at build time. Both services use the same image. |
| `docker/Dockerfile` | `sudo docker compose build --no-cache api && sudo docker compose up -d api app` | Image must rebuild. |
| `docker-compose.yml` (any service) | `sudo docker compose up -d` | Compose detects changed services and recreates them. |
| New env var in `.env` | `sudo docker compose up -d <service>` | Env is read at container start. `restart` keeps old env; `up -d` recreates. |
| New prompt file (e.g. `direct_escalate.md`) | drop the file, `--force-recreate api` | Bind mount picks it up; recreate clears `get_prompt` cache. |
| New PDF on disk | nothing (use UI upload or `rag-ingest`) | Ingestion is its own pipeline. |
| `src/rag_exp/config/settings.py` | `sudo docker compose up -d --force-recreate api app` | `get_settings()` is `lru_cache`d at module level. |

Rule of thumb: **logic-only edit -> `--force-recreate api`. Dep or Dockerfile edit -> rebuild. Streamlit-only edit -> nothing.**

## Extending the agent (reusable nodes)

Each LangGraph node is built by a factory in [src/rag_exp/agent/nodes.py](src/rag_exp/agent/nodes.py). Override any node via `build_graph(nodes={name: factory(...)})`:

```python
from rag_exp.agent.graph import build_graph, run_agent_on
from rag_exp.agent.nodes import (
    make_router, make_retrieve, make_answer, make_direct_answer,
    make_persist_turn, make_update_summary,
)
from langchain_openai import ChatOpenAI

# Cheap LLM for routing, big LLM for answers, custom retriever, custom citation shape.
graph = build_graph(nodes={
    "route":    make_router(llm_factory=lambda: ChatOpenAI(model="gpt-4o-mini")),
    "retrieve": make_retrieve(
                    retriever=legal_filtered_retriever,
                    citation_builder=lambda chunks: [
                        {"index": i+1, "quote": c.text[:120], "doi": c.source_path,
                         "page_range": f"{c.page_number}"} for i, c in enumerate(chunks)
                    ],
                ),
    "answer":   make_answer(llm_factory=lambda: ChatOpenAI(model="gpt-4o"),
                            prompt_name="legal_answerer"),
    "direct_answer": make_direct_answer(prompt_names={
        "chitchat": "direct_chitchat",
        "clarify":  "direct_clarify",
        "escalate": "direct_escalate",   # drop a prompts/direct_escalate.md
    }),
})

result = run_agent_on(graph, "Indemnity cap?", session_id="...")
```

Override surface:
- **LLM per node** via `llm_factory`.
- **Prompt per node** via `prompt_name` (resolves filesystem first, Langfuse override second).
- **Retriever** via `retriever` callable in `make_retrieve`.
- **Citation shape** via `citation_builder` in `make_retrieve`.
- **Memory backend** via `MemoryBackend(load_summary, save_summary, append_turn, init)` passed to `build_graph(memory=...)`.
- **Routes** via `allowed_routes` in `make_router` and matching `prompt_names` in `make_direct_answer`.
- **State schema** via `build_graph(state_schema=MyTypedDict)`.
- **Memory split** controlled separately: override `persist_turn` and/or `update_summary` independently if you want e.g. summary updates only every Nth turn.

## Observability

With Langfuse keys configured (`.env` populated and `app` recreated):

- Every turn emits a Langfuse trace covering routing decision, query rewrite, vector search, rerank scores, LLM calls (with token counts and latency).
- Traces are session-scoped (`st.session_state.session_id`) so a whole conversation groups together.
- Prompt edits made in Langfuse take effect on the next call (an `app` recreate clears the `get_prompt` lru_cache if needed).

## Tests

```bash
pip install -e .[dev]
pytest -q
```

## GPU layout and portability

GPU pinning is env-driven for portability across hosts:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_GPU` | 0 | Card for Qwen2.5-7B + BGE-M3 (~16 GB VRAM) |
| `PADDLEOCR_GPU` | 0 | Card for PaddleOCR-VL via vLLM (~10 GB VRAM) |
| `APP_GPU` | 0 | Card for the BGE reranker in the API container (~2 GB VRAM) |

Default `0/0/0` means **single-GPU hosts work without any config change** — all three services share GPU 0. On a multi-GPU host, set them to different cards in `.env` to eliminate contention:

```
OLLAMA_GPU=0
PADDLEOCR_GPU=1
APP_GPU=2
```

## Known limitations and next steps

- **Handwriting and complex graphics are not reliably extracted, and this is not detected at ingest.** PaddleOCR-VL is a print/document OCR model. On handwritten classroom notes, dense diagrams, or figure-heavy pages it does not fail cleanly — it returns plausible-looking but inaccurate text. The ingest gate is purely quantitative (`len(text) >= OCR_MIN_CHARS_PER_PAGE`), so a garbled handwritten page passes the threshold, gets embedded, and silently degrades retrieval for that document. There is currently **no OCR-quality/confidence signal** surfaced to the user on load. Mitigations (not yet implemented): a per-page confidence gate combining a VL self-report (`[[UNREADABLE]]` sentinel) with a heuristic textiness score, an `ocr_confidence` payload field, default exclusion of low-confidence pages, and an explicit `rag-ingest` summary line (e.g. "4 pages flagged handwritten/unreadable and excluded"). Truly reading handwriting would require a handwriting-specialised model or a larger VLM — out of scope for this stack. Assumption for this assignment: the corpus is print-quality EN/AR PDFs.
- Per-tenant collection routing. Only one collection today.
- Old chunks for deleted/renamed PDFs are not pruned (re-upload with the same name overwrites in place; rename or delete needs a `client.delete(filter=...)`).
- Baidu's PaddleOCR-VL image is large (~10 GB) and pulled from `ccr-2vdh3abv-pub.cnc.bj.baidubce.com`. The inline alternative in [docker-compose.yml](docker-compose.yml) (stock `vllm/vllm-openai:v0.11.1+`) is a drop-in if that registry is blocked.
- Streaming tokens to the chat UI is single-chunk SSE today. Per-token streaming would need `make_answer` to expose a streaming LLM call.
