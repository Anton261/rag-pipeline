# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A from-scratch RAG pipeline over PDFs using the Mistral AI API. **No external RAG/search frameworks** (no LangChain, LlamaIndex, FAISS, Chroma, etc.) — vector search, BM25, and hybrid retrieval are all implemented in-repo. Two services: FastAPI backend (`:8000`) and Streamlit UI (`:8501`).

## Commands

### Run (Docker, recommended)

```bash
cp .env.example .env   # then set MISTRAL_API_KEY in .env
docker-compose up --build
```

### Run (local, no Docker)

```bash
pip install -r backend/requirements.txt -r frontend/requirements.txt
cp .env backend/.env   # backend reads .env from its own working directory

# Terminal 1 — backend (must be run from ./backend so data/ paths resolve correctly)
cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000

# Terminal 2 — frontend
cd frontend && streamlit run app.py --server.port 8501
```

The Streamlit frontend reads `BACKEND_URL` from env (defaults to `http://localhost:8000`; docker-compose sets it to `http://backend:8000`).

### Bulk-ingest sample PDFs

`scripts/load_sample_pdfs.py` POSTs PDFs to `/api/ingest`. The `OPENSTAX_DIR` constant at the top of the file is hardcoded to a Windows path — edit it before running.

### Tests / lint

There is no test suite, linter config, or CI in this repo. Don't claim "tests pass" — there are none to run. Validate changes by exercising the API (`http://localhost:8000/docs`) or the Streamlit UI.

## Architecture

### Service topology

- `backend/` — FastAPI app, the entire RAG pipeline lives here
- `frontend/` — single-file Streamlit chat UI (`frontend/app.py`) that talks to the backend over HTTP
- `backend/data/` — persisted state (mounted as a Docker volume); destroying it wipes the knowledge base

### Persistence layout (`backend/data/`)

- `rag.db` — SQLite (via `aiosqlite`); document + chunk metadata, schema in `app/db/schema.sql`
- `embeddings/{chunk_id}.npy` — one float32 numpy file per chunk (1024-dim `mistral-embed` vectors)
- `bm25_index.json` — BM25 inverted index, serialized on shutdown

Embeddings are NOT stored in SQLite. SQLite holds metadata; the `.npy` files are the source of truth for vectors and are stacked into an in-memory matrix at startup.

### Startup lifespan (`backend/app/main.py`)

`lifespan()` is the single source of truth for wiring. On startup it:

1. Creates the SQLite `Repository` and runs schema migrations
2. Constructs a single `mistralai.Mistral` client and shares it across components
3. Builds `VectorStore` and calls `load(all_chunks)` — reads every `.npy` from disk into a single numpy matrix
4. Loads `BM25` from `bm25_index.json` if present, else rebuilds from chunks in SQLite
5. Instantiates every pipeline component (intent detector, query transformer, retriever, reranker, generator, hallucination filter, safety checker) and attaches them to `app.state`
6. On shutdown, persists the BM25 index to disk (the vector store is already disk-backed per chunk)

`app/dependencies.py` exposes `get_*` factories that just read singletons off `request.app.state` — there is no per-request construction of heavy components.

### Query pipeline (the critical path)

The full sequence in `app/api/query.py`, with each stage in `app/core/`:

```
safety.check_query     → regex PII / injection / length gate (no LLM call, ~0.1ms)
intent_detector.detect → mistral-small classifies into 1 of 7 intents (LRU-cached)
   ├─ CONVERSATIONAL / IRRELEVANT → direct reply, no retrieval
   └─ FACTUAL_QA / LIST / TABLE / SUMMARY / CALCULATION:
        query_transformer.transform → HyDE passage + BM25 expansion (asyncio.gather, parallel)
        retriever.retrieve          → vector top-20 + BM25 top-20 → RRF → top-8
        reranker.rerank             → single mistral-small call scores all 8 (0–10)
        generator.generate          → mistral-large; intent picks the prompt template
        hallucination_filter.filter → batched sentence-level entailment vs source passages
```

Two safety gates exist beyond the regex layer:
- **Insufficient-evidence gate** in `Generator`: if `max(semantic_scores) < similarity_threshold` (default 0.80) it short-circuits with a structured refusal *before* calling the chat model.
- **Hallucination filter** runs *after* generation, tags unsupported sentences with `{{UNVERIFIED}}...{{/UNVERIFIED}}` (double braces to avoid clashing with `[N]` citation markers), and sets `has_hallucination_warning` when >40% of sentences are unsupported.

Medical/legal disclaimers are appended **after** the hallucination filter so the disclaimer itself is never checked against passages.

### Ingestion pipeline (`app/api/ingest.py`)

`safety → chunker.chunk_pdf → embedder.embed_texts (batched ×32) → vector_store.add + bm25.update + repo.insert`.

The chunker (`app/core/chunker.py`) does semantic-boundary-aware sliding window over PyMuPDF `get_text("blocks")` output, with special-casing for tables (`[TABLE]` prefix, emitted atomically) and section headings (prepended to the next paragraph). Tokens are approximated as `len(text)/4` — there is no real tokenizer in the chunker.

### Mistral models used

Three model slots, all configured in `app/config.py` and overridable via env:
- `mistral_embed_model` (`mistral-embed`) — embeddings only
- `mistral_fast_model` (`mistral-small-latest`) — intent, query transform, reranking, hallucination check (everything that doesn't need deep reasoning)
- `mistral_chat_model` (`mistral-large-latest`) — final answer generation only

When adding a new pipeline stage, default to `mistral_fast_model` unless the task genuinely needs `mistral-large`.

### Hybrid retrieval & RRF

`HybridRetriever` retrieves 20 candidates from each path and fuses with Reciprocal Rank Fusion (`k=60`, `rrf_k` in config). Don't replace RRF with weighted score combination — cosine and BM25 scores live in incomparable spaces. Custom BM25 (`app/core/bm25.py`) uses Robertson's formula with `k₁=1.5, b=0.75`, hardcoded ~80-word English stopword set, and supports incremental `update()` so ingestion doesn't rebuild the index.

`VectorStore` L2-normalizes at insert time so cosine reduces to a BLAS dot product (`matrix @ query`), uses `np.argpartition` for O(N) top-k, and doubles capacity when full (dynamic-array growth).

## Conventions

- All pipeline components take the shared `Mistral` client by constructor injection — do not instantiate clients inside core modules.
- Mistral SDK calls are synchronous; wrap them in `asyncio.to_thread` when called from async handlers (existing components already do this).
- Pydantic request/response shapes live in `app/models/`; keep API schemas there, don't define ad-hoc dicts in route handlers.
- `app.state.*` is the dependency container. New singletons should be built in `lifespan()` and exposed via a `get_*` in `dependencies.py`.
- `processing_steps` in the query response is a debugging breadcrumb trail — append to it when adding stages.

## Things that look wrong but aren't

- `VectorStore` keeps the entire matrix in RAM. This is intentional (≈400 MB per 100K chunks at 1024-dim float32); don't "fix" it by adding an external vector DB.
- All 8 reranker scores come from a single LLM call, not 8 calls. If parsing fails, the original RRF order is preserved — the reranker is designed to never make results worse.
- The hallucination filter strips `[N]` citation markers before entailment checking. Citations themselves are not trusted — the underlying claim is what gets verified.
- The safety layer is pure regex on purpose. An LLM-based safety check could itself be prompt-injected.
