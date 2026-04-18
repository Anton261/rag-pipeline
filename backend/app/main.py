"""FastAPI application factory with lifespan startup/shutdown."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mistralai import Mistral
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.api.documents import router as documents_router
from app.api.ingest import router as ingest_router
from app.api.query import router as query_router
from app.config import settings
from app.core.bm25 import BM25
from app.core.embedder import MistralEmbedder
from app.core.generator import Generator
from app.core.hallucination import HallucinationFilter
from app.core.intent_detector import IntentDetector
from app.core.query_transformer import QueryTransformer
from app.core.reranker import Reranker
from app.core.retriever import HybridRetriever
from app.core.safety import SafetyChecker
from app.core.vector_store import VectorStore
from app.db.repository import Repository

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan (startup + shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up RAG pipeline…")

    # Ensure data directories exist
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Database
    repo = Repository(settings.db_path)
    await repo.init_db()
    app.state.repo = repo

    # Mistral client (sync; wrapped with asyncio.to_thread where needed)
    mistral_client = Mistral(api_key=settings.mistral_api_key)

    # Core components
    embedder = MistralEmbedder(mistral_client, settings.mistral_embed_model)
    app.state.embedder = embedder

    # Vector store — load persisted embeddings from disk
    vector_store = VectorStore(settings.embeddings_dir)
    all_chunks = await repo.get_all_chunks()
    await vector_store.load(all_chunks)
    logger.info("Vector store loaded: %d vectors", len(vector_store))
    app.state.vector_store = vector_store

    # BM25 index — load from disk if available, else rebuild from DB
    bm25 = BM25()
    if settings.bm25_index_path.exists():
        bm25 = BM25.load(settings.bm25_index_path)
        logger.info("BM25 index loaded: %d documents", len(bm25))
    else:
        bm25.fit([(c.id, c.text) for c in all_chunks])
        logger.info("BM25 index built: %d documents", len(bm25))
    app.state.bm25 = bm25

    # Pipeline components
    intent_detector = IntentDetector(mistral_client, settings.mistral_fast_model)
    query_transformer = QueryTransformer(mistral_client, settings.mistral_fast_model)
    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25=bm25,
        embedder=embedder,
        repo=repo,
        top_k=settings.top_k,
        rrf_k=settings.rrf_k,
    )
    reranker = Reranker(mistral_client, settings.mistral_fast_model)
    generator = Generator(
        mistral_client,
        chat_model=settings.mistral_chat_model,
        similarity_threshold=settings.similarity_threshold,
    )
    hallucination_filter = HallucinationFilter(mistral_client, settings.mistral_fast_model)
    safety_checker = SafetyChecker(
        max_query_length=settings.max_query_length,
        max_file_size_mb=settings.max_file_size_mb,
    )

    app.state.intent_detector = intent_detector
    app.state.query_transformer = query_transformer
    app.state.retriever = retriever
    app.state.reranker = reranker
    app.state.generator = generator
    app.state.hallucination_filter = hallucination_filter
    app.state.safety_checker = safety_checker

    logger.info("RAG pipeline ready.")
    yield

    # Shutdown: persist BM25 index
    logger.info("Shutting down — saving BM25 index…")
    bm25.save(settings.bm25_index_path)
    logger.info("Shutdown complete.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit])

app = FastAPI(
    title="RAG Pipeline API",
    description="Retrieval-Augmented Generation over PDF documents using Mistral AI.",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Streamlit on :8501 + local dev
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router, prefix="/api", tags=["Ingestion"])
app.include_router(query_router, prefix="/api", tags=["Query"])
app.include_router(documents_router, prefix="/api", tags=["Documents"])


@app.get("/health", tags=["Health"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
