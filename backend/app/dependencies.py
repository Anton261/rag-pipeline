"""FastAPI dependency factories — all singletons stored on app.state."""

from __future__ import annotations

from fastapi import Request

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


def get_repo(request: Request) -> Repository:
    return request.app.state.repo


def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store


def get_bm25(request: Request) -> BM25:
    return request.app.state.bm25


def get_embedder(request: Request) -> MistralEmbedder:
    return request.app.state.embedder


def get_intent_detector(request: Request) -> IntentDetector:
    return request.app.state.intent_detector


def get_query_transformer(request: Request) -> QueryTransformer:
    return request.app.state.query_transformer


def get_retriever(request: Request) -> HybridRetriever:
    return request.app.state.retriever


def get_reranker(request: Request) -> Reranker:
    return request.app.state.reranker


def get_generator(request: Request) -> Generator:
    return request.app.state.generator


def get_hallucination_filter(request: Request) -> HallucinationFilter:
    return request.app.state.hallucination_filter


def get_safety_checker(request: Request) -> SafetyChecker:
    return request.app.state.safety_checker
