"""
Hybrid retriever: semantic search + BM25 fused with Reciprocal Rank Fusion (RRF).

Why RRF over weighted score combination:
  BM25 and cosine similarity scores live in incomparable spaces. BM25 scores are
  unbounded and corpus-dependent; cosine scores are in [0, 1] but not linearly
  comparable to BM25. Any fixed weighting requires corpus-specific calibration.
  RRF (Cormack et al., 2009) is parameter-free, robust to score distribution
  differences, and has strong empirical performance in TREC benchmarks.

  RRF_score(d) = Σ_r  1 / (k + rank_r(d))   where k = 60 (standard constant)

  Documents appearing in only one list get rank = len(list) + 1 in the other.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.core.bm25 import BM25
from app.core.embedder import MistralEmbedder
from app.core.query_transformer import TransformedQuery
from app.core.vector_store import VectorStore
from app.db.repository import Repository

logger = logging.getLogger(__name__)

_CANDIDATE_POOL = 20   # retrieve top-20 from each source before fusion


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    filename: str
    page_number: int
    text: str
    semantic_score: float
    bm25_score: float
    rrf_score: float


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        bm25: BM25,
        embedder: MistralEmbedder,
        repo: Repository,
        top_k: int = 8,
        rrf_k: int = 60,
    ) -> None:
        self._vector_store = vector_store
        self._bm25 = bm25
        self._embedder = embedder
        self._repo = repo
        self._top_k = top_k
        self._rrf_k = rrf_k

    async def retrieve(
        self,
        transformed: TransformedQuery,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Run hybrid search and return *top_k* chunks sorted by RRF score."""
        k = top_k or self._top_k

        # Embed the HyDE passage for semantic search
        query_embedding = await self._embedder.embed_query(transformed.hyde_passage)

        # Semantic search
        semantic_results = self._vector_store.search(query_embedding, top_k=_CANDIDATE_POOL)
        semantic_scores = {r.chunk_id: r.score for r in semantic_results}

        # BM25 search on expanded query
        bm25_results = self._bm25.search(transformed.bm25_query, top_k=_CANDIDATE_POOL)
        bm25_scores = {chunk_id: score for chunk_id, score in bm25_results}

        # RRF fusion
        fused = self._rrf_fuse(
            [r.chunk_id for r in semantic_results],
            [chunk_id for chunk_id, _ in bm25_results],
        )

        # Take top_k fused chunk IDs
        top_ids = [chunk_id for chunk_id, _ in fused[:k]]
        rrf_map = {chunk_id: score for chunk_id, score in fused}

        # Fetch chunk text and metadata
        chunk_records = await self._repo.get_chunks_by_ids(top_ids)

        # Build a doc_id → filename map
        doc_ids = list({c.doc_id for c in chunk_records})
        documents = await self._repo.list_documents()
        doc_filename_map = {d.id: d.filename for d in documents}

        results: list[RetrievedChunk] = []
        for chunk in chunk_records:
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    doc_id=chunk.doc_id,
                    filename=doc_filename_map.get(chunk.doc_id, "unknown"),
                    page_number=chunk.page_number,
                    text=chunk.text,
                    semantic_score=semantic_scores.get(chunk.id, 0.0),
                    bm25_score=bm25_scores.get(chunk.id, 0.0),
                    rrf_score=rrf_map.get(chunk.id, 0.0),
                )
            )

        # Sort by RRF score descending
        results.sort(key=lambda c: c.rrf_score, reverse=True)
        return results

    def _rrf_fuse(
        self,
        semantic_ids: list[str],
        bm25_ids: list[str],
    ) -> list[tuple[str, float]]:
        """Reciprocal Rank Fusion of two ranked lists."""
        all_ids: set[str] = set(semantic_ids) | set(bm25_ids)
        sem_rank = {cid: i + 1 for i, cid in enumerate(semantic_ids)}
        bm25_rank = {cid: i + 1 for i, cid in enumerate(bm25_ids)}

        fallback_sem = len(semantic_ids) + 1
        fallback_bm25 = len(bm25_ids) + 1

        scores: dict[str, float] = {}
        for cid in all_ids:
            r_sem = sem_rank.get(cid, fallback_sem)
            r_bm25 = bm25_rank.get(cid, fallback_bm25)
            scores[cid] = 1.0 / (self._rrf_k + r_sem) + 1.0 / (self._rrf_k + r_bm25)

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
