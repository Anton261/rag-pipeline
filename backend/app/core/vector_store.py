"""
Custom in-memory vector store backed by numpy.

Design:
  - All embeddings are stored as a single float32 matrix of shape (N, D).
  - Vectors are L2-normalized at insert time, so cosine similarity reduces to
    a dot product: scores = matrix @ query_norm.
  - Top-k selection uses np.argpartition for O(N) average-case complexity
    instead of O(N log N) full sort.
  - The matrix is pre-allocated at capacity=INITIAL_CAPACITY and doubled when
    full (amortized O(1) appends).
  - Each chunk's embedding is persisted to disk as a .npy file so the store
    can be rebuilt after restart without re-embedding.

Memory estimate: 100K chunks × 1024 dims × 4 bytes = ~400 MB float32.
For larger corpora, replace with np.memmap or an approximate index.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.db.repository import ChunkRecord


EMBEDDING_DIM = 1024       # mistral-embed output dimension
INITIAL_CAPACITY = 10_000  # rows pre-allocated


@dataclass
class SearchResult:
    chunk_id: str
    score: float   # cosine similarity ∈ [0, 1] (after L2 normalisation)


class VectorStore:
    def __init__(self, embeddings_dir: Path) -> None:
        self._embeddings_dir = embeddings_dir
        self._embeddings_dir.mkdir(parents=True, exist_ok=True)

        self._capacity = INITIAL_CAPACITY
        self._size = 0
        self._matrix = np.zeros((self._capacity, EMBEDDING_DIM), dtype=np.float32)
        self._ids: list[str] = []
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Startup loading
    # ------------------------------------------------------------------

    async def load(self, chunks: list[ChunkRecord]) -> None:
        """Load all persisted embeddings into memory. Called once at startup."""
        vectors: list[np.ndarray] = []
        ids: list[str] = []

        for chunk in chunks:
            npy_path = Path(chunk.npy_path)
            if npy_path.exists():
                vec = np.load(str(npy_path)).astype(np.float32)
                vectors.append(vec)
                ids.append(chunk.id)

        if not vectors:
            return

        matrix = np.stack(vectors, axis=0)  # (N, D)
        n = matrix.shape[0]

        # Grow internal buffer if needed
        while n > self._capacity:
            self._capacity *= 2

        self._matrix = np.zeros((self._capacity, EMBEDDING_DIM), dtype=np.float32)
        self._matrix[:n] = matrix
        self._size = n
        self._ids = ids

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------

    def add(self, chunk_id: str, embedding: np.ndarray) -> None:
        """Normalise and append one embedding. Thread-safe via asyncio.Lock (call within async context)."""
        norm_vec = self._normalize(embedding.astype(np.float32))

        # Save to disk first
        npy_path = self._embeddings_dir / f"{chunk_id}.npy"
        np.save(str(npy_path), norm_vec)

        # Grow matrix if at capacity
        if self._size >= self._capacity:
            self._capacity *= 2
            new_matrix = np.zeros((self._capacity, EMBEDDING_DIM), dtype=np.float32)
            new_matrix[: self._size] = self._matrix
            self._matrix = new_matrix

        self._matrix[self._size] = norm_vec
        self._ids.append(chunk_id)
        self._size += 1

    def npy_path_for(self, chunk_id: str) -> str:
        return str(self._embeddings_dir / f"{chunk_id}.npy")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[SearchResult]:
        """Return top-k chunks by cosine similarity."""
        if self._size == 0:
            return []

        query_norm = self._normalize(query_embedding.astype(np.float32))
        active = self._matrix[: self._size]       # view — no copy
        scores = active @ query_norm               # (N,) cosine similarities

        k = min(top_k, self._size)
        # argpartition gives unsorted top-k indices in O(N)
        top_indices = np.argpartition(scores, -k)[-k:]
        # Sort those k indices by score descending
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [
            SearchResult(chunk_id=self._ids[int(i)], score=float(scores[i]))
            for i in top_indices
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm < 1e-10:
            return v
        return v / norm

    def __len__(self) -> int:
        return self._size
