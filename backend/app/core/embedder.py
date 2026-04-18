"""Mistral embedding wrapper with batching and retry logic."""

from __future__ import annotations

import asyncio
import logging

import numpy as np
from mistralai import Mistral

logger = logging.getLogger(__name__)

_BATCH_SIZE = 32        # Mistral embed batch limit
_MAX_RETRIES = 3
_RETRY_BASE_SECONDS = 2.0


class MistralEmbedder:
    def __init__(self, client: Mistral, model: str) -> None:
        self._client = client
        self._model = model

    async def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts. Returns shape (N, 1024) float32."""
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            embeddings = await self._embed_batch(batch)
            all_embeddings.extend(embeddings)

        return np.array(all_embeddings, dtype=np.float32)

    async def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string. Returns shape (1024,)."""
        result = await self.embed_texts([text])
        return result[0]

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        for attempt in range(_MAX_RETRIES):
            try:
                response = await asyncio.to_thread(
                    self._client.embeddings.create,
                    model=self._model,
                    inputs=texts,
                )
                return [item.embedding for item in response.data]
            except Exception as exc:
                if attempt == _MAX_RETRIES - 1:
                    logger.error("Embedding failed after %d attempts: %s", _MAX_RETRIES, exc)
                    raise
                wait = _RETRY_BASE_SECONDS * (2 ** attempt)
                logger.warning("Embedding attempt %d failed, retrying in %.1fs: %s", attempt + 1, wait, exc)
                await asyncio.sleep(wait)

        raise RuntimeError("Unreachable")  # pragma: no cover
