"""
LLM-based pointwise reranker.

All top-k chunks are sent in a single Mistral call, asking for a relevance
score 0-10 for each. This is more expensive than RRF alone but improves
precision — the LLM can reason about semantic relevance beyond keyword overlap.

A truncated passage (200 chars) is used in the reranking prompt to control
context length while preserving enough signal for scoring.

Falls back to the original RRF order if the LLM response cannot be parsed.
"""

from __future__ import annotations

import asyncio
import json
import logging

from mistralai import Mistral

from app.core.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

_PASSAGE_PREVIEW_LEN = 200   # chars shown to the reranker per chunk

_SYSTEM_PROMPT = """\
You are a relevance judge for a document retrieval system.
Given a question and a numbered list of text passages, rate the relevance of \
each passage to the question on a scale of 0 (not relevant) to 10 (highly relevant).
Return ONLY valid JSON: {"scores": [<int>, <int>, ...]} — one integer per passage, in order."""


class Reranker:
    def __init__(self, client: Mistral, model: str) -> None:
        self._client = client
        self._model = model

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Return *chunks* sorted by LLM relevance score descending."""
        if not chunks:
            return chunks

        passages = self._format_passages(chunks)
        user_message = f"Question: {query}\n\nPassages:\n{passages}"

        scores = await self._score(user_message, len(chunks))

        if len(scores) != len(chunks):
            logger.warning(
                "Reranker returned %d scores for %d chunks; using original order",
                len(scores),
                len(chunks),
            )
            return chunks

        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in ranked]

    async def _score(self, user_message: str, expected: int) -> list[int]:
        try:
            response = await asyncio.to_thread(
                self._client.chat.complete,
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.0,
                max_tokens=128,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or "{}"
            data = json.loads(raw)
            scores = data.get("scores", [])
            return [int(s) for s in scores[:expected]]
        except Exception as exc:
            logger.warning("Reranking failed: %s", exc)
            return []

    @staticmethod
    def _format_passages(chunks: list[RetrievedChunk]) -> str:
        lines: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            preview = chunk.text[:_PASSAGE_PREVIEW_LEN].replace("\n", " ")
            lines.append(f"[{i}] {preview}")
        return "\n".join(lines)
