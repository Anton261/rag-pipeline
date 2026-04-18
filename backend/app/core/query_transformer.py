"""
Query transformation: HyDE + query expansion.

Two parallel Mistral calls:

1. HyDE (Hypothetical Document Embeddings)
   Instead of embedding the raw query (short, information-sparse), we ask the LLM
   to generate a short hypothetical answer paragraph, then embed that. The embedding
   of a plausible answer is geometrically much closer to real document chunks than
   the query's own embedding. This consistently improves recall for factual questions.

2. Query expansion
   Extract 4 related terms / synonyms / alternative phrasings that might appear in
   source documents. These are appended to the BM25 query string to improve keyword
   recall without changing the semantic search path.

Both calls are fired with asyncio.gather so latency is bounded by the slower one.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass

from mistralai import Mistral

from app.core.intent_detector import Intent

logger = logging.getLogger(__name__)

_HYDE_SYSTEM = """\
You are a knowledgeable assistant. Given a question, write a short passage \
(2-4 sentences) that would directly answer it, as if excerpted from an authoritative \
textbook. Write the passage only — no preamble, no labels."""

_EXPANSION_SYSTEM = """\
You are a search query optimizer. Given a search query, return 4 alternative phrasings, \
synonyms, or related technical terms that might appear in academic source documents. \
Respond with valid JSON only: {"terms": ["term1", "term2", "term3", "term4"]}"""


@dataclass
class TransformedQuery:
    original: str
    hyde_passage: str          # embedded for vector search
    expanded_terms: list[str]  # appended to BM25 query
    bm25_query: str            # original + " " + expanded terms


class QueryTransformer:
    def __init__(self, client: Mistral, model: str) -> None:
        self._client = client
        self._model = model

    async def transform(self, query: str, intent: Intent) -> TransformedQuery:
        """Run HyDE and expansion concurrently and return a TransformedQuery."""
        if intent in (Intent.CONVERSATIONAL, Intent.IRRELEVANT):
            return TransformedQuery(
                original=query,
                hyde_passage=query,
                expanded_terms=[],
                bm25_query=query,
            )

        hyde_task = asyncio.create_task(self._hyde(query))
        expansion_task = asyncio.create_task(self._expand(query))

        hyde_passage, expanded_terms = await asyncio.gather(hyde_task, expansion_task)

        bm25_query = query
        if expanded_terms:
            bm25_query = query + " " + " ".join(expanded_terms)

        return TransformedQuery(
            original=query,
            hyde_passage=hyde_passage,
            expanded_terms=expanded_terms,
            bm25_query=bm25_query,
        )

    async def _hyde(self, query: str) -> str:
        try:
            response = await asyncio.to_thread(
                self._client.chat.complete,
                model=self._model,
                messages=[
                    {"role": "system", "content": _HYDE_SYSTEM},
                    {"role": "user", "content": query},
                ],
                temperature=0.3,
                max_tokens=200,
            )
            return response.choices[0].message.content or query
        except Exception as exc:
            logger.warning("HyDE generation failed: %s", exc)
            return query

    async def _expand(self, query: str) -> list[str]:
        try:
            response = await asyncio.to_thread(
                self._client.chat.complete,
                model=self._model,
                messages=[
                    {"role": "system", "content": _EXPANSION_SYSTEM},
                    {"role": "user", "content": query},
                ],
                temperature=0.2,
                max_tokens=80,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or "{}"
            data = json.loads(raw)
            terms = data.get("terms", [])
            return [str(t) for t in terms[:4]]
        except Exception as exc:
            logger.warning("Query expansion failed: %s", exc)
            return []
