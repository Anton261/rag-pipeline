"""
Intent detection via Mistral (mistral-small for cost efficiency).

Intents:
  CONVERSATIONAL  — greetings, thanks, small talk → skip retrieval, friendly reply
  FACTUAL_QA      — direct factual question → standard Q&A template
  LIST_REQUEST    — "list all / what are the..." → bullet list template
  TABLE_REQUEST   — comparative / structured data → markdown table template
  SUMMARY         — "summarize / overview of..." → summary template
  CALCULATION     — numbers, dates, quantities → step-by-step template
  IRRELEVANT      — completely off-topic → polite refusal

LRU cache avoids duplicate API calls for identical queries within a session.
"""

from __future__ import annotations

import asyncio
import json
import logging
from enum import Enum
from functools import lru_cache

from mistralai import Mistral

logger = logging.getLogger(__name__)


class Intent(str, Enum):
    CONVERSATIONAL = "CONVERSATIONAL"
    FACTUAL_QA = "FACTUAL_QA"
    LIST_REQUEST = "LIST_REQUEST"
    TABLE_REQUEST = "TABLE_REQUEST"
    SUMMARY = "SUMMARY"
    CALCULATION = "CALCULATION"
    IRRELEVANT = "IRRELEVANT"


_NEEDS_RETRIEVAL: set[Intent] = {
    Intent.FACTUAL_QA,
    Intent.LIST_REQUEST,
    Intent.TABLE_REQUEST,
    Intent.SUMMARY,
    Intent.CALCULATION,
}

_SYSTEM_PROMPT = """\
You are an intent classifier for a document Q&A assistant.
Classify the user query into exactly ONE of these intents:

  CONVERSATIONAL  - greetings, thanks, small talk, compliments, questions about the assistant itself
  FACTUAL_QA      - a question seeking a specific fact or explanation
  LIST_REQUEST    - requests for a list of items, examples, types, or steps
  TABLE_REQUEST   - requests for a comparison, table, or structured data
  SUMMARY         - requests to summarize, overview, or give a brief of a topic
  CALCULATION     - requests involving numbers, dates, quantities, rates, or computations
  IRRELEVANT      - completely off-topic, harmful, or impossible to answer from documents

Respond with valid JSON only, no other text:
{"intent": "<INTENT>", "confidence": <0.0-1.0>}"""


class IntentDetector:
    def __init__(self, client: Mistral, model: str) -> None:
        self._client = client
        self._model = model

    async def detect(self, query: str) -> Intent:
        """Classify *query* and return an Intent enum member."""
        intent_str = await self._cached_detect(query)
        try:
            return Intent(intent_str)
        except ValueError:
            logger.warning("Unknown intent '%s', defaulting to FACTUAL_QA", intent_str)
            return Intent.FACTUAL_QA

    async def _cached_detect(self, query: str) -> str:
        # LRU cache is sync; run in thread to keep async interface clean
        return await asyncio.to_thread(self._sync_detect, query)

    @lru_cache(maxsize=256)
    def _sync_detect(self, query: str) -> str:
        import time
        for attempt in range(3):
            try:
                response = self._client.chat.complete(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": query},
                    ],
                    temperature=0.0,
                    max_tokens=64,
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content or "{}"
                data = json.loads(raw)
                return data.get("intent", "FACTUAL_QA").upper()
            except Exception as exc:
                if attempt == 2:
                    logger.warning("Intent detection failed after 3 attempts: %s", exc)
                    return "FACTUAL_QA"
                time.sleep(2.0 * (2 ** attempt))
        return "FACTUAL_QA"

    @staticmethod
    def needs_retrieval(intent: Intent) -> bool:
        return intent in _NEEDS_RETRIEVAL
