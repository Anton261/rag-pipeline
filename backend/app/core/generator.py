"""
Answer generator with intent-based prompt templates.

Templates:
  FACTUAL_QA      — concise, citation-referenced answer
  LIST_REQUEST    — bullet-point list
  TABLE_REQUEST   — markdown table
  SUMMARY         — 3-5 sentence synthesis
  CALCULATION     — step-by-step reasoning
  CONVERSATIONAL  — friendly chat reply (no retrieval passages)

Multi-turn:
  The last 10 turns of conversation_history are prepended to the Mistral
  messages array so the model can answer follow-up questions coherently.

Insufficient evidence gate:
  If the maximum cosine similarity score across all retrieved chunks is below
  `similarity_threshold`, we return a structured refusal without calling the LLM.

Medical / legal disclaimer:
  If the safety checker flagged the query, a disclaimer is prepended to the answer.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from mistralai import Mistral

from app.core.intent_detector import Intent
from app.core.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

_MAX_HISTORY_TURNS = 10    # keep last N conversation turns
_PASSAGE_MAX_CHARS = 500   # per passage in the prompt

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_BASE = """\
You are a precise document assistant. Your answers must be grounded EXCLUSIVELY \
in the provided passages.

STRICT RULES:
1. Use ONLY information explicitly stated in the passages. Do NOT add dates, names, \
   numbers, or details from your own knowledge — even if you know them to be correct.
2. Always cite sources using [N] notation where N is the passage number.
3. If the answer is not present in the passages, say "I cannot find this in the \
   provided documents."
4. Do NOT paraphrase in ways that introduce new claims. Stick closely to the \
   wording of the passages.
5. Never use phrases like "it is well known", "historically", or "first invented by" \
   unless those exact phrases appear in the passages."""

_TEMPLATES: dict[Intent, str] = {
    Intent.FACTUAL_QA: _SYSTEM_BASE,

    Intent.LIST_REQUEST: _SYSTEM_BASE + """
Format your answer as a bullet list using "- " prefix for each item.
Group related items if logical. Only include items supported by the passages.""",

    Intent.TABLE_REQUEST: _SYSTEM_BASE + """
Format your answer as a Markdown table with a header row.
Only use data that is explicitly present in the passages.""",

    Intent.SUMMARY: _SYSTEM_BASE + """
Write a concise summary of 3-5 sentences covering the key points relevant to the request.
Draw exclusively from the provided passages.""",

    Intent.CALCULATION: _SYSTEM_BASE + """
Extract the relevant numbers or dates from the passages and show your reasoning step by step.
Label each step clearly.""",

    Intent.CONVERSATIONAL: """\
You are a helpful and friendly document assistant.
Respond naturally to the user's message.
CRITICAL RULES:
- NEVER invent, guess, or list specific document names or categories.
- If the user asks what documents are available, use ONLY the document list provided below.
  If no document list is provided, say you don't have that information and suggest they check the sidebar.
- You can mention that you answer questions about uploaded PDF documents.""",

    Intent.IRRELEVANT: """\
You are a helpful document assistant.
Politely decline and explain that you can only answer questions about the ingested documents.""",
}

_MEDICAL_LEGAL_DISCLAIMER = (
    "\n\n---\n"
    "*Disclaimer: This answer is for informational purposes only and does not constitute "
    "medical, legal, or professional advice. Always consult a qualified professional.*"
)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    answer: str
    intent: Intent
    sources: list[RetrievedChunk]
    top_similarity_score: float
    insufficient_evidence: bool


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class Generator:
    def __init__(
        self,
        client: Mistral,
        chat_model: str,
        similarity_threshold: float = 0.35,
    ) -> None:
        self._client = client
        self._model = chat_model
        self._threshold = similarity_threshold

    async def generate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        intent: Intent,
        conversation_history: list[dict[str, str]] | None = None,
        add_disclaimer: bool = False,
    ) -> GenerationResult:
        # Handle intents that don't need retrieval
        if intent in (Intent.CONVERSATIONAL, Intent.IRRELEVANT):
            answer = await self._call_llm(
                system=_TEMPLATES[intent],
                user=query,
                history=conversation_history,
            )
            return GenerationResult(
                answer=answer,
                intent=intent,
                sources=[],
                top_similarity_score=0.0,
                insufficient_evidence=False,
            )

        # Insufficient evidence gate
        top_score = max((c.semantic_score for c in chunks), default=0.0)
        if top_score < self._threshold:
            return GenerationResult(
                answer=(
                    "I cannot provide a reliable answer because the available documents "
                    "do not contain sufficiently relevant information for your question. "
                    "Please try rephrasing or upload more relevant documents."
                ),
                intent=intent,
                sources=[],
                top_similarity_score=top_score,
                insufficient_evidence=True,
            )

        passages_text = self._format_passages(chunks)
        user_message = f"Passages:\n{passages_text}\n\nQuestion: {query}"

        answer = await self._call_llm(
            system=_TEMPLATES.get(intent, _TEMPLATES[Intent.FACTUAL_QA]),
            user=user_message,
            history=conversation_history,
        )

        if add_disclaimer:
            answer += _MEDICAL_LEGAL_DISCLAIMER

        return GenerationResult(
            answer=answer,
            intent=intent,
            sources=chunks,
            top_similarity_score=top_score,
            insufficient_evidence=False,
        )

    async def _call_llm(
        self,
        system: str,
        user: str,
        history: list[dict[str, str]] | None,
    ) -> str:
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]

        if history:
            for msg in history[-(_MAX_HISTORY_TURNS * 2):]:
                messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": user})

        for attempt in range(3):
            try:
                response = await asyncio.to_thread(
                    self._client.chat.complete,
                    model=self._model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=1024,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                if attempt == 2:
                    logger.error("Generation failed after 3 attempts: %s", exc)
                    raise
                wait = 2.0 * (2 ** attempt)
                logger.warning("Generation attempt %d failed, retrying in %.1fs: %s", attempt + 1, wait, exc)
                await asyncio.sleep(wait)

    @staticmethod
    def _format_passages(chunks: list[RetrievedChunk]) -> str:
        lines: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            text = chunk.text[:_PASSAGE_MAX_CHARS].replace("\n", " ")
            lines.append(f"[{i}] ({chunk.filename}, p.{chunk.page_number}): {text}")
        return "\n\n".join(lines)
