"""
Post-generation hallucination filter.

Strategy: sentence-level LLM entailment.

For each sentence in the generated answer we ask the LLM whether it is directly
supported by the retrieved passages. Unsupported sentences are wrapped in an
[UNVERIFIED: ...] tag. If more than 40 % of sentences are unsupported the
response carries a `has_hallucination_warning` flag.

To keep latency acceptable, all sentences are sent in a single batched Mistral
call rather than one call per sentence.

Uses mistral-small (fast + cheap) since this is a binary classification task.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass

from mistralai import Mistral

from app.core.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

_UNVERIFIED_THRESHOLD = 0.4   # fraction of unsupported sentences that triggers the warning
_PASSAGE_MAX_CHARS = 600      # per passage — must be >= generator's 500 chars to see the same text

_SYSTEM_PROMPT = """\
You are a fact-checking assistant. Given a set of source passages and a list of sentences, \
determine whether each sentence is supported by the passages.

Respond ONLY with valid JSON:
{"results": [{"idx": 0, "supported": true}, {"idx": 1, "supported": false}, ...]}

Rules:
- Mark as supported (true) if the sentence is a reasonable paraphrase or summary of \
  information in the passages. Minor rewording is acceptable.
- Mark as unsupported (false) ONLY if the sentence introduces specific facts, dates, \
  names, numbers, or claims that are NOT present in and cannot be inferred from the passages.
- General statements that summarize the passages' content should be marked as supported.
- Numbering starts at 0."""


@dataclass
class FilteredAnswer:
    text: str                        # answer with [UNVERIFIED: ...] tags injected
    unsupported_count: int
    total_sentences: int
    has_hallucination_warning: bool


class HallucinationFilter:
    def __init__(self, client: Mistral, model: str) -> None:
        self._client = client
        self._model = model

    async def filter(
        self,
        answer: str,
        source_chunks: list[RetrievedChunk],
    ) -> FilteredAnswer:
        """Check every sentence of *answer* against *source_chunks*."""
        sentences = self._split_sentences(answer)
        if not sentences or not source_chunks:
            return FilteredAnswer(
                text=answer,
                unsupported_count=0,
                total_sentences=len(sentences),
                has_hallucination_warning=False,
            )

        # Strip [N] citation markers before checking — they confuse the entailment model
        # but still check ALL sentences (citations can be hallucinated too)
        clean_sentences = [re.sub(r"\s*\[\d+\]", "", s).strip() for s in sentences]
        clean_sentences = [s for s in clean_sentences if s]  # drop empty

        passages_text = self._format_passages(source_chunks)
        supported_flags_list = await self._check_sentences(clean_sentences, passages_text)

        annotated_parts: list[str] = []
        unsupported_count = 0
        for i, sentence in enumerate(sentences):
            is_supported = supported_flags_list[i] if i < len(supported_flags_list) else True
            if is_supported:
                annotated_parts.append(sentence)
            else:
                # Use {{UNVERIFIED}} delimiter to avoid conflict with [N] citations
                annotated_parts.append(f"{{{{UNVERIFIED}}}}{sentence}{{{{/UNVERIFIED}}}}")
                unsupported_count += 1

        total = len(sentences)
        has_warning = total > 0 and (unsupported_count / total) > _UNVERIFIED_THRESHOLD

        return FilteredAnswer(
            text=" ".join(annotated_parts),
            unsupported_count=unsupported_count,
            total_sentences=total,
            has_hallucination_warning=has_warning,
        )

    async def _check_sentences(
        self,
        sentences: list[str],
        passages_text: str,
    ) -> list[bool]:
        """Single Mistral call to check all sentences at once."""
        numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences))
        user_message = f"Source passages:\n{passages_text}\n\nSentences to verify:\n{numbered}"

        try:
            response = await asyncio.to_thread(
                self._client.chat.complete,
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.0,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or "{}"
            data = json.loads(raw)
            results = data.get("results", [])
            # Build index map in case results come out of order
            support_map: dict[int, bool] = {
                int(r["idx"]): bool(r["supported"])
                for r in results
                if "idx" in r and "supported" in r
            }
            return [support_map.get(i, True) for i in range(len(sentences))]
        except Exception as exc:
            logger.warning("Hallucination filter failed: %s", exc)
            # On failure, assume all sentences are supported
            return [True] * len(sentences)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences on . ? ! boundaries."""
        raw = re.split(r"(?<=[.?!])\s+", text.strip())
        return [s.strip() for s in raw if s.strip()]

    @staticmethod
    def _format_passages(chunks: list[RetrievedChunk]) -> str:
        lines: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            preview = chunk.text[:_PASSAGE_MAX_CHARS].replace("\n", " ")
            lines.append(f"[{i}] {preview}")
        return "\n".join(lines)
