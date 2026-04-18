"""
PDF text extraction and chunking.

Strategy: semantic-boundary-aware sliding window.

1.  Extract text blocks from each page using PyMuPDF (preserves reading order).
2.  Detect section headings and table-like blocks.
3.  Merge blocks into paragraphs separated by blank lines.
4.  Greedily merge paragraphs into chunks up to `chunk_size_tokens`.
5.  On overflow carry forward the last `overlap_tokens` worth of text so context
    is not lost at chunk boundaries.
6.  Each chunk is tagged with its source file, page number, and position.

Chunk size rationale (512 tokens, 64-token overlap):

  mistral-embed accepts up to 8 192 tokens, so chunk size is not a model
  constraint.  The choice of 512 tokens is based on RAG retrieval research:

  - Factoid queries retrieve best at 256-512 tokens; analytical queries benefit
    from 1 024+ (Rethinking Chunk Size for Long-Document Retrieval, 2025).
  - The general recommendation in the literature is to start at 512 tokens with
    10-20 % overlap, then measure on a representative query set (Unstructured
    best-practices guide; NVIDIA benchmarks 2024).
  - Smaller chunks (< 128 tokens) fragment context and scored only 54 % accuracy
    in FloTorch 2026 benchmarks.  Larger chunks (> 1 024) dilute the relevance
    signal by mixing multiple topics into one embedding vector.
  - 512 tokens is therefore a balanced starting default for a mixed-topic
    knowledge base (textbooks covering biology, philosophy, economics, etc.).

  Overlap of 64 tokens (12.5 %) ensures sentences spanning chunk boundaries
  appear intact in at least one chunk.  10-15 % is the empirical range most
  often cited; higher overlap nearly doubles storage without proportional gain.

Token approximation:
  We approximate 1 token ≈ 4 characters (a well-known heuristic documented by
  OpenAI and observed across GPT / LLaMA / PaLM tokenizers for English text).
  This avoids adding a tokeniser dependency for chunking decisions; the
  embedding model handles exact tokenisation internally.
"""

from __future__ import annotations

import re
import unicodedata
import uuid
from dataclasses import dataclass, field

import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ChunkData:
    chunk_id: str
    doc_id: str
    chunk_index: int
    page_number: int   # 1-based
    text: str
    token_count: int   # approximate


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(\d+(\.\d+)*\.?\s|#{1,4}\s|[A-Z][A-Z\s]{3,}$)")
_TABLE_RE = re.compile(r"\t.+\t|\s{4,}\S+\s{4,}")  # tab or wide-space alignment


class PDFChunker:
    def __init__(self, chunk_size_tokens: int = 512, overlap_tokens: int = 64) -> None:
        self._chunk_chars = chunk_size_tokens * 4   # ~4 chars per token
        self._overlap_chars = overlap_tokens * 4

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_pdf(self, pdf_bytes: bytes, doc_id: str) -> list[ChunkData]:
        """Extract text from *pdf_bytes* and return a list of ChunkData."""
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_count = doc.page_count
        paragraphs: list[tuple[str, int]] = []  # (text, page_number)

        for page_idx in range(page_count):
            page = doc[page_idx]
            page_paragraphs = self._extract_paragraphs(page, page_idx + 1)
            paragraphs.extend(page_paragraphs)

        doc.close()
        return self._build_chunks(paragraphs, doc_id)

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_paragraphs(self, page: fitz.Page, page_num: int) -> list[tuple[str, int]]:
        """Return a list of (paragraph_text, page_num) from one PDF page."""
        blocks = page.get_text("blocks")  # list of (x0,y0,x1,y1,text,block_no,block_type)
        paragraphs: list[tuple[str, int]] = []
        current_lines: list[str] = []

        for block in sorted(blocks, key=lambda b: (b[1], b[0])):  # sort top→bottom, left→right
            block_type = block[6]
            if block_type != 0:  # skip image blocks
                continue
            raw = block[4].strip()
            if not raw:
                continue
            raw = self._clean_text(raw)

            # Tables: emit as atomic chunk with prefix
            if _TABLE_RE.search(raw):
                if current_lines:
                    paragraphs.append((" ".join(current_lines), page_num))
                    current_lines = []
                paragraphs.append(("[TABLE] " + raw, page_num))
                continue

            # Heading: start a new paragraph
            if _HEADING_RE.match(raw) and current_lines:
                paragraphs.append((" ".join(current_lines), page_num))
                current_lines = [raw]
                continue

            current_lines.append(raw)

        if current_lines:
            paragraphs.append((" ".join(current_lines), page_num))

        return paragraphs

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize unicode, fix soft hyphens, collapse whitespace."""
        text = unicodedata.normalize("NFKC", text)
        text = text.replace("\u00ad", "")   # soft hyphen
        text = text.replace("\u2019", "'")  # right single quote
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def _build_chunks(
        self,
        paragraphs: list[tuple[str, int]],
        doc_id: str,
    ) -> list[ChunkData]:
        chunks: list[ChunkData] = []
        current_text = ""
        current_page = 1
        carry_over = ""   # overlap text carried from previous chunk

        for text, page_num in paragraphs:
            if not current_text:
                current_page = page_num
                current_text = carry_over + (" " if carry_over else "") + text
                continue

            candidate = current_text + "\n\n" + text
            if self._token_estimate(candidate) <= self._chunk_chars:
                current_text = candidate
            else:
                # Emit current chunk
                if current_text.strip():
                    chunks.append(self._make_chunk(current_text.strip(), doc_id, len(chunks), current_page))
                    carry_over = current_text[-self._overlap_chars:].strip()

                current_page = page_num
                current_text = carry_over + (" " if carry_over else "") + text

        # Flush remaining
        if current_text.strip():
            chunks.append(self._make_chunk(current_text.strip(), doc_id, len(chunks), current_page))

        return chunks

    @staticmethod
    def _make_chunk(text: str, doc_id: str, index: int, page_num: int) -> ChunkData:
        return ChunkData(
            chunk_id=str(uuid.uuid4()),
            doc_id=doc_id,
            chunk_index=index,
            page_number=page_num,
            text=text,
            token_count=len(text) // 4,
        )

    @staticmethod
    def _token_estimate(text: str) -> int:
        """Approximate token count: 1 token ≈ 4 characters."""
        return len(text)
