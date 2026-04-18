"""
Custom BM25 implementation — no external search libraries.

Formula (Robertson BM25):
    BM25(q, d) = Σ_{t in q}  IDF(t) * tf_norm(t, d)

    IDF(t)       = log( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )   # always positive
    tf_norm(t,d) = tf(t,d) * (k1 + 1) / (tf(t,d) + k1*(1 - b + b*|d|/avgdl))

Parameters:  k1 = 1.5,  b = 0.75  (standard defaults)

Tokenization:
    lowercase → NFKD normalize → regex word-tokens → remove English stopwords

Data structures:
    _inverted : dict[term, dict[chunk_id, raw_tf]]   — sparse, query-time scoring only
    _doc_lengths : dict[chunk_id, int]               — token count per doc
    _df : dict[term, int]                            — document frequency
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path


# ---------------------------------------------------------------------------
# English stopwords (hardcoded — no external library)
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "as", "is", "was", "are", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "may", "might", "must", "can",
    "could", "not", "no", "nor", "so", "yet", "both", "either", "neither",
    "each", "few", "more", "most", "other", "some", "such", "than", "too",
    "very", "just", "that", "this", "these", "those", "it", "its", "itself",
    "he", "him", "his", "she", "her", "hers", "they", "them", "their",
    "we", "us", "our", "i", "me", "my", "you", "your", "what", "which",
    "who", "whom", "how", "when", "where", "why", "all", "any", "both",
    "here", "there", "up", "down", "out", "off", "over", "under", "again",
    "then", "once", "only", "own", "same", "also", "into", "through",
    "during", "before", "after", "above", "below", "between", "about",
    "against", "while", "because", "until", "although", "however",
})


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

class BM25:
    """Robertson BM25 with incremental index updates."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

        self._inverted: dict[str, dict[str, int]] = {}   # term → {chunk_id → tf}
        self._doc_lengths: dict[str, int] = {}           # chunk_id → length
        self._df: dict[str, int] = {}                    # term → doc count
        self._N: int = 0                                  # total documents
        self._avgdl: float = 0.0                          # average doc length

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def fit(self, documents: list[tuple[str, str]]) -> None:
        """Build index from scratch. *documents* is a list of (chunk_id, text)."""
        self._inverted = {}
        self._doc_lengths = {}
        self._df = {}
        self._N = 0

        for chunk_id, text in documents:
            self._index_document(chunk_id, text)

        self._recompute_avgdl()

    def update(self, chunk_id: str, text: str) -> None:
        """Incrementally add one document to the index."""
        if chunk_id in self._doc_lengths:
            return  # already indexed
        self._index_document(chunk_id, text)
        self._recompute_avgdl()

    def _index_document(self, chunk_id: str, text: str) -> None:
        tokens = self._tokenize(text)
        if not tokens:
            return

        self._doc_lengths[chunk_id] = len(tokens)
        self._N += 1

        tf: dict[str, int] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1

        for term, count in tf.items():
            if term not in self._inverted:
                self._inverted[term] = {}
                self._df[term] = 0
            self._inverted[term][chunk_id] = count
            self._df[term] += 1

    def _recompute_avgdl(self) -> None:
        if self._N == 0:
            self._avgdl = 0.0
        else:
            self._avgdl = sum(self._doc_lengths.values()) / self._N

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Return list of (chunk_id, score) sorted by score descending."""
        if self._N == 0:
            return []

        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        scores: dict[str, float] = {}
        for term in set(query_terms):
            if term not in self._inverted:
                continue
            idf = self._idf(term)
            for chunk_id, raw_tf in self._inverted[term].items():
                dl = self._doc_lengths[chunk_id]
                tf_norm = (raw_tf * (self.k1 + 1)) / (
                    raw_tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                )
                scores[chunk_id] = scores.get(chunk_id, 0.0) + idf * tf_norm

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        return math.log((self._N - df + 0.5) / (df + 0.5) + 1)

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase → NFKD normalize → extract word tokens → remove stopwords."""
        normalized = unicodedata.normalize("NFKD", text.lower())
        tokens = re.findall(r"\b[a-z0-9]+\b", normalized)
        return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        payload = {
            "k1": self.k1,
            "b": self.b,
            "N": self._N,
            "avgdl": self._avgdl,
            "doc_lengths": self._doc_lengths,
            "df": self._df,
            "inverted": self._inverted,
        }
        path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BM25":
        payload = json.loads(path.read_text(encoding="utf-8"))
        instance = cls(k1=payload["k1"], b=payload["b"])
        instance._N = payload["N"]
        instance._avgdl = payload["avgdl"]
        instance._doc_lengths = payload["doc_lengths"]
        instance._df = payload["df"]
        instance._inverted = {
            term: dict(postings)
            for term, postings in payload["inverted"].items()
        }
        return instance

    def __len__(self) -> int:
        return self._N
