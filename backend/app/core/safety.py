"""
Safety checker — pure Python, no LLM calls (must be fast).

Checks:
  1. PII regex patterns (SSN, credit card, email, phone, IP address)
  2. Legal / medical sensitive keywords → allow with disclaimer flag
  3. Query length limit
  4. Prompt injection patterns
  5. File upload validation (extension, MIME type, size)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# PII patterns
# ---------------------------------------------------------------------------

_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "ssn":         re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "email":       re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    "phone_us":    re.compile(r"\b(?:\+1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b"),
    "ip_address":  re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}

# ---------------------------------------------------------------------------
# Legal / medical keywords
# ---------------------------------------------------------------------------

_LEGAL_MEDICAL_KEYWORDS: frozenset[str] = frozenset({
    "diagnose", "diagnosis", "diagnoses", "treatment", "prescription",
    "dosage", "drug interaction", "symptom", "symptoms", "disease",
    "medical advice", "clinical", "therapy", "therapies",
    "legal advice", "sue", "lawsuit", "litigation", "liability",
    "contract law", "criminal", "attorney", "solicitor", "barrister",
    "malpractice", "negligence", "indemnity",
})

# ---------------------------------------------------------------------------
# Prompt injection patterns
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore (previous|all|above|prior|earlier) (instructions?|prompts?|context)", re.I),
    re.compile(r"(you are|act as|pretend to be|roleplay as)\s+(a\s+)?(?!an? (helpful|assistant))\w+", re.I),
    re.compile(r"(system|human|assistant)\s*:\s*", re.I),
    re.compile(r"<\|im_(start|end)\|>", re.I),
    re.compile(r"\[INST\]|\[/INST\]", re.I),
]

_ALLOWED_MIME_TYPES = {"application/pdf"}
_ALLOWED_EXTENSIONS = {".pdf"}


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class SafetyResult:
    is_safe: bool
    refusal_reason: str | None = None
    pii_types_found: list[str] = field(default_factory=list)
    add_medical_legal_disclaimer: bool = False


# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------

class SafetyChecker:
    def __init__(self, max_query_length: int = 2000, max_file_size_mb: int = 20) -> None:
        self._max_query_length = max_query_length
        self._max_file_size_bytes = max_file_size_mb * 1024 * 1024

    def check_query(self, query: str) -> SafetyResult:
        # Length guard
        if len(query) > self._max_query_length:
            return SafetyResult(
                is_safe=False,
                refusal_reason=f"Query is too long (max {self._max_query_length} characters).",
            )

        # Prompt injection
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(query):
                return SafetyResult(
                    is_safe=False,
                    refusal_reason="The query contains patterns that are not allowed.",
                )

        # PII detection
        pii_found: list[str] = []
        for pii_type, pattern in _PII_PATTERNS.items():
            if pattern.search(query):
                pii_found.append(pii_type)

        if pii_found:
            return SafetyResult(
                is_safe=False,
                refusal_reason=(
                    "Your query appears to contain personal information "
                    f"({', '.join(pii_found)}). Please remove it and try again."
                ),
                pii_types_found=pii_found,
            )

        # Legal / medical keywords — allow but flag for disclaimer
        query_lower = query.lower()
        add_disclaimer = any(kw in query_lower for kw in _LEGAL_MEDICAL_KEYWORDS)

        return SafetyResult(
            is_safe=True,
            add_medical_legal_disclaimer=add_disclaimer,
        )

    def check_upload(
        self,
        filename: str,
        content_type: str,
        size_bytes: int,
    ) -> SafetyResult:
        # Strip path components (prevent directory traversal)
        safe_name = os.path.basename(filename)
        _, ext = os.path.splitext(safe_name)

        if ext.lower() not in _ALLOWED_EXTENSIONS:
            return SafetyResult(
                is_safe=False,
                refusal_reason=f"Only PDF files are accepted (got '{ext}').",
            )

        if content_type not in _ALLOWED_MIME_TYPES:
            return SafetyResult(
                is_safe=False,
                refusal_reason=f"Invalid MIME type '{content_type}'. Only PDFs are accepted.",
            )

        if size_bytes > self._max_file_size_bytes:
            limit_mb = self._max_file_size_bytes // (1024 * 1024)
            return SafetyResult(
                is_safe=False,
                refusal_reason=f"File exceeds {limit_mb} MB size limit.",
            )

        return SafetyResult(is_safe=True)
