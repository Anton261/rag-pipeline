"""
Load a subset of local OpenStax PDFs into the running RAG pipeline.

Usage:
    python scripts/load_sample_pdfs.py

Adjust OPENSTAX_DIR and BOOKS_TO_LOAD below as needed.
The backend must be running at BACKEND_URL (default: http://localhost:8000).
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration — edit these paths / book selection as needed
# ---------------------------------------------------------------------------

OPENSTAX_DIR = Path(
    r"C:\Users\Antonio Santamaria\MIT Dropbox\Antonio Escobar"
    r"\Antonio Santamaría\MIT\RA\openstax"
)

# Select which books to load for the demo
BOOKS_TO_LOAD = [
    "Biology.pdf",
    "Economics.pdf",
    "Philosophy.pdf",
    "Computer Science.pdf",
]

BACKEND_URL = "http://localhost:8000"
API_INGEST = f"{BACKEND_URL}/api/ingest"

# ---------------------------------------------------------------------------


def main() -> None:
    if not OPENSTAX_DIR.exists():
        print(f"ERROR: OpenStax directory not found: {OPENSTAX_DIR}")
        sys.exit(1)

    files_to_send: list[Path] = []
    for book in BOOKS_TO_LOAD:
        path = OPENSTAX_DIR / book
        if path.exists():
            files_to_send.append(path)
        else:
            print(f"  WARNING: {book} not found, skipping")

    if not files_to_send:
        print("No files found to ingest.")
        sys.exit(1)

    print(f"Ingesting {len(files_to_send)} PDFs into {BACKEND_URL}…")
    print("(This may take several minutes due to embedding generation)\n")

    for pdf_path in files_to_send:
        print(f"  Sending {pdf_path.name}…", end=" ", flush=True)
        with pdf_path.open("rb") as f:
            try:
                resp = requests.post(
                    API_INGEST,
                    files=[("files", (pdf_path.name, f, "application/pdf"))],
                    timeout=600,
                )
                resp.raise_for_status()
                data = resp.json()
                docs = data.get("documents", [])
                if docs:
                    chunks = docs[0]["chunk_count"]
                    pages = docs[0]["page_count"]
                    print(f"OK ({chunks} chunks, {pages} pages)")
                else:
                    print("already ingested or empty")
            except requests.HTTPError as e:
                detail = ""
                try:
                    detail = e.response.json().get("detail", "")
                except Exception:
                    pass
                print(f"FAILED: {detail or e}")
            except Exception as e:
                print(f"ERROR: {e}")

    print("\nDone. Visit http://localhost:8501 to start chatting.")


if __name__ == "__main__":
    main()
