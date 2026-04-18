"""
Streamlit chat UI for the RAG pipeline.

Layout:
  Left sidebar  — document list + PDF upload
  Main area     — multi-turn chat with citation expanders and warning badges
"""

from __future__ import annotations

import os

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
API_INGEST = f"{BACKEND_URL}/api/ingest"
API_QUERY = f"{BACKEND_URL}/api/query"
API_DOCS = f"{BACKEND_URL}/api/documents"

st.set_page_config(
    page_title="RAG Pipeline",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages: list[dict] = []

if "documents" not in st.session_state:
    st.session_state.documents: list[dict] = []

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def render_answer(text: str) -> None:
    """Render answer with inline yellow highlights for unverified claims."""
    import re
    if "{{UNVERIFIED}}" not in text:
        st.markdown(text)
        return

    has_table = bool(re.search(r"^\|.+\|", text, re.MULTILINE))

    if has_table:
        # Tables break with HTML — render clean markdown + warnings below
        unverified = re.findall(r"\{\{UNVERIFIED\}\}(.+?)\{\{/UNVERIFIED\}\}", text)
        clean = re.sub(r"\{\{UNVERIFIED\}\}(.+?)\{\{/UNVERIFIED\}\}", r"\1", text)
        st.markdown(clean)
        for sentence in unverified:
            st.warning(f"**Unverified claim** (not found in sources): {sentence}", icon="⚠️")
    else:
        # Prose/lists — inline yellow highlighting
        def _highlight(match):
            return (
                f'<span style="background-color: #fff3cd; padding: 1px 3px; '
                f'border-radius: 3px; border-left: 3px solid #ffc107;">'
                f'⚠️ {match.group(1)}</span>'
            )
        styled = re.sub(
            r"\{\{UNVERIFIED\}\}(.+?)\{\{/UNVERIFIED\}\}",
            _highlight,
            text,
        )
        st.markdown(styled, unsafe_allow_html=True)


def get_cited_indices(answer: str) -> set[int]:
    """Extract [N] citation indices from the answer text."""
    import re
    return {int(m) for m in re.findall(r"\[(\d+)\]", answer)}


def get_relevant_snippet(text: str, answer: str, max_len: int = 300) -> str:
    """Find the most relevant snippet of a chunk by matching capitalized names and key terms from the answer."""
    import re
    if len(text) <= max_len:
        return text

    # Extract capitalized names/terms (e.g. "Philippa Foot", "Trolley") and long words
    proper_nouns = set(re.findall(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b", answer))
    key_words = {w.lower() for w in re.findall(r"\b[a-z]{5,}\b", answer.lower())}

    # First try: find a window containing a proper noun from the answer
    for noun in proper_nouns:
        idx = text.find(noun)
        if idx >= 0:
            start = max(0, idx - 50)
            snippet = text[start:start + max_len]
            prefix = "…" if start > 0 else ""
            suffix = "…" if start + max_len < len(text) else ""
            return f"{prefix}{snippet}{suffix}"

    # Fallback: slide a window and score by keyword overlap
    best_start = 0
    best_score = 0
    for start in range(0, max(1, len(text) - max_len), 40):
        window = text[start:start + max_len].lower()
        score = sum(1 for w in key_words if w in window)
        if score > best_score:
            best_score = score
            best_start = start

    snippet = text[best_start:best_start + max_len]
    prefix = "…" if best_start > 0 else ""
    suffix = "…" if best_start + max_len < len(text) else ""
    return f"{prefix}{snippet}{suffix}"


_source_render_counter = 0

def render_sources(sources: list[dict], answer: str = "") -> None:
    """Render only the sources actually cited in the answer."""
    global _source_render_counter
    _source_render_counter += 1
    render_id = _source_render_counter

    cited = get_cited_indices(answer)

    if cited:
        cited_sources = [(idx, src) for idx, src in enumerate(sources, start=1) if idx in cited]
    else:
        cited_sources = [(idx, src) for idx, src in enumerate(sources, start=1) if src.get("score", 0) > 0.01]

    if not cited_sources:
        return

    with st.expander(f"📎 Sources ({len(cited_sources)})"):
        for i, (orig_idx, src) in enumerate(cited_sources):
            st.markdown(
                f"**[{orig_idx}]** `{src['filename']}` — page {src['page']} "
                f"*(similarity: {src['score']:.3f})*"
            )
            full_text = src.get("text", "")
            snippet = get_relevant_snippet(full_text, answer)
            st.markdown(f"> {snippet}")
            # Toggle to show full chunk text
            if len(full_text) > 320:
                key = f"full_{render_id}_{orig_idx}"
                if st.toggle("Show full passage", key=key, value=False):
                    st.text(full_text)
            if i < len(cited_sources) - 1:
                st.divider()


def fetch_documents() -> list[dict]:
    try:
        resp = requests.get(API_DOCS, timeout=10)
        resp.raise_for_status()
        return resp.json().get("documents", [])
    except Exception:
        return []


def ingest_files(uploaded_files) -> tuple[bool, str]:
    files = [("files", (f.name, f.read(), "application/pdf")) for f in uploaded_files]
    try:
        resp = requests.post(API_INGEST, files=files, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        docs = data.get("documents", [])
        total = data.get("total_chunks_added", 0)
        if not docs:
            return False, "Files were already ingested or contained no extractable text."
        names = ", ".join(d["filename"] for d in docs)
        return True, f"Ingested {len(docs)} file(s): {names} ({total} chunks total)"
    except requests.HTTPError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            pass
        return False, f"Ingestion failed: {detail or str(e)}"
    except Exception as e:
        return False, f"Ingestion error: {e}"


def send_query(question: str) -> dict:
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]
    payload = {
        "question": question,
        "conversation_history": history,
        "top_k": 8,
    }
    resp = requests.post(API_QUERY, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def render_message(msg: dict) -> None:
    """Render a single chat message with optional metadata."""
    role = msg["role"]
    with st.chat_message(role):
        content = msg["content"]
        if role == "assistant" and msg.get("meta"):
            render_answer(content)
        else:
            st.markdown(content)

        if role == "assistant" and msg.get("meta"):
            meta = msg["meta"]

            # Insufficient evidence banner
            if meta.get("insufficient_evidence"):
                st.warning("No sufficiently relevant passages found in the knowledge base.", icon="⚠️")

            # Hallucination warning
            if meta.get("has_hallucination_warning"):
                st.warning(
                    "Some parts of this answer could not be verified against the source documents.",
                    icon="🔍",
                )

            # Intent badge
            intent = meta.get("intent", "")
            if intent:
                st.caption(f"Intent: `{intent}`")

            # Citations (only sources actually cited in the answer)
            render_sources(meta.get("sources", []), content)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📚 RAG Pipeline")
    st.caption("Chat with your PDF knowledge base")
    st.divider()

    # Upload section
    st.subheader("Upload Documents")
    uploaded = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded:
        if st.button("Ingest", type="primary", use_container_width=True):
            with st.spinner("Processing PDFs…"):
                success, message = ingest_files(uploaded)
            if success:
                st.success(message)
                st.session_state.documents = fetch_documents()
            else:
                st.error(message)

    st.divider()

    # Document list
    st.subheader("Ingested Documents")
    if st.button("Refresh", use_container_width=True):
        st.session_state.documents = fetch_documents()

    if not st.session_state.documents:
        st.session_state.documents = fetch_documents()

    if st.session_state.documents:
        for doc in st.session_state.documents:
            st.markdown(
                f"**{doc['filename']}**  \n"
                f"*{doc['chunk_count']} chunks · {doc['page_count']} pages*"
            )
    else:
        st.info("No documents ingested yet.")

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.header("Chat")

# Render existing messages
for msg in st.session_state.messages:
    render_message(msg)

# Chat input
question = st.chat_input("Ask a question about your documents…")

if question:
    # Display user message immediately
    user_msg = {"role": "user", "content": question, "meta": None}
    st.session_state.messages.append(user_msg)

    # Show the user message right away before the spinner
    with st.chat_message("user"):
        st.markdown(question)

    # Call backend — show spinner, save result, then rerun so render_message handles display
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = send_query(question)

                answer = result.get("answer", "")
                meta = {
                    "intent": result.get("intent", ""),
                    "insufficient_evidence": result.get("insufficient_evidence", False),
                    "has_hallucination_warning": result.get("has_hallucination_warning", False),
                    "sources": result.get("sources", []),
                }

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "meta": meta}
                )

            except requests.HTTPError as e:
                detail = ""
                try:
                    detail = e.response.json().get("detail", "")
                except Exception:
                    pass
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"Error: {detail or str(e)}", "meta": None}
                )
            except Exception as e:
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"Error connecting to backend: {e}", "meta": None}
                )

    st.rerun()
