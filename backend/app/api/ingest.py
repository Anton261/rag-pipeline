"""POST /api/ingest — upload and process one or more PDF files."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from app.core.chunker import PDFChunker
from app.core.embedder import MistralEmbedder
from app.core.safety import SafetyChecker
from app.core.vector_store import VectorStore
from app.core.bm25 import BM25
from app.db.repository import ChunkRecord, DocumentRecord, Repository
from app.dependencies import (
    get_bm25,
    get_embedder,
    get_repo,
    get_safety_checker,
    get_vector_store,
)
from app.models.responses import IngestedDocument, IngestResponse

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest_files(
    files: list[UploadFile],
    repo: Repository = Depends(get_repo),
    vector_store: VectorStore = Depends(get_vector_store),
    bm25: BM25 = Depends(get_bm25),
    embedder: MistralEmbedder = Depends(get_embedder),
    safety: SafetyChecker = Depends(get_safety_checker),
) -> IngestResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    chunker = PDFChunker()
    ingested: list[IngestedDocument] = []
    total_chunks = 0

    for upload in files:
        filename = upload.filename or "unknown.pdf"
        content_type = upload.content_type or "application/octet-stream"
        pdf_bytes = await upload.read()

        # Safety / validation
        safety_result = safety.check_upload(
            filename=filename,
            content_type=content_type,
            size_bytes=len(pdf_bytes),
        )
        if not safety_result.is_safe:
            raise HTTPException(status_code=400, detail=f"{filename}: {safety_result.refusal_reason}")

        # Skip duplicates
        if await repo.document_exists(filename):
            continue

        doc_id = str(uuid.uuid4())

        # Chunk the PDF
        chunks = chunker.chunk_pdf(pdf_bytes, doc_id)
        if not chunks:
            continue

        page_count = max(c.page_number for c in chunks)

        # Embed all chunks (batched)
        texts = [c.text for c in chunks]
        embeddings = await embedder.embed_texts(texts)

        # Store embeddings in vector store and on disk
        chunk_records: list[ChunkRecord] = []
        for chunk_data, embedding in zip(chunks, embeddings):
            vector_store.add(chunk_data.chunk_id, embedding)
            bm25.update(chunk_data.chunk_id, chunk_data.text)
            chunk_records.append(
                ChunkRecord(
                    id=chunk_data.chunk_id,
                    doc_id=doc_id,
                    chunk_index=chunk_data.chunk_index,
                    page_number=chunk_data.page_number,
                    text=chunk_data.text,
                    token_count=chunk_data.token_count,
                    npy_path=vector_store.npy_path_for(chunk_data.chunk_id),
                )
            )

        # Persist to database
        doc_record = DocumentRecord(
            id=doc_id,
            filename=filename,
            page_count=page_count,
            chunk_count=len(chunks),
            ingested_at=datetime.now(timezone.utc).isoformat(),
        )
        await repo.insert_document(doc_record)
        await repo.insert_chunks(chunk_records)

        ingested.append(
            IngestedDocument(
                doc_id=doc_id,
                filename=filename,
                chunk_count=len(chunks),
                page_count=page_count,
            )
        )
        total_chunks += len(chunks)

    return IngestResponse(documents=ingested, total_chunks_added=total_chunks)
