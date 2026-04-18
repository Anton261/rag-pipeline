"""GET /api/documents — list all ingested documents."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.db.repository import Repository
from app.dependencies import get_repo
from app.models.responses import DocumentInfo, DocumentsResponse

router = APIRouter()


@router.get("/documents", response_model=DocumentsResponse)
async def list_documents(repo: Repository = Depends(get_repo)) -> DocumentsResponse:
    docs = await repo.list_documents()
    return DocumentsResponse(
        documents=[
            DocumentInfo(
                doc_id=d.id,
                filename=d.filename,
                chunk_count=d.chunk_count,
                page_count=d.page_count,
                ingested_at=d.ingested_at,
            )
            for d in docs
        ]
    )
