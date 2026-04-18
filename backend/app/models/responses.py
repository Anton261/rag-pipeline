from pydantic import BaseModel


class IngestedDocument(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    page_count: int


class IngestResponse(BaseModel):
    documents: list[IngestedDocument]
    total_chunks_added: int


class SourceChunk(BaseModel):
    filename: str
    page: int
    chunk_id: str
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    intent: str
    insufficient_evidence: bool
    has_hallucination_warning: bool
    sources: list[SourceChunk]
    query_used: str
    processing_steps: list[str]


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    page_count: int
    ingested_at: str


class DocumentsResponse(BaseModel):
    documents: list[DocumentInfo]
