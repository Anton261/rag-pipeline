"""POST /api/query — answer a user question using the RAG pipeline."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.core.generator import Generator
from app.core.hallucination import HallucinationFilter
from app.core.intent_detector import Intent, IntentDetector
from app.core.query_transformer import QueryTransformer
from app.core.reranker import Reranker
from app.core.retriever import HybridRetriever
from app.core.safety import SafetyChecker
from app.db.repository import Repository
from app.dependencies import (
    get_generator,
    get_hallucination_filter,
    get_intent_detector,
    get_query_transformer,
    get_reranker,
    get_repo,
    get_retriever,
    get_safety_checker,
)
from app.models.requests import QueryRequest
from app.models.responses import QueryResponse, SourceChunk

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    repo: Repository = Depends(get_repo),
    safety: SafetyChecker = Depends(get_safety_checker),
    intent_detector: IntentDetector = Depends(get_intent_detector),
    query_transformer: QueryTransformer = Depends(get_query_transformer),
    retriever: HybridRetriever = Depends(get_retriever),
    reranker: Reranker = Depends(get_reranker),
    generator: Generator = Depends(get_generator),
    hallucination_filter: HallucinationFilter = Depends(get_hallucination_filter),
) -> QueryResponse:
    steps: list[str] = []

    # 1. Safety check
    safety_result = safety.check_query(request.question)
    if not safety_result.is_safe:
        raise HTTPException(status_code=400, detail=safety_result.refusal_reason)
    steps.append("safety_passed")

    # 2. Intent detection
    intent = await intent_detector.detect(request.question)
    steps.append(f"intent_detected:{intent.value}")

    # 3. Conversational — skip retrieval but inject document context
    if intent == Intent.CONVERSATIONAL:
        docs = await repo.list_documents()
        doc_context = ""
        if docs:
            doc_list = "\n".join(f"- {d.filename} ({d.chunk_count} chunks, {d.page_count} pages)" for d in docs)
            doc_context = f"\n\nCurrently ingested documents:\n{doc_list}"
        query_with_context = request.question + doc_context
        gen_result = await generator.generate(
            query=query_with_context,
            chunks=[],
            intent=intent,
            conversation_history=[m.model_dump() for m in request.conversation_history],
        )
        steps.append("conversational_reply")
        return _build_response(gen_result.answer, intent, [], False, False, request.question, steps)

    # 4. Irrelevant — polite refusal
    if intent == Intent.IRRELEVANT:
        steps.append("query_refused")
        return _build_response(
            "I can only answer questions about the documents that have been uploaded to the system. "
            "Please ask a question related to the available knowledge base.",
            intent, [], False, False, request.question, steps,
        )

    # 5. Query transformation (HyDE + expansion)
    transformed = await query_transformer.transform(request.question, intent)
    steps.append(f"query_transformed:hyde={len(transformed.hyde_passage)}chars,terms={len(transformed.expanded_terms)}")

    # 6. Hybrid retrieval
    chunks = await retriever.retrieve(transformed, top_k=request.top_k)
    steps.append(f"retrieved:{len(chunks)}_chunks")

    # 7. Reranking
    if chunks:
        chunks = await reranker.rerank(request.question, chunks)
        steps.append("reranked")

    # 8. Generation (with insufficient-evidence gate)
    history = [m.model_dump() for m in request.conversation_history]
    gen_result = await generator.generate(
        query=request.question,
        chunks=chunks,
        intent=intent,
        conversation_history=history,
        add_disclaimer=False,
    )

    if gen_result.insufficient_evidence:
        steps.append("insufficient_evidence")
        return _build_response(
            gen_result.answer, intent, [], True, False,
            transformed.bm25_query, steps,
        )

    steps.append("generated")

    # 9. Hallucination filter
    filtered = await hallucination_filter.filter(gen_result.answer, chunks)
    steps.append(
        f"hallucination_check:{filtered.unsupported_count}/{filtered.total_sentences}_unverified"
    )

    # 10. Append disclaimer AFTER hallucination filter so it doesn't get checked
    answer_text = filtered.text
    if safety_result.add_medical_legal_disclaimer:
        answer_text += (
            "\n\n---\n"
            "*Disclaimer: This answer is for informational purposes only and does not constitute "
            "medical, legal, or professional advice. Always consult a qualified professional.*"
        )

    sources = [
        SourceChunk(
            filename=c.filename,
            page=c.page_number,
            chunk_id=c.chunk_id,
            score=round(c.semantic_score, 4),
            text=c.text[:400],
        )
        for c in gen_result.sources
    ]

    return _build_response(
        answer_text,
        intent,
        sources,
        False,
        filtered.has_hallucination_warning,
        transformed.bm25_query,
        steps,
    )


def _build_response(
    answer: str,
    intent: Intent,
    sources: list[SourceChunk],
    insufficient_evidence: bool,
    has_hallucination_warning: bool,
    query_used: str,
    steps: list[str],
) -> QueryResponse:
    return QueryResponse(
        answer=answer,
        intent=intent.value,
        insufficient_evidence=insufficient_evidence,
        has_hallucination_warning=has_hallucination_warning,
        sources=sources,
        query_used=query_used,
        processing_steps=steps,
    )
