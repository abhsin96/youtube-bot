from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator

import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from langsmith import traceable

from graphs.ingest_graph import (
    IngestState,
    build_graph,
    chunk_node,
    embed_node,
    fetch_metadata_node,
    fetch_transcript_node,
    idempotency_check_node,
    store_node,
)
from src.config import Settings
from src.dependencies import _resolve_key, get_chroma_client, get_settings, limiter
from src.embeddings import get_embeddings
from src.error_envelope import AppError
from src.schemas import IngestRequest, IngestResponse

logger = structlog.get_logger(__name__)

router = APIRouter()


@traceable(name="ingest_video", metadata={"operation": "ingest"})
def _run_ingest_graph(
    initial_state: IngestState,
    *,
    video_id: str,
    embed_model: str,
) -> dict:
    """Run the ingestion graph; expose video_id, embed_model, chunk_count to LangSmith."""
    result = build_graph().invoke(initial_state)
    return {
        "status": result["status"],
        "chunk_count": len(result["chunks"]),
        "video_id": video_id,
        "embed_model": embed_model,
        "error": result["error"],
        "_result": result,
    }


async def _stream_ingest_progress(
    initial_state: IngestState,
    *,
    video_id: str,
) -> AsyncIterator[str]:
    """SSE progress-stream generator for the ingestion pipeline.

    Runs each graph node in the threadpool and emits a progress event before
    each expensive stage, then a final 'done' event on success.  Errors that
    arise after the 200 OK header has been flushed are surfaced as SSE error
    frames rather than HTTP errors.
    """
    t0 = time.perf_counter()
    try:
        state: dict = dict(initial_state)

        result = await run_in_threadpool(idempotency_check_node, state)
        state.update(result)

        if state["status"] == "skipped":
            logger.info(
                "ingest_completed",
                video_id=video_id,
                duration_ms=round((time.perf_counter() - t0) * 1000),
                chunk_count=0,
                cached=True,
                stream=True,
            )
            yield f"data: {json.dumps({'type': 'done', 'status': 'done', 'chunk_count': 0, 'cached': True})}\n\n"
            return

        if state["status"] == "error":
            yield f"data: {json.dumps({'type': 'error', 'code': 'INGEST_FAILED', 'message': state.get('error') or 'Ingestion failed'})}\n\n"
            return

        meta_result = await run_in_threadpool(fetch_metadata_node, state)
        state.update(meta_result)

        yield f"data: {json.dumps({'type': 'progress', 'step': 'fetch_transcript', 'pct': 20})}\n\n"
        result = await run_in_threadpool(fetch_transcript_node, state)
        state.update(result)
        if state["status"] == "error":
            yield f"data: {json.dumps({'type': 'error', 'code': 'INGEST_FAILED', 'message': state.get('error') or 'Transcript fetch failed'})}\n\n"
            return

        yield f"data: {json.dumps({'type': 'progress', 'step': 'chunk', 'pct': 40})}\n\n"
        result = await run_in_threadpool(chunk_node, state)
        state.update(result)
        if state["status"] == "error":
            yield f"data: {json.dumps({'type': 'error', 'code': 'INGEST_FAILED', 'message': state.get('error') or 'Chunking failed'})}\n\n"
            return

        yield f"data: {json.dumps({'type': 'progress', 'step': 'embed', 'pct': 70})}\n\n"
        result = await run_in_threadpool(embed_node, state)
        state.update(result)
        if state["status"] == "error":
            yield f"data: {json.dumps({'type': 'error', 'code': 'INGEST_FAILED', 'message': state.get('error') or 'Embedding failed'})}\n\n"
            return

        yield f"data: {json.dumps({'type': 'progress', 'step': 'store', 'pct': 90})}\n\n"
        result = await run_in_threadpool(store_node, state)
        state.update(result)
        if state["status"] == "error":
            yield f"data: {json.dumps({'type': 'error', 'code': 'INGEST_FAILED', 'message': state.get('error') or 'Store failed'})}\n\n"
            return

        chunk_count = len(state.get("chunks") or [])
        logger.info(
            "ingest_completed",
            video_id=video_id,
            duration_ms=round((time.perf_counter() - t0) * 1000),
            chunk_count=chunk_count,
            cached=False,
            stream=True,
        )
        yield f"data: {json.dumps({'type': 'done', 'status': 'done', 'chunk_count': chunk_count, 'cached': False})}\n\n"

    except Exception:
        logger.exception("stream_ingest_error", video_id=video_id)
        yield f"data: {json.dumps({'type': 'error', 'code': 'INGEST_FAILED', 'message': 'An unexpected error occurred.'})}\n\n"


@router.post("/ingest")
@limiter.limit("10/minute")
async def ingest(
    request: Request,
    body: IngestRequest,
    settings: Settings = Depends(get_settings),
    chroma_client=Depends(get_chroma_client),
):
    key = _resolve_key(settings)
    embeddings = get_embeddings(settings, api_key=key)

    logger.info("ingest_started", video_id=body.video_id, stream=body.stream)
    t0 = time.perf_counter()

    initial_state: IngestState = {
        "video_id": body.video_id,
        "force": body.force,
        "embeddings": embeddings,
        "chroma_client": chroma_client,
        "segments": [],
        "chunks": [],
        "embedded_chunks": [],
        "channel_metadata": {},
        "status": "pending",
        "error": None,
    }

    if body.stream:
        return StreamingResponse(
            _stream_ingest_progress(initial_state, video_id=body.video_id),
            media_type="text/event-stream",
        )

    traced = await run_in_threadpool(
        _run_ingest_graph,
        initial_state,
        video_id=body.video_id,
        embed_model=settings.embed_model,
    )

    if traced["status"] == "error":
        raise AppError(502, "INGEST_FAILED", traced["error"] or "Ingestion failed")

    response = IngestResponse(
        status=traced["status"],
        chunk_count=traced["chunk_count"],
        cached=traced["status"] == "skipped",
    )
    logger.info(
        "ingest_completed",
        video_id=body.video_id,
        duration_ms=round((time.perf_counter() - t0) * 1000),
        chunk_count=response.chunk_count,
        cached=response.cached,
        stream=False,
    )
    return response
