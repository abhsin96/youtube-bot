from __future__ import annotations

import hashlib
import json
import time
from collections.abc import AsyncIterator

import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from langsmith import traceable, uuid7

from graphs.ask_graph import (
    VIDEO_NOT_INGESTED,
    AskState,
    build_ask_graph,
    make_initial_state,
)
from src.chain import answer_question
from src.config import Settings
from src.dependencies import (
    _resolve_key,
    get_ask_graph_cache,
    get_chroma_client,
    get_settings,
    get_thread_store,
    limiter,
)
from src.embeddings import get_embeddings
from src.error_envelope import AppError
from src.schemas import AskResponse, QueryRequest, QuestionResponse
from src.stores.chroma_vector_store import ChromaVectorStore
from src.stores.thread_store import AnyThreadStore, _resolve_history
from src.stores.vector_store_abc import VectorStorePort

logger = structlog.get_logger(__name__)

router = APIRouter()


@traceable(name="ask_question", metadata={"operation": "ask"})
def _run_ask_graph(graph, state: AskState, *, video_id: str, thread_id: str | None = None) -> dict:
    """Invoke the ask graph; exposes video_id to LangSmith as a run tag."""
    config = {"run_name": "ask_question"}
    if thread_id:
        config["metadata"] = {"thread_id": thread_id}
    return graph.invoke(state, config=config)


@traceable(name="answer_question", metadata={"operation": "chat"})
def _run_answer_question(*args, **kwargs):
    """Bridge for running answer_question in a threadpool with tracing."""
    return answer_question(*args, **kwargs)


async def _stream_graph_answer(
    graph,
    state: AskState,
    thread_id: str,
    video_id: str,
    *,
    thread_store: AnyThreadStore | None = None,
    question: str = "",
    max_turns: int = 10,
) -> AsyncIterator[str]:
    """SSE token-stream generator using graph.astream_events().

    Callers must validate that the collection exists and raise HTTP errors
    *before* iterating this generator — any error raised here arrives after
    the 200 OK header has already been flushed, so it becomes an SSE error
    frame rather than an HTTP error response.

    When *thread_store* is provided, the (question, answer) pair is appended to
    the store before the ``done`` SSE frame is emitted so that the next request
    on the same thread immediately sees the updated history.
    """
    t0 = time.perf_counter()
    final_output = None
    try:
        async for event in graph.astream_events(state, version="v2"):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                node = event.get("metadata", {}).get("langgraph_node", "")
                if node == "generate":
                    chunk = event["data"].get("chunk")
                    token = chunk.content if chunk and hasattr(chunk, "content") else ""
                    if token:
                        yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            elif kind == "on_chain_end" and event.get("name") == "LangGraph":
                final_output = event["data"].get("output")

        if final_output is None:
            yield f"data: {json.dumps({'type': 'error', 'code': 'INTERNAL_ERROR', 'message': 'Graph produced no output.'})}\n\n"
            return

        if final_output.get("error"):
            yield f"data: {json.dumps({'type': 'error', 'code': 'INTERNAL_ERROR', 'message': final_output['error']})}\n\n"
            return

        answer = final_output.get("answer", "")
        if thread_store is not None and answer and question:
            thread_store.append(thread_id, question, answer, max_turns=max_turns)

        tokens_used = final_output.get("tokens_used")
        logger.info(
            "query_completed",
            video_id=video_id,
            question=question[:80],
            duration_ms=round((time.perf_counter() - t0) * 1000),
            tokens_used=tokens_used,
            stream=True,
        )
        yield f"data: {json.dumps({'type': 'done', 'answer': answer, 'citations': final_output.get('citations', []), 'tokens_used': tokens_used, 'refused': final_output.get('refused', False), 'thread_id': thread_id})}\n\n"
    except Exception:
        logger.exception("stream_graph_error", video_id=video_id)
        yield f"data: {json.dumps({'type': 'error', 'code': 'INTERNAL_ERROR', 'message': 'An unexpected error occurred.'})}\n\n"


def _get_or_rebuild_ask_graph(
    settings: Settings,
    key: str,
    emb,
    *,
    cache: dict,
    chroma_client,
):
    """Return the cached compiled ask graph, rebuilding when the API key changes.

    The graph is compiled once per unique API key and stored in the cache dict
    (which lives in app.state) so the expensive LangGraph compilation only runs
    on the first request or after a key rotation.  The raw key is never stored
    — only a short hex digest used for change detection.
    """
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    if cache["key_hash"] != key_hash:
        logger.info(
            "building ask graph",
            reason="api_key_changed" if cache["key_hash"] else "first_build",
        )
        cache["graph"] = build_ask_graph(
            settings, emb, api_key=key, vector_store=ChromaVectorStore(chroma_client)
        )
        cache["key_hash"] = key_hash
    return cache["graph"]


async def _handle_simple_query(
    body: QueryRequest,
    settings: Settings,
    key: str,
    emb,
    thread_id: str,
    vector_store: VectorStorePort,
) -> QuestionResponse:
    try:
        result = await run_in_threadpool(
            _run_answer_question,
            video_id=body.video_id,
            question=body.question,
            embeddings=emb,
            vector_store=vector_store,
            chat_model=settings.chat_model,
            openai_api_key=key,
            k=body.k,
            score_threshold=settings.min_similarity_threshold,
            context_budget_tokens=settings.context_budget_tokens,
            openai_api_base=settings.openai_api_base,
        )
    except Exception as exc:
        logger.exception("query_simple_error", video_id=body.video_id, thread_id=thread_id)
        raise AppError(500, "INTERNAL_ERROR", "An unexpected error occurred.") from exc

    return QuestionResponse(
        answer=result.answer,
        sources=[
            {
                "chunk_id": doc.metadata.get("chunk_id"),
                "start_ts": doc.metadata.get("start_ts"),
                "end_ts": doc.metadata.get("end_ts"),
                "text": doc.page_content,
            }
            for doc in result.sources
        ],
    )


async def _handle_advanced_query(
    body: QueryRequest,
    settings: Settings,
    key: str,
    emb,
    thread_id: str,
    thread_store: AnyThreadStore,
    chroma_client,
    ask_graph_cache: dict,
) -> AskResponse:
    graph = _get_or_rebuild_ask_graph(
        settings, key, emb, cache=ask_graph_cache, chroma_client=chroma_client
    )

    state = make_initial_state(
        video_id=body.video_id,
        question=body.question,
        history=_resolve_history(thread_store, thread_id, body, settings),
        k=body.k,
    )

    @traceable(name="Chat Bot", metadata={"thread_id": thread_id, "endpoint": "/query"})
    def run_ask_with_thread():
        return _run_ask_graph(graph, state, video_id=body.video_id, thread_id=thread_id)

    result = await run_in_threadpool(run_ask_with_thread)

    if result.get("error") == VIDEO_NOT_INGESTED:
        raise AppError(404, "VIDEO_NOT_INGESTED", "Video has not been ingested yet")
    if result.get("error"):
        raise AppError(500, "INTERNAL_ERROR", result["error"])

    thread_store.append(
        thread_id,
        body.question,
        result["answer"],
        max_turns=settings.max_history_turns,
    )

    return AskResponse(
        answer=result["answer"],
        citations=result["citations"],
        tokens_used=result.get("tokens_used"),
        refused=result.get("refused", False),
        thread_id=thread_id,
    )


async def _handle_streaming_query(
    body: QueryRequest,
    settings: Settings,
    key: str,
    emb,
    thread_id: str,
    thread_store: AnyThreadStore,
    chroma_client,
    ask_graph_cache: dict,
) -> StreamingResponse:
    vector_store = ChromaVectorStore(chroma_client)
    if not vector_store.collection_exists(body.video_id):
        raise AppError(404, "VIDEO_NOT_INGESTED", "Video has not been ingested yet")

    graph = _get_or_rebuild_ask_graph(
        settings, key, emb, cache=ask_graph_cache, chroma_client=chroma_client
    )
    state = make_initial_state(
        video_id=body.video_id,
        question=body.question,
        history=_resolve_history(thread_store, thread_id, body, settings),
        k=body.k,
    )
    return StreamingResponse(
        _stream_graph_answer(
            graph,
            state,
            thread_id,
            body.video_id,
            thread_store=thread_store,
            question=body.question,
            max_turns=settings.max_history_turns,
        ),
        media_type="text/event-stream",
    )


@router.post("/query")
@limiter.limit("30/minute")
async def query(
    request: Request,
    body: QueryRequest,
    settings: Settings = Depends(get_settings),
    thread_store: AnyThreadStore = Depends(get_thread_store),
    chroma_client=Depends(get_chroma_client),
    ask_graph_cache: dict = Depends(get_ask_graph_cache),
):
    """Unified Q&A endpoint with streaming and advanced mode support.

    Flags:
    - stream: Enable Server-Sent Events streaming (default: False)
    - advanced: Use LangGraph pipeline with guardrails (default: True)
    """
    key = _resolve_key(settings)
    emb = get_embeddings(settings, api_key=key)
    thread_id = body.thread_id or str(uuid7())
    t0 = time.perf_counter()
    logger.info(
        "query_started",
        video_id=body.video_id,
        question=body.question[:80],
        thread_id=thread_id,
        stream=body.stream,
        advanced=body.advanced,
    )

    if body.stream:
        return await _handle_streaming_query(
            body, settings, key, emb, thread_id, thread_store, chroma_client, ask_graph_cache
        )

    if body.advanced:
        result = await _handle_advanced_query(
            body, settings, key, emb, thread_id, thread_store, chroma_client, ask_graph_cache
        )
    else:
        result = await _handle_simple_query(
            body, settings, key, emb, thread_id, ChromaVectorStore(chroma_client)
        )

    logger.info(
        "query_completed",
        video_id=body.video_id,
        question=body.question[:80],
        duration_ms=round((time.perf_counter() - t0) * 1000),
        tokens_used=getattr(result, "tokens_used", None),
    )
    return result


@router.delete("/threads/{thread_id}", status_code=204)
async def delete_thread(
    thread_id: str,
    thread_store: AnyThreadStore = Depends(get_thread_store),
):
    """Clear server-side conversation history for a thread."""
    if not thread_store.delete(thread_id):
        raise AppError(404, "THREAD_NOT_FOUND", f"Thread '{thread_id}' not found")
