from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

import chromadb
import structlog
import uvicorn
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langsmith import traceable, uuid7
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.exceptions import HTTPException as StarletteHTTPException

from graphs.ask_graph import (
    VIDEO_NOT_INGESTED,
    AskState,
    build_ask_graph,
    make_initial_state,
)
from graphs.ingest_graph import IngestState, build_graph
from src.api_secrets import clear_api_key, get_openai_key, set_api_key
from src.chain import answer_question
from src.config import Settings, load_settings
from src.embeddings import get_embeddings
from src.error_envelope import (
    AppError,
    app_error_handler,
    global_exception_handler,
    http_exception_handler,
    validation_error_handler,
)
from src.logging_config import configure_logging
from src.middleware import RequestIDMiddleware
from src.vector_store import collection_exists, get_chroma_client

logger = structlog.get_logger(__name__)


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
        # keep full result accessible to the caller via a side-channel key
        "_result": result,
    }


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
    thread_store: ThreadStore | None = None,
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

        yield f"data: {json.dumps({'type': 'done', 'answer': answer, 'citations': final_output.get('citations', []), 'tokens_used': final_output.get('tokens_used'), 'refused': final_output.get('refused', False), 'thread_id': thread_id})}\n\n"
    except Exception:
        logger.exception("stream_graph_error", video_id=video_id)
        yield f"data: {json.dumps({'type': 'error', 'code': 'INTERNAL_ERROR', 'message': 'An unexpected error occurred.'})}\n\n"


def _get_or_rebuild_ask_graph(app: FastAPI, settings: Settings, key: str, emb):
    """Return the cached compiled ask graph, rebuilding when the API key changes.

    The graph is compiled once per unique API key and stored in app.state so
    the expensive LangGraph compilation only runs on the first request (or after
    a key rotation via /config/api-key).  The raw key is never stored — only a
    short hex digest used for change detection.
    """
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    cache = app.state.ask_graph_cache
    if cache["key_hash"] != key_hash:
        logger.info(
            "building ask graph",
            reason="api_key_changed" if cache["key_hash"] else "first_build",
        )
        cache["graph"] = build_ask_graph(
            settings, emb, api_key=key, chroma_client=app.state.chroma_client
        )
        cache["key_hash"] = key_hash
    return cache["graph"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    if app.state.chroma_client is None:
        app.state.chroma_client = get_chroma_client(settings.chroma_host, settings.chroma_port)
    logger.info("server starting", project=settings.langsmith_project)
    yield
    logger.info("server stopped")


# --- request / response models ---


class IngestRequest(BaseModel):
    video_id: str = Field(..., min_length=1, description="YouTube video ID")
    force: bool = Field(False, description="Re-ingest even if collection already exists")


class IngestResponse(BaseModel):
    status: str
    chunk_count: int
    cached: bool


class QuestionResponse(BaseModel):
    answer: str
    sources: list[dict]


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"] = Field(..., description="Speaker role")
    content: str = Field(..., min_length=1, description="Message text")


class QueryRequest(BaseModel):
    video_id: str = Field(..., min_length=1, description="YouTube video ID")
    question: str = Field(..., min_length=1, description="Question about the video")
    stream: bool = Field(False, description="Enable streaming response")
    advanced: bool = Field(True, description="Use advanced graph-based pipeline")
    conversation_history: list[ConversationTurn] = Field(
        default_factory=list, description="Prior turn messages"
    )
    k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")
    thread_id: str | None = Field(None, description="Thread ID for conversation tracking")


class CitationSchema(BaseModel):
    chunk_id: str | None = None
    start_ts: float | None = None
    end_ts: float | None = None
    text: str


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationSchema]
    tokens_used: int | None = None
    refused: bool = False
    thread_id: str


class ApiKeyRequest(BaseModel):
    api_key: str = Field(..., min_length=1, description="OpenAI API key")


class ConfigStatusResponse(BaseModel):
    has_key: bool


# --- thread store ---


class ThreadStore:
    """In-memory conversation history store keyed by thread_id.

    Each stored entry is a flat list of alternating HumanMessage / AIMessage
    objects.  The list is capped at *max_turns* turn-pairs (2 × max_turns
    messages) by evicting the oldest pair whenever the cap is exceeded.
    """

    def __init__(self) -> None:
        self._store: dict[str, list[BaseMessage]] = {}

    def get(self, thread_id: str) -> list[BaseMessage]:
        """Return a copy of the stored history, or [] if the thread is unknown."""
        return list(self._store.get(thread_id, []))

    def has(self, thread_id: str) -> bool:
        return thread_id in self._store

    def append(self, thread_id: str, question: str, answer: str, *, max_turns: int) -> None:
        """Append a (question, answer) pair and evict the oldest when over cap."""
        msgs = self._store.setdefault(thread_id, [])
        msgs.append(HumanMessage(content=question))
        msgs.append(AIMessage(content=answer))
        cap = max_turns * 2
        if len(msgs) > cap:
            self._store[thread_id] = msgs[-cap:]

    def delete(self, thread_id: str) -> bool:
        """Remove a thread.  Returns True if it existed, False otherwise."""
        return self._store.pop(thread_id, None) is not None


# --- history helpers ---


def _to_langchain_history(
    turns: list[ConversationTurn],
    max_turns: int,
) -> list[BaseMessage]:
    """Convert validated ConversationTurn objects to LangChain message objects.

    Applies the server-side cap (last *max_turns* messages) as defense-in-depth
    so runaway front-end history can never blow the context budget.
    """
    capped = turns[-max_turns:] if len(turns) > max_turns else turns
    return [
        HumanMessage(content=t.content) if t.role == "user" else AIMessage(content=t.content)
        for t in capped
    ]


def _resolve_history(
    thread_store: ThreadStore,
    thread_id: str,
    body: QueryRequest,
    settings: Settings,
) -> list[BaseMessage]:
    """Return the conversation history to pass to the ask graph.

    When the thread already exists in the server store the stored history is
    used and the client-supplied conversation_history is ignored (O(1) payload
    instead of O(n)).  For brand-new threads the client-supplied history is
    converted and used to seed the thread.
    """
    if thread_store.has(thread_id):
        return thread_store.get(thread_id)
    return _to_langchain_history(body.conversation_history, settings.max_history_turns)


# --- key resolution ---


def _resolve_key(settings: Settings) -> str:
    """Return the active OpenAI key (keyring → settings) or raise 400."""
    key = get_openai_key(settings)
    if not key:
        raise AppError(
            400,
            "API_KEY_MISSING",
            "No OpenAI API key configured. POST to /config/api-key to set one.",
        )
    return key


# --- app factory ---


def create_app(
    settings: Settings | None = None,
    *,
    chroma_client: chromadb.ClientAPI | None = None,
) -> FastAPI:
    if settings is None:
        settings = load_settings()

    configure_logging(settings.log_level, json_logs=settings.json_logs)

    limiter = Limiter(key_func=get_remote_address)

    app = FastAPI(title="youtube-extention", lifespan=lifespan)
    app.state.settings = settings
    app.state.chroma_client = chroma_client  # None means lazy-init in lifespan
    app.state.ask_graph_cache = {"key_hash": None, "graph": None}
    app.state.thread_store = ThreadStore()
    app.state.limiter = limiter

    async def _rate_limit_handler(_request: Request, _exc: RateLimitExceeded) -> JSONResponse:
        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "code": "RATE_LIMITED",
                    "message": "Rate limit exceeded. Try again later.",
                }
            },
        )

    app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)
    app.add_exception_handler(AppError, app_error_handler)
    # Register for both the Starlette base class (raised by the router, e.g. 405)
    # and FastAPI's subclass so all HTTP errors go through our envelope.
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    # Last-resort fallback: any unhandled Exception → generic 500 envelope.
    app.add_exception_handler(Exception, global_exception_handler)

    # RequestIDMiddleware must be outermost so request_id is bound before CORS
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_origin_regex=r"(chrome-extension://.*|http://localhost(:\d+)?)",
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        s: Settings = app.state.settings
        return {
            "status": "ok",
            "version": s.version,
            "langsmith_enabled": s.langsmith_tracing.lower() == "true",
            "has_api_key": bool(get_openai_key(s)),
        }

    # --- /config routes ---

    @app.post("/config/api-key", status_code=204)
    @limiter.limit("5/minute")
    async def config_set_api_key(request: Request, body: ApiKeyRequest):
        set_api_key(body.api_key)

    @app.delete("/config/api-key", status_code=204)
    async def config_clear_api_key():
        clear_api_key()

    @app.get("/config/status", response_model=ConfigStatusResponse)
    async def config_status():
        s: Settings = app.state.settings
        return ConfigStatusResponse(has_key=bool(get_openai_key(s)))

    # --- /threads ---

    @app.delete("/threads/{thread_id}", status_code=204)
    async def delete_thread(thread_id: str):
        """Clear server-side conversation history for a thread."""
        store: ThreadStore = app.state.thread_store
        if not store.delete(thread_id):
            raise AppError(404, "THREAD_NOT_FOUND", f"Thread '{thread_id}' not found")

    # --- /ingest ---

    @app.post("/ingest", response_model=IngestResponse)
    @limiter.limit("10/minute")
    async def ingest(request: Request, body: IngestRequest):
        s: Settings = app.state.settings
        key = _resolve_key(s)
        embeddings = get_embeddings(s, api_key=key)

        initial_state: IngestState = {
            "video_id": body.video_id,
            "force": body.force,
            "embeddings": embeddings,
            "chroma_client": app.state.chroma_client,
            "segments": [],
            "chunks": [],
            "embedded_chunks": [],
            "channel_metadata": {},
            "status": "pending",
            "error": None,
        }

        traced = await run_in_threadpool(
            _run_ingest_graph,
            initial_state,
            video_id=body.video_id,
            embed_model=s.embed_model,
        )

        if traced["status"] == "error":
            raise AppError(502, "INGEST_FAILED", traced["error"] or "Ingestion failed")

        return IngestResponse(
            status=traced["status"],
            chunk_count=traced["chunk_count"],
            cached=traced["status"] == "skipped",
        )

    # --- /query endpoint ---

    @app.post("/query")
    @limiter.limit("30/minute")
    async def query(request: Request, body: QueryRequest):
        """Unified Q&A endpoint with streaming and advanced mode support.

        Flags:
        - stream: Enable Server-Sent Events streaming (default: False)
        - advanced: Use LangGraph pipeline with guardrails (default: True)

        Routes internally to:
        - Simple mode (!advanced): Direct retrieval + generation
        - Advanced mode (advanced): LangGraph with validation + guardrails
        - Streaming (stream=True): SSE token-by-token response
        """
        s: Settings = app.state.settings
        key = _resolve_key(s)
        emb = get_embeddings(s, api_key=key)

        # Generate or use provided thread_id for conversation tracking
        thread_id = body.thread_id or str(uuid7())
        logger.info(
            "query_started",
            video_id=body.video_id,
            thread_id=thread_id,
            stream=body.stream,
            advanced=body.advanced,
        )

        # Route to streaming if requested
        if body.stream:
            return await _handle_streaming_query(body, s, key, emb, thread_id)

        # Route to advanced (graph-based) or simple mode
        if body.advanced:
            return await _handle_advanced_query(body, s, key, emb, thread_id, request)
        else:
            return await _handle_simple_query(body, s, key, emb, thread_id, request)

    async def _handle_simple_query(
        body: QueryRequest, s: Settings, key: str, emb, thread_id: str, request: Request
    ) -> QuestionResponse:
        """Simple Q&A without graph pipeline."""
        try:
            result = await run_in_threadpool(
                _run_answer_question,
                video_id=body.video_id,
                question=body.question,
                embeddings=emb,
                chroma_client=app.state.chroma_client,
                chat_model=s.chat_model,
                openai_api_key=key,
                k=body.k,
                score_threshold=s.min_similarity_threshold,
                context_budget_tokens=s.context_budget_tokens,
                openai_api_base=s.openai_api_base,
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
        body: QueryRequest, s: Settings, key: str, emb, thread_id: str, request: Request
    ) -> AskResponse:
        """Advanced Q&A with LangGraph pipeline and guardrails."""
        thread_store: ThreadStore = app.state.thread_store
        graph = _get_or_rebuild_ask_graph(app, s, key, emb)

        state = make_initial_state(
            video_id=body.video_id,
            question=body.question,
            history=_resolve_history(thread_store, thread_id, body, s),
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
            max_turns=s.max_history_turns,
        )

        return AskResponse(
            answer=result["answer"],
            citations=result["citations"],
            tokens_used=result.get("tokens_used"),
            refused=result.get("refused", False),
            thread_id=thread_id,
        )

    async def _handle_streaming_query(
        body: QueryRequest, s: Settings, key: str, emb, thread_id: str
    ) -> StreamingResponse:
        """Streaming Q&A via the ask graph with Server-Sent Events."""
        if not collection_exists(body.video_id, app.state.chroma_client):
            raise AppError(404, "VIDEO_NOT_INGESTED", "Video has not been ingested yet")

        thread_store: ThreadStore = app.state.thread_store
        graph = _get_or_rebuild_ask_graph(app, s, key, emb)
        state = make_initial_state(
            video_id=body.video_id,
            question=body.question,
            history=_resolve_history(thread_store, thread_id, body, s),
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
                max_turns=s.max_history_turns,
            ),
            media_type="text/event-stream",
        )

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="127.0.0.1", port=8000, reload=True)
