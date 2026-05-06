import json
from contextlib import asynccontextmanager
from typing import Literal

import structlog
import uvicorn
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langsmith import traceable, uuid7
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from graphs.ask_graph import REFUSAL_SENTINEL as _REFUSAL_SENTINEL
from graphs.ask_graph import (
    VIDEO_NOT_INGESTED,
    AskState,
    build_ask_graph,
    make_initial_state,
)
from graphs.ingest_graph import IngestState, build_graph
from src.chain import (
    _format_context,
    _load_system_prompt,
    _trim_to_budget,
    answer_question,
    build_chat_prompt,
)
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
from src.retriever import build_retriever
from src.secrets import clear_api_key, get_openai_key, set_api_key
from src.vector_store import collection_exists

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
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


class QuestionRequest(BaseModel):
    question: str
    k: int = 5


class QuestionResponse(BaseModel):
    answer: str
    sources: list[dict]


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"] = Field(..., description="Speaker role")
    content: str = Field(..., min_length=1, description="Message text")


class AskRequest(BaseModel):
    video_id: str = Field(..., min_length=1, description="YouTube video ID")
    question: str = Field(..., min_length=1, description="Question about the video")
    conversation_history: list[ConversationTurn] = Field(
        default_factory=list, description="Prior turn messages"
    )
    k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")
    thread_id: str | None = Field(None, description="Thread ID for conversation tracking")


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


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = load_settings()

    configure_logging(settings.log_level, json_logs=settings.json_logs)

    app = FastAPI(title="youtube-extention", lifespan=lifespan)
    app.state.settings = settings

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
    async def config_set_api_key(body: ApiKeyRequest):
        set_api_key(body.api_key)

    @app.delete("/config/api-key", status_code=204)
    async def config_clear_api_key():
        clear_api_key()

    @app.get("/config/status", response_model=ConfigStatusResponse)
    async def config_status():
        s: Settings = app.state.settings
        return ConfigStatusResponse(has_key=bool(get_openai_key(s)))

    # --- /ingest ---

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest(body: IngestRequest):
        s: Settings = app.state.settings
        key = _resolve_key(s)
        embeddings = get_embeddings(s, api_key=key)

        initial_state: IngestState = {
            "video_id": body.video_id,
            "force": body.force,
            "embeddings": embeddings,
            "vector_db_path": s.vector_db_path,
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

    @app.post("/chat/{video_id}", response_model=QuestionResponse)
    @traceable(name="Chat Bot", metadata={"endpoint": "/chat"})
    async def chat(video_id: str, body: QuestionRequest, request: Request):
        s: Settings = app.state.settings
        key = _resolve_key(s)
        embeddings = get_embeddings(s, api_key=key)

        # Use request ID as thread_id for tracing
        thread_id = getattr(request.state, "request_id", None)

        try:
            result = await run_in_threadpool(
                _run_answer_question,
                video_id=video_id,
                question=body.question,
                embeddings=embeddings,
                vector_db_path=s.vector_db_path,
                chat_model=s.chat_model,
                openai_api_key=key,
                k=body.k,
                score_threshold=s.min_similarity_threshold,
                context_budget_tokens=s.context_budget_tokens,
                openai_api_base=s.openai_api_base,
            )
        except Exception as exc:
            logger.exception("chat_error", video_id=video_id, thread_id=thread_id)
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

    @app.post("/ask", response_model=AskResponse)
    async def ask(body: AskRequest, request: Request):
        s: Settings = app.state.settings
        key = _resolve_key(s)
        emb = get_embeddings(s, api_key=key)
        graph = build_ask_graph(s, emb, api_key=key)

        # Generate or use provided thread_id for conversation tracking
        thread_id = body.thread_id or str(uuid7())
        logger.info("ask_started", video_id=body.video_id, thread_id=thread_id)

        state = make_initial_state(
            video_id=body.video_id,
            question=body.question,
            history=_to_langchain_history(body.conversation_history, s.max_history_turns),
            k=body.k,
        )

        # Wrap the graph invocation with @traceable and thread metadata
        @traceable(name="Chat Bot", metadata={"thread_id": thread_id, "endpoint": "/ask"})
        def run_ask_with_thread():
            return _run_ask_graph(graph, state, video_id=body.video_id, thread_id=thread_id)

        result = await run_in_threadpool(run_ask_with_thread)

        if result.get("error") == VIDEO_NOT_INGESTED:
            raise AppError(404, "VIDEO_NOT_INGESTED", "Video has not been ingested yet")
        if result.get("error"):
            raise AppError(500, "INTERNAL_ERROR", result["error"])

        return AskResponse(
            answer=result["answer"],
            citations=result["citations"],
            tokens_used=result.get("tokens_used"),
            refused=result.get("refused", False),
            thread_id=thread_id,
        )

    @app.post("/ask/stream")
    async def ask_stream(body: AskRequest, request: Request):
        s: Settings = app.state.settings
        key = _resolve_key(s)
        emb = get_embeddings(s, api_key=key)

        # Generate or use provided thread_id for conversation tracking
        thread_id = body.thread_id or str(uuid7())
        logger.info("ask_stream_started", video_id=body.video_id, thread_id=thread_id)

        # Validate + retrieve before opening the SSE channel so we can raise
        # HTTP errors instead of embedding them mid-stream.
        if not collection_exists(body.video_id, s.vector_db_path):
            raise AppError(404, "VIDEO_NOT_INGESTED", "Video has not been ingested yet")

        retriever = build_retriever(
            body.video_id,
            emb,
            s.vector_db_path,
            k=body.k,
            score_threshold=s.min_similarity_threshold,
        )
        chunks = retriever.invoke(body.question)

        history = _to_langchain_history(body.conversation_history, s.max_history_turns)
        system = _load_system_prompt()
        trimmed = (
            _trim_to_budget(
                chunks, system, history, body.question, s.chat_model, s.context_budget_tokens
            )
            if chunks
            else []
        )

        citations = [
            {
                "chunk_id": doc.metadata.get("chunk_id"),
                "start_ts": doc.metadata.get("start_ts"),
                "end_ts": doc.metadata.get("end_ts"),
                "text": doc.page_content,
            }
            for doc in trimmed
        ]

        @traceable(
            name="Chat Bot Stream", metadata={"thread_id": thread_id, "endpoint": "/ask/stream"}
        )
        async def event_stream():
            try:
                if not trimmed:
                    refusal = "I'm sorry, that information isn't available in the video transcript."
                    yield f"data: {json.dumps({'type': 'done', 'answer': refusal, 'citations': [], 'tokens_used': None, 'refused': True, 'thread_id': thread_id})}\n\n"
                    return

                stream_kwargs: dict = {
                    "model": s.chat_model,
                    "openai_api_key": key,
                    "temperature": 0.2,
                    "streaming": True,
                }
                if s.openai_api_base:
                    stream_kwargs["base_url"] = s.openai_api_base
                streaming_llm = ChatOpenAI(**stream_kwargs)
                chain = build_chat_prompt() | streaming_llm | StrOutputParser()
                full_answer = ""
                async for token in chain.astream(
                    {
                        "context": _format_context(trimmed),
                        "question": body.question,
                        "history": history,
                    }
                ):
                    full_answer += token
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

                refused = _REFUSAL_SENTINEL.lower() in full_answer.lower()
                yield f"data: {json.dumps({'type': 'done', 'answer': full_answer, 'citations': citations, 'tokens_used': None, 'refused': refused, 'thread_id': thread_id})}\n\n"
            except Exception:
                logger.exception("stream_error", video_id=body.video_id)
                yield f"data: {json.dumps({'type': 'error', 'code': 'INTERNAL_ERROR', 'message': 'An unexpected error occurred.'})}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # --- Unified /query endpoint ---

    @app.post("/query")
    async def query(body: QueryRequest, request: Request):
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
                vector_db_path=s.vector_db_path,
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
        graph = build_ask_graph(s, emb, api_key=key)

        state = make_initial_state(
            video_id=body.video_id,
            question=body.question,
            history=_to_langchain_history(body.conversation_history, s.max_history_turns),
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
        """Streaming Q&A with Server-Sent Events."""
        # Validate + retrieve before opening the SSE channel
        if not collection_exists(body.video_id, s.vector_db_path):
            raise AppError(404, "VIDEO_NOT_INGESTED", "Video has not been ingested yet")

        retriever = build_retriever(
            body.video_id,
            emb,
            s.vector_db_path,
            k=body.k,
            score_threshold=s.min_similarity_threshold,
        )
        chunks = retriever.invoke(body.question)

        history = _to_langchain_history(body.conversation_history, s.max_history_turns)
        system = _load_system_prompt()
        trimmed = (
            _trim_to_budget(
                chunks, system, history, body.question, s.chat_model, s.context_budget_tokens
            )
            if chunks
            else []
        )

        citations = [
            {
                "chunk_id": doc.metadata.get("chunk_id"),
                "start_ts": doc.metadata.get("start_ts"),
                "end_ts": doc.metadata.get("end_ts"),
                "text": doc.page_content,
            }
            for doc in trimmed
        ]

        @traceable(name="Chat Bot Stream", metadata={"thread_id": thread_id, "endpoint": "/query"})
        async def event_stream():
            try:
                if not trimmed:
                    refusal = "I'm sorry, that information isn't available in the video transcript."
                    yield f"data: {json.dumps({'type': 'done', 'answer': refusal, 'citations': [], 'tokens_used': None, 'refused': True, 'thread_id': thread_id})}\n\n"
                    return

                stream_kwargs: dict = {
                    "model": s.chat_model,
                    "openai_api_key": key,
                    "temperature": 0.2,
                    "streaming": True,
                }
                if s.openai_api_base:
                    stream_kwargs["base_url"] = s.openai_api_base
                streaming_llm = ChatOpenAI(**stream_kwargs)
                chain = build_chat_prompt() | streaming_llm | StrOutputParser()
                full_answer = ""
                async for token in chain.astream(
                    {
                        "context": _format_context(trimmed),
                        "question": body.question,
                        "history": history,
                    }
                ):
                    full_answer += token
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

                refused = _REFUSAL_SENTINEL.lower() in full_answer.lower()
                yield f"data: {json.dumps({'type': 'done', 'answer': full_answer, 'citations': citations, 'tokens_used': None, 'refused': refused, 'thread_id': thread_id})}\n\n"
            except Exception:
                logger.exception("stream_error", video_id=body.video_id)
                yield f"data: {json.dumps({'type': 'error', 'code': 'INTERNAL_ERROR', 'message': 'An unexpected error occurred.'})}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="127.0.0.1", port=8000, reload=True)
