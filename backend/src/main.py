import json
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langsmith import traceable
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
    http_exception_handler,
    validation_error_handler,
)
from src.logging_config import configure_logging
from src.middleware import RequestIDMiddleware
from src.retriever import build_retriever
from src.secrets import clear_api_key, get_openai_key, set_api_key
from src.vector_store import collection_exists

logger = structlog.get_logger(__name__)


@traceable(name="ingest_video")
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


@traceable(name="ask_question")
def _run_ask_graph(graph, state: AskState, *, video_id: str) -> dict:
    """Invoke the ask graph; exposes video_id to LangSmith as a run tag."""
    return graph.invoke(state, config={"run_name": "ask_question"})


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


class AskRequest(BaseModel):
    video_id: str = Field(..., min_length=1, description="YouTube video ID")
    question: str = Field(..., min_length=1, description="Question about the video")
    conversation_history: list = Field(default_factory=list, description="Prior turn messages")
    k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")


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


class ApiKeyRequest(BaseModel):
    api_key: str = Field(..., min_length=1, description="OpenAI API key")


class ConfigStatusResponse(BaseModel):
    has_key: bool


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
            "status": "pending",
            "error": None,
        }

        traced = _run_ingest_graph(
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
    async def chat(video_id: str, body: QuestionRequest):
        s: Settings = app.state.settings
        key = _resolve_key(s)
        embeddings = get_embeddings(s, api_key=key)
        result = answer_question(
            video_id=video_id,
            question=body.question,
            embeddings=embeddings,
            vector_db_path=s.vector_db_path,
            chat_model=s.chat_model,
            openai_api_key=key,
            k=body.k,
            score_threshold=s.min_similarity_threshold,
            context_budget_tokens=s.context_budget_tokens,
        )
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
    async def ask(body: AskRequest):
        s: Settings = app.state.settings
        key = _resolve_key(s)
        emb = get_embeddings(s, api_key=key)
        graph = build_ask_graph(s, emb, api_key=key)
        state = make_initial_state(
            video_id=body.video_id,
            question=body.question,
            history=body.conversation_history,
            k=body.k,
        )
        result = _run_ask_graph(graph, state, video_id=body.video_id)

        if result.get("error") == VIDEO_NOT_INGESTED:
            raise AppError(404, "VIDEO_NOT_INGESTED", "Video has not been ingested yet")
        if result.get("error"):
            raise AppError(500, "INTERNAL_ERROR", result["error"])

        return AskResponse(
            answer=result["answer"],
            citations=result["citations"],
            tokens_used=result.get("tokens_used"),
            refused=result.get("refused", False),
        )

    @app.post("/ask/stream")
    async def ask_stream(body: AskRequest):
        s: Settings = app.state.settings
        key = _resolve_key(s)
        emb = get_embeddings(s, api_key=key)

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

        history = body.conversation_history or []
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

        async def event_stream():
            if not trimmed:
                refusal = "I'm sorry, that information isn't available in the video transcript."
                yield f"data: {json.dumps({'type': 'done', 'answer': refusal, 'citations': [], 'tokens_used': None, 'refused': True})}\n\n"
                return

            streaming_llm = ChatOpenAI(
                model=s.chat_model,
                openai_api_key=key,
                temperature=0.2,
                streaming=True,
            )
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
            yield f"data: {json.dumps({'type': 'done', 'answer': full_answer, 'citations': citations, 'tokens_used': None, 'refused': refused})}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="127.0.0.1", port=8000, reload=True)
