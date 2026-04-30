from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.chain import answer_question
from src.config import Settings, load_settings
from src.embeddings import get_embeddings
from src.ingestion import ingest_video
from src.logging_config import configure_logging
from src.middleware import RequestIDMiddleware
from src.transcript_service import ErrorCode, TranscriptError

logger = structlog.get_logger(__name__)

_TRANSCRIPT_ERROR_STATUS: dict[ErrorCode, int] = {
    ErrorCode.VIDEO_NOT_FOUND: 404,
    ErrorCode.TRANSCRIPT_DISABLED: 422,
    ErrorCode.RATE_LIMITED: 429,
    ErrorCode.UNKNOWN: 502,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    logger.info("server starting", project=settings.langsmith_project)
    yield
    logger.info("server stopped")


# --- request / response models ---


class IngestResponse(BaseModel):
    video_id: str
    segments: int
    chunks: int
    already_existed: bool


class QuestionRequest(BaseModel):
    question: str
    k: int = 5


class QuestionResponse(BaseModel):
    answer: str
    sources: list[dict]


# --- app factory ---


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = load_settings()

    configure_logging(settings.log_level, json_logs=settings.json_logs)

    app = FastAPI(title="youtube-extention", lifespan=lifespan)
    app.state.settings = settings

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
            "has_api_key": bool(s.openai_api_key),
        }

    @app.post("/ingest/{video_id}", response_model=IngestResponse)
    async def ingest(video_id: str, force: bool = False):
        s: Settings = app.state.settings
        embeddings = get_embeddings(s)
        try:
            result = ingest_video(video_id, embeddings, s.vector_db_path, force=force)
        except TranscriptError as exc:
            raise HTTPException(
                status_code=_TRANSCRIPT_ERROR_STATUS.get(exc.code, 502),
                detail=str(exc),
            ) from exc
        return IngestResponse(
            video_id=result.video_id,
            segments=result.segments,
            chunks=result.chunks,
            already_existed=result.already_existed,
        )

    @app.post("/chat/{video_id}", response_model=QuestionResponse)
    async def chat(video_id: str, body: QuestionRequest):
        s: Settings = app.state.settings
        embeddings = get_embeddings(s)
        result = answer_question(
            video_id=video_id,
            question=body.question,
            embeddings=embeddings,
            vector_db_path=s.vector_db_path,
            chat_model=s.chat_model,
            openai_api_key=s.openai_api_key,
            k=body.k,
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

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="127.0.0.1", port=8000, reload=True)
