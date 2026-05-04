from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langsmith import traceable
from pydantic import BaseModel, Field

from graphs.ingest_graph import IngestState, build_graph
from src.chain import answer_question
from src.config import Settings, load_settings
from src.embeddings import get_embeddings
from src.logging_config import configure_logging
from src.middleware import RequestIDMiddleware

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

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest(body: IngestRequest):
        s: Settings = app.state.settings
        embeddings = get_embeddings(s)

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
            raise HTTPException(status_code=502, detail=traced["error"])

        return IngestResponse(
            status=traced["status"],
            chunk_count=traced["chunk_count"],
            cached=traced["status"] == "skipped",
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
