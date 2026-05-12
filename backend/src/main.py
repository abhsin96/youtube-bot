from __future__ import annotations

import os
from contextlib import asynccontextmanager

import chromadb
import redis as redis_lib
import structlog
import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.config import Settings, load_settings
from src.dependencies import limiter
from src.error_envelope import (
    AppError,
    app_error_handler,
    global_exception_handler,
    http_exception_handler,
    validation_error_handler,
)
from src.logging_config import configure_logging
from src.middleware import RequestIDMiddleware
from src.routers import config, health, ingest, query
from src.stores.thread_store import RedisThreadStore, ThreadStore
from src.vector_store import get_chroma_client

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    if app.state.chroma_client is None:
        app.state.chroma_client = get_chroma_client(settings.chroma_host, settings.chroma_port)
    logger.info("server starting", project=settings.langsmith_project)
    yield
    if (rc := getattr(app.state, "redis_client", None)) is not None:
        rc.close()
    logger.info("server stopped")


def create_app(
    settings: Settings | None = None,
    *,
    chroma_client: chromadb.ClientAPI | None = None,
) -> FastAPI:
    if settings is None:
        settings = load_settings()

    # pydantic_settings reads .env into the Settings object but does NOT write
    # to os.environ.  The LangSmith SDK reads os.environ directly, so we
    # propagate the relevant keys here before any LangSmith client is created.
    # setdefault means explicitly-set shell variables always win over .env.
    if settings.langsmith_api_key:
        os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
    os.environ.setdefault("LANGSMITH_TRACING", settings.langsmith_tracing)
    os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)

    configure_logging(settings.log_level, json_logs=settings.json_logs)

    app = FastAPI(title="youtube-extention", lifespan=lifespan)
    app.state.settings = settings
    app.state.chroma_client = chroma_client  # None means lazy-init in lifespan
    app.state.ask_graph_cache = {"key_hash": None, "graph": None}
    app.state.limiter = limiter

    if settings.redis_url:
        rc = redis_lib.from_url(settings.redis_url, decode_responses=True)
        app.state.thread_store = RedisThreadStore(rc)
        app.state.redis_client = rc
        logger.info("thread_store", backend="redis", url=settings.redis_url)
    else:
        app.state.thread_store = ThreadStore()
        app.state.redis_client = None
        logger.info("thread_store", backend="memory")

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

    app.include_router(health.router)
    app.include_router(ingest.router)
    app.include_router(query.router)
    app.include_router(config.router)

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="127.0.0.1", port=8000, reload=True)
