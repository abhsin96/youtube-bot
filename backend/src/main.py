import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import Settings, load_settings
from src.logging_config import configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    logger.info("server starting", extra={"project": settings.langsmith_project})
    yield
    logger.info("server stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = load_settings()

    configure_logging(settings.log_level)

    app = FastAPI(title="youtube-extention", lifespan=lifespan)
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_origin_regex=r"chrome-extension://.*",
        allow_credentials=True,
        allow_methods=["*"],
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

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="127.0.0.1", port=8000, reload=True)
