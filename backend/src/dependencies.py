from __future__ import annotations

import structlog
from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.api_secrets import get_openai_key
from src.config import Settings
from src.error_envelope import AppError
from src.stores.thread_store import AnyThreadStore

logger = structlog.get_logger(__name__)

limiter = Limiter(key_func=get_remote_address)


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_thread_store(request: Request) -> AnyThreadStore:
    return request.app.state.thread_store


def get_chroma_client(request: Request):
    return request.app.state.chroma_client


def get_ask_graph_cache(request: Request) -> dict:
    return request.app.state.ask_graph_cache


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
