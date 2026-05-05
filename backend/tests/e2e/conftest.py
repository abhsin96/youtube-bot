"""E2E test fixtures.

Session-scoped mock OpenAI server (uvicorn in a background thread) +
function-scoped fixtures that pre-populate a temporary Chroma DB with
fixture documents embedded via the mock server.
"""

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from fastapi.testclient import TestClient
from langchain_openai import OpenAIEmbeddings

from src.config import Settings
from src.main import create_app
from src.vector_store import add_documents
from tests.e2e.fixture_data import FIXTURE_VIDEO_ID, make_fixture_docs
from tests.mock_openai.server import app as _mock_openai_app
from tests.mock_openai.server import clear_responses

# ---------------------------------------------------------------------------
# Session-scoped mock OpenAI server
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def mock_openai_url() -> str:  # type: ignore[return]
    """Start the mock OpenAI server once for the whole test session."""
    port = _free_port()
    config = uvicorn.Config(_mock_openai_app, host="127.0.0.1", port=port, log_level="critical")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Poll until the server responds (max 5 s)
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            import httpx

            httpx.get(f"{base_url}/docs", timeout=0.5)
            break
        except Exception:
            time.sleep(0.05)

    yield base_url

    server.should_exit = True


@pytest.fixture(autouse=True)
def _reset_mock_responses():
    """Ensure no registered responses bleed between tests."""
    clear_responses()
    yield
    clear_responses()


# ---------------------------------------------------------------------------
# Function-scoped: fresh temp Chroma DB + pre-populated fixture docs
# ---------------------------------------------------------------------------


@pytest.fixture()
def e2e_settings(mock_openai_url: str, tmp_path: Path) -> Settings:
    """Settings that point to the mock OpenAI server and a temp Chroma dir."""
    return Settings(
        openai_api_key="sk-e2e-mock",
        openai_api_base=mock_openai_url,
        vector_db_path=tmp_path / "chroma",
        min_similarity_threshold=0.0,
        _env_file=None,
    )


@pytest.fixture()
def populated_db(e2e_settings: Settings) -> Path:
    """Pre-populate a fresh Chroma DB with fixture docs; return its path."""
    from src.vector_store import delete_collection

    embeddings = OpenAIEmbeddings(
        model=e2e_settings.embed_model,
        openai_api_key=e2e_settings.openai_api_key,
        base_url=e2e_settings.openai_api_base,
    )
    # Delete any existing collection to ensure fresh creation with cosine metric
    delete_collection(FIXTURE_VIDEO_ID, e2e_settings.vector_db_path)

    add_documents(
        FIXTURE_VIDEO_ID,
        make_fixture_docs(),
        embeddings,
        e2e_settings.vector_db_path,
    )
    return e2e_settings.vector_db_path


@pytest.fixture()
def e2e_client(e2e_settings: Settings, populated_db: Path) -> TestClient:
    """TestClient wired to the mock OpenAI server and pre-populated Chroma."""
    app = create_app(e2e_settings)
    return TestClient(app, raise_server_exceptions=False)
