import chromadb
import pytest
from chromadb.config import Settings as ChromaSettings

from src.dependencies import limiter

_LANGSMITH_ENV_KEYS = (
    "LANGSMITH_TRACING",
    "LANGCHAIN_TRACING_V2",
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
)


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Reset the singleton rate-limiter between tests.

    The limiter moved from a per-app instance (created inside create_app) to a
    module-level singleton in src.dependencies.  Without this reset, hit counts
    accumulate across the test session and cause false 429 failures.
    """
    if hasattr(limiter, "_storage") and hasattr(limiter._storage, "reset"):
        limiter._storage.reset()
    yield


@pytest.fixture(autouse=True)
def _reset_langsmith_env(monkeypatch):
    """Undo the os.environ.setdefault calls that app = create_app() makes at
    import time (from the .env file).  Without this, tests that create
    Settings(_env_file=None) still pick up LANGSMITH_TRACING=true from the
    process environment and fail their assertions about default values."""
    for key in _LANGSMITH_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def _ephemeral_chroma(monkeypatch, tmp_path):
    """Replace the Chroma HTTP client factory with an isolated in-memory client.

    Applied to every test so that tests never attempt to connect to a real
    Chroma server.  A fresh PersistentClient in a unique temp directory is used
    instead of EphemeralClient to avoid chromadb's shared in-process singleton.

    Tests that inject their own client (e.g. e2e tests, or create_app calls
    that pass chroma_client=...) are unaffected because create_app skips
    get_chroma_client when a client is already provided.
    """

    def _make_client(host: str, port: int) -> chromadb.ClientAPI:
        return chromadb.PersistentClient(
            path=str(tmp_path / "chroma_main"),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    monkeypatch.setattr("src.main.get_chroma_client", _make_client)
