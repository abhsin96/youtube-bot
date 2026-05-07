import chromadb
import pytest
from chromadb.config import Settings as ChromaSettings


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
