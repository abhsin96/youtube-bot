"""
Integration tests for POST /ingest.

These tests let the full LangGraph pipeline run; only the external I/O
boundaries are mocked:
  - graphs.ingest_graph.collection_exists / add_documents / delete_collection
  - graphs.ingest_graph.fetch_transcript
  - graphs.ingest_graph.segments_to_documents / _chunk_documents
  - graphs.ingest_graph.embed_chunks
  - src.main.get_embeddings   ← prevents real OpenAI client construction
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from src.config import Settings
from src.main import create_app

# ---------------------------------------------------------------------------
# Shared fixtures / test data
# ---------------------------------------------------------------------------

_SEGMENTS = [
    {"text": "Hello world", "start": 0.0, "duration": 2.0},
    {"text": "Second segment", "start": 2.0, "duration": 3.0},
]

_CHUNKS = [
    Document(
        page_content="Hello world",
        metadata={"video_id": "vid1", "start_ts": 0.0, "end_ts": 2.0, "chunk_id": "c1"},
    ),
    Document(
        page_content="Second segment",
        metadata={"video_id": "vid1", "start_ts": 2.0, "end_ts": 5.0, "chunk_id": "c2"},
    ),
]

_EMBEDDED = [(_CHUNKS[0], [0.1] * 4), (_CHUNKS[1], [0.2] * 4)]


def _fake_embeddings():
    emb = MagicMock()
    emb.embed_documents.side_effect = lambda texts: [[0.0] * 4 for _ in texts]
    emb.embed_query.return_value = [0.0] * 4
    return emb


@pytest.fixture()
def client():
    settings = Settings(openai_api_key="sk-test", _env_file=None)
    return TestClient(create_app(settings))


# Patch targets — external I/O only; graph wiring is exercised for real
_GRAPH = "graphs.ingest_graph"
_MAIN = "src.main"


# ---------------------------------------------------------------------------
# Happy path — new video ingested end-to-end
# ---------------------------------------------------------------------------


def test_happy_path_returns_done(client):
    with (
        patch(f"{_GRAPH}.collection_exists", return_value=False),
        patch(f"{_GRAPH}.fetch_transcript", return_value=_SEGMENTS),
        patch(f"{_GRAPH}.segments_to_documents", return_value=_CHUNKS),
        patch(f"{_GRAPH}._chunk_documents", return_value=_CHUNKS),
        patch(f"{_GRAPH}.embed_chunks", return_value=_EMBEDDED),
        patch(f"{_GRAPH}.add_documents"),
        patch(f"{_MAIN}.get_embeddings", return_value=_fake_embeddings()),
    ):
        resp = client.post("/ingest", json={"video_id": "vid1"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "done"
    assert body["cached"] is False


def test_happy_path_chunk_count_matches_graph_output(client):
    with (
        patch(f"{_GRAPH}.collection_exists", return_value=False),
        patch(f"{_GRAPH}.fetch_transcript", return_value=_SEGMENTS),
        patch(f"{_GRAPH}.segments_to_documents", return_value=_CHUNKS),
        patch(f"{_GRAPH}._chunk_documents", return_value=_CHUNKS),
        patch(f"{_GRAPH}.embed_chunks", return_value=_EMBEDDED),
        patch(f"{_GRAPH}.add_documents"),
        patch(f"{_MAIN}.get_embeddings", return_value=_fake_embeddings()),
    ):
        resp = client.post("/ingest", json={"video_id": "vid1"})

    assert resp.json()["chunk_count"] == len(_CHUNKS)


def test_happy_path_add_documents_called_once(client):
    with (
        patch(f"{_GRAPH}.collection_exists", return_value=False),
        patch(f"{_GRAPH}.fetch_transcript", return_value=_SEGMENTS),
        patch(f"{_GRAPH}.segments_to_documents", return_value=_CHUNKS),
        patch(f"{_GRAPH}._chunk_documents", return_value=_CHUNKS),
        patch(f"{_GRAPH}.embed_chunks", return_value=_EMBEDDED),
        patch(f"{_GRAPH}.add_documents") as mock_add,
        patch(f"{_MAIN}.get_embeddings", return_value=_fake_embeddings()),
    ):
        client.post("/ingest", json={"video_id": "vid1"})

    mock_add.assert_called_once()


def test_happy_path_delete_collection_not_called(client):
    with (
        patch(f"{_GRAPH}.collection_exists", return_value=False),
        patch(f"{_GRAPH}.fetch_transcript", return_value=_SEGMENTS),
        patch(f"{_GRAPH}.segments_to_documents", return_value=_CHUNKS),
        patch(f"{_GRAPH}._chunk_documents", return_value=_CHUNKS),
        patch(f"{_GRAPH}.embed_chunks", return_value=_EMBEDDED),
        patch(f"{_GRAPH}.add_documents"),
        patch(f"{_GRAPH}.delete_collection") as mock_del,
        patch(f"{_MAIN}.get_embeddings", return_value=_fake_embeddings()),
    ):
        client.post("/ingest", json={"video_id": "vid1"})

    mock_del.assert_not_called()


# ---------------------------------------------------------------------------
# Idempotency — video already in the store
# ---------------------------------------------------------------------------


def test_cached_returns_200_with_cached_true(client):
    with (
        patch(f"{_GRAPH}.collection_exists", return_value=True),
        patch(f"{_MAIN}.get_embeddings", return_value=_fake_embeddings()),
    ):
        resp = client.post("/ingest", json={"video_id": "vid1"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["cached"] is True
    assert body["status"] == "skipped"
    assert body["chunk_count"] == 0


def test_cached_fetch_transcript_never_called(client):
    with (
        patch(f"{_GRAPH}.collection_exists", return_value=True),
        patch(f"{_GRAPH}.fetch_transcript") as mock_fetch,
        patch(f"{_MAIN}.get_embeddings", return_value=_fake_embeddings()),
    ):
        client.post("/ingest", json={"video_id": "vid1"})

    mock_fetch.assert_not_called()


def test_cached_add_documents_never_called(client):
    with (
        patch(f"{_GRAPH}.collection_exists", return_value=True),
        patch(f"{_GRAPH}.add_documents") as mock_add,
        patch(f"{_MAIN}.get_embeddings", return_value=_fake_embeddings()),
    ):
        client.post("/ingest", json={"video_id": "vid1"})

    mock_add.assert_not_called()


# ---------------------------------------------------------------------------
# Force flag — bypass idempotency check
# ---------------------------------------------------------------------------


def test_force_reruns_when_collection_exists(client):
    with (
        patch(f"{_GRAPH}.collection_exists", return_value=True),
        patch(f"{_GRAPH}.fetch_transcript", return_value=_SEGMENTS),
        patch(f"{_GRAPH}.segments_to_documents", return_value=_CHUNKS),
        patch(f"{_GRAPH}._chunk_documents", return_value=_CHUNKS),
        patch(f"{_GRAPH}.embed_chunks", return_value=_EMBEDDED),
        patch(f"{_GRAPH}.add_documents"),
        patch(f"{_MAIN}.get_embeddings", return_value=_fake_embeddings()),
    ):
        resp = client.post("/ingest", json={"video_id": "vid1", "force": True})

    assert resp.status_code == 200
    assert resp.json()["cached"] is False
    assert resp.json()["status"] == "done"


# ---------------------------------------------------------------------------
# Failure + rollback — transcript unavailable
# ---------------------------------------------------------------------------


def test_transcript_failure_returns_502(client):
    with (
        patch(f"{_GRAPH}.collection_exists", return_value=False),
        patch(f"{_GRAPH}.fetch_transcript", side_effect=RuntimeError("no captions")),
        patch(f"{_GRAPH}.delete_collection"),
        patch(f"{_MAIN}.get_embeddings", return_value=_fake_embeddings()),
    ):
        resp = client.post("/ingest", json={"video_id": "vid1"})

    assert resp.status_code == 502
    assert "no captions" in resp.json()["detail"]


def test_transcript_failure_triggers_rollback(client):
    with (
        patch(f"{_GRAPH}.collection_exists", return_value=False),
        patch(f"{_GRAPH}.fetch_transcript", side_effect=RuntimeError("no captions")),
        patch(f"{_GRAPH}.delete_collection") as mock_del,
        patch(f"{_MAIN}.get_embeddings", return_value=_fake_embeddings()),
    ):
        client.post("/ingest", json={"video_id": "vid1"})

    mock_del.assert_called_once_with("vid1", mock_del.call_args[0][1])


def test_transcript_failure_add_documents_never_called(client):
    with (
        patch(f"{_GRAPH}.collection_exists", return_value=False),
        patch(f"{_GRAPH}.fetch_transcript", side_effect=RuntimeError("no captions")),
        patch(f"{_GRAPH}.delete_collection"),
        patch(f"{_GRAPH}.add_documents") as mock_add,
        patch(f"{_MAIN}.get_embeddings", return_value=_fake_embeddings()),
    ):
        client.post("/ingest", json={"video_id": "vid1"})

    mock_add.assert_not_called()


# ---------------------------------------------------------------------------
# Failure + rollback — embed step fails mid-pipeline
# ---------------------------------------------------------------------------


def test_embed_failure_returns_502(client):
    with (
        patch(f"{_GRAPH}.collection_exists", return_value=False),
        patch(f"{_GRAPH}.fetch_transcript", return_value=_SEGMENTS),
        patch(f"{_GRAPH}.segments_to_documents", return_value=_CHUNKS),
        patch(f"{_GRAPH}._chunk_documents", return_value=_CHUNKS),
        patch(f"{_GRAPH}.embed_chunks", side_effect=RuntimeError("rate limit")),
        patch(f"{_GRAPH}.delete_collection"),
        patch(f"{_MAIN}.get_embeddings", return_value=_fake_embeddings()),
    ):
        resp = client.post("/ingest", json={"video_id": "vid1"})

    assert resp.status_code == 502
    assert "rate limit" in resp.json()["detail"]


def test_embed_failure_triggers_rollback(client):
    with (
        patch(f"{_GRAPH}.collection_exists", return_value=False),
        patch(f"{_GRAPH}.fetch_transcript", return_value=_SEGMENTS),
        patch(f"{_GRAPH}.segments_to_documents", return_value=_CHUNKS),
        patch(f"{_GRAPH}._chunk_documents", return_value=_CHUNKS),
        patch(f"{_GRAPH}.embed_chunks", side_effect=RuntimeError("rate limit")),
        patch(f"{_GRAPH}.delete_collection") as mock_del,
        patch(f"{_MAIN}.get_embeddings", return_value=_fake_embeddings()),
    ):
        client.post("/ingest", json={"video_id": "vid1"})

    mock_del.assert_called_once_with("vid1", mock_del.call_args[0][1])


def test_embed_failure_add_documents_never_called(client):
    with (
        patch(f"{_GRAPH}.collection_exists", return_value=False),
        patch(f"{_GRAPH}.fetch_transcript", return_value=_SEGMENTS),
        patch(f"{_GRAPH}.segments_to_documents", return_value=_CHUNKS),
        patch(f"{_GRAPH}._chunk_documents", return_value=_CHUNKS),
        patch(f"{_GRAPH}.embed_chunks", side_effect=RuntimeError("rate limit")),
        patch(f"{_GRAPH}.delete_collection"),
        patch(f"{_GRAPH}.add_documents") as mock_add,
        patch(f"{_MAIN}.get_embeddings", return_value=_fake_embeddings()),
    ):
        client.post("/ingest", json={"video_id": "vid1"})

    mock_add.assert_not_called()


# ---------------------------------------------------------------------------
# Failure + rollback — store step fails
# ---------------------------------------------------------------------------


def test_store_failure_returns_502(client):
    with (
        patch(f"{_GRAPH}.collection_exists", return_value=False),
        patch(f"{_GRAPH}.fetch_transcript", return_value=_SEGMENTS),
        patch(f"{_GRAPH}.segments_to_documents", return_value=_CHUNKS),
        patch(f"{_GRAPH}._chunk_documents", return_value=_CHUNKS),
        patch(f"{_GRAPH}.embed_chunks", return_value=_EMBEDDED),
        patch(f"{_GRAPH}.add_documents", side_effect=RuntimeError("chroma down")),
        patch(f"{_GRAPH}.delete_collection"),
        patch(f"{_MAIN}.get_embeddings", return_value=_fake_embeddings()),
    ):
        resp = client.post("/ingest", json={"video_id": "vid1"})

    assert resp.status_code == 502
    assert "chroma down" in resp.json()["detail"]


def test_store_failure_triggers_rollback(client):
    with (
        patch(f"{_GRAPH}.collection_exists", return_value=False),
        patch(f"{_GRAPH}.fetch_transcript", return_value=_SEGMENTS),
        patch(f"{_GRAPH}.segments_to_documents", return_value=_CHUNKS),
        patch(f"{_GRAPH}._chunk_documents", return_value=_CHUNKS),
        patch(f"{_GRAPH}.embed_chunks", return_value=_EMBEDDED),
        patch(f"{_GRAPH}.add_documents", side_effect=RuntimeError("chroma down")),
        patch(f"{_GRAPH}.delete_collection") as mock_del,
        patch(f"{_MAIN}.get_embeddings", return_value=_fake_embeddings()),
    ):
        client.post("/ingest", json={"video_id": "vid1"})

    mock_del.assert_called_once_with("vid1", mock_del.call_args[0][1])


# ---------------------------------------------------------------------------
# Rollback resilience — rollback itself fails
# ---------------------------------------------------------------------------


def test_rollback_failure_does_not_mask_original_error(client):
    with (
        patch(f"{_GRAPH}.collection_exists", return_value=False),
        patch(f"{_GRAPH}.fetch_transcript", side_effect=RuntimeError("no captions")),
        patch(f"{_GRAPH}.delete_collection", side_effect=OSError("disk full")),
        patch(f"{_MAIN}.get_embeddings", return_value=_fake_embeddings()),
    ):
        resp = client.post("/ingest", json={"video_id": "vid1"})

    assert resp.status_code == 502
    assert "no captions" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Input validation (exercised via real route, no graph needed)
# ---------------------------------------------------------------------------


def test_missing_video_id_returns_422(client):
    resp = client.post("/ingest", json={})
    assert resp.status_code == 422


def test_empty_video_id_returns_422(client):
    resp = client.post("/ingest", json={"video_id": ""})
    assert resp.status_code == 422
