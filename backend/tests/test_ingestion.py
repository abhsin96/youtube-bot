"""
Tests for the ingestion pipeline via the LangGraph path.

Previously exercised src/ingestion.py (legacy linear pipeline, now deleted).
These tests verify the same behavioural contracts using build_graph().invoke(),
which is the path the API uses.
"""

from unittest.mock import MagicMock, patch

import chromadb
import pytest
from chromadb.config import Settings as ChromaSettings

from graphs.ingest_graph import IngestState, build_graph
from src.stores.chroma_vector_store import ChromaVectorStore
from src.transcript_service import ErrorCode, TranscriptError
from src.vector_store import collection_exists

_SEGMENTS = [
    {"text": "Hello world", "start": 0.0, "duration": 2.0},
    {"text": "This is a test", "start": 2.0, "duration": 3.0},
]

_DIM = 4


def _fake_embeddings(dim: int = _DIM):
    emb = MagicMock()
    emb.embed_documents.side_effect = lambda texts: [[0.0] * dim for _ in texts]
    emb.embed_query.return_value = [0.0] * dim
    return emb


@pytest.fixture()
def tmp_db(tmp_path):
    return chromadb.PersistentClient(
        path=str(tmp_path / "chroma"),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def _run_graph(video_id, embeddings, db, *, force=False) -> dict:
    state: IngestState = {
        "video_id": video_id,
        "force": force,
        "embeddings": embeddings,
        "vector_store": ChromaVectorStore(db),
        "segments": [],
        "chunks": [],
        "embedded_chunks": [],
        "channel_metadata": {},
        "status": "pending",
        "error": None,
    }
    return build_graph().invoke(state)


# --- happy path ---


def test_ingest_returns_done(tmp_db):
    with (
        patch("graphs.ingest_graph.fetch_transcript", return_value=_SEGMENTS),
        patch("graphs.ingest_graph.fetch_video_metadata", return_value={}),
    ):
        result = _run_graph("vid1", _fake_embeddings(), tmp_db)
    assert result["status"] == "done"
    assert result["error"] is None
    assert len(result["segments"]) == 2
    assert len(result["chunks"]) > 0


def test_ingest_stores_collection(tmp_db):
    with (
        patch("graphs.ingest_graph.fetch_transcript", return_value=_SEGMENTS),
        patch("graphs.ingest_graph.fetch_video_metadata", return_value={}),
    ):
        _run_graph("vid1", _fake_embeddings(), tmp_db)
    assert collection_exists("vid1", tmp_db)


# --- skip if already ingested ---


def test_ingest_skips_if_collection_exists(tmp_db):
    emb = _fake_embeddings()
    with (
        patch("graphs.ingest_graph.fetch_transcript", return_value=_SEGMENTS) as mock_fetch,
        patch("graphs.ingest_graph.fetch_video_metadata", return_value={}),
    ):
        _run_graph("vid1", emb, tmp_db)
        result = _run_graph("vid1", emb, tmp_db)

    assert result["status"] == "skipped"
    assert result["segments"] == []
    assert result["chunks"] == []
    assert mock_fetch.call_count == 1  # transcript fetched only once


def test_ingest_force_re_ingests(tmp_db):
    emb = _fake_embeddings()
    with (
        patch("graphs.ingest_graph.fetch_transcript", return_value=_SEGMENTS) as mock_fetch,
        patch("graphs.ingest_graph.fetch_video_metadata", return_value={}),
    ):
        _run_graph("vid1", emb, tmp_db)
        result = _run_graph("vid1", emb, tmp_db, force=True)

    assert result["status"] == "done"
    assert mock_fetch.call_count == 2


# --- transcript errors set error state ---


@pytest.mark.parametrize(
    "code",
    [ErrorCode.VIDEO_NOT_FOUND, ErrorCode.TRANSCRIPT_DISABLED, ErrorCode.RATE_LIMITED],
)
def test_ingest_transcript_error_sets_error_state(tmp_db, code):
    with (
        patch(
            "graphs.ingest_graph.fetch_transcript",
            side_effect=TranscriptError("fail", code),
        ),
        patch("graphs.ingest_graph.fetch_video_metadata", return_value={}),
    ):
        result = _run_graph("vid1", _fake_embeddings(), tmp_db)

    assert result["status"] == "error"
    assert "fail" in result["error"]
