from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from graphs.ingest_graph import (
    IngestState,
    build_graph,
    chunk_node,
    embed_node,
    fetch_transcript_node,
    idempotency_check_node,
    rollback_node,
    route_idempotency,
    route_on_error,
    store_node,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_SEGMENTS = [
    {"text": "Hello world", "start": 0.0, "duration": 2.0},
    {"text": "Second segment", "start": 2.0, "duration": 3.0},
]

_CHUNKS = [
    Document(
        page_content="Hello world",
        metadata={"video_id": "v1", "start_ts": 0.0, "end_ts": 2.0, "chunk_id": "c1"},
    )
]


def _fake_embeddings(dim: int = 4):
    emb = MagicMock()
    emb.embed_documents.side_effect = lambda texts: [[0.0] * dim for _ in texts]
    emb.embed_query.return_value = [0.0] * dim
    return emb


def _state(**overrides) -> IngestState:
    base: IngestState = {
        "video_id": "vid1",
        "force": False,
        "embeddings": _fake_embeddings(),
        "vector_db_path": "/tmp/db",
        "segments": [],
        "chunks": [],
        "embedded_chunks": [],
        "status": "pending",
        "error": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# idempotency_check_node
# ---------------------------------------------------------------------------


def test_idempotency_skips_when_exists():
    with patch("graphs.ingest_graph.collection_exists", return_value=True):
        result = idempotency_check_node(_state(force=False))
    assert result["status"] == "skipped"


def test_idempotency_runs_when_not_exists():
    with patch("graphs.ingest_graph.collection_exists", return_value=False):
        result = idempotency_check_node(_state(force=False))
    assert result["status"] == "running"


def test_idempotency_force_runs_even_when_exists():
    with patch("graphs.ingest_graph.collection_exists", return_value=True):
        result = idempotency_check_node(_state(force=True))
    assert result["status"] == "running"


def test_idempotency_error_on_storage_failure():
    with patch("graphs.ingest_graph.collection_exists", side_effect=OSError("disk full")):
        result = idempotency_check_node(_state())
    assert result["status"] == "error"
    assert "disk full" in result["error"]


# ---------------------------------------------------------------------------
# fetch_transcript_node
# ---------------------------------------------------------------------------


def test_fetch_transcript_sets_segments():
    with patch("graphs.ingest_graph.fetch_transcript", return_value=_SEGMENTS):
        result = fetch_transcript_node(_state())
    assert result["segments"] == _SEGMENTS


def test_fetch_transcript_error_on_failure():
    with patch("graphs.ingest_graph.fetch_transcript", side_effect=RuntimeError("no transcript")):
        result = fetch_transcript_node(_state())
    assert result["status"] == "error"
    assert "no transcript" in result["error"]


def test_fetch_transcript_does_not_set_status_on_success():
    with patch("graphs.ingest_graph.fetch_transcript", return_value=_SEGMENTS):
        result = fetch_transcript_node(_state())
    assert "status" not in result


# ---------------------------------------------------------------------------
# chunk_node
# ---------------------------------------------------------------------------


def test_chunk_node_produces_chunks():
    with (
        patch("graphs.ingest_graph.segments_to_documents", return_value=_CHUNKS),
        patch("graphs.ingest_graph._chunk_documents", return_value=_CHUNKS),
    ):
        result = chunk_node(_state(segments=_SEGMENTS))
    assert result["chunks"] == _CHUNKS


def test_chunk_node_error_on_failure():
    with patch("graphs.ingest_graph.segments_to_documents", side_effect=ValueError("bad")):
        result = chunk_node(_state(segments=_SEGMENTS))
    assert result["status"] == "error"


def test_chunk_node_does_not_set_status_on_success():
    with (
        patch("graphs.ingest_graph.segments_to_documents", return_value=_CHUNKS),
        patch("graphs.ingest_graph._chunk_documents", return_value=_CHUNKS),
    ):
        result = chunk_node(_state(segments=_SEGMENTS))
    assert "status" not in result


# ---------------------------------------------------------------------------
# embed_node
# ---------------------------------------------------------------------------


def test_embed_node_returns_pairs():
    pairs = [(_CHUNKS[0], [0.1, 0.2, 0.3, 0.4])]
    with patch("graphs.ingest_graph.embed_chunks", return_value=pairs):
        result = embed_node(_state(chunks=_CHUNKS))
    assert result["embedded_chunks"] == pairs


def test_embed_node_error_on_failure():
    with patch("graphs.ingest_graph.embed_chunks", side_effect=RuntimeError("rate limit")):
        result = embed_node(_state(chunks=_CHUNKS))
    assert result["status"] == "error"
    assert "rate limit" in result["error"]


def test_embed_node_does_not_set_status_on_success():
    with patch("graphs.ingest_graph.embed_chunks", return_value=[]):
        result = embed_node(_state(chunks=[]))
    assert "status" not in result


# ---------------------------------------------------------------------------
# store_node
# ---------------------------------------------------------------------------


def test_store_node_sets_status_done():
    with patch("graphs.ingest_graph.add_documents"):
        result = store_node(_state(chunks=_CHUNKS))
    assert result["status"] == "done"
    assert result["error"] is None


def test_store_node_uses_precomputed_vectors():
    pairs = [(_CHUNKS[0], [0.1, 0.2, 0.3, 0.4])]
    with patch("graphs.ingest_graph.add_documents") as mock_add:
        emb = _fake_embeddings()
        store_node(
            _state(chunks=_CHUNKS, embedded_chunks=pairs, embeddings=emb, vector_db_path="/db")
        )
    mock_add.assert_called_once_with(
        "vid1", [_CHUNKS[0]], emb, "/db", precomputed_vectors=[[0.1, 0.2, 0.3, 0.4]]
    )


def test_store_node_falls_back_when_no_embedded_chunks():
    with patch("graphs.ingest_graph.add_documents") as mock_add:
        emb = _fake_embeddings()
        store_node(_state(chunks=_CHUNKS, embedded_chunks=[], embeddings=emb, vector_db_path="/db"))
    mock_add.assert_called_once_with("vid1", _CHUNKS, emb, "/db")


def test_store_node_error_on_failure():
    with patch("graphs.ingest_graph.add_documents", side_effect=RuntimeError("chroma down")):
        result = store_node(_state(chunks=_CHUNKS))
    assert result["status"] == "error"
    assert "chroma down" in result["error"]


# ---------------------------------------------------------------------------
# route_idempotency
# ---------------------------------------------------------------------------


def test_route_idempotency_skipped_goes_to_end():
    from langgraph.graph import END

    assert route_idempotency(_state(status="skipped")) == END


def test_route_idempotency_error_goes_to_rollback():
    assert route_idempotency(_state(status="error")) == "rollback_node"


def test_route_idempotency_running_goes_to_fetch():
    assert route_idempotency(_state(status="running")) == "fetch_transcript_node"


# ---------------------------------------------------------------------------
# route_on_error
# ---------------------------------------------------------------------------


def test_route_on_error_returns_rollback_on_error():
    assert route_on_error(_state(status="error")) == "rollback_node"


def test_route_on_error_returns_ok_on_running():
    assert route_on_error(_state(status="running")) == "_ok"


def test_route_on_error_returns_ok_on_pending():
    assert route_on_error(_state(status="pending")) == "_ok"


# ---------------------------------------------------------------------------
# build_graph — full pipeline (mocked I/O)
# ---------------------------------------------------------------------------


def test_graph_happy_path(tmp_path):
    db = tmp_path / "chroma"
    with (
        patch("graphs.ingest_graph.collection_exists", return_value=False),
        patch("graphs.ingest_graph.fetch_transcript", return_value=_SEGMENTS),
        patch("graphs.ingest_graph.segments_to_documents", return_value=_CHUNKS),
        patch("graphs.ingest_graph._chunk_documents", return_value=_CHUNKS),
        patch("graphs.ingest_graph.embed_chunks", return_value=[(_CHUNKS[0], [0.0] * 4)]),
        patch("graphs.ingest_graph.add_documents"),
    ):
        result = build_graph().invoke(_state(vector_db_path=str(db)))

    assert result["status"] == "done"
    assert result["error"] is None
    assert result["segments"] == _SEGMENTS
    assert result["chunks"] == _CHUNKS


def test_graph_skips_when_collection_exists():
    with patch("graphs.ingest_graph.collection_exists", return_value=True):
        result = build_graph().invoke(_state(force=False))

    assert result["status"] == "skipped"
    assert result["segments"] == []  # never populated


def test_graph_force_bypasses_idempotency(tmp_path):
    db = tmp_path / "chroma"
    with (
        patch("graphs.ingest_graph.collection_exists", return_value=True),
        patch("graphs.ingest_graph.fetch_transcript", return_value=_SEGMENTS),
        patch("graphs.ingest_graph.segments_to_documents", return_value=_CHUNKS),
        patch("graphs.ingest_graph._chunk_documents", return_value=_CHUNKS),
        patch("graphs.ingest_graph.embed_chunks", return_value=[(_CHUNKS[0], [0.0] * 4)]),
        patch("graphs.ingest_graph.add_documents"),
    ):
        result = build_graph().invoke(_state(force=True, vector_db_path=str(db)))

    assert result["status"] == "done"


def test_graph_stops_at_fetch_error():
    with (
        patch("graphs.ingest_graph.collection_exists", return_value=False),
        patch("graphs.ingest_graph.fetch_transcript", side_effect=RuntimeError("no captions")),
    ):
        result = build_graph().invoke(_state())

    assert result["status"] == "error"
    assert "no captions" in result["error"]
    assert result["chunks"] == []  # chunk_node never ran


def test_graph_stops_at_embed_error():
    with (
        patch("graphs.ingest_graph.collection_exists", return_value=False),
        patch("graphs.ingest_graph.fetch_transcript", return_value=_SEGMENTS),
        patch("graphs.ingest_graph.segments_to_documents", return_value=_CHUNKS),
        patch("graphs.ingest_graph._chunk_documents", return_value=_CHUNKS),
        patch("graphs.ingest_graph.embed_chunks", side_effect=RuntimeError("rate limit")),
        patch("graphs.ingest_graph.delete_collection"),
    ):
        result = build_graph().invoke(_state())

    assert result["status"] == "error"
    assert result["embedded_chunks"] == []  # store_node never ran


# ---------------------------------------------------------------------------
# rollback_node (unit)
# ---------------------------------------------------------------------------


def test_rollback_calls_delete_collection():
    with patch("graphs.ingest_graph.delete_collection") as mock_del:
        rollback_node(_state(video_id="vid1", vector_db_path="/db", status="error"))
    mock_del.assert_called_once_with("vid1", "/db")


def test_rollback_returns_empty_dict():
    with patch("graphs.ingest_graph.delete_collection"):
        result = rollback_node(_state(status="error", error="something failed"))
    assert result == {}


def test_rollback_preserves_error_in_state():
    # rollback returns {}, so LangGraph keeps the existing status/error untouched
    with patch("graphs.ingest_graph.delete_collection"):
        delta = rollback_node(_state(status="error", error="original error"))
    assert "status" not in delta
    assert "error" not in delta


def test_rollback_tolerates_delete_failure():
    with patch("graphs.ingest_graph.delete_collection", side_effect=RuntimeError("disk full")):
        # must not raise — original error should remain surfaceable
        result = rollback_node(_state(status="error"))
    assert result == {}


# ---------------------------------------------------------------------------
# Rollback integration: embed_node failure → collection deleted
# ---------------------------------------------------------------------------


def test_rollback_called_on_embed_failure(tmp_path):
    db = tmp_path / "chroma"
    with (
        patch("graphs.ingest_graph.collection_exists", return_value=False),
        patch("graphs.ingest_graph.fetch_transcript", return_value=_SEGMENTS),
        patch("graphs.ingest_graph.segments_to_documents", return_value=_CHUNKS),
        patch("graphs.ingest_graph._chunk_documents", return_value=_CHUNKS),
        patch("graphs.ingest_graph.embed_chunks", side_effect=RuntimeError("rate limit")),
        patch("graphs.ingest_graph.delete_collection") as mock_del,
    ):
        result = build_graph().invoke(_state(vector_db_path=str(db)))

    assert result["status"] == "error"
    mock_del.assert_called_once_with("vid1", str(db))


def test_rollback_called_on_store_failure(tmp_path):
    db = tmp_path / "chroma"
    with (
        patch("graphs.ingest_graph.collection_exists", return_value=False),
        patch("graphs.ingest_graph.fetch_transcript", return_value=_SEGMENTS),
        patch("graphs.ingest_graph.segments_to_documents", return_value=_CHUNKS),
        patch("graphs.ingest_graph._chunk_documents", return_value=_CHUNKS),
        patch("graphs.ingest_graph.embed_chunks", return_value=[(_CHUNKS[0], [0.0] * 4)]),
        patch("graphs.ingest_graph.add_documents", side_effect=RuntimeError("chroma down")),
        patch("graphs.ingest_graph.delete_collection") as mock_del,
    ):
        result = build_graph().invoke(_state(vector_db_path=str(db)))

    assert result["status"] == "error"
    mock_del.assert_called_once_with("vid1", str(db))


def test_rollback_not_called_on_success(tmp_path):
    db = tmp_path / "chroma"
    with (
        patch("graphs.ingest_graph.collection_exists", return_value=False),
        patch("graphs.ingest_graph.fetch_transcript", return_value=_SEGMENTS),
        patch("graphs.ingest_graph.segments_to_documents", return_value=_CHUNKS),
        patch("graphs.ingest_graph._chunk_documents", return_value=_CHUNKS),
        patch("graphs.ingest_graph.embed_chunks", return_value=[(_CHUNKS[0], [0.0] * 4)]),
        patch("graphs.ingest_graph.add_documents"),
        patch("graphs.ingest_graph.delete_collection") as mock_del,
    ):
        result = build_graph().invoke(_state(vector_db_path=str(db)))

    assert result["status"] == "done"
    mock_del.assert_not_called()


def test_retry_succeeds_after_rollback(tmp_path):
    """After a failed run (collection deleted), a second run completes."""
    db = tmp_path / "chroma"
    base = _state(vector_db_path=str(db))

    # First run: embed fails → rollback deletes collection
    with (
        patch("graphs.ingest_graph.collection_exists", return_value=False),
        patch("graphs.ingest_graph.fetch_transcript", return_value=_SEGMENTS),
        patch("graphs.ingest_graph.segments_to_documents", return_value=_CHUNKS),
        patch("graphs.ingest_graph._chunk_documents", return_value=_CHUNKS),
        patch("graphs.ingest_graph.embed_chunks", side_effect=RuntimeError("transient")),
        patch("graphs.ingest_graph.delete_collection"),
    ):
        first = build_graph().invoke(base)
    assert first["status"] == "error"

    # Second run: everything succeeds
    with (
        patch("graphs.ingest_graph.collection_exists", return_value=False),
        patch("graphs.ingest_graph.fetch_transcript", return_value=_SEGMENTS),
        patch("graphs.ingest_graph.segments_to_documents", return_value=_CHUNKS),
        patch("graphs.ingest_graph._chunk_documents", return_value=_CHUNKS),
        patch("graphs.ingest_graph.embed_chunks", return_value=[(_CHUNKS[0], [0.0] * 4)]),
        patch("graphs.ingest_graph.add_documents"),
    ):
        second = build_graph().invoke(base)
    assert second["status"] == "done"
