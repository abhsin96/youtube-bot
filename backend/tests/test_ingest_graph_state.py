from typing import get_args, get_type_hints
from unittest.mock import MagicMock

from graphs.ingest_graph import IngestState, IngestStatus

_REQUIRED_KEYS = {
    "video_id",
    "force",
    "embeddings",
    "chroma_client",
    "segments",
    "chunks",
    "embedded_chunks",
    "channel_metadata",
    "status",
    "error",
}


def _valid_state() -> IngestState:
    return IngestState(
        video_id="abc123",
        force=False,
        embeddings=MagicMock(),
        chroma_client=MagicMock(),
        segments=[],
        chunks=[],
        embedded_chunks=[],
        status="pending",
        error=None,
    )


# --- schema completeness ---


def test_state_has_all_required_keys():
    hints = get_type_hints(IngestState)
    assert set(hints.keys()) == _REQUIRED_KEYS


def test_state_can_be_constructed():
    state = _valid_state()
    assert state["video_id"] == "abc123"


# --- field types ---


def test_video_id_is_str():
    hints = get_type_hints(IngestState)
    assert hints["video_id"] is str


def test_force_is_bool():
    hints = get_type_hints(IngestState)
    assert hints["force"] is bool


def test_error_is_optional():
    hints = get_type_hints(IngestState)
    args = get_args(hints["error"])
    assert type(None) in args


def test_status_literal_values():
    allowed = set(get_args(IngestStatus))
    assert allowed == {"pending", "running", "done", "error", "skipped"}


def test_segments_is_list():
    hints = get_type_hints(IngestState)
    origin = getattr(hints["segments"], "__origin__", None)
    assert origin is list


def test_chunks_is_list():
    hints = get_type_hints(IngestState)
    origin = getattr(hints["chunks"], "__origin__", None)
    assert origin is list


def test_embedded_chunks_is_list():
    hints = get_type_hints(IngestState)
    origin = getattr(hints["embedded_chunks"], "__origin__", None)
    assert origin is list


# --- status transitions (runtime values) ---


def test_initial_status_pending():
    assert _valid_state()["status"] == "pending"


def test_status_can_be_set_to_done():
    s = _valid_state()
    s["status"] = "done"
    assert s["status"] == "done"


def test_status_can_be_set_to_error_with_message():
    s = _valid_state()
    s["status"] = "error"
    s["error"] = "rate limit hit"
    assert s["error"] == "rate limit hit"


def test_status_skipped_has_empty_pipeline_fields():
    s = _valid_state()
    s["status"] = "skipped"
    assert s["segments"] == []
    assert s["chunks"] == []
    assert s["embedded_chunks"] == []
