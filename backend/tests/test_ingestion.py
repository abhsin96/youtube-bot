from unittest.mock import MagicMock, patch

import pytest

from src.ingestion import IngestionResult, ingest_video
from src.transcript_service import ErrorCode, TranscriptError

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
    return tmp_path / "chroma"


# --- happy path ---


def test_ingest_returns_result(tmp_db):
    with patch("src.ingestion.fetch_transcript", return_value=_SEGMENTS):
        result = ingest_video("vid1", _fake_embeddings(), tmp_db)
    assert isinstance(result, IngestionResult)
    assert result.video_id == "vid1"
    assert result.segments == 2
    assert result.chunks > 0
    assert result.already_existed is False


def test_ingest_stores_collection(tmp_db):
    from src.vector_store import collection_exists

    with patch("src.ingestion.fetch_transcript", return_value=_SEGMENTS):
        ingest_video("vid1", _fake_embeddings(), tmp_db)
    assert collection_exists("vid1", tmp_db)


# --- skip if already ingested ---


def test_ingest_skips_if_collection_exists(tmp_db):
    emb = _fake_embeddings()
    with patch("src.ingestion.fetch_transcript", return_value=_SEGMENTS) as mock_fetch:
        ingest_video("vid1", emb, tmp_db)
        result = ingest_video("vid1", emb, tmp_db)

    assert result.already_existed is True
    assert result.segments == 0
    assert result.chunks == 0
    assert mock_fetch.call_count == 1  # transcript fetched only once


def test_ingest_force_re_ingests(tmp_db):
    emb = _fake_embeddings()
    with patch("src.ingestion.fetch_transcript", return_value=_SEGMENTS) as mock_fetch:
        ingest_video("vid1", emb, tmp_db)
        result = ingest_video("vid1", emb, tmp_db, force=True)

    assert result.already_existed is False
    assert mock_fetch.call_count == 2


# --- transcript errors propagate ---


@pytest.mark.parametrize(
    "code",
    [ErrorCode.VIDEO_NOT_FOUND, ErrorCode.TRANSCRIPT_DISABLED, ErrorCode.RATE_LIMITED],
)
def test_ingest_raises_transcript_error(tmp_db, code):
    with (
        patch("src.ingestion.fetch_transcript", side_effect=TranscriptError("fail", code)),
        pytest.raises(TranscriptError) as exc_info,
    ):
        ingest_video("vid1", _fake_embeddings(), tmp_db)
    assert exc_info.value.code == code
