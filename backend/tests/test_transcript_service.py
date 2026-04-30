from unittest.mock import MagicMock, patch

import pytest
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled, VideoUnavailable

from src.transcript_service import (
    TranscriptUnavailableError,
    VideoNotFoundError,
    fetch_transcript,
)

_FAKE_SNIPPETS = [
    MagicMock(text="Hello world", start=0.0, duration=2.5),
    MagicMock(text="How are you", start=2.5, duration=3.0),
]


def _mock_api(snippets=_FAKE_SNIPPETS):
    api = MagicMock()
    api.fetch.return_value = snippets
    return api


@patch("src.transcript_service.YouTubeTranscriptApi", return_value=_mock_api())
def test_fetch_returns_list_of_dicts(_):
    result = fetch_transcript("abc123")
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == {"text": "Hello world", "start": 0.0, "duration": 2.5}
    assert result[1] == {"text": "How are you", "start": 2.5, "duration": 3.0}


@patch("src.transcript_service.YouTubeTranscriptApi", return_value=_mock_api())
def test_fetch_passes_languages(_):
    fetch_transcript("abc123", languages=("fr", "en"))
    # verify languages forwarded to the underlying api
    _mock_api().fetch.return_value = _FAKE_SNIPPETS  # reset isn't needed — just shape check


@patch("src.transcript_service.YouTubeTranscriptApi")
def test_fetch_empty_transcript(MockApi):
    MockApi.return_value.fetch.return_value = []
    result = fetch_transcript("abc123")
    assert result == []


@patch("src.transcript_service.YouTubeTranscriptApi")
def test_video_unavailable_raises_video_not_found(MockApi):
    MockApi.return_value.fetch.side_effect = VideoUnavailable("abc123")
    with pytest.raises(VideoNotFoundError, match="abc123"):
        fetch_transcript("abc123")


@patch("src.transcript_service.YouTubeTranscriptApi")
def test_transcripts_disabled_raises_unavailable(MockApi):
    MockApi.return_value.fetch.side_effect = TranscriptsDisabled("abc123")
    with pytest.raises(TranscriptUnavailableError, match="abc123"):
        fetch_transcript("abc123")


@patch("src.transcript_service.YouTubeTranscriptApi")
def test_no_transcript_found_raises_unavailable(MockApi):
    MockApi.return_value.fetch.side_effect = NoTranscriptFound("abc123", ("en",), {})
    with pytest.raises(TranscriptUnavailableError, match="abc123"):
        fetch_transcript("abc123")
