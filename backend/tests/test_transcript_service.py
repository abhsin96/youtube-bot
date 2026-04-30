from unittest.mock import MagicMock, patch

import pytest
from youtube_transcript_api._errors import (
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
)

from src.transcript_service import ErrorCode, TranscriptError, fetch_transcript

_FAKE_SNIPPETS = [
    MagicMock(text="Hello world", start=0.0, duration=2.5),
    MagicMock(text="How are you", start=2.5, duration=3.0),
]


@patch("src.transcript_service.YouTubeTranscriptApi")
def test_fetch_returns_list_of_dicts(MockApi):
    MockApi.return_value.fetch.return_value = _FAKE_SNIPPETS
    result = fetch_transcript("abc123")
    assert result == [
        {"text": "Hello world", "start": 0.0, "duration": 2.5},
        {"text": "How are you", "start": 2.5, "duration": 3.0},
    ]


@patch("src.transcript_service.YouTubeTranscriptApi")
def test_fetch_empty_transcript(MockApi):
    MockApi.return_value.fetch.return_value = []
    assert fetch_transcript("abc123") == []


@pytest.mark.parametrize(
    "exc_class, expected_code",
    [
        (VideoUnavailable, ErrorCode.VIDEO_NOT_FOUND),
        (TranscriptsDisabled, ErrorCode.TRANSCRIPT_DISABLED),
        (NoTranscriptFound, ErrorCode.TRANSCRIPT_DISABLED),
        (IpBlocked, ErrorCode.RATE_LIMITED),
        (RequestBlocked, ErrorCode.RATE_LIMITED),
        (RuntimeError, ErrorCode.UNKNOWN),
    ],
)
@patch("src.transcript_service.YouTubeTranscriptApi")
def test_exception_maps_to_error_code(MockApi, exc_class, expected_code):
    if exc_class in (IpBlocked, RequestBlocked, TranscriptsDisabled, VideoUnavailable):
        MockApi.return_value.fetch.side_effect = exc_class("abc123")
    elif exc_class is NoTranscriptFound:
        MockApi.return_value.fetch.side_effect = exc_class("abc123", ("en",), {})
    else:
        MockApi.return_value.fetch.side_effect = exc_class("boom")

    with pytest.raises(TranscriptError) as exc_info:
        fetch_transcript("abc123")

    assert exc_info.value.code == expected_code


@patch("src.transcript_service.YouTubeTranscriptApi")
def test_error_message_contains_video_id(MockApi):
    MockApi.return_value.fetch.side_effect = VideoUnavailable("abc123")
    with pytest.raises(TranscriptError) as exc_info:
        fetch_transcript("abc123")
    assert "abc123" in str(exc_info.value)


@patch("src.transcript_service.YouTubeTranscriptApi")
def test_original_exception_chained(MockApi):
    original = VideoUnavailable("abc123")
    MockApi.return_value.fetch.side_effect = original
    with pytest.raises(TranscriptError) as exc_info:
        fetch_transcript("abc123")
    assert exc_info.value.__cause__ is original
