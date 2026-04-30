from enum import StrEnum
from threading import Lock
from typing import TypedDict

import structlog
from cachetools import TTLCache, cached
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
)

logger = structlog.get_logger(__name__)

_CACHE: TTLCache = TTLCache(maxsize=50, ttl=3600)
_LOCK = Lock()


class TranscriptSegment(TypedDict):
    text: str
    start: float
    duration: float


class ErrorCode(StrEnum):
    TRANSCRIPT_DISABLED = "TRANSCRIPT_DISABLED"
    VIDEO_NOT_FOUND = "VIDEO_NOT_FOUND"
    RATE_LIMITED = "RATE_LIMITED"
    UNKNOWN = "UNKNOWN"


class TranscriptError(Exception):
    def __init__(self, message: str, code: ErrorCode) -> None:
        super().__init__(message)
        self.code = code


@cached(cache=_CACHE, lock=_LOCK, key=lambda video_id, languages=("en",): (video_id, languages))
def fetch_transcript(
    video_id: str,
    languages: tuple[str, ...] = ("en",),
) -> list[TranscriptSegment]:
    log = logger.bind(video_id=video_id)
    try:
        api = YouTubeTranscriptApi()
        fetched = api.fetch(video_id, languages=languages)
        segments: list[TranscriptSegment] = [
            {"text": s.text, "start": s.start, "duration": s.duration} for s in fetched
        ]
        log.info("transcript fetched", segment_count=len(segments))
        return segments
    except VideoUnavailable as exc:
        raise TranscriptError(f"Video not found: {video_id}", ErrorCode.VIDEO_NOT_FOUND) from exc
    except TranscriptsDisabled as exc:
        raise TranscriptError(
            f"Transcripts are disabled for video: {video_id}", ErrorCode.TRANSCRIPT_DISABLED
        ) from exc
    except NoTranscriptFound as exc:
        raise TranscriptError(
            f"No transcript found for video: {video_id}", ErrorCode.TRANSCRIPT_DISABLED
        ) from exc
    except (IpBlocked, RequestBlocked) as exc:
        raise TranscriptError(
            f"Rate limited fetching transcript for video: {video_id}", ErrorCode.RATE_LIMITED
        ) from exc
    except Exception as exc:
        raise TranscriptError(
            f"Unexpected error fetching transcript for video: {video_id}", ErrorCode.UNKNOWN
        ) from exc
