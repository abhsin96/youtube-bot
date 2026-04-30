from typing import TypedDict

import structlog
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

logger = structlog.get_logger(__name__)


class TranscriptSegment(TypedDict):
    text: str
    start: float
    duration: float


class TranscriptError(Exception):
    pass


class TranscriptUnavailableError(TranscriptError):
    pass


class VideoNotFoundError(TranscriptError):
    pass


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
        raise VideoNotFoundError(f"Video not found: {video_id}") from exc
    except (TranscriptsDisabled, NoTranscriptFound) as exc:
        raise TranscriptUnavailableError(f"No transcript available for video: {video_id}") from exc
