"""
YouTube video/channel metadata via the oEmbed endpoint — no API key required.

oEmbed returns: title, author_name (channel name), author_url (channel URL).
"""

from __future__ import annotations

import httpx
import structlog

logger = structlog.get_logger(__name__)

_OEMBED_URL = (
    "https://www.youtube.com/oembed" "?url=https://www.youtube.com/watch?v={video_id}&format=json"
)


def fetch_video_metadata(video_id: str) -> dict:
    """Return {title, channel_name, channel_url} for *video_id*.

    Never raises — returns empty strings on any failure so callers don't need
    to guard against network errors.
    """
    url = _OEMBED_URL.format(video_id=video_id)
    try:
        resp = httpx.get(url, timeout=8, follow_redirects=True)
        resp.raise_for_status()
        data = resp.json()
        return {
            "title": data.get("title", ""),
            "channel_name": data.get("author_name", ""),
            "channel_url": data.get("author_url", ""),
        }
    except Exception as exc:
        logger.warning("metadata_fetch_failed", video_id=video_id, error=str(exc))
        return {"title": "", "channel_name": "", "channel_url": ""}
