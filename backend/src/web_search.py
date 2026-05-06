"""
Web search helper for creator/channel information lookup.

Uses DuckDuckGo (no API key) via the duckduckgo-search package.
Returns a plain-text block ready to be injected into the LLM context.
"""

from __future__ import annotations

import structlog
from ddgs import DDGS

logger = structlog.get_logger(__name__)

_MAX_RESULTS = 4
_SNIPPET_MAX_CHARS = 400


def search_creator_info(channel_name: str, question: str) -> str:
    """Search for *channel_name* on the web and return formatted snippets.

    Returns an empty string when the search fails or yields no results so
    callers can safely check ``if creator_info:`` without extra guards.
    """
    query = f"{channel_name} YouTube channel {question}"
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=_MAX_RESULTS))
    except Exception as exc:
        logger.warning("web_search_failed", channel=channel_name, error=str(exc))
        return ""

    if not results:
        logger.info("web_search_no_results", channel=channel_name)
        return ""

    lines = []
    for r in results:
        title = r.get("title", "").strip()
        body = (r.get("body", "") or "")[:_SNIPPET_MAX_CHARS].strip()
        href = r.get("href", "")
        if title or body:
            lines.append(f"- {title}: {body}" + (f" ({href})" if href else ""))

    info = "\n".join(lines)
    logger.info("web_search_done", channel=channel_name, snippets=len(lines))
    return info
