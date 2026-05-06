"""
Web search helper using DuckDuckGo (no API key required).

Exposes a single ``search(query)`` function used by the LangChain tool
defined in ask_graph.py.  The LLM constructs the query; this module just
executes it and returns formatted plain-text snippets.
"""

from __future__ import annotations

import structlog
from ddgs import DDGS

logger = structlog.get_logger(__name__)

_MAX_RESULTS = 4
_SNIPPET_MAX_CHARS = 400


def search(query: str) -> str:
    """Run *query* through DuckDuckGo and return formatted plain-text snippets.

    Returns an empty string on failure so callers can check ``if result:``.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=_MAX_RESULTS))
    except Exception as exc:
        logger.warning("web_search_failed", query=query, error=str(exc))
        return ""

    if not results:
        logger.info("web_search_no_results", query=query)
        return ""

    lines = []
    for r in results:
        title = (r.get("title") or "").strip()
        body = (r.get("body") or "")[:_SNIPPET_MAX_CHARS].strip()
        href = r.get("href") or ""
        if title or body:
            lines.append(f"- {title}: {body}" + (f" ({href})" if href else ""))

    logger.info("web_search_done", query=query, snippets=len(lines))
    return "\n".join(lines)
