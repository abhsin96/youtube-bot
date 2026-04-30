"""
LangGraph ingestion pipeline.

Graph shape:
    fetch_transcript → convert_documents → chunk_documents → embed_chunks → store → END
    Each node writes its output into IngestState; any exception sets status="error"
    and short-circuits to END via the error edge on every node.

State contract
──────────────
video_id         str                       YouTube video ID (immutable input)
segments         list[TranscriptSegment]   raw transcript segments from youtube-transcript-api
chunks           list[Document]            chunked LangChain Documents with timestamp metadata
embedded_chunks  list[tuple[Document, list[float]]]  (doc, vector) pairs
status           Literal["pending", "running", "done", "error", "skipped"]
error            str | None                exception message if status == "error"

Allowed status transitions
──────────────────────────
  pending → running  (on graph entry)
  running → done     (on successful store)
  running → error    (on any node exception)
  pending → skipped  (collection already exists and force=False)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, TypedDict

import structlog
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.transcript_service import TranscriptSegment  # noqa: E402

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Status literal — explicit set keeps mypy and LangGraph schema checks honest
# ---------------------------------------------------------------------------

IngestStatus = Literal["pending", "running", "done", "error", "skipped"]

# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


class IngestState(TypedDict):
    """Shared state threaded through every node of the ingestion graph.

    Fields are populated progressively as each pipeline stage completes.
    Nodes MUST NOT mutate fields owned by earlier nodes — return a fresh dict
    with only the keys they produce.
    """

    # ── inputs (set once by the caller, never modified by nodes) ──────────
    video_id: str
    """YouTube video ID, e.g. 'dQw4w9WgXcQ'."""

    force: bool
    """Re-ingest even if a collection already exists when True."""

    embeddings: Embeddings
    """Embedding model instance; passed through without modification."""

    vector_db_path: str | Path
    """Directory path for the Chroma persistent store."""

    # ── pipeline outputs (written by exactly one node each) ───────────────
    segments: list[TranscriptSegment]
    """Raw transcript segments from youtube-transcript-api.
    Populated by: fetch_transcript_node."""

    chunks: list[Document]
    """Chunked LangChain Documents with video_id / start_ts / end_ts / chunk_id metadata.
    Populated by: chunk_documents_node."""

    embedded_chunks: list[tuple[Document, list[float]]]
    """(Document, embedding-vector) pairs ready for storage.
    Populated by: embed_chunks_node."""

    # ── control fields ─────────────────────────────────────────────────────
    status: IngestStatus
    """Current pipeline status.  Starts as 'pending'; terminal values are
    'done', 'error', and 'skipped'."""

    error: str | None
    """Human-readable error message when status == 'error', else None."""
