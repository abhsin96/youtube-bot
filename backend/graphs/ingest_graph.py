"""
LangGraph ingestion pipeline.

Graph shape:
    idempotency_check_node
        │ "run"                     │ "skip"
        ▼                           ▼
    fetch_transcript_node          END
        │ ok / error
        ▼
    chunk_node
        │ ok / error
        ▼
    embed_node
        │ ok / error
        ▼
    store_node
        │
        ▼
       END

Each node returns only the keys it owns.  On any exception the node sets
status="error" and error=<message>; the router then sends execution to END.

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
  pending → skipped  idempotency_check_node (collection exists, force=False)
  pending → running  idempotency_check_node (needs ingestion)
  running → done     store_node (success)
  running → error    any node (exception)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, TypedDict

import structlog
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunker import chunk_documents as _chunk_documents  # noqa: E402
from src.document_converter import segments_to_documents  # noqa: E402
from src.embeddings import embed_chunks  # noqa: E402
from src.transcript_service import TranscriptSegment, fetch_transcript  # noqa: E402
from src.vector_store import add_documents, collection_exists  # noqa: E402

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _error(message: str) -> dict:
    """Shorthand for the error terminal state."""
    return {"status": "error", "error": message}


# ---------------------------------------------------------------------------
# Nodes  (pure functions: IngestState → dict)
# ---------------------------------------------------------------------------


def idempotency_check_node(state: IngestState) -> dict:
    """Skip ingestion if the collection already exists and force is False.

    Returns:
        status="skipped"  — collection present, force=False
        status="running"  — needs ingestion (collection absent OR force=True)
    """
    try:
        exists = collection_exists(state["video_id"], state["vector_db_path"])
    except Exception as exc:  # storage unavailable
        logger.error("idempotency check failed", video_id=state["video_id"], error=str(exc))
        return _error(f"idempotency check failed: {exc}")

    if exists and not state["force"]:
        logger.info("collection exists — skipping", video_id=state["video_id"])
        return {"status": "skipped"}

    logger.info("starting ingestion", video_id=state["video_id"], force=state["force"])
    return {"status": "running"}


def fetch_transcript_node(state: IngestState) -> dict:
    """Fetch raw transcript segments for the video.

    Owns: segments
    """
    try:
        segments = fetch_transcript(state["video_id"])
    except Exception as exc:
        logger.error("transcript fetch failed", video_id=state["video_id"], error=str(exc))
        return _error(str(exc))

    logger.info("transcript fetched", video_id=state["video_id"], segments=len(segments))
    return {"segments": segments}


def chunk_node(state: IngestState) -> dict:
    """Convert segments → Documents, then chunk with timestamp preservation.

    Owns: chunks
    """
    try:
        docs = segments_to_documents(state["video_id"], state["segments"])
        chunks = _chunk_documents(docs)
    except Exception as exc:
        logger.error("chunking failed", video_id=state["video_id"], error=str(exc))
        return _error(str(exc))

    logger.info("chunks produced", video_id=state["video_id"], chunks=len(chunks))
    return {"chunks": chunks}


def embed_node(state: IngestState) -> dict:
    """Embed each chunk; return (Document, vector) pairs.

    Owns: embedded_chunks
    """
    try:
        pairs = embed_chunks(state["chunks"], state["embeddings"])
    except Exception as exc:
        logger.error("embedding failed", video_id=state["video_id"], error=str(exc))
        return _error(str(exc))

    logger.info("chunks embedded", video_id=state["video_id"], count=len(pairs))
    return {"embedded_chunks": pairs}


def store_node(state: IngestState) -> dict:
    """Persist chunks into the vector store.

    Uses the plain chunk list (not embedded_chunks) because add_documents
    re-embeds internally via the Chroma integration — the embedded_chunks
    field is available for downstream consumers (e.g. returning vectors to
    the caller) but Chroma manages its own index.

    Owns: status (→ "done")
    """
    try:
        add_documents(
            state["video_id"],
            state["chunks"],
            state["embeddings"],
            state["vector_db_path"],
        )
    except Exception as exc:
        logger.error("store failed", video_id=state["video_id"], error=str(exc))
        return _error(str(exc))

    logger.info(
        "ingestion done",
        video_id=state["video_id"],
        chunks=len(state["chunks"]),
    )
    return {"status": "done", "error": None}


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------


def route_idempotency(state: IngestState) -> str:
    """Conditional edge after idempotency_check_node.

    skipped → END  (collection already exists; caller reads status="skipped")
    error   → END  (storage unavailable)
    running → "fetch_transcript_node"
    """
    from langgraph.graph import END

    if state["status"] in ("skipped", "error"):
        return END
    return "fetch_transcript_node"


def route_on_error(state: IngestState) -> str:
    """Conditional edge after fetch_transcript_node, chunk_node, embed_node.

    error → END
    *     → next node (caller supplies the target via the edge mapping)
    """
    from langgraph.graph import END

    if state["status"] == "error":
        return END
    return "_ok"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def build_graph():
    """Compile and return the ingestion StateGraph.

    Topology
    ────────
    idempotency_check_node
        ├─ skipped/error → END
        └─ running       → fetch_transcript_node
                               ├─ error → END
                               └─ ok    → chunk_node
                                              ├─ error → END
                                              └─ ok    → embed_node
                                                             ├─ error → END
                                                             └─ ok    → store_node → END
    """
    from langgraph.graph import END, StateGraph

    g = StateGraph(IngestState)

    g.add_node("idempotency_check_node", idempotency_check_node)
    g.add_node("fetch_transcript_node", fetch_transcript_node)
    g.add_node("chunk_node", chunk_node)
    g.add_node("embed_node", embed_node)
    g.add_node("store_node", store_node)

    g.set_entry_point("idempotency_check_node")

    # idempotency → skip to END or proceed to fetch
    g.add_conditional_edges(
        "idempotency_check_node",
        route_idempotency,
        {"fetch_transcript_node": "fetch_transcript_node", END: END},
    )

    # each subsequent node short-circuits to END on error, else advances
    g.add_conditional_edges(
        "fetch_transcript_node",
        route_on_error,
        {"_ok": "chunk_node", END: END},
    )
    g.add_conditional_edges(
        "chunk_node",
        route_on_error,
        {"_ok": "embed_node", END: END},
    )
    g.add_conditional_edges(
        "embed_node",
        route_on_error,
        {"_ok": "store_node", END: END},
    )
    g.add_edge("store_node", END)

    return g.compile()
