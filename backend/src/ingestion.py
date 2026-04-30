from dataclasses import dataclass

import structlog
from langchain_core.embeddings import Embeddings

from src.chunker import chunk_documents
from src.document_converter import segments_to_documents
from src.transcript_service import fetch_transcript
from src.vector_store import add_documents, collection_exists

logger = structlog.get_logger(__name__)


@dataclass
class IngestionResult:
    video_id: str
    segments: int
    chunks: int
    already_existed: bool


def ingest_video(
    video_id: str,
    embeddings: Embeddings,
    vector_db_path,
    *,
    force: bool = False,
) -> IngestionResult:
    """Full pipeline: transcript → documents → chunks → embed → store.

    Skips ingestion if the collection already exists unless *force* is True.
    Raises TranscriptError on transcript fetch failures.
    """
    if not force and collection_exists(video_id, vector_db_path):
        logger.info("collection already exists, skipping", video_id=video_id)
        return IngestionResult(video_id=video_id, segments=0, chunks=0, already_existed=True)

    segments = fetch_transcript(video_id)
    docs = segments_to_documents(video_id, segments)
    chunks = chunk_documents(docs)
    add_documents(video_id, chunks, embeddings, vector_db_path)

    logger.info(
        "ingestion complete",
        video_id=video_id,
        segments=len(segments),
        chunks=len(chunks),
    )
    return IngestionResult(
        video_id=video_id,
        segments=len(segments),
        chunks=len(chunks),
        already_existed=False,
    )
