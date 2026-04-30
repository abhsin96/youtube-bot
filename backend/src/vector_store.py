from pathlib import Path

import chromadb
import structlog
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = structlog.get_logger(__name__)


def _sanitize_video_id(video_id: str) -> str:
    """Replace every non-alphanumeric, non-underscore character with '_'."""
    import re

    return re.sub(r"[^a-zA-Z0-9_]", "_", video_id)


def _collection_name(video_id: str) -> str:
    # Chroma names: 3-63 chars, alphanumeric + hyphens/underscores.
    # We use only alphanumeric + underscore; prefix guarantees >= 7 chars.
    return f"video_{_sanitize_video_id(video_id)}"[:63]


def _make_store(
    video_id: str,
    embedding: Embeddings,
    persist_directory: str | Path,
    *,
    create: bool = True,
) -> Chroma:
    return Chroma(
        collection_name=_collection_name(video_id),
        embedding_function=embedding,
        persist_directory=str(persist_directory),
        create_collection_if_not_exists=create,
    )


def add_documents(
    video_id: str,
    docs: list[Document],
    embedding: Embeddings,
    persist_directory: str | Path,
) -> None:
    """Embed and persist *docs* into the collection for *video_id*."""
    store = _make_store(video_id, embedding, persist_directory)
    store.add_documents(docs)
    logger.info("documents added", video_id=video_id, count=len(docs))


def query(
    video_id: str,
    embedded_query: list[float],
    embedding: Embeddings,
    persist_directory: str | Path,
    k: int = 4,
) -> list[Document]:
    """Return up to *k* Documents nearest to *embedded_query*."""
    store = _make_store(video_id, embedding, persist_directory)
    results = store.similarity_search_by_vector(embedded_query, k=k)
    logger.info("query executed", video_id=video_id, k=k, results=len(results))
    return results


def collection_exists(video_id: str, persist_directory: str | Path) -> bool:
    """Return True if a collection for *video_id* already exists on disk."""
    client = chromadb.PersistentClient(path=str(persist_directory))
    name = _collection_name(video_id)
    existing = {c.name for c in client.list_collections()}
    return name in existing


def delete_collection(video_id: str, persist_directory: str | Path) -> None:
    """Delete the Chroma collection for *video_id* (no-op if absent)."""
    client = chromadb.PersistentClient(path=str(persist_directory))
    name = _collection_name(video_id)
    existing = {c.name for c in client.list_collections()}
    if name in existing:
        client.delete_collection(name)
        logger.info("collection deleted", video_id=video_id)
    else:
        logger.info("collection not found, skipping delete", video_id=video_id)
