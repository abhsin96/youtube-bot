import json
import re
import uuid
from pathlib import Path

import chromadb
import structlog
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = structlog.get_logger(__name__)

_NO_TELEMETRY = ChromaSettings(anonymized_telemetry=False)


def _sanitize_video_id(video_id: str) -> str:
    """Replace every non-alphanumeric, non-underscore character with '_'."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", video_id)


def _collection_name(video_id: str) -> str:
    # Chroma names: 3-63 chars, alphanumeric + hyphens/underscores.
    # We use only alphanumeric + underscore; prefix guarantees >= 7 chars.
    return f"video_{_sanitize_video_id(video_id)}"[:63]


def _get_client(persist_directory: str | Path) -> chromadb.ClientAPI:
    """Return a PersistentClient with telemetry disabled; create folder if absent."""
    db_path = Path(persist_directory)
    db_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(db_path), settings=_NO_TELEMETRY)


def _make_store(
    video_id: str,
    embedding: Embeddings,
    persist_directory: str | Path,
    *,
    create: bool = True,
) -> Chroma:
    client = _get_client(persist_directory)
    collection_name = _collection_name(video_id)

    # For existing collections, check and potentially recreate with correct metric
    existing_collections = {c.name: c for c in client.list_collections()}

    if collection_name in existing_collections:
        existing_coll = existing_collections[collection_name]
        # Check if it has the wrong distance metric
        config = existing_coll.configuration
        hnsw_space = config.get("hnsw", {}).get("space") if config else None

        if hnsw_space and hnsw_space != "cosine":
            # Delete and recreate with correct metric
            logger.info(
                "recreating collection with cosine metric",
                video_id=video_id,
                old_metric=hnsw_space,
            )
            client.delete_collection(collection_name)
            # Now create new collection with cosine metric
            if create:
                client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
    elif create:
        # Collection doesn't exist, create it with cosine metric
        client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    return Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embedding,
    )


def add_documents(
    video_id: str,
    docs: list[Document],
    embedding: Embeddings,
    persist_directory: str | Path,
    *,
    precomputed_vectors: list[list[float]] | None = None,
) -> None:
    """Embed and persist *docs* into the collection for *video_id*.

    Pass *precomputed_vectors* (parallel list of embedding vectors) to skip the
    embedding API call.  When provided, vectors are inserted directly into the
    Chroma collection, avoiding the double-embedding that would otherwise occur
    when the caller has already run embed_chunks().
    """
    store = _make_store(video_id, embedding, persist_directory)
    if precomputed_vectors is not None:
        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        ids = [str(uuid.uuid4()) for _ in docs]
        store._collection.add(
            ids=ids,
            embeddings=precomputed_vectors,
            documents=texts,
            metadatas=metadatas,
        )
    else:
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
    client = _get_client(persist_directory)
    existing = {c.name for c in client.list_collections()}
    return _collection_name(video_id) in existing


def delete_collection(video_id: str, persist_directory: str | Path) -> None:
    """Delete the Chroma collection for *video_id* (no-op if absent)."""
    client = _get_client(persist_directory)
    name = _collection_name(video_id)
    existing = {c.name for c in client.list_collections()}
    if name in existing:
        client.delete_collection(name)
        logger.info("collection deleted", video_id=video_id)
    else:
        logger.info("collection not found, skipping delete", video_id=video_id)


def recreate_collection_with_correct_metric(video_id: str, persist_directory: str | Path) -> None:
    """Delete and recreate collection to ensure correct distance metric."""
    delete_collection(video_id, persist_directory)
    logger.info("collection recreated with cosine metric", video_id=video_id)


# ---------------------------------------------------------------------------
# Channel metadata  (JSON sidecar alongside the Chroma DB directory)
# ---------------------------------------------------------------------------


def _metadata_path(video_id: str, persist_directory: str | Path) -> Path:
    return Path(persist_directory) / f"{_sanitize_video_id(video_id)}_meta.json"


def save_channel_metadata(video_id: str, persist_directory: str | Path, metadata: dict) -> None:
    """Persist channel metadata (title, channel_name, channel_url) to disk."""
    path = _metadata_path(video_id, persist_directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, ensure_ascii=False))
    logger.info("channel_metadata_saved", video_id=video_id, channel=metadata.get("channel_name"))


def get_channel_metadata(video_id: str, persist_directory: str | Path) -> dict:
    """Return saved channel metadata, or an empty dict if not yet stored."""
    path = _metadata_path(video_id, persist_directory)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        logger.warning("channel_metadata_read_failed", video_id=video_id, error=str(exc))
        return {}
