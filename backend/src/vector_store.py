import re
import uuid

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


def get_chroma_client(host: str, port: int) -> chromadb.ClientAPI:
    """Return an HTTP client connected to the running Chroma server."""
    return chromadb.HttpClient(host=host, port=port, settings=_NO_TELEMETRY)


def _make_store(
    video_id: str,
    embedding: Embeddings,
    client: chromadb.ClientAPI,
    *,
    create: bool = True,
) -> Chroma:
    collection_name = _collection_name(video_id)
    if create:
        client.get_or_create_collection(
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
    client: chromadb.ClientAPI,
    *,
    precomputed_vectors: list[list[float]] | None = None,
) -> None:
    """Embed and persist *docs* into the collection for *video_id*."""
    store = _make_store(video_id, embedding, client)
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
    client: chromadb.ClientAPI,
    k: int = 4,
) -> list[Document]:
    """Return up to *k* Documents nearest to *embedded_query*."""
    store = _make_store(video_id, embedding, client)
    results = store.similarity_search_by_vector(embedded_query, k=k)
    logger.info("query executed", video_id=video_id, k=k, results=len(results))
    return results


def collection_exists(video_id: str, client: chromadb.ClientAPI) -> bool:
    """Return True if a collection for *video_id* already exists."""
    existing = {c.name for c in client.list_collections()}
    return _collection_name(video_id) in existing


def delete_collection(video_id: str, client: chromadb.ClientAPI) -> None:
    """Delete the Chroma collection for *video_id* (no-op if absent)."""
    name = _collection_name(video_id)
    existing = {c.name for c in client.list_collections()}
    if name in existing:
        client.delete_collection(name)
        logger.info("collection deleted", video_id=video_id)
    else:
        logger.info("collection not found, skipping delete", video_id=video_id)


# ---------------------------------------------------------------------------
# Channel metadata  (stored in Chroma collection metadata)
# ---------------------------------------------------------------------------


def save_channel_metadata(video_id: str, client: chromadb.ClientAPI, metadata: dict) -> None:
    """Persist channel metadata (title, channel_name, channel_url) in the Chroma collection."""
    name = _collection_name(video_id)
    collection = client.get_collection(name)
    existing = collection.metadata or {}
    # hnsw:* keys are set at creation time; modify() rejects them even unchanged
    non_hnsw = {k: v for k, v in existing.items() if not k.startswith("hnsw:")}
    collection.modify(metadata={**non_hnsw, **metadata})
    logger.info("channel_metadata_saved", video_id=video_id, channel=metadata.get("channel_name"))


def get_channel_metadata(video_id: str, client: chromadb.ClientAPI) -> dict:
    """Return saved channel metadata, or an empty dict if not yet stored."""
    name = _collection_name(video_id)
    try:
        collection = client.get_collection(name)
        meta = collection.metadata or {}
        return {k: v for k, v in meta.items() if not k.startswith("hnsw:")}
    except Exception as exc:
        logger.warning("channel_metadata_read_failed", video_id=video_id, error=str(exc))
        return {}
