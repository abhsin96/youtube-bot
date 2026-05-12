from __future__ import annotations

import chromadb
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

import src.retriever as _retriever_mod
import src.vector_store as _vs_mod


class ChromaVectorStore:
    """Adapter that wraps module-level vector_store functions behind a single object.

    Satisfies VectorStorePort structurally (no inheritance required).
    Uses module-level attribute access (not local aliases) so that
    patch("src.vector_store.*") intercepts calls correctly in tests.
    """

    def __init__(self, client: chromadb.ClientAPI) -> None:
        self._client = client

    def collection_exists(self, video_id: str) -> bool:
        return _vs_mod.collection_exists(video_id, self._client)

    def delete_collection(self, video_id: str) -> None:
        _vs_mod.delete_collection(video_id, self._client)

    def add_documents(
        self,
        video_id: str,
        docs: list[Document],
        embedding: Embeddings,
        *,
        precomputed_vectors: list[list[float]] | None = None,
    ) -> None:
        _vs_mod.add_documents(
            video_id, docs, embedding, self._client, precomputed_vectors=precomputed_vectors
        )

    def query(
        self,
        video_id: str,
        embedded_query: list[float],
        embedding: Embeddings,
        k: int = 4,
    ) -> list[Document]:
        return _vs_mod.query(video_id, embedded_query, embedding, self._client, k=k)

    def save_channel_metadata(self, video_id: str, metadata: dict) -> None:
        _vs_mod.save_channel_metadata(video_id, self._client, metadata)

    def get_channel_metadata(self, video_id: str) -> dict:
        return _vs_mod.get_channel_metadata(video_id, self._client)

    def build_retriever(
        self,
        video_id: str,
        embeddings: Embeddings,
        *,
        k: int = 4,
        score_threshold: float = 0.0,
    ) -> BaseRetriever:
        return _retriever_mod.build_retriever(
            video_id, embeddings, self._client, k=k, score_threshold=score_threshold
        )
