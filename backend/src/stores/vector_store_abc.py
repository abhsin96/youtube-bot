from __future__ import annotations

from typing import Protocol, runtime_checkable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever


@runtime_checkable
class VectorStorePort(Protocol):
    def collection_exists(self, video_id: str) -> bool: ...

    def delete_collection(self, video_id: str) -> None: ...

    def add_documents(
        self,
        video_id: str,
        docs: list[Document],
        embedding: Embeddings,
        *,
        precomputed_vectors: list[list[float]] | None = None,
    ) -> None: ...

    def query(
        self,
        video_id: str,
        embedded_query: list[float],
        embedding: Embeddings,
        k: int = 4,
    ) -> list[Document]: ...

    def save_channel_metadata(self, video_id: str, metadata: dict) -> None: ...

    def get_channel_metadata(self, video_id: str) -> dict: ...

    def build_retriever(
        self,
        video_id: str,
        embeddings: Embeddings,
        *,
        k: int = 4,
        score_threshold: float = 0.0,
    ) -> BaseRetriever: ...
