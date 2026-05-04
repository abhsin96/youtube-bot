"""
VideoRetriever — LangChain Runnable retriever backed by a per-video Chroma collection.

Wraps Chroma.similarity_search_with_score (via similarity_search_with_relevance_scores,
which normalises raw distances to a [0, 1] relevance score) and filters results against
a configurable threshold.  Because it subclasses BaseRetriever it is a full Runnable
and composes with any LCEL chain via the | operator.

    retriever = build_retriever(video_id, embeddings, vector_db_path, k=5, score_threshold=0.25)
    chain = retriever | format_docs | prompt | llm | StrOutputParser()
"""

from pathlib import Path

import structlog
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

from src.vector_store import _make_store

logger = structlog.get_logger(__name__)


class VideoRetriever(BaseRetriever):
    """Retrieve the top-k most relevant chunks from a video's Chroma collection.

    `invoke(query_string)` returns a list[Document] — identical to the Runnable
    interface expected by LCEL graphs.

    Scores are relevance scores in [0, 1] (1 = most similar); only documents
    at or above *score_threshold* are returned.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    video_id: str
    embeddings: Embeddings
    vector_db_path: str | Path
    k: int = Field(default=4, ge=1)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        store = _make_store(self.video_id, self.embeddings, self.vector_db_path, create=False)

        # similarity_search_with_relevance_scores wraps similarity_search_with_score
        # and normalises raw Chroma distances to a [0, 1] relevance score.
        pairs: list[tuple[Document, float]] = store.similarity_search_with_relevance_scores(
            query, k=self.k
        )

        # Sort highest relevance first so callers get a stable, predictable order
        # regardless of the underlying store's iteration order.
        pairs.sort(key=lambda p: p[1], reverse=True)

        docs = [doc for doc, score in pairs if score >= self.score_threshold]

        logger.info(
            "retriever query",
            video_id=self.video_id,
            k=self.k,
            threshold=self.score_threshold,
            candidates=len(pairs),
            returned=len(docs),
        )
        return docs


def build_retriever(
    video_id: str,
    embeddings: Embeddings,
    vector_db_path: str | Path,
    *,
    k: int = 4,
    score_threshold: float = 0.0,
) -> VideoRetriever:
    """Factory that returns a configured VideoRetriever."""
    return VideoRetriever(
        video_id=video_id,
        embeddings=embeddings,
        vector_db_path=vector_db_path,
        k=k,
        score_threshold=score_threshold,
    )
