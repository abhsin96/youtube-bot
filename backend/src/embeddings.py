from functools import lru_cache

import structlog
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from openai import APIConnectionError, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import Settings

logger = structlog.get_logger(__name__)

_RETRY = dict(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    stop=stop_after_attempt(3),
    reraise=True,
)


@lru_cache(maxsize=4)
def _build(model: str, api_key: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=model, openai_api_key=api_key)


def get_embeddings(settings: Settings) -> OpenAIEmbeddings:
    """Return a cached OpenAIEmbeddings instance for the model in *settings*."""
    return _build(settings.embed_model, settings.openai_api_key)


@retry(**_RETRY)
def _call_embed(embeddings: Embeddings, texts: list[str]) -> list[list[float]]:
    return embeddings.embed_documents(texts)


def embed_chunks(
    chunks: list[Document],
    embeddings: Embeddings,
    batch_size: int = 100,
) -> list[tuple[Document, list[float]]]:
    """Embed *chunks* in batches; return (Document, vector) pairs."""
    results: list[tuple[Document, list[float]]] = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vectors = _call_embed(embeddings, [doc.page_content for doc in batch])
        results.extend(zip(batch, vectors, strict=True))
        logger.debug("batch embedded", offset=i, size=len(batch))
    return results
