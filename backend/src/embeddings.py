from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from src.config import Settings


@lru_cache(maxsize=4)
def _build(model: str, api_key: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=model, openai_api_key=api_key)


def get_embeddings(settings: Settings) -> OpenAIEmbeddings:
    """Return a cached OpenAIEmbeddings instance for the model in *settings*."""
    return _build(settings.embed_model, settings.openai_api_key)
