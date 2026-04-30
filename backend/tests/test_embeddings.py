from unittest.mock import patch

from langchain_openai import OpenAIEmbeddings

from src.config import Settings
from src.embeddings import get_embeddings


def _settings(**overrides) -> Settings:
    overrides.setdefault("openai_api_key", "sk-test")
    return Settings(_env_file=None, **overrides)


# --- model and key wired correctly ---


def test_uses_embed_model_from_settings():
    s = _settings(embed_model="text-embedding-3-large")
    emb = get_embeddings(s)
    assert emb.model == "text-embedding-3-large"


def test_uses_default_embed_model():
    s = _settings()
    emb = get_embeddings(s)
    assert emb.model == "text-embedding-3-small"


def test_api_key_injected():
    s = _settings(openai_api_key="sk-secret-key")
    emb = get_embeddings(s)
    assert emb.openai_api_key.get_secret_value() == "sk-secret-key"


def test_returns_openai_embeddings_instance():
    assert isinstance(get_embeddings(_settings()), OpenAIEmbeddings)


# --- caching ---


def test_same_model_and_key_returns_same_instance():
    s = _settings(embed_model="text-embedding-3-small", openai_api_key="sk-abc")
    assert get_embeddings(s) is get_embeddings(s)


def test_different_model_returns_different_instance():
    s1 = _settings(embed_model="text-embedding-3-small")
    s2 = _settings(embed_model="text-embedding-3-large")
    assert get_embeddings(s1) is not get_embeddings(s2)


def test_different_key_returns_different_instance():
    s1 = _settings(openai_api_key="sk-key-one")
    s2 = _settings(openai_api_key="sk-key-two")
    assert get_embeddings(s1) is not get_embeddings(s2)


# --- no live network calls at construction time ---


def test_construction_does_not_call_openai():
    with patch("httpx.Client.send") as mock_send:
        get_embeddings(_settings())
    mock_send.assert_not_called()
