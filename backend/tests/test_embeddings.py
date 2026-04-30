from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from openai import APIConnectionError, RateLimitError

from src.config import Settings
from src.embeddings import embed_chunks, get_embeddings

_DIM = 4


def _fake_embedding(dim: int = _DIM):
    emb = MagicMock()
    emb.embed_documents.side_effect = lambda texts: [[float(i)] * dim for i in range(len(texts))]
    return emb


def _docs(n: int) -> list[Document]:
    return [Document(page_content=f"chunk {i}") for i in range(n)]


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


# --- embed_chunks ---


def test_embed_chunks_returns_doc_vector_pairs():
    pairs = embed_chunks(_docs(3), _fake_embedding())
    assert len(pairs) == 3
    assert all(isinstance(doc, Document) for doc, _ in pairs)
    assert all(isinstance(vec, list) for _, vec in pairs)


def test_embed_chunks_vector_length():
    pairs = embed_chunks(_docs(5), _fake_embedding())
    assert all(len(vec) == _DIM for _, vec in pairs)


def test_embed_chunks_preserves_document_identity():
    docs = _docs(3)
    pairs = embed_chunks(docs, _fake_embedding())
    assert [doc for doc, _ in pairs] == docs


def test_embed_chunks_empty_input():
    assert embed_chunks([], _fake_embedding()) == []


def test_embed_chunks_single_batch_calls_embed_documents_once():
    emb = _fake_embedding()
    embed_chunks(_docs(5), emb, batch_size=10)
    assert emb.embed_documents.call_count == 1


def test_embed_chunks_batching_splits_calls():
    emb = _fake_embedding()
    embed_chunks(_docs(7), emb, batch_size=3)
    # 7 docs / batch_size 3 → 3 batches (3+3+1)
    assert emb.embed_documents.call_count == 3


def test_embed_chunks_exact_batch_boundary():
    emb = _fake_embedding()
    embed_chunks(_docs(6), emb, batch_size=3)
    assert emb.embed_documents.call_count == 2


def test_embed_chunks_batch_size_one():
    emb = _fake_embedding()
    embed_chunks(_docs(4), emb, batch_size=1)
    assert emb.embed_documents.call_count == 4


def test_embed_chunks_vectors_match_order():
    # _fake_embedding returns [float(i)]*dim for the i-th text in the batch
    pairs = embed_chunks(_docs(3), _fake_embedding())
    for idx, (_, vec) in enumerate(pairs):
        assert vec == [float(idx)] * _DIM


# --- retry behaviour ---

_GOOD_VECTORS = [[1.0, 2.0], [3.0, 4.0]]


def _rate_limit_error() -> RateLimitError:
    return RateLimitError("rate limited", response=MagicMock(), body={})


def _connection_error() -> APIConnectionError:
    return APIConnectionError(request=MagicMock())


@patch("time.sleep")
def test_retry_succeeds_after_two_rate_limit_errors(mock_sleep):
    emb = MagicMock()
    emb.embed_documents.side_effect = [
        _rate_limit_error(),
        _rate_limit_error(),
        _GOOD_VECTORS,
    ]
    pairs = embed_chunks([Document(page_content="a"), Document(page_content="b")], emb)
    assert emb.embed_documents.call_count == 3
    assert [vec for _, vec in pairs] == _GOOD_VECTORS


@patch("time.sleep")
def test_retry_succeeds_after_one_connection_error(mock_sleep):
    emb = MagicMock()
    emb.embed_documents.side_effect = [_connection_error(), _GOOD_VECTORS]
    pairs = embed_chunks([Document(page_content="a"), Document(page_content="b")], emb)
    assert emb.embed_documents.call_count == 2
    assert [vec for _, vec in pairs] == _GOOD_VECTORS


@patch("time.sleep")
def test_retry_reraises_after_three_failures(mock_sleep):
    emb = MagicMock()
    emb.embed_documents.side_effect = [
        _rate_limit_error(),
        _rate_limit_error(),
        _rate_limit_error(),
    ]
    with pytest.raises(RateLimitError):
        embed_chunks([Document(page_content="x")], emb)
    assert emb.embed_documents.call_count == 3


@patch("time.sleep")
def test_non_transient_error_not_retried(mock_sleep):
    emb = MagicMock()
    emb.embed_documents.side_effect = ValueError("bad input")
    with pytest.raises(ValueError, match="bad input"):
        embed_chunks([Document(page_content="x")], emb)
    assert emb.embed_documents.call_count == 1


@patch("time.sleep")
def test_retry_result_correct_after_two_failures(mock_sleep):
    expected = [[0.1, 0.2]]
    emb = MagicMock()
    emb.embed_documents.side_effect = [_rate_limit_error(), _rate_limit_error(), expected]
    pairs = embed_chunks([Document(page_content="hello")], emb)
    assert pairs[0][1] == [0.1, 0.2]
