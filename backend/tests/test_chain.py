from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.chain import RAGResult, _format_context, answer_question, build_rag_chain

_DIM = 4
_VIDEO_ID = "vid1"
_QUESTION = "What is this video about?"


def _fake_embeddings():
    emb = MagicMock()
    emb.embed_query.return_value = [0.0] * _DIM
    emb.embed_documents.side_effect = lambda texts: [[0.0] * _DIM for _ in texts]
    return emb


def _doc(text: str, start_ts: float = 0.0) -> Document:
    return Document(
        page_content=text,
        metadata={"start_ts": start_ts, "end_ts": start_ts + 5.0, "chunk_id": "c1"},
    )


def _mock_retriever(docs: list[Document]):
    """Return a mock retriever whose invoke() returns *docs*."""
    r = MagicMock()
    r.invoke.return_value = docs
    return r


# --- _format_context ---


def test_format_context_includes_text():
    docs = [_doc("hello world", start_ts=10.0)]
    ctx = _format_context(docs)
    assert "hello world" in ctx


def test_format_context_includes_timestamp():
    docs = [_doc("hello", start_ts=42.0)]
    ctx = _format_context(docs)
    assert "42.0" in ctx


def test_format_context_multiple_docs_separated():
    docs = [_doc("first", 0.0), _doc("second", 5.0)]
    ctx = _format_context(docs)
    assert "first" in ctx
    assert "second" in ctx


def test_format_context_empty():
    assert _format_context([]) == ""


# --- build_rag_chain ---


def test_build_rag_chain_returns_callable():
    chain = build_rag_chain("gpt-4o-mini", "sk-test")
    assert callable(chain.invoke)


# --- answer_question ---


@pytest.fixture()
def tmp_db(tmp_path):
    return tmp_path / "chroma"


def test_answer_question_returns_rag_result(tmp_db):
    docs = [_doc("LangChain is a framework", start_ts=1.0)]

    with (
        patch("src.chain.build_retriever", return_value=_mock_retriever(docs)),
        patch("src.chain.build_rag_chain") as mock_chain_fn,
    ):
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "LangChain is great."
        mock_chain_fn.return_value = mock_chain

        result = answer_question(
            _VIDEO_ID, _QUESTION, _fake_embeddings(), tmp_db, "gpt-4o-mini", "sk-test"
        )

    assert isinstance(result, RAGResult)
    assert result.answer == "LangChain is great."
    assert result.sources == docs


def test_answer_question_no_sources_returns_fallback(tmp_db):
    with patch("src.chain.build_retriever", return_value=_mock_retriever([])):
        result = answer_question(
            _VIDEO_ID, _QUESTION, _fake_embeddings(), tmp_db, "gpt-4o-mini", "sk-test"
        )
    assert "couldn't find" in result.answer.lower()
    assert result.sources == []


def test_answer_question_passes_k_to_retriever(tmp_db):
    with patch("src.chain.build_retriever", return_value=_mock_retriever([])) as mock_build:
        answer_question(
            _VIDEO_ID, _QUESTION, _fake_embeddings(), tmp_db, "gpt-4o-mini", "sk-test", k=3
        )
    assert mock_build.call_args.kwargs["k"] == 3


def test_answer_question_passes_question_to_retriever(tmp_db):
    retriever = _mock_retriever([])
    with patch("src.chain.build_retriever", return_value=retriever):
        answer_question(_VIDEO_ID, _QUESTION, _fake_embeddings(), tmp_db, "gpt-4o-mini", "sk-test")
    retriever.invoke.assert_called_once_with(_QUESTION)


def test_answer_question_context_passed_to_chain(tmp_db):
    docs = [_doc("unique phrase xyz", start_ts=7.0)]

    with (
        patch("src.chain.build_retriever", return_value=_mock_retriever(docs)),
        patch("src.chain.build_rag_chain") as mock_chain_fn,
    ):
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "ok"
        mock_chain_fn.return_value = mock_chain

        answer_question(_VIDEO_ID, _QUESTION, _fake_embeddings(), tmp_db, "gpt-4o-mini", "sk-test")

    invoke_kwargs = mock_chain.invoke.call_args[0][0]
    assert "unique phrase xyz" in invoke_kwargs["context"]
    assert invoke_kwargs["question"] == _QUESTION


# --- score_threshold ---


def test_answer_question_default_score_threshold_zero(tmp_db):
    with patch("src.chain.build_retriever", return_value=_mock_retriever([])) as mock_build:
        answer_question(_VIDEO_ID, _QUESTION, _fake_embeddings(), tmp_db, "gpt-4o-mini", "sk-test")
    assert mock_build.call_args.kwargs["score_threshold"] == 0.0


def test_answer_question_passes_score_threshold_to_retriever(tmp_db):
    with patch("src.chain.build_retriever", return_value=_mock_retriever([])) as mock_build:
        answer_question(
            _VIDEO_ID,
            _QUESTION,
            _fake_embeddings(),
            tmp_db,
            "gpt-4o-mini",
            "sk-test",
            score_threshold=0.25,
        )
    assert mock_build.call_args.kwargs["score_threshold"] == 0.25


def test_answer_question_threshold_filters_propagate(tmp_db):
    """Docs that don't meet the threshold come back empty from the retriever."""
    with (
        patch("src.chain.build_retriever", return_value=_mock_retriever([])),
        patch("src.chain.build_rag_chain"),
    ):
        result = answer_question(
            _VIDEO_ID,
            _QUESTION,
            _fake_embeddings(),
            tmp_db,
            "gpt-4o-mini",
            "sk-test",
            score_threshold=0.9,
        )
    assert result.sources == []
    assert "couldn't find" in result.answer.lower()
