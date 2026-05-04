from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda

from src.retriever import VideoRetriever, build_retriever

_VIDEO_ID = "vid1"
_DB_PATH = "/tmp/db"

_DOCS = [
    Document(page_content="first chunk", metadata={"start_ts": 0.0, "chunk_id": "c1"}),
    Document(page_content="second chunk", metadata={"start_ts": 5.0, "chunk_id": "c2"}),
    Document(page_content="third chunk", metadata={"start_ts": 10.0, "chunk_id": "c3"}),
]


class _FakeEmbeddings(Embeddings):
    """Minimal concrete Embeddings subclass — passes Pydantic isinstance validation."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 4 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0] * 4


def _fake_embeddings() -> _FakeEmbeddings:
    return _FakeEmbeddings()


def _mock_store(pairs: list[tuple[Document, float]]):
    """Return a mock Chroma store whose similarity_search_with_relevance_scores returns *pairs*."""
    store = MagicMock()
    store.similarity_search_with_relevance_scores.return_value = pairs
    return store


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_build_retriever_returns_video_retriever():
    r = build_retriever(_VIDEO_ID, _fake_embeddings(), _DB_PATH)
    assert isinstance(r, VideoRetriever)


def test_build_retriever_sets_video_id():
    r = build_retriever(_VIDEO_ID, _fake_embeddings(), _DB_PATH)
    assert r.video_id == _VIDEO_ID


def test_build_retriever_default_k():
    r = build_retriever(_VIDEO_ID, _fake_embeddings(), _DB_PATH)
    assert r.k == 4


def test_build_retriever_custom_k():
    r = build_retriever(_VIDEO_ID, _fake_embeddings(), _DB_PATH, k=10)
    assert r.k == 10


def test_build_retriever_default_score_threshold():
    r = build_retriever(_VIDEO_ID, _fake_embeddings(), _DB_PATH)
    assert r.score_threshold == 0.0


def test_build_retriever_custom_score_threshold():
    r = build_retriever(_VIDEO_ID, _fake_embeddings(), _DB_PATH, score_threshold=0.25)
    assert r.score_threshold == 0.25


# ---------------------------------------------------------------------------
# _get_relevant_documents / invoke
# ---------------------------------------------------------------------------


def test_invoke_returns_list_of_documents():
    pairs = [(_DOCS[0], 0.9), (_DOCS[1], 0.8)]
    with patch("src.retriever._make_store", return_value=_mock_store(pairs)):
        r = build_retriever(_VIDEO_ID, _fake_embeddings(), _DB_PATH)
        result = r.invoke("what is this about?")
    assert result == [_DOCS[0], _DOCS[1]]


def test_invoke_returns_empty_when_store_empty():
    with patch("src.retriever._make_store", return_value=_mock_store([])):
        r = build_retriever(_VIDEO_ID, _fake_embeddings(), _DB_PATH)
        result = r.invoke("anything")
    assert result == []


def test_k_passed_to_similarity_search():
    store = _mock_store([(_DOCS[0], 0.9)])
    with patch("src.retriever._make_store", return_value=store):
        r = build_retriever(_VIDEO_ID, _fake_embeddings(), _DB_PATH, k=7)
        r.invoke("query")
    store.similarity_search_with_relevance_scores.assert_called_once_with("query", k=7)


def test_query_string_passed_to_similarity_search():
    store = _mock_store([])
    with patch("src.retriever._make_store", return_value=store):
        r = build_retriever(_VIDEO_ID, _fake_embeddings(), _DB_PATH)
        r.invoke("specific question text")
    store.similarity_search_with_relevance_scores.assert_called_once_with(
        "specific question text", k=r.k
    )


# ---------------------------------------------------------------------------
# Score threshold filtering
# ---------------------------------------------------------------------------


def test_threshold_filters_low_score_docs():
    pairs = [(_DOCS[0], 0.9), (_DOCS[1], 0.3), (_DOCS[2], 0.1)]
    with patch("src.retriever._make_store", return_value=_mock_store(pairs)):
        r = build_retriever(_VIDEO_ID, _fake_embeddings(), _DB_PATH, score_threshold=0.5)
        result = r.invoke("query")
    assert result == [_DOCS[0]]


def test_threshold_exact_match_is_included():
    pairs = [(_DOCS[0], 0.25)]
    with patch("src.retriever._make_store", return_value=_mock_store(pairs)):
        r = build_retriever(_VIDEO_ID, _fake_embeddings(), _DB_PATH, score_threshold=0.25)
        result = r.invoke("query")
    assert result == [_DOCS[0]]


def test_threshold_zero_returns_all_docs():
    pairs = [(_DOCS[0], 0.9), (_DOCS[1], 0.0)]
    with patch("src.retriever._make_store", return_value=_mock_store(pairs)):
        r = build_retriever(_VIDEO_ID, _fake_embeddings(), _DB_PATH, score_threshold=0.0)
        result = r.invoke("query")
    assert len(result) == 2


def test_threshold_one_returns_only_perfect_matches():
    pairs = [(_DOCS[0], 1.0), (_DOCS[1], 0.99)]
    with patch("src.retriever._make_store", return_value=_mock_store(pairs)):
        r = build_retriever(_VIDEO_ID, _fake_embeddings(), _DB_PATH, score_threshold=1.0)
        result = r.invoke("query")
    assert result == [_DOCS[0]]


def test_threshold_all_filtered_returns_empty():
    pairs = [(_DOCS[0], 0.1), (_DOCS[1], 0.2)]
    with patch("src.retriever._make_store", return_value=_mock_store(pairs)):
        r = build_retriever(_VIDEO_ID, _fake_embeddings(), _DB_PATH, score_threshold=0.5)
        result = r.invoke("query")
    assert result == []


# ---------------------------------------------------------------------------
# Runnable interface
# ---------------------------------------------------------------------------


def test_retriever_is_runnable():
    """BaseRetriever inherits RunnableSerializable; invoke must exist."""
    r = build_retriever(_VIDEO_ID, _fake_embeddings(), _DB_PATH)
    assert callable(r.invoke)


def test_retriever_composes_with_lcel_pipe():
    """VideoRetriever | RunnableLambda should form a valid chain."""
    pairs = [(_DOCS[0], 0.9)]
    format_fn = RunnableLambda(lambda docs: " ".join(d.page_content for d in docs))

    with patch("src.retriever._make_store", return_value=_mock_store(pairs)):
        r = build_retriever(_VIDEO_ID, _fake_embeddings(), _DB_PATH)
        chain = r | format_fn
        result = chain.invoke("test question")

    assert result == "first chunk"


def test_retriever_batch_returns_list_per_query():
    pairs = [(_DOCS[0], 0.9)]
    with patch("src.retriever._make_store", return_value=_mock_store(pairs)):
        r = build_retriever(_VIDEO_ID, _fake_embeddings(), _DB_PATH)
        results = r.batch(["q1", "q2"])
    assert len(results) == 2
    assert all(isinstance(res, list) for res in results)


# ---------------------------------------------------------------------------
# _make_store wired correctly
# ---------------------------------------------------------------------------


def test_make_store_called_with_video_id_and_path():
    with patch("src.retriever._make_store", return_value=_mock_store([])) as mock_make:
        r = build_retriever(_VIDEO_ID, _fake_embeddings(), _DB_PATH)
        r.invoke("q")
    args = mock_make.call_args[0]
    assert args[0] == _VIDEO_ID
    assert args[2] == _DB_PATH


def test_make_store_called_with_create_false():
    with patch("src.retriever._make_store", return_value=_mock_store([])) as mock_make:
        r = build_retriever(_VIDEO_ID, _fake_embeddings(), _DB_PATH)
        r.invoke("q")
    assert mock_make.call_args.kwargs.get("create") is False
