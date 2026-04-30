from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from src.vector_store import (
    _collection_name,
    _sanitize_video_id,
    add_documents,
    collection_exists,
    delete_collection,
    query,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIM = 4


def _fake_embedding(dim: int = _DIM):
    """Embedding that returns a fixed zero-vector per text — no OpenAI calls."""
    emb = MagicMock()
    emb.embed_documents.side_effect = lambda texts: [[0.0] * dim for _ in texts]
    emb.embed_query.return_value = [0.0] * dim
    return emb


def _docs(n: int = 3) -> list[Document]:
    return [
        Document(
            page_content=f"chunk {i}",
            metadata={
                "video_id": "vid1",
                "start_ts": float(i),
                "end_ts": float(i + 1),
                "chunk_id": f"c{i}",
            },
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# _collection_name
# ---------------------------------------------------------------------------


def test_collection_name_format():
    assert _collection_name("abc123") == "video_abc123"


def test_collection_name_prefixed_with_video_underscore():
    assert _collection_name("abc").startswith("video_")


def test_collection_name_max_63_chars():
    assert len(_collection_name("x" * 100)) <= 63


def test_collection_name_deterministic():
    assert _collection_name("vid123") == _collection_name("vid123")


def test_sanitize_replaces_hyphens():
    assert "-" not in _sanitize_video_id("abc-def")
    assert _sanitize_video_id("abc-def") == "abc_def"


def test_sanitize_replaces_dots():
    assert _sanitize_video_id("a.b.c") == "a_b_c"


def test_sanitize_replaces_slashes():
    assert _sanitize_video_id("a/b") == "a_b"


def test_sanitize_preserves_alphanumeric_and_underscore():
    assert _sanitize_video_id("abc_123_XYZ") == "abc_123_XYZ"


def test_sanitize_replaces_spaces():
    assert " " not in _sanitize_video_id("hello world")


def test_collection_name_only_alphanumeric_and_underscore():
    import re

    name = _collection_name("a.b-c/d e!f")
    # strip the known "video_" prefix then check the rest
    assert re.fullmatch(r"[a-zA-Z0-9_]+", name)


# ---------------------------------------------------------------------------
# add_documents / query / collection_exists / delete_collection
# Use a temporary directory for each test (no shared state).
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_db(tmp_path):
    return tmp_path / "chroma"


def test_add_documents_does_not_raise(tmp_db):
    add_documents("vid1", _docs(), _fake_embedding(), tmp_db)


def test_collection_exists_false_before_add(tmp_db):
    assert collection_exists("new-vid", tmp_db) is False


def test_collection_exists_true_after_add(tmp_db):
    add_documents("vid1", _docs(), _fake_embedding(), tmp_db)
    assert collection_exists("vid1", tmp_db) is True


def test_query_returns_list_of_documents(tmp_db):
    emb = _fake_embedding()
    add_documents("vid1", _docs(3), emb, tmp_db)
    results = query("vid1", [0.0] * _DIM, emb, tmp_db, k=2)
    assert isinstance(results, list)
    assert all(isinstance(d, Document) for d in results)


def test_query_respects_k(tmp_db):
    emb = _fake_embedding()
    add_documents("vid1", _docs(3), emb, tmp_db)
    results = query("vid1", [0.0] * _DIM, emb, tmp_db, k=2)
    assert len(results) <= 2


def test_delete_collection_removes_it(tmp_db):
    add_documents("vid1", _docs(), _fake_embedding(), tmp_db)
    assert collection_exists("vid1", tmp_db)
    delete_collection("vid1", tmp_db)
    assert not collection_exists("vid1", tmp_db)


def test_delete_collection_noop_when_absent(tmp_db):
    # Must not raise
    delete_collection("nonexistent", tmp_db)


def test_add_then_delete_then_add_again(tmp_db):
    emb = _fake_embedding()
    add_documents("vid1", _docs(2), emb, tmp_db)
    delete_collection("vid1", tmp_db)
    add_documents("vid1", _docs(2), emb, tmp_db)
    assert collection_exists("vid1", tmp_db)


def test_separate_video_ids_use_separate_collections(tmp_db):
    emb = _fake_embedding()
    add_documents("vid-a", _docs(2), emb, tmp_db)
    add_documents("vid-b", _docs(2), emb, tmp_db)
    assert collection_exists("vid-a", tmp_db)
    assert collection_exists("vid-b", tmp_db)
    delete_collection("vid-a", tmp_db)
    assert not collection_exists("vid-a", tmp_db)
    assert collection_exists("vid-b", tmp_db)
