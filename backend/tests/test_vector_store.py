import re
from unittest.mock import MagicMock

import chromadb
import pytest
from langchain_core.documents import Document

from src.vector_store import (
    _collection_name,
    _sanitize_video_id,
    add_documents,
    collection_exists,
    delete_collection,
    get_channel_metadata,
    query,
    save_channel_metadata,
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
    name = _collection_name("a.b-c/d e!f")
    # strip the known "video_" prefix then check the rest
    assert re.fullmatch(r"[a-zA-Z0-9_]+", name)


# ---------------------------------------------------------------------------
# add_documents / query / collection_exists / delete_collection
# Each test gets a fresh in-memory EphemeralClient — no shared state.
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_db(tmp_path):
    from chromadb.config import Settings as ChromaSettings

    return chromadb.PersistentClient(
        path=str(tmp_path / "chroma"),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def test_add_documents_does_not_raise(tmp_db):
    add_documents("vid1", _docs(), _fake_embedding(), tmp_db)


def test_add_documents_with_precomputed_vectors_skips_embedding(tmp_db):
    emb = _fake_embedding()
    docs = _docs(3)
    vectors = [[float(i)] * _DIM for i in range(len(docs))]
    add_documents("vid1", docs, emb, tmp_db, precomputed_vectors=vectors)
    emb.embed_documents.assert_not_called()
    assert collection_exists("vid1", tmp_db)


def test_add_documents_precomputed_vectors_queryable(tmp_db):
    emb = _fake_embedding()
    docs = _docs(3)
    # Use distinct non-zero unit vectors so cosine similarity is well-defined.
    vectors = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    add_documents("vid1", docs, emb, tmp_db, precomputed_vectors=vectors)
    results = query("vid1", [1.0, 0.0, 0.0, 0.0], emb, tmp_db, k=3)
    assert {d.page_content for d in results} == {d.page_content for d in docs}


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


# ---------------------------------------------------------------------------
# Round-trip: add then query — verify content comes back
# ---------------------------------------------------------------------------


def test_roundtrip_query_returns_added_content(tmp_db):
    emb = _fake_embedding()
    docs = _docs(3)
    add_documents("vid1", docs, emb, tmp_db)
    results = query("vid1", [0.0] * _DIM, emb, tmp_db, k=3)
    returned_texts = {d.page_content for d in results}
    added_texts = {d.page_content for d in docs}
    assert returned_texts == added_texts


def test_roundtrip_metadata_preserved(tmp_db):
    emb = _fake_embedding()
    docs = _docs(2)
    add_documents("vid1", docs, emb, tmp_db)
    results = query("vid1", [0.0] * _DIM, emb, tmp_db, k=2)
    for result in results:
        assert "video_id" in result.metadata
        assert "start_ts" in result.metadata
        assert "end_ts" in result.metadata
        assert "chunk_id" in result.metadata


def test_roundtrip_k_limits_results(tmp_db):
    emb = _fake_embedding()
    add_documents("vid1", _docs(5), emb, tmp_db)
    for k in (1, 2, 3):
        results = query("vid1", [0.0] * _DIM, emb, tmp_db, k=k)
        assert len(results) <= k


# ---------------------------------------------------------------------------
# Collection isolation: querying vid-a never returns vid-b documents
# ---------------------------------------------------------------------------


def test_query_isolation_between_collections(tmp_db):
    emb = _fake_embedding()
    docs_a = [
        Document(
            page_content="alpha content unique to video A",
            metadata={"video_id": "vid-a", "start_ts": 0.0, "end_ts": 1.0, "chunk_id": "a0"},
        )
    ]
    docs_b = [
        Document(
            page_content="beta content unique to video B",
            metadata={"video_id": "vid-b", "start_ts": 0.0, "end_ts": 1.0, "chunk_id": "b0"},
        )
    ]
    add_documents("vid-a", docs_a, emb, tmp_db)
    add_documents("vid-b", docs_b, emb, tmp_db)

    results_a = query("vid-a", [0.0] * _DIM, emb, tmp_db, k=5)
    texts_a = {d.page_content for d in results_a}
    assert all("alpha" in t for t in texts_a), "vid-a results should only contain vid-a docs"
    assert not any("beta" in t for t in texts_a)


def test_delete_one_collection_does_not_affect_other(tmp_db):
    emb = _fake_embedding()
    add_documents("vid-a", _docs(2), emb, tmp_db)
    add_documents("vid-b", _docs(2), emb, tmp_db)
    delete_collection("vid-a", tmp_db)
    # vid-b should still be queryable
    results = query("vid-b", [0.0] * _DIM, emb, tmp_db, k=2)
    assert len(results) > 0


# ---------------------------------------------------------------------------
# Channel metadata — stored in Chroma collection metadata
# ---------------------------------------------------------------------------


def test_save_and_get_channel_metadata_roundtrip(tmp_db):
    add_documents("vid1", _docs(1), _fake_embedding(), tmp_db)
    meta = {
        "channel_name": "Test Channel",
        "channel_url": "https://example.com",
        "title": "My Video",
    }
    save_channel_metadata("vid1", tmp_db, meta)
    result = get_channel_metadata("vid1", tmp_db)
    assert result == meta


def test_get_channel_metadata_returns_empty_when_collection_absent(tmp_db):
    result = get_channel_metadata("nonexistent", tmp_db)
    assert result == {}


def test_get_channel_metadata_filters_hnsw_keys(tmp_db):
    add_documents("vid1", _docs(1), _fake_embedding(), tmp_db)
    save_channel_metadata("vid1", tmp_db, {"channel_name": "Chan"})
    result = get_channel_metadata("vid1", tmp_db)
    assert all(not k.startswith("hnsw:") for k in result)


def test_collection_still_queryable_after_save_channel_metadata(tmp_db):
    emb = _fake_embedding()
    add_documents("vid1", _docs(2), emb, tmp_db)
    save_channel_metadata("vid1", tmp_db, {"channel_name": "Chan"})
    results = query("vid1", [0.0] * _DIM, emb, tmp_db, k=2)
    assert len(results) > 0


def test_save_channel_metadata_overwrites_previous(tmp_db):
    add_documents("vid1", _docs(1), _fake_embedding(), tmp_db)
    save_channel_metadata("vid1", tmp_db, {"channel_name": "Old"})
    save_channel_metadata("vid1", tmp_db, {"channel_name": "New"})
    result = get_channel_metadata("vid1", tmp_db)
    assert result["channel_name"] == "New"


def test_channel_metadata_isolated_between_videos(tmp_db):
    emb = _fake_embedding()
    add_documents("vid-a", _docs(1), emb, tmp_db)
    add_documents("vid-b", _docs(1), emb, tmp_db)
    save_channel_metadata("vid-a", tmp_db, {"channel_name": "Alpha"})
    save_channel_metadata("vid-b", tmp_db, {"channel_name": "Beta"})
    assert get_channel_metadata("vid-a", tmp_db)["channel_name"] == "Alpha"
    assert get_channel_metadata("vid-b", tmp_db)["channel_name"] == "Beta"
