import re

import tiktoken
from langchain_core.documents import Document

from src.chunker import TimestampAwareTextSplitter, chunk_documents

UUID4_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


def _make_doc(text: str, start: float, end: float, video_id: str = "vid1") -> Document:
    return Document(
        page_content=text,
        metadata={"video_id": video_id, "start_ts": start, "end_ts": end, "segment_index": 0},
    )


def _short_segments() -> list[Document]:
    return [
        _make_doc("Hello world.", 0.0, 2.0),
        _make_doc("How are you today?", 2.0, 4.5),
        _make_doc("I am doing well.", 4.5, 7.0),
    ]


def _long_segments(n: int = 40) -> list[Document]:
    return [
        _make_doc(
            f"Segment number {i} contains some transcript text here.",
            float(i * 5),
            float(i * 5 + 4),
        )
        for i in range(n)
    ]


# --- chunk_documents public contract ---


def test_chunk_documents_returns_list_of_documents():
    chunks = chunk_documents(_long_segments())
    assert all(isinstance(d, Document) for d in chunks)


def test_chunk_documents_empty_input():
    assert chunk_documents([]) == []


def test_chunk_documents_respects_target_tokens():
    enc = tiktoken.get_encoding("cl100k_base")
    target = 30
    for doc in chunk_documents(_long_segments(60), target_tokens=target, overlap_tokens=5):
        assert len(enc.encode(doc.page_content)) <= target + 5


def test_chunk_documents_default_params_produce_chunks():
    chunks = chunk_documents(_long_segments(100))
    assert len(chunks) > 1


# --- metadata contract ---


def test_each_chunk_has_required_metadata_keys():
    required = {"start_ts", "end_ts", "chunk_id", "video_id"}
    for doc in chunk_documents(_long_segments()):
        assert required <= doc.metadata.keys()


def test_chunk_id_is_valid_uuid4():
    for doc in chunk_documents(_long_segments()):
        assert UUID4_RE.match(doc.metadata["chunk_id"]), doc.metadata["chunk_id"]


def test_chunk_ids_are_unique():
    chunks = chunk_documents(_long_segments(60), target_tokens=30, overlap_tokens=5)
    ids = [d.metadata["chunk_id"] for d in chunks]
    assert len(ids) == len(set(ids))


def test_video_id_propagated_to_all_chunks():
    for doc in chunk_documents(_long_segments()):
        assert doc.metadata["video_id"] == "vid1"


def test_start_ts_from_first_source_segment():
    segs = _short_segments()
    chunks = chunk_documents(segs, target_tokens=500)
    assert chunks[0].metadata["start_ts"] == segs[0].metadata["start_ts"]


def test_end_ts_from_last_source_segment():
    segs = _short_segments()
    chunks = chunk_documents(segs, target_tokens=500)
    assert chunks[-1].metadata["end_ts"] == segs[-1].metadata["end_ts"]


def test_end_ts_always_gte_start_ts():
    for doc in chunk_documents(_long_segments(), target_tokens=50, overlap_tokens=5):
        assert doc.metadata["end_ts"] >= doc.metadata["start_ts"]


# --- splitting behaviour ---


def test_large_input_produces_multiple_chunks():
    assert len(chunk_documents(_long_segments(), target_tokens=30, overlap_tokens=5)) > 1


def test_overlap_increases_chunk_count():
    segs = _long_segments()
    assert len(chunk_documents(segs, target_tokens=50, overlap_tokens=20)) >= len(
        chunk_documents(segs, target_tokens=50, overlap_tokens=0)
    )


# --- splitter class still works ---


def test_splitter_empty_input():
    assert TimestampAwareTextSplitter().split_documents([]) == []


def test_splitter_single_small_doc():
    docs = TimestampAwareTextSplitter(chunk_size=500).split_documents(
        [_make_doc("Short text.", 0.0, 1.0)]
    )
    assert len(docs) == 1
    assert docs[0].page_content == "Short text."
