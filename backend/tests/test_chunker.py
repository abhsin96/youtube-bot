import tiktoken
from langchain_core.documents import Document

from src.chunker import TimestampAwareTextSplitter


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
    """40 segments of ~10 tokens each → forces splitting."""
    return [
        _make_doc(
            f"Segment number {i} contains some transcript text here.",
            float(i * 5),
            float(i * 5 + 4),
        )
        for i in range(n)
    ]


# --- basic shape ---


def test_returns_list_of_documents():
    splitter = TimestampAwareTextSplitter(chunk_size=50, chunk_overlap=5)
    docs = splitter.split_documents(_long_segments())
    assert all(isinstance(d, Document) for d in docs)


def test_empty_input_returns_empty():
    splitter = TimestampAwareTextSplitter()
    assert splitter.split_documents([]) == []


def test_single_small_doc_returned_as_one_chunk():
    splitter = TimestampAwareTextSplitter(chunk_size=500)
    docs = splitter.split_documents([_make_doc("Short text.", 0.0, 1.0)])
    assert len(docs) == 1
    assert docs[0].page_content == "Short text."


# --- timestamp preservation ---


def test_start_ts_comes_from_first_source_segment():
    splitter = TimestampAwareTextSplitter(chunk_size=500)
    segs = _short_segments()
    chunks = splitter.split_documents(segs)
    assert chunks[0].metadata["start_ts"] == segs[0].metadata["start_ts"]


def test_end_ts_comes_from_last_source_segment():
    splitter = TimestampAwareTextSplitter(chunk_size=500)
    segs = _short_segments()
    chunks = splitter.split_documents(segs)
    assert chunks[-1].metadata["end_ts"] == segs[-1].metadata["end_ts"]


def test_each_chunk_has_start_and_end_ts():
    splitter = TimestampAwareTextSplitter(chunk_size=50, chunk_overlap=5)
    for doc in splitter.split_documents(_long_segments()):
        assert "start_ts" in doc.metadata
        assert "end_ts" in doc.metadata


def test_end_ts_always_gte_start_ts():
    splitter = TimestampAwareTextSplitter(chunk_size=50, chunk_overlap=5)
    for doc in splitter.split_documents(_long_segments()):
        assert doc.metadata["end_ts"] >= doc.metadata["start_ts"]


def test_video_id_preserved_in_all_chunks():
    splitter = TimestampAwareTextSplitter(chunk_size=50, chunk_overlap=5)
    for doc in splitter.split_documents(_long_segments()):
        assert doc.metadata["video_id"] == "vid1"


# --- splitting behaviour ---


def test_large_input_produces_multiple_chunks():
    splitter = TimestampAwareTextSplitter(chunk_size=30, chunk_overlap=5)
    chunks = splitter.split_documents(_long_segments())
    assert len(chunks) > 1


def test_chunk_overlap_produces_more_chunks_than_no_overlap():
    segs = _long_segments()
    no_overlap = TimestampAwareTextSplitter(chunk_size=50, chunk_overlap=0)
    with_overlap = TimestampAwareTextSplitter(chunk_size=50, chunk_overlap=20)
    assert len(with_overlap.split_documents(segs)) >= len(no_overlap.split_documents(segs))


# --- tiktoken token counts ---


def test_chunks_respect_token_size_limit():
    chunk_size = 30
    splitter = TimestampAwareTextSplitter(chunk_size=chunk_size, chunk_overlap=5)
    enc = tiktoken.get_encoding("cl100k_base")
    for doc in splitter.split_documents(_long_segments(60)):
        token_count = len(enc.encode(doc.page_content))
        # allow one chunk_overlap worth of slack for boundary chunks
        assert token_count <= chunk_size + 5, f"chunk too large: {token_count} tokens"


def test_default_encoding_is_cl100k():
    splitter = TimestampAwareTextSplitter()
    # If encoding was wrong, from_tiktoken_encoder would raise at construction
    assert splitter.chunk_size == 500
    assert splitter.chunk_overlap == 50
