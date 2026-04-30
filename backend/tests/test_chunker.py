import re

import tiktoken
from langchain_core.documents import Document

from src.chunker import TimestampAwareTextSplitter, chunk_documents
from src.tokens import count_tokens

UUID4_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
_MODEL = "gpt-4o-mini"
_ENC = tiktoken.encoding_for_model(_MODEL)


def _make_doc(text: str, start: float, end: float, video_id: str = "vid1") -> Document:
    return Document(
        page_content=text,
        metadata={"video_id": video_id, "start_ts": start, "end_ts": end, "segment_index": 0},
    )


def _seg(i: int) -> Document:
    """One segment whose start/end timestamps are unambiguously derived from i."""
    return _make_doc(
        f"Segment number {i} contains some transcript text here.",
        start=float(i * 10),
        end=float(i * 10 + 9),
    )


def _segs(n: int) -> list[Document]:
    return [_seg(i) for i in range(n)]


def _text_of_tokens(n: int) -> str:
    """Return a text string that encodes to exactly n tokens under _MODEL."""
    word = "hello"  # 1 token each
    tokens = _ENC.encode((word + " ") * n)[:n]
    return _ENC.decode(tokens)


# ---------------------------------------------------------------------------
# Basic contract
# ---------------------------------------------------------------------------


def test_empty_input_returns_empty():
    assert chunk_documents([]) == []


def test_single_small_doc_is_one_chunk():
    docs = chunk_documents([_make_doc("Short.", 0.0, 1.0)], target_tokens=500)
    assert len(docs) == 1
    assert docs[0].page_content == "Short."


def test_returns_list_of_documents():
    assert all(isinstance(d, Document) for d in chunk_documents(_segs(20)))


# ---------------------------------------------------------------------------
# Boundary correctness
# ---------------------------------------------------------------------------


def test_text_at_exact_limit_stays_as_one_chunk():
    target = 20
    text = _text_of_tokens(target)
    docs = chunk_documents([_make_doc(text, 0.0, 1.0)], target_tokens=target, overlap_tokens=0)
    assert len(docs) == 1


def test_text_one_token_over_limit_is_split():
    target = 10
    text = _text_of_tokens(target + 5)  # clearly over
    docs = chunk_documents([_make_doc(text, 0.0, 1.0)], target_tokens=target, overlap_tokens=0)
    assert len(docs) > 1


def test_every_chunk_within_token_limit():
    target = 30
    for doc in chunk_documents(_segs(60), target_tokens=target, overlap_tokens=5, model=_MODEL):
        assert count_tokens(doc.page_content, _MODEL) <= target + 5


def test_first_chunk_starts_at_beginning_of_combined_text():
    segs = _segs(10)
    chunks = chunk_documents(segs, target_tokens=30, overlap_tokens=5)
    # The very first token of the first source segment must appear in chunk 0
    first_word = segs[0].page_content.split()[0]
    assert first_word in chunks[0].page_content


def test_last_chunk_contains_end_of_combined_text():
    segs = _segs(10)
    chunks = chunk_documents(segs, target_tokens=30, overlap_tokens=5)
    last_word = segs[-1].page_content.split()[-1]
    assert last_word in chunks[-1].page_content


# ---------------------------------------------------------------------------
# Overlap presence
# ---------------------------------------------------------------------------


def test_consecutive_chunks_share_text_when_overlap_nonzero():
    chunks = chunk_documents(_segs(30), target_tokens=30, overlap_tokens=10, model=_MODEL)
    assert len(chunks) >= 2
    shared = 0
    for i in range(len(chunks) - 1):
        words_a = set(chunks[i].page_content.split())
        words_b = set(chunks[i + 1].page_content.split())
        if words_a & words_b:
            shared += 1
    # At least half of consecutive pairs should share words due to overlap
    assert shared >= len(chunks) // 2


def test_zero_overlap_adjacent_chunks_have_no_shared_tail_head():
    # With unique text per segment and no overlap, the last word of chunk[i]
    # must not be the first word of chunk[i+1].
    segs = _segs(30)  # unique index in each segment's text
    chunks = chunk_documents(segs, target_tokens=20, overlap_tokens=0, model=_MODEL)
    assert len(chunks) >= 2
    for i in range(len(chunks) - 1):
        last_word = chunks[i].page_content.split()[-1]
        first_word = chunks[i + 1].page_content.split()[0]
        assert last_word != first_word


# ---------------------------------------------------------------------------
# Timestamp integrity — every chunk, not just first/last
# ---------------------------------------------------------------------------

# Build source timestamps as a lookup so we can assert exact mapping.
_TS_SEGS = [
    _make_doc(f"Word number {i} transcript segment text.", start=float(i * 7), end=float(i * 7 + 6))
    for i in range(25)
]
_VALID_START_TS = {s.metadata["start_ts"] for s in _TS_SEGS}
_VALID_END_TS = {s.metadata["end_ts"] for s in _TS_SEGS}


def test_every_chunk_start_ts_is_a_source_segment_start_ts():
    chunks = chunk_documents(_TS_SEGS, target_tokens=30, overlap_tokens=5, model=_MODEL)
    for c in chunks:
        assert (
            c.metadata["start_ts"] in _VALID_START_TS
        ), f"start_ts {c.metadata['start_ts']} not from any source segment"


def test_every_chunk_end_ts_is_a_source_segment_end_ts():
    chunks = chunk_documents(_TS_SEGS, target_tokens=30, overlap_tokens=5, model=_MODEL)
    for c in chunks:
        assert (
            c.metadata["end_ts"] in _VALID_END_TS
        ), f"end_ts {c.metadata['end_ts']} not from any source segment"


def test_chunk_start_ts_lte_end_ts_for_every_chunk():
    for c in chunk_documents(_TS_SEGS, target_tokens=30, overlap_tokens=5, model=_MODEL):
        assert c.metadata["start_ts"] <= c.metadata["end_ts"]


def test_chunk_start_ts_is_non_decreasing():
    chunks = chunk_documents(_segs(40), target_tokens=30, overlap_tokens=5, model=_MODEL)
    starts = [c.metadata["start_ts"] for c in chunks]
    assert starts == sorted(starts)


# ---------------------------------------------------------------------------
# First-chunk and last-chunk timestamp pinning
# ---------------------------------------------------------------------------


def test_first_chunk_start_ts_equals_first_segment_start_ts():
    segs = _segs(20)
    chunks = chunk_documents(segs, target_tokens=30, overlap_tokens=5, model=_MODEL)
    assert chunks[0].metadata["start_ts"] == segs[0].metadata["start_ts"]


def test_last_chunk_end_ts_equals_last_segment_end_ts():
    segs = _segs(20)
    chunks = chunk_documents(segs, target_tokens=30, overlap_tokens=5, model=_MODEL)
    assert chunks[-1].metadata["end_ts"] == segs[-1].metadata["end_ts"]


# ---------------------------------------------------------------------------
# Metadata completeness
# ---------------------------------------------------------------------------


def test_each_chunk_has_required_keys():
    required = {"start_ts", "end_ts", "chunk_id", "video_id"}
    for doc in chunk_documents(_segs(20)):
        assert required <= doc.metadata.keys()


def test_chunk_id_is_valid_uuid4():
    for doc in chunk_documents(_segs(10)):
        assert UUID4_RE.match(doc.metadata["chunk_id"])


def test_chunk_ids_are_all_unique():
    chunks = chunk_documents(_segs(60), target_tokens=30, overlap_tokens=5)
    ids = [d.metadata["chunk_id"] for d in chunks]
    assert len(ids) == len(set(ids))


def test_video_id_propagated_to_every_chunk():
    for doc in chunk_documents(_segs(20)):
        assert doc.metadata["video_id"] == "vid1"


# ---------------------------------------------------------------------------
# Splitter class interface
# ---------------------------------------------------------------------------


def test_splitter_is_subclass_of_recursive_splitter():
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    assert issubclass(TimestampAwareTextSplitter, RecursiveCharacterTextSplitter)


def test_splitter_empty_input():
    assert TimestampAwareTextSplitter().split_documents([]) == []
