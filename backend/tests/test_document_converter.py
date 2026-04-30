from langchain_core.documents import Document

from src.document_converter import segments_to_documents

_SEGMENTS = [
    {"text": "Hello world", "start": 0.0, "duration": 2.5},
    {"text": "How are you", "start": 2.5, "duration": 3.0},
    {"text": "Goodbye", "start": 5.5, "duration": 1.2},
]


def test_returns_list_of_documents():
    docs = segments_to_documents("vid123", _SEGMENTS)
    assert all(isinstance(d, Document) for d in docs)


def test_length_matches_segments():
    docs = segments_to_documents("vid123", _SEGMENTS)
    assert len(docs) == len(_SEGMENTS)


def test_page_content_is_segment_text():
    docs = segments_to_documents("vid123", _SEGMENTS)
    assert [d.page_content for d in docs] == [s["text"] for s in _SEGMENTS]


def test_metadata_fields():
    docs = segments_to_documents("vid123", _SEGMENTS)
    for idx, doc in enumerate(docs):
        m = doc.metadata
        assert m["video_id"] == "vid123"
        assert m["segment_index"] == idx
        assert m["start_ts"] == _SEGMENTS[idx]["start"]
        expected_end = _SEGMENTS[idx]["start"] + _SEGMENTS[idx]["duration"]
        assert abs(m["end_ts"] - expected_end) < 1e-9


def test_end_ts_greater_than_start_ts():
    docs = segments_to_documents("vid123", _SEGMENTS)
    assert all(d.metadata["end_ts"] > d.metadata["start_ts"] for d in docs)


def test_segment_indices_are_sequential():
    docs = segments_to_documents("vid123", _SEGMENTS)
    assert [d.metadata["segment_index"] for d in docs] == list(range(len(_SEGMENTS)))


def test_empty_segments_returns_empty_list():
    assert segments_to_documents("vid123", []) == []


def test_video_id_propagated_to_all_docs():
    docs = segments_to_documents("unique-id-xyz", _SEGMENTS)
    assert all(d.metadata["video_id"] == "unique-id-xyz" for d in docs)
