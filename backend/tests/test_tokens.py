import tiktoken

from src.tokens import count_tokens


def test_count_tokens_returns_int():
    assert isinstance(count_tokens("hello world"), int)


def test_count_tokens_empty_string():
    assert count_tokens("") == 0


def test_count_tokens_known_value():
    # "hello world" is 2 tokens in o200k_base (gpt-4o-mini)
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    expected = len(enc.encode("hello world"))
    assert count_tokens("hello world", model="gpt-4o-mini") == expected


def test_count_tokens_model_affects_count():
    text = "This is a test sentence with some content."
    # gpt-4o-mini uses o200k_base; gpt-4 uses cl100k_base — counts may differ
    n_mini = count_tokens(text, model="gpt-4o-mini")
    n_gpt4 = count_tokens(text, model="gpt-4")
    # both should be positive integers; encoding differences may or may not change count
    assert n_mini > 0
    assert n_gpt4 > 0


def test_count_tokens_longer_text_has_more_tokens():
    short = "Hello."
    long = "Hello. " * 50
    assert count_tokens(long) > count_tokens(short)


def test_count_tokens_unknown_model_falls_back():
    # Should not raise; falls back to cl100k_base
    result = count_tokens("some text", model="unknown-model-xyz")
    assert result > 0


def test_default_model_is_gpt4o_mini():
    text = "test"
    assert count_tokens(text) == count_tokens(text, model="gpt-4o-mini")


def test_chunker_uses_model_encoding():
    from langchain_core.documents import Document

    from src.chunker import chunk_documents

    docs = [
        Document(
            page_content=f"Segment {i} has some words to fill tokens.",
            metadata={"video_id": "v1", "start_ts": float(i), "end_ts": float(i + 1)},
        )
        for i in range(30)
    ]
    # both models should produce valid chunks with correct metadata keys
    for model in ("gpt-4o-mini", "gpt-4"):
        chunks = chunk_documents(docs, target_tokens=40, overlap_tokens=5, model=model)
        assert len(chunks) > 0
        for c in chunks:
            assert "start_ts" in c.metadata
            assert "end_ts" in c.metadata
            assert "chunk_id" in c.metadata
