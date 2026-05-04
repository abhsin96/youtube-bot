"""
Unit tests for context-budget enforcement.

Strategy: use real tiktoken counting with a known system prompt + real documents
so the token arithmetic is deterministic.  A tight budget forces _trim_to_budget
to drop the lowest-score (tail) chunks; we assert on what survives.
"""

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from src.chain import _count_prompt_tokens, _trim_to_budget
from src.tokens import count_tokens

_MODEL = "gpt-4o-mini"
_SYSTEM = "You answer questions."  # short, stable baseline
_QUESTION = "What does the speaker say?"


def _doc(text: str, start_ts: float = 0.0, chunk_id: str = "c") -> Document:
    return Document(
        page_content=text,
        metadata={"start_ts": start_ts, "chunk_id": chunk_id},
    )


def _budget_for(*sources: Document, padding: int = 0) -> int:
    """Return the exact token count for the given sources + a padding offset."""
    return _count_prompt_tokens(list(sources), _SYSTEM, [], _QUESTION, _MODEL) + padding


# ---------------------------------------------------------------------------
# _count_prompt_tokens
# ---------------------------------------------------------------------------


def test_count_prompt_tokens_returns_int():
    docs = [_doc("hello", chunk_id="c1")]
    result = _count_prompt_tokens(docs, _SYSTEM, [], _QUESTION, _MODEL)
    assert isinstance(result, int)


def test_count_prompt_tokens_increases_with_more_chunks():
    one = _count_prompt_tokens([_doc("a")], _SYSTEM, [], _QUESTION, _MODEL)
    two = _count_prompt_tokens([_doc("a"), _doc("b")], _SYSTEM, [], _QUESTION, _MODEL)
    assert two > one


def test_count_prompt_tokens_increases_with_history():
    no_history = _count_prompt_tokens([_doc("x")], _SYSTEM, [], _QUESTION, _MODEL)
    with_history = _count_prompt_tokens(
        [_doc("x")],
        _SYSTEM,
        [HumanMessage(content="old q"), AIMessage(content="old a")],
        _QUESTION,
        _MODEL,
    )
    assert with_history > no_history


def test_count_prompt_tokens_empty_sources():
    result = _count_prompt_tokens([], _SYSTEM, [], _QUESTION, _MODEL)
    # system + question tokens still counted
    expected_min = count_tokens(_SYSTEM, _MODEL) + count_tokens(_QUESTION, _MODEL)
    assert result >= expected_min


def test_count_prompt_tokens_longer_question_raises_count():
    short = _count_prompt_tokens([_doc("x")], _SYSTEM, [], "Why?", _MODEL)
    long = _count_prompt_tokens([_doc("x")], _SYSTEM, [], "Why? " * 50, _MODEL)
    assert long > short


# ---------------------------------------------------------------------------
# _trim_to_budget — happy path (already fits)
# ---------------------------------------------------------------------------


def test_trim_returns_all_when_within_budget():
    docs = [_doc("short", chunk_id="c1")]
    budget = _budget_for(*docs, padding=100)
    result = _trim_to_budget(docs, _SYSTEM, [], _QUESTION, _MODEL, budget)
    assert result == docs


def test_trim_preserves_order_when_no_trimming_needed():
    docs = [_doc("first", chunk_id="c1"), _doc("second", chunk_id="c2")]
    budget = _budget_for(*docs, padding=200)
    result = _trim_to_budget(docs, _SYSTEM, [], _QUESTION, _MODEL, budget)
    assert [d.metadata["chunk_id"] for d in result] == ["c1", "c2"]


def test_trim_empty_sources_returns_empty():
    result = _trim_to_budget([], _SYSTEM, [], _QUESTION, _MODEL, budget=500)
    assert result == []


# ---------------------------------------------------------------------------
# _trim_to_budget — trimming (oversized input)
# ---------------------------------------------------------------------------


def test_trim_drops_chunks_when_over_budget():
    # Budget only enough for one chunk
    docs = [_doc("high score chunk", chunk_id="c1"), _doc("low score chunk", chunk_id="c2")]
    # Budget = exact tokens for just the first doc
    budget = _budget_for(docs[0], padding=0)
    result = _trim_to_budget(docs, _SYSTEM, [], _QUESTION, _MODEL, budget)
    assert len(result) < len(docs)


def test_trim_keeps_highest_score_chunks():
    # sources are sorted highest-score first; trim should keep the head
    docs = [
        _doc("important content", chunk_id="high"),
        _doc("less important", chunk_id="mid"),
        _doc("least important", chunk_id="low"),
    ]
    # Budget fits only one chunk
    budget = _budget_for(docs[0], padding=0)
    result = _trim_to_budget(docs, _SYSTEM, [], _QUESTION, _MODEL, budget)
    ids = [d.metadata["chunk_id"] for d in result]
    assert "high" in ids
    assert "low" not in ids


def test_trim_drops_from_tail():
    docs = [_doc("keep me", chunk_id="keep"), _doc("drop me", chunk_id="drop")]
    budget = _budget_for(docs[0], padding=0)
    result = _trim_to_budget(docs, _SYSTEM, [], _QUESTION, _MODEL, budget)
    assert any(d.metadata["chunk_id"] == "keep" for d in result)
    assert all(d.metadata["chunk_id"] != "drop" for d in result)


def test_trim_removes_minimum_necessary_chunks():
    # Three chunks; budget fits two — only the third should be dropped.
    docs = [
        _doc("chunk one", chunk_id="c1"),
        _doc("chunk two", chunk_id="c2"),
        _doc("chunk three", chunk_id="c3"),
    ]
    budget = _budget_for(docs[0], docs[1], padding=0)
    result = _trim_to_budget(docs, _SYSTEM, [], _QUESTION, _MODEL, budget)
    ids = [d.metadata["chunk_id"] for d in result]
    assert "c1" in ids
    assert "c2" in ids
    assert "c3" not in ids


def test_trim_result_fits_within_budget():
    docs = [_doc(f"chunk number {i} with some text", chunk_id=f"c{i}") for i in range(5)]
    budget = _budget_for(docs[0], padding=0)
    result = _trim_to_budget(docs, _SYSTEM, [], _QUESTION, _MODEL, budget)
    actual_tokens = _count_prompt_tokens(result, _SYSTEM, [], _QUESTION, _MODEL)
    assert actual_tokens <= budget


def test_trim_all_chunks_when_even_one_exceeds_budget():
    # Budget so small that even a single chunk won't fit
    docs = [_doc("some text here", chunk_id="c1")]
    budget = count_tokens(_SYSTEM, _MODEL) + count_tokens(_QUESTION, _MODEL)  # no room for chunks
    result = _trim_to_budget(docs, _SYSTEM, [], _QUESTION, _MODEL, budget)
    assert result == []


def test_trim_with_history_still_fits():
    history = [HumanMessage(content="previous question"), AIMessage(content="previous answer")]
    docs = [_doc("content A", chunk_id="c1"), _doc("content B", chunk_id="c2")]
    # Budget sized for one doc + history
    budget = _budget_for(docs[0], padding=0) + count_tokens(
        "previous question\nprevious answer", _MODEL
    )
    result = _trim_to_budget(docs, _SYSTEM, history, _QUESTION, _MODEL, budget)
    actual_tokens = _count_prompt_tokens(result, _SYSTEM, history, _QUESTION, _MODEL)
    assert actual_tokens <= budget


def test_trim_does_not_mutate_original_list():
    docs = [_doc("a", chunk_id="c1"), _doc("b", chunk_id="c2")]
    original_len = len(docs)
    budget = _budget_for(docs[0], padding=0)
    _trim_to_budget(docs, _SYSTEM, [], _QUESTION, _MODEL, budget)
    assert len(docs) == original_len
