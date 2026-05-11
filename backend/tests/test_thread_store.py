"""Unit tests for ThreadStore — in-memory conversation history keyed by thread_id."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.main import ThreadStore


@pytest.fixture()
def store():
    return ThreadStore()


# ---------------------------------------------------------------------------
# Empty / missing threads
# ---------------------------------------------------------------------------


def test_get_unknown_thread_returns_empty_list(store):
    assert store.get("nonexistent") == []


def test_has_returns_false_for_unknown_thread(store):
    assert not store.has("nonexistent")


def test_delete_nonexistent_returns_false(store):
    assert store.delete("nonexistent") is False


# ---------------------------------------------------------------------------
# append / get / has
# ---------------------------------------------------------------------------


def test_append_creates_two_messages(store):
    store.append("t1", "What is it?", "It is 42.", max_turns=10)
    history = store.get("t1")
    assert len(history) == 2


def test_append_human_message_first(store):
    store.append("t1", "question", "answer", max_turns=10)
    assert isinstance(store.get("t1")[0], HumanMessage)
    assert store.get("t1")[0].content == "question"


def test_append_ai_message_second(store):
    store.append("t1", "question", "answer", max_turns=10)
    assert isinstance(store.get("t1")[1], AIMessage)
    assert store.get("t1")[1].content == "answer"


def test_has_returns_true_after_append(store):
    store.append("t1", "q", "a", max_turns=10)
    assert store.has("t1")


def test_multiple_appends_accumulate(store):
    store.append("t1", "q1", "a1", max_turns=10)
    store.append("t1", "q2", "a2", max_turns=10)
    assert len(store.get("t1")) == 4


def test_append_preserves_order(store):
    store.append("t1", "q1", "a1", max_turns=10)
    store.append("t1", "q2", "a2", max_turns=10)
    msgs = store.get("t1")
    assert msgs[0].content == "q1"
    assert msgs[1].content == "a1"
    assert msgs[2].content == "q2"
    assert msgs[3].content == "a2"


def test_different_threads_are_isolated(store):
    store.append("t1", "q1", "a1", max_turns=10)
    store.append("t2", "q2", "a2", max_turns=10)
    assert len(store.get("t1")) == 2
    assert store.get("t1")[0].content == "q1"
    assert store.get("t2")[0].content == "q2"


# ---------------------------------------------------------------------------
# Cap enforcement
# ---------------------------------------------------------------------------


def test_cap_at_one_turn_keeps_one_pair(store):
    store.append("t1", "q1", "a1", max_turns=1)
    store.append("t1", "q2", "a2", max_turns=1)
    msgs = store.get("t1")
    assert len(msgs) == 2
    assert msgs[0].content == "q2"
    assert msgs[1].content == "a2"


def test_cap_at_two_turns_evicts_oldest_after_third(store):
    store.append("t1", "q1", "a1", max_turns=2)
    store.append("t1", "q2", "a2", max_turns=2)
    store.append("t1", "q3", "a3", max_turns=2)
    msgs = store.get("t1")
    assert len(msgs) == 4  # 2 turns × 2 messages
    assert msgs[0].content == "q2"
    assert msgs[1].content == "a2"
    assert msgs[2].content == "q3"
    assert msgs[3].content == "a3"


def test_cap_keeps_most_recent_turns(store):
    for i in range(15):
        store.append("t1", f"q{i}", f"a{i}", max_turns=10)
    msgs = store.get("t1")
    assert len(msgs) == 20  # cap = 10 × 2
    assert msgs[0].content == "q5"  # oldest kept after eviction
    assert msgs[-1].content == "a14"


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


def test_delete_existing_thread_returns_true(store):
    store.append("t1", "q", "a", max_turns=10)
    assert store.delete("t1") is True


def test_delete_removes_thread(store):
    store.append("t1", "q", "a", max_turns=10)
    store.delete("t1")
    assert not store.has("t1")
    assert store.get("t1") == []


def test_double_delete_returns_false_on_second(store):
    store.append("t1", "q", "a", max_turns=10)
    store.delete("t1")
    assert store.delete("t1") is False


# ---------------------------------------------------------------------------
# get returns a copy
# ---------------------------------------------------------------------------


def test_get_returns_copy_not_internal_reference(store):
    store.append("t1", "q", "a", max_turns=10)
    copy = store.get("t1")
    copy.clear()
    assert len(store.get("t1")) == 2
