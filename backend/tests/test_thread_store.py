"""Tests for ThreadStore (in-memory) and RedisThreadStore (Redis-backed).

All behaviour tests run against both implementations via the parametrized
``store`` fixture.  Redis-specific tests (persistence across client instances)
live in a separate section and use ``fakeredis.FakeServer``.
"""

import fakeredis
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.main import RedisThreadStore, ThreadStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_redis_store() -> RedisThreadStore:
    return RedisThreadStore(fakeredis.FakeRedis(decode_responses=True))


@pytest.fixture(params=["memory", "redis"])
def store(request):
    if request.param == "memory":
        return ThreadStore()
    return _make_redis_store()


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
# get returns a copy (memory store) / fresh data (redis store)
# ---------------------------------------------------------------------------


def test_get_returns_independent_list(store):
    store.append("t1", "q", "a", max_turns=10)
    copy = store.get("t1")
    copy.clear()
    assert len(store.get("t1")) == 2


# ---------------------------------------------------------------------------
# Redis-specific: data survives creating a new client on the same server
# ---------------------------------------------------------------------------


def test_redis_data_survives_new_client_instance():
    server = fakeredis.FakeServer()
    store1 = RedisThreadStore(fakeredis.FakeRedis(server=server, decode_responses=True))
    store1.append("t1", "hello", "world", max_turns=10)

    store2 = RedisThreadStore(fakeredis.FakeRedis(server=server, decode_responses=True))
    assert store2.has("t1")
    msgs = store2.get("t1")
    assert len(msgs) == 2
    assert msgs[0].content == "hello"
    assert msgs[1].content == "world"


def test_redis_delete_is_visible_to_other_clients():
    server = fakeredis.FakeServer()
    store1 = RedisThreadStore(fakeredis.FakeRedis(server=server, decode_responses=True))
    store1.append("t1", "q", "a", max_turns=10)
    store1.delete("t1")

    store2 = RedisThreadStore(fakeredis.FakeRedis(server=server, decode_responses=True))
    assert not store2.has("t1")
    assert store2.get("t1") == []


def test_redis_append_is_visible_across_clients():
    server = fakeredis.FakeServer()
    store1 = RedisThreadStore(fakeredis.FakeRedis(server=server, decode_responses=True))
    store2 = RedisThreadStore(fakeredis.FakeRedis(server=server, decode_responses=True))

    store1.append("t1", "q1", "a1", max_turns=10)
    store2.append("t1", "q2", "a2", max_turns=10)

    msgs = store1.get("t1")
    assert len(msgs) == 4
    assert msgs[0].content == "q1"
    assert msgs[2].content == "q2"


def test_redis_cap_uses_pipeline_atomically():
    server = fakeredis.FakeServer()
    store = RedisThreadStore(fakeredis.FakeRedis(server=server, decode_responses=True))

    for i in range(5):
        store.append("t1", f"q{i}", f"a{i}", max_turns=2)

    msgs = store.get("t1")
    assert len(msgs) == 4  # capped at 2 turns = 4 messages
    assert msgs[0].content == "q3"
    assert msgs[-1].content == "a4"


def test_redis_message_types_round_trip():
    store = _make_redis_store()
    store.append("t1", "What is 2+2?", "It is 4.", max_turns=10)
    msgs = store.get("t1")
    assert isinstance(msgs[0], HumanMessage)
    assert isinstance(msgs[1], AIMessage)
    assert msgs[0].content == "What is 2+2?"
    assert msgs[1].content == "It is 4."
