from __future__ import annotations

import json

import redis as redis_lib
import structlog
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    messages_from_dict,
    messages_to_dict,
)

from src.config import Settings
from src.schemas import ConversationTurn

logger = structlog.get_logger(__name__)


class ThreadStore:
    """In-memory conversation history store keyed by thread_id.

    Each stored entry is a flat list of alternating HumanMessage / AIMessage
    objects.  The list is capped at *max_turns* turn-pairs (2 × max_turns
    messages) by evicting the oldest pair whenever the cap is exceeded.
    """

    def __init__(self) -> None:
        self._store: dict[str, list[BaseMessage]] = {}

    def get(self, thread_id: str) -> list[BaseMessage]:
        return list(self._store.get(thread_id, []))

    def has(self, thread_id: str) -> bool:
        return thread_id in self._store

    def append(self, thread_id: str, question: str, answer: str, *, max_turns: int) -> None:
        msgs = self._store.setdefault(thread_id, [])
        msgs.append(HumanMessage(content=question))
        msgs.append(AIMessage(content=answer))
        cap = max_turns * 2
        if len(msgs) > cap:
            self._store[thread_id] = msgs[-cap:]

    def delete(self, thread_id: str) -> bool:
        return self._store.pop(thread_id, None) is not None


class RedisThreadStore:
    """Redis-backed conversation history store with the same interface as ThreadStore.

    Messages are serialized with langchain_core's ``messages_to_dict`` /
    ``messages_from_dict`` and stored as individual JSON strings in a Redis list.

    List layout (LPUSH → newest pair at the head):
        [q_new, a_new, q_prev, a_prev, ...]

    ``get()`` reads the list and reconstructs chronological order by reading
    pairs from the tail.  ``append()`` uses a pipeline so the LPUSH and LTRIM
    are sent in a single round-trip.
    """

    _KEY_PREFIX = "thread:"

    def __init__(self, client: redis_lib.Redis) -> None:  # type: ignore[type-arg]
        self._client = client

    def _key(self, thread_id: str) -> str:
        return f"{self._KEY_PREFIX}{thread_id}"

    def get(self, thread_id: str) -> list[BaseMessage]:
        raw: list[str] = self._client.lrange(self._key(thread_id), 0, -1)
        if not raw:
            return []
        ordered: list[str] = []
        for i in range(len(raw) - 2, -1, -2):
            ordered.append(raw[i])
            ordered.append(raw[i + 1])
        return messages_from_dict([json.loads(item) for item in ordered])

    def has(self, thread_id: str) -> bool:
        return bool(self._client.exists(self._key(thread_id)))

    def append(self, thread_id: str, question: str, answer: str, *, max_turns: int) -> None:
        key = self._key(thread_id)
        cap = max_turns * 2
        q_json = json.dumps(messages_to_dict([HumanMessage(content=question)])[0])
        a_json = json.dumps(messages_to_dict([AIMessage(content=answer)])[0])
        with self._client.pipeline() as pipe:
            pipe.lpush(key, a_json, q_json)
            pipe.ltrim(key, 0, cap - 1)
            pipe.execute()

    def delete(self, thread_id: str) -> bool:
        return bool(self._client.delete(self._key(thread_id)))


AnyThreadStore = ThreadStore | RedisThreadStore


def _to_langchain_history(
    turns: list[ConversationTurn],
    max_turns: int,
) -> list[BaseMessage]:
    """Convert validated ConversationTurn objects to LangChain message objects.

    Applies the server-side cap (last *max_turns* messages) as defense-in-depth
    so runaway front-end history can never blow the context budget.
    """
    capped = turns[-max_turns:] if len(turns) > max_turns else turns
    return [
        HumanMessage(content=t.content) if t.role == "user" else AIMessage(content=t.content)
        for t in capped
    ]


def _resolve_history(
    thread_store: AnyThreadStore,
    thread_id: str,
    body,
    settings: Settings,
) -> list[BaseMessage]:
    """Return conversation history for the ask graph.

    Uses server-stored history when the thread exists (avoids O(n) client
    payload); otherwise converts the client-supplied turns.
    """
    if thread_store.has(thread_id):
        return thread_store.get(thread_id)
    return _to_langchain_history(body.conversation_history, settings.max_history_turns)
