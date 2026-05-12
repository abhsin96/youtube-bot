"""Integration tests for POST /query (non-streaming and streaming modes).

All external I/O (vector store, LLM) is mocked; only the HTTP layer and
request/response shapes are exercised end-to-end against a real FastAPI app.
"""

import contextlib
import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from graphs.ask_graph import VIDEO_NOT_INGESTED
from src.config import Settings
from src.main import create_app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def settings():
    return Settings(openai_api_key="sk-test", redis_url="", _env_file=None)


@pytest.fixture()
def client(settings):
    app = create_app(settings)
    return TestClient(app)


def _graph_result(
    answer="The answer is 42.",
    citations=None,
    tokens_used=150,
    refused=False,
    error=None,
):
    if citations is None:
        citations = [{"chunk_id": "c1", "start_ts": 5.0, "end_ts": 10.0, "text": "hello world"}]
    return {
        "answer": answer,
        "citations": citations,
        "tokens_used": tokens_used,
        "refused": refused,
        "error": error,
    }


# ---------------------------------------------------------------------------
# POST /query (advanced mode, non-streaming)
# ---------------------------------------------------------------------------


class TestQueryEndpoint:
    @patch("src.routers.query.build_ask_graph")
    @patch("src.routers.query.get_embeddings")
    def test_happy_path_returns_200(self, _mock_emb, mock_build, client):
        mock_build.return_value.invoke.return_value = _graph_result()
        resp = client.post("/query", json={"video_id": "vid1", "question": "q?"})
        assert resp.status_code == 200

    @patch("src.routers.query.build_ask_graph")
    @patch("src.routers.query.get_embeddings")
    def test_happy_path_response_shape(self, _mock_emb, mock_build, client):
        mock_build.return_value.invoke.return_value = _graph_result()
        data = client.post("/query", json={"video_id": "vid1", "question": "q?"}).json()

        assert data["answer"] == "The answer is 42."
        assert data["tokens_used"] == 150
        assert data["refused"] is False
        assert len(data["citations"]) == 1
        cit = data["citations"][0]
        assert cit["chunk_id"] == "c1"
        assert cit["start_ts"] == 5.0
        assert cit["end_ts"] == 10.0
        assert cit["text"] == "hello world"

    @patch("src.routers.query.build_ask_graph")
    @patch("src.routers.query.get_embeddings")
    def test_citations_are_subset_of_retrieved_chunks(self, _mock_emb, mock_build, client):
        all_citations = [
            {"chunk_id": "c1", "start_ts": 5.0, "end_ts": 10.0, "text": "chunk one"},
            {"chunk_id": "c2", "start_ts": 20.0, "end_ts": 30.0, "text": "chunk two"},
        ]
        mock_build.return_value.invoke.return_value = _graph_result(citations=all_citations)
        data = client.post("/query", json={"video_id": "vid1", "question": "q?"}).json()

        returned_ids = {c["chunk_id"] for c in data["citations"]}
        all_ids = {"c1", "c2"}
        assert returned_ids.issubset(all_ids)

    @patch("src.routers.query.build_ask_graph")
    @patch("src.routers.query.get_embeddings")
    def test_not_ingested_returns_404(self, _mock_emb, mock_build, client):
        mock_build.return_value.invoke.return_value = _graph_result(
            answer="", citations=[], tokens_used=None, error=VIDEO_NOT_INGESTED
        )
        resp = client.post("/query", json={"video_id": "missing", "question": "q?"})
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "VIDEO_NOT_INGESTED"

    @patch("src.routers.query.build_ask_graph")
    @patch("src.routers.query.get_embeddings")
    def test_refusal_when_no_relevant_chunks(self, _mock_emb, mock_build, client):
        mock_build.return_value.invoke.return_value = _graph_result(
            answer="I'm sorry, that information isn't available in the video transcript.",
            citations=[],
            tokens_used=None,
            refused=True,
        )
        data = client.post("/query", json={"video_id": "vid1", "question": "q?"}).json()
        assert data["refused"] is True
        assert data["citations"] == []
        assert "isn't available" in data["answer"]

    @patch("src.routers.query.build_ask_graph")
    @patch("src.routers.query.get_embeddings")
    def test_conversation_history_forwarded_to_graph(self, _mock_emb, mock_build, client):
        from langchain_core.messages import HumanMessage

        mock_graph = mock_build.return_value
        mock_graph.invoke.return_value = _graph_result()
        history = [{"role": "user", "content": "prior question"}]

        client.post(
            "/query",
            json={"video_id": "vid1", "question": "follow up?", "conversation_history": history},
        )

        call_state = mock_graph.invoke.call_args[0][0]
        assert len(call_state["history"]) == 1
        assert isinstance(call_state["history"][0], HumanMessage)
        assert call_state["history"][0].content == "prior question"

    @patch("src.routers.query.build_ask_graph")
    @patch("src.routers.query.get_embeddings")
    def test_k_forwarded_to_graph(self, _mock_emb, mock_build, client):
        mock_graph = mock_build.return_value
        mock_graph.invoke.return_value = _graph_result()

        client.post("/query", json={"video_id": "vid1", "question": "q?", "k": 3})

        call_state = mock_graph.invoke.call_args[0][0]
        assert call_state["k"] == 3

    @patch("src.routers.query.build_ask_graph")
    @patch("src.routers.query.get_embeddings")
    def test_tokens_used_none_is_allowed(self, _mock_emb, mock_build, client):
        mock_build.return_value.invoke.return_value = _graph_result(tokens_used=None)
        data = client.post("/query", json={"video_id": "vid1", "question": "q?"}).json()
        assert data["tokens_used"] is None


# ---------------------------------------------------------------------------
# Conversation history — schema validation, conversion, and cap
# ---------------------------------------------------------------------------


class TestConversationHistory:
    @patch("src.routers.query.build_ask_graph")
    @patch("src.routers.query.get_embeddings")
    def test_invalid_role_returns_422(self, _mock_emb, _mock_build, client):
        resp = client.post(
            "/query",
            json={
                "video_id": "vid1",
                "question": "q?",
                "conversation_history": [{"role": "human", "content": "hi"}],
            },
        )
        assert resp.status_code == 422

    @patch("src.routers.query.build_ask_graph")
    @patch("src.routers.query.get_embeddings")
    def test_user_turn_becomes_human_message(self, _mock_emb, mock_build, client):
        from langchain_core.messages import HumanMessage

        mock_graph = mock_build.return_value
        mock_graph.invoke.return_value = _graph_result()

        client.post(
            "/query",
            json={
                "video_id": "vid1",
                "question": "q?",
                "conversation_history": [{"role": "user", "content": "hello"}],
            },
        )

        history = mock_graph.invoke.call_args[0][0]["history"]
        assert isinstance(history[0], HumanMessage)
        assert history[0].content == "hello"

    @patch("src.routers.query.build_ask_graph")
    @patch("src.routers.query.get_embeddings")
    def test_assistant_turn_becomes_ai_message(self, _mock_emb, mock_build, client):
        from langchain_core.messages import AIMessage

        mock_graph = mock_build.return_value
        mock_graph.invoke.return_value = _graph_result()

        client.post(
            "/query",
            json={
                "video_id": "vid1",
                "question": "q?",
                "conversation_history": [{"role": "assistant", "content": "hi back"}],
            },
        )

        history = mock_graph.invoke.call_args[0][0]["history"]
        assert isinstance(history[0], AIMessage)
        assert history[0].content == "hi back"

    @patch("src.routers.query.build_ask_graph")
    @patch("src.routers.query.get_embeddings")
    def test_cap_enforced_30_turns_yields_10(self, _mock_emb, mock_build, client):
        """Server caps history at max_history_turns (default 10) even when FE sends 30."""
        mock_graph = mock_build.return_value
        mock_graph.invoke.return_value = _graph_result()

        # Alternate user/assistant for 30 messages
        turns = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(30)
        ]

        client.post(
            "/query",
            json={"video_id": "vid1", "question": "q?", "conversation_history": turns},
        )

        history = mock_graph.invoke.call_args[0][0]["history"]
        assert len(history) == 10  # capped to max_history_turns

    @patch("src.routers.query.build_ask_graph")
    @patch("src.routers.query.get_embeddings")
    def test_cap_keeps_most_recent_turns(self, _mock_emb, mock_build, client):
        """After capping, the last N messages are kept (not the first N)."""
        mock_graph = mock_build.return_value
        mock_graph.invoke.return_value = _graph_result()

        turns = [{"role": "user", "content": f"msg {i}"} for i in range(30)]

        client.post(
            "/query",
            json={"video_id": "vid1", "question": "q?", "conversation_history": turns},
        )

        history = mock_graph.invoke.call_args[0][0]["history"]
        # The last 10 messages should be msg 20–29
        assert history[-1].content == "msg 29"
        assert history[0].content == "msg 20"

    @patch("src.routers.query.build_ask_graph")
    @patch("src.routers.query.get_embeddings")
    def test_empty_history_is_valid(self, _mock_emb, mock_build, client):
        mock_graph = mock_build.return_value
        mock_graph.invoke.return_value = _graph_result()

        client.post("/query", json={"video_id": "vid1", "question": "q?"})

        history = mock_graph.invoke.call_args[0][0]["history"]
        assert history == []


# ---------------------------------------------------------------------------
# Ask graph caching
# ---------------------------------------------------------------------------


class TestGraphCaching:
    @patch("src.routers.query.build_ask_graph")
    @patch("src.routers.query.get_embeddings")
    def test_graph_built_once_for_same_key(self, _mock_emb, mock_build, client):
        """build_ask_graph is called only once when successive requests share the same key."""
        mock_build.return_value.invoke.return_value = _graph_result()
        client.post("/query", json={"video_id": "v", "question": "q?"})
        client.post("/query", json={"video_id": "v", "question": "q?"})
        assert mock_build.call_count == 1

    @patch("src.routers.query.build_ask_graph")
    @patch("src.routers.query.get_embeddings")
    @patch("src.dependencies.get_openai_key")
    def test_graph_rebuilt_when_api_key_changes(self, mock_get_key, _mock_emb, mock_build, client):
        """build_ask_graph is called again after the API key is rotated."""
        mock_build.return_value.invoke.return_value = _graph_result()

        mock_get_key.return_value = "sk-key-a"
        client.post("/query", json={"video_id": "v", "question": "q?"})
        assert mock_build.call_count == 1

        mock_get_key.return_value = "sk-key-b"
        client.post("/query", json={"video_id": "v", "question": "q?"})
        assert mock_build.call_count == 2


# ---------------------------------------------------------------------------
# POST /query?stream=true  (streaming mode)
# ---------------------------------------------------------------------------


class TestQueryStreamEndpoint:
    @patch("src.vector_store.collection_exists", return_value=False)
    @patch("src.routers.query.get_embeddings")
    def test_not_ingested_returns_404(self, _mock_emb, _mock_ce, client):
        resp = client.post("/query", json={"video_id": "missing", "question": "q?", "stream": True})
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "VIDEO_NOT_INGESTED"

    @patch("src.routers.query.build_ask_graph")
    @patch("src.vector_store.collection_exists", return_value=True)
    @patch("src.routers.query.get_embeddings")
    def test_no_chunks_emits_refusal_done_event(self, _mock_emb, _mock_ce, mock_build, client):
        async def fake_astream_events(state, **kwargs):
            yield {
                "event": "on_chain_end",
                "name": "LangGraph",
                "data": {
                    "output": {
                        "error": None,
                        "answer": "I'm sorry, that information isn't available in the video transcript.",
                        "refused": True,
                        "citations": [],
                        "tokens_used": None,
                    }
                },
            }

        mock_build.return_value.astream_events = fake_astream_events

        resp = client.post("/query", json={"video_id": "vid1", "question": "q?", "stream": True})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        events = _parse_sse(resp.text)
        done = next(e for e in events if e.get("type") == "done")
        assert done["refused"] is True
        assert done["citations"] == []
        assert "isn't available" in done["answer"]

    @patch("src.routers.query.build_ask_graph")
    @patch("src.vector_store.collection_exists", return_value=True)
    @patch("src.routers.query.get_embeddings")
    def test_happy_path_emits_token_and_done_events(self, _mock_emb, _mock_ce, mock_build, client):
        from langchain_core.messages import AIMessageChunk

        async def fake_astream_events(state, **kwargs):
            for content in ["The ", "answer."]:
                yield {
                    "event": "on_chat_model_stream",
                    "metadata": {"langgraph_node": "generate"},
                    "data": {"chunk": AIMessageChunk(content=content)},
                }
            yield {
                "event": "on_chain_end",
                "name": "LangGraph",
                "data": {
                    "output": {
                        "error": None,
                        "answer": "The answer.",
                        "refused": False,
                        "citations": [
                            {
                                "chunk_id": "c1",
                                "start_ts": 5.0,
                                "end_ts": 10.0,
                                "text": "hello world",
                            }
                        ],
                        "tokens_used": 50,
                    }
                },
            }

        mock_build.return_value.astream_events = fake_astream_events

        resp = client.post("/query", json={"video_id": "vid1", "question": "q?", "stream": True})
        assert resp.status_code == 200
        events = _parse_sse(resp.text)
        types = [e["type"] for e in events]
        assert "token" in types
        assert "done" in types

        done = next(e for e in events if e["type"] == "done")
        assert "citations" in done
        assert "answer" in done
        assert "tokens_used" in done

    @patch("src.routers.query.build_ask_graph")
    @patch("src.vector_store.collection_exists", return_value=True)
    @patch("src.routers.query.get_embeddings")
    def test_done_event_citations_match_retrieved_chunks(
        self, _mock_emb, _mock_ce, mock_build, client
    ):
        async def fake_astream_events(state, **kwargs):
            yield {
                "event": "on_chain_end",
                "name": "LangGraph",
                "data": {
                    "output": {
                        "error": None,
                        "answer": "Answer text.",
                        "refused": False,
                        "citations": [
                            {
                                "chunk_id": "cX",
                                "start_ts": 1.0,
                                "end_ts": 5.0,
                                "text": "transcript text",
                            }
                        ],
                        "tokens_used": None,
                    }
                },
            }

        mock_build.return_value.astream_events = fake_astream_events

        resp = client.post("/query", json={"video_id": "vid1", "question": "q?", "stream": True})
        events = _parse_sse(resp.text)
        done = next(e for e in events if e["type"] == "done")

        assert len(done["citations"]) == 1
        assert done["citations"][0]["chunk_id"] == "cX"

    @patch("src.routers.query.build_ask_graph")
    @patch("src.vector_store.collection_exists", return_value=True)
    @patch("src.routers.query.get_embeddings")
    def test_llm_exception_emits_error_sse_event(self, _mock_emb, _mock_ce, mock_build, client):
        """LLM failure inside the graph stream emits an error SSE event instead of crashing."""

        async def raising_astream_events(state, **kwargs):
            raise RuntimeError("openai network error")
            yield  # pragma: no cover — makes this an async generator

        mock_build.return_value.astream_events = raising_astream_events

        resp = client.post("/query", json={"video_id": "vid1", "question": "q?", "stream": True})
        assert resp.status_code == 200
        events = _parse_sse(resp.text)
        error_events = [e for e in events if e.get("type") == "error"]
        assert len(error_events) == 1
        assert error_events[0]["code"] == "INTERNAL_ERROR"


# ---------------------------------------------------------------------------
# Server-side history (thread store)
# ---------------------------------------------------------------------------


class TestServerSideHistory:
    @patch("src.routers.query.build_ask_graph")
    @patch("src.routers.query.get_embeddings")
    def test_history_appended_after_successful_query(self, _mock_emb, mock_build, client):
        """A successful /query stores the turn so the next request sees it."""
        from langchain_core.messages import AIMessage, HumanMessage

        mock_graph = mock_build.return_value
        mock_graph.invoke.side_effect = [
            _graph_result(answer="First answer."),
            _graph_result(answer="Second answer."),
        ]

        client.post("/query", json={"video_id": "v", "question": "q1?", "thread_id": "th1"})
        client.post("/query", json={"video_id": "v", "question": "q2?", "thread_id": "th1"})

        second_call_state = mock_graph.invoke.call_args_list[1][0][0]
        assert len(second_call_state["history"]) == 2
        assert isinstance(second_call_state["history"][0], HumanMessage)
        assert second_call_state["history"][0].content == "q1?"
        assert isinstance(second_call_state["history"][1], AIMessage)
        assert second_call_state["history"][1].content == "First answer."

    @patch("src.routers.query.build_ask_graph")
    @patch("src.routers.query.get_embeddings")
    def test_server_history_overrides_client_history(self, _mock_emb, mock_build, client):
        """When a thread exists in the store, client-supplied history is ignored."""
        mock_graph = mock_build.return_value
        mock_graph.invoke.side_effect = [
            _graph_result(answer="stored answer"),
            _graph_result(answer="second"),
        ]

        client.post("/query", json={"video_id": "v", "question": "stored q?", "thread_id": "th2"})
        client.post(
            "/query",
            json={
                "video_id": "v",
                "question": "q2?",
                "thread_id": "th2",
                "conversation_history": [{"role": "user", "content": "WRONG HISTORY"}],
            },
        )

        second_state = mock_graph.invoke.call_args_list[1][0][0]
        assert second_state["history"][0].content == "stored q?"
        assert second_state["history"][1].content == "stored answer"

    @patch("src.routers.query.build_ask_graph")
    @patch("src.routers.query.get_embeddings")
    def test_client_history_seeds_brand_new_thread(self, _mock_emb, mock_build, client):
        """conversation_history in the request seeds a thread that doesn't exist yet."""
        from langchain_core.messages import HumanMessage

        mock_build.return_value.invoke.return_value = _graph_result()
        client.post(
            "/query",
            json={
                "video_id": "v",
                "question": "q?",
                "thread_id": "brand-new",
                "conversation_history": [{"role": "user", "content": "seeded turn"}],
            },
        )

        state = mock_build.return_value.invoke.call_args[0][0]
        assert isinstance(state["history"][0], HumanMessage)
        assert state["history"][0].content == "seeded turn"

    @patch("src.routers.query.build_ask_graph")
    @patch("src.routers.query.get_embeddings")
    def test_history_not_stored_on_graph_error(self, _mock_emb, mock_build, client):
        """A failed /query (e.g. VIDEO_NOT_INGESTED) does not write to the store."""
        mock_graph = mock_build.return_value
        mock_graph.invoke.side_effect = [
            _graph_result(error=VIDEO_NOT_INGESTED),
            _graph_result(answer="ok"),
        ]

        client.post("/query", json={"video_id": "v", "question": "q1?", "thread_id": "th3"})
        client.post("/query", json={"video_id": "v", "question": "q2?", "thread_id": "th3"})

        second_state = mock_graph.invoke.call_args_list[1][0][0]
        assert second_state["history"] == []

    @patch("src.routers.query.build_ask_graph")
    @patch("src.routers.query.get_embeddings")
    def test_history_cap_per_thread(self, _mock_emb, mock_build, client):
        """History is evicted when it exceeds max_history_turns pairs per thread."""
        import chromadb

        from src.config import Settings
        from src.main import create_app

        small_settings = Settings(openai_api_key="sk-test", max_history_turns=2, _env_file=None)
        small_client = TestClient(
            create_app(small_settings, chroma_client=chromadb.EphemeralClient())
        )

        with (
            patch("src.routers.query.build_ask_graph") as mock_b,
            patch("src.routers.query.get_embeddings"),
        ):
            mock_b.return_value.invoke.return_value = _graph_result(answer="ans")
            # 3 turns; after turn 3 the store should evict turn 1 (cap = 2 pairs)
            for i in range(3):
                small_client.post(
                    "/query",
                    json={"video_id": "v", "question": f"q{i}?", "thread_id": "cap-t"},
                )
            # The 4th request receives capped history: only turns 2 and 3
            small_client.post(
                "/query",
                json={"video_id": "v", "question": "q3?", "thread_id": "cap-t"},
            )

        fourth_state = mock_b.return_value.invoke.call_args_list[3][0][0]
        assert len(fourth_state["history"]) == 4  # 2 pairs × 2 messages
        assert fourth_state["history"][0].content == "q1?"


# ---------------------------------------------------------------------------
# DELETE /threads/{thread_id}
# ---------------------------------------------------------------------------


class TestDeleteThread:
    @patch("src.routers.query.build_ask_graph")
    @patch("src.routers.query.get_embeddings")
    def test_delete_existing_thread_returns_204(self, _mock_emb, mock_build, client):
        mock_build.return_value.invoke.return_value = _graph_result()
        client.post("/query", json={"video_id": "v", "question": "q?", "thread_id": "del-t1"})

        resp = client.delete("/threads/del-t1")
        assert resp.status_code == 204

    @patch("src.routers.query.build_ask_graph")
    @patch("src.routers.query.get_embeddings")
    def test_delete_clears_history_for_next_request(self, _mock_emb, mock_build, client):
        """After DELETE, the next /query sees no history for that thread."""
        mock_graph = mock_build.return_value
        mock_graph.invoke.side_effect = [
            _graph_result(answer="stored ans"),
            _graph_result(answer="fresh ans"),
        ]

        client.post("/query", json={"video_id": "v", "question": "q1?", "thread_id": "del-t2"})
        client.delete("/threads/del-t2")
        client.post("/query", json={"video_id": "v", "question": "q2?", "thread_id": "del-t2"})

        second_state = mock_graph.invoke.call_args_list[1][0][0]
        assert second_state["history"] == []

    def test_delete_nonexistent_thread_returns_404(self, client):
        resp = client.delete("/threads/this-does-not-exist")
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "THREAD_NOT_FOUND"

    @patch("src.routers.query.build_ask_graph")
    @patch("src.routers.query.get_embeddings")
    def test_delete_is_idempotent_second_call_returns_404(self, _mock_emb, mock_build, client):
        mock_build.return_value.invoke.return_value = _graph_result()
        client.post("/query", json={"video_id": "v", "question": "q?", "thread_id": "del-t3"})
        client.delete("/threads/del-t3")
        resp = client.delete("/threads/del-t3")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_sse(body: str) -> list[dict]:
    """Parse SSE body into a list of JSON event payloads."""
    events = []
    for line in body.splitlines():
        if line.startswith("data: "):
            with contextlib.suppress(json.JSONDecodeError):
                events.append(json.loads(line[6:]))
    return events
