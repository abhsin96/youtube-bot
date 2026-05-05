"""Integration tests for POST /ask and POST /ask/stream.

All external I/O (vector store, LLM) is mocked; only the HTTP layer and
request/response shapes are exercised end-to-end against a real FastAPI app.
"""

import contextlib
import json
from unittest.mock import MagicMock, patch

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
    return Settings(openai_api_key="sk-test", _env_file=None)


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
# POST /ask
# ---------------------------------------------------------------------------


class TestAskEndpoint:
    @patch("src.main.build_ask_graph")
    @patch("src.main.get_embeddings")
    def test_happy_path_returns_200(self, _mock_emb, mock_build, client):
        mock_build.return_value.invoke.return_value = _graph_result()
        resp = client.post("/ask", json={"video_id": "vid1", "question": "q?"})
        assert resp.status_code == 200

    @patch("src.main.build_ask_graph")
    @patch("src.main.get_embeddings")
    def test_happy_path_response_shape(self, _mock_emb, mock_build, client):
        mock_build.return_value.invoke.return_value = _graph_result()
        data = client.post("/ask", json={"video_id": "vid1", "question": "q?"}).json()

        assert data["answer"] == "The answer is 42."
        assert data["tokens_used"] == 150
        assert data["refused"] is False
        assert len(data["citations"]) == 1
        cit = data["citations"][0]
        assert cit["chunk_id"] == "c1"
        assert cit["start_ts"] == 5.0
        assert cit["end_ts"] == 10.0
        assert cit["text"] == "hello world"

    @patch("src.main.build_ask_graph")
    @patch("src.main.get_embeddings")
    def test_citations_are_subset_of_retrieved_chunks(self, _mock_emb, mock_build, client):
        all_citations = [
            {"chunk_id": "c1", "start_ts": 5.0, "end_ts": 10.0, "text": "chunk one"},
            {"chunk_id": "c2", "start_ts": 20.0, "end_ts": 30.0, "text": "chunk two"},
        ]
        mock_build.return_value.invoke.return_value = _graph_result(citations=all_citations)
        data = client.post("/ask", json={"video_id": "vid1", "question": "q?"}).json()

        returned_ids = {c["chunk_id"] for c in data["citations"]}
        all_ids = {"c1", "c2"}
        assert returned_ids.issubset(all_ids)

    @patch("src.main.build_ask_graph")
    @patch("src.main.get_embeddings")
    def test_not_ingested_returns_404(self, _mock_emb, mock_build, client):
        mock_build.return_value.invoke.return_value = _graph_result(
            answer="", citations=[], tokens_used=None, error=VIDEO_NOT_INGESTED
        )
        resp = client.post("/ask", json={"video_id": "missing", "question": "q?"})
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "VIDEO_NOT_INGESTED"

    @patch("src.main.build_ask_graph")
    @patch("src.main.get_embeddings")
    def test_refusal_when_no_relevant_chunks(self, _mock_emb, mock_build, client):
        mock_build.return_value.invoke.return_value = _graph_result(
            answer="I'm sorry, that information isn't available in the video transcript.",
            citations=[],
            tokens_used=None,
            refused=True,
        )
        data = client.post("/ask", json={"video_id": "vid1", "question": "q?"}).json()
        assert data["refused"] is True
        assert data["citations"] == []
        assert "isn't available" in data["answer"]

    @patch("src.main.build_ask_graph")
    @patch("src.main.get_embeddings")
    def test_conversation_history_forwarded_to_graph(self, _mock_emb, mock_build, client):
        from langchain_core.messages import HumanMessage

        mock_graph = mock_build.return_value
        mock_graph.invoke.return_value = _graph_result()
        history = [{"role": "user", "content": "prior question"}]

        client.post(
            "/ask",
            json={"video_id": "vid1", "question": "follow up?", "conversation_history": history},
        )

        call_state = mock_graph.invoke.call_args[0][0]
        assert len(call_state["history"]) == 1
        assert isinstance(call_state["history"][0], HumanMessage)
        assert call_state["history"][0].content == "prior question"

    @patch("src.main.build_ask_graph")
    @patch("src.main.get_embeddings")
    def test_k_forwarded_to_graph(self, _mock_emb, mock_build, client):
        mock_graph = mock_build.return_value
        mock_graph.invoke.return_value = _graph_result()

        client.post("/ask", json={"video_id": "vid1", "question": "q?", "k": 3})

        call_state = mock_graph.invoke.call_args[0][0]
        assert call_state["k"] == 3

    @patch("src.main.build_ask_graph")
    @patch("src.main.get_embeddings")
    def test_tokens_used_none_is_allowed(self, _mock_emb, mock_build, client):
        mock_build.return_value.invoke.return_value = _graph_result(tokens_used=None)
        data = client.post("/ask", json={"video_id": "vid1", "question": "q?"}).json()
        assert data["tokens_used"] is None


# ---------------------------------------------------------------------------
# Conversation history — schema validation, conversion, and cap
# ---------------------------------------------------------------------------


class TestConversationHistory:
    @patch("src.main.build_ask_graph")
    @patch("src.main.get_embeddings")
    def test_invalid_role_returns_422(self, _mock_emb, _mock_build, client):
        resp = client.post(
            "/ask",
            json={
                "video_id": "vid1",
                "question": "q?",
                "conversation_history": [{"role": "human", "content": "hi"}],
            },
        )
        assert resp.status_code == 422

    @patch("src.main.build_ask_graph")
    @patch("src.main.get_embeddings")
    def test_user_turn_becomes_human_message(self, _mock_emb, mock_build, client):
        from langchain_core.messages import HumanMessage

        mock_graph = mock_build.return_value
        mock_graph.invoke.return_value = _graph_result()

        client.post(
            "/ask",
            json={
                "video_id": "vid1",
                "question": "q?",
                "conversation_history": [{"role": "user", "content": "hello"}],
            },
        )

        history = mock_graph.invoke.call_args[0][0]["history"]
        assert isinstance(history[0], HumanMessage)
        assert history[0].content == "hello"

    @patch("src.main.build_ask_graph")
    @patch("src.main.get_embeddings")
    def test_assistant_turn_becomes_ai_message(self, _mock_emb, mock_build, client):
        from langchain_core.messages import AIMessage

        mock_graph = mock_build.return_value
        mock_graph.invoke.return_value = _graph_result()

        client.post(
            "/ask",
            json={
                "video_id": "vid1",
                "question": "q?",
                "conversation_history": [{"role": "assistant", "content": "hi back"}],
            },
        )

        history = mock_graph.invoke.call_args[0][0]["history"]
        assert isinstance(history[0], AIMessage)
        assert history[0].content == "hi back"

    @patch("src.main.build_ask_graph")
    @patch("src.main.get_embeddings")
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
            "/ask",
            json={"video_id": "vid1", "question": "q?", "conversation_history": turns},
        )

        history = mock_graph.invoke.call_args[0][0]["history"]
        assert len(history) == 10  # capped to max_history_turns

    @patch("src.main.build_ask_graph")
    @patch("src.main.get_embeddings")
    def test_cap_keeps_most_recent_turns(self, _mock_emb, mock_build, client):
        """After capping, the last N messages are kept (not the first N)."""
        mock_graph = mock_build.return_value
        mock_graph.invoke.return_value = _graph_result()

        turns = [{"role": "user", "content": f"msg {i}"} for i in range(30)]

        client.post(
            "/ask",
            json={"video_id": "vid1", "question": "q?", "conversation_history": turns},
        )

        history = mock_graph.invoke.call_args[0][0]["history"]
        # The last 10 messages should be msg 20–29
        assert history[-1].content == "msg 29"
        assert history[0].content == "msg 20"

    @patch("src.main.build_ask_graph")
    @patch("src.main.get_embeddings")
    def test_empty_history_is_valid(self, _mock_emb, mock_build, client):
        mock_graph = mock_build.return_value
        mock_graph.invoke.return_value = _graph_result()

        client.post("/ask", json={"video_id": "vid1", "question": "q?"})

        history = mock_graph.invoke.call_args[0][0]["history"]
        assert history == []


# ---------------------------------------------------------------------------
# POST /ask/stream
# ---------------------------------------------------------------------------


class TestAskStreamEndpoint:
    @patch("src.main.collection_exists", return_value=False)
    @patch("src.main.get_embeddings")
    def test_not_ingested_returns_404(self, _mock_emb, _mock_ce, client):
        resp = client.post("/ask/stream", json={"video_id": "missing", "question": "q?"})
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "VIDEO_NOT_INGESTED"

    @patch("src.main.collection_exists", return_value=True)
    @patch("src.main.build_retriever")
    @patch("src.main.get_embeddings")
    def test_no_chunks_emits_refusal_done_event(self, _mock_emb, mock_br, _mock_ce, client):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_br.return_value = mock_retriever

        resp = client.post("/ask/stream", json={"video_id": "vid1", "question": "q?"})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        events = _parse_sse(resp.text)
        done = next(e for e in events if e.get("type") == "done")
        assert done["refused"] is True
        assert done["citations"] == []
        assert "isn't available" in done["answer"]

    @patch("src.main.collection_exists", return_value=True)
    @patch("src.main.build_retriever")
    @patch("src.main.get_embeddings")
    def test_happy_path_emits_token_and_done_events(self, _mock_emb, mock_br, _mock_ce, client):
        from langchain_core.documents import Document
        from langchain_core.messages import AIMessageChunk
        from langchain_core.outputs import ChatGenerationChunk
        from langchain_openai import ChatOpenAI

        doc = Document(
            page_content="hello world",
            metadata={"chunk_id": "c1", "start_ts": 5.0, "end_ts": 10.0},
        )
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [doc]
        mock_br.return_value = mock_retriever

        async def fake_astream(self_llm, messages, stop=None, run_manager=None, **kwargs):
            for content in ["The ", "answer."]:
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=content))
                if run_manager:
                    await run_manager.on_llm_new_token(content, chunk=chunk)
                yield chunk

        with patch.object(ChatOpenAI, "_astream", fake_astream):
            resp = client.post("/ask/stream", json={"video_id": "vid1", "question": "q?"})

        assert resp.status_code == 200
        events = _parse_sse(resp.text)
        types = [e["type"] for e in events]
        assert "done" in types

        done = next(e for e in events if e["type"] == "done")
        assert "citations" in done
        assert "answer" in done
        assert "tokens_used" in done

    @patch("src.main.collection_exists", return_value=True)
    @patch("src.main.build_retriever")
    @patch("src.main.get_embeddings")
    def test_done_event_citations_match_retrieved_chunks(
        self, _mock_emb, mock_br, _mock_ce, client
    ):
        from langchain_core.documents import Document
        from langchain_core.messages import AIMessageChunk
        from langchain_core.outputs import ChatGenerationChunk
        from langchain_openai import ChatOpenAI

        doc = Document(
            page_content="transcript text",
            metadata={"chunk_id": "cX", "start_ts": 1.0, "end_ts": 5.0},
        )
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [doc]
        mock_br.return_value = mock_retriever

        async def fake_astream(self_llm, messages, stop=None, run_manager=None, **kwargs):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content="Answer text."))
            if run_manager:
                await run_manager.on_llm_new_token("Answer text.", chunk=chunk)
            yield chunk

        with patch.object(ChatOpenAI, "_astream", fake_astream):
            resp = client.post("/ask/stream", json={"video_id": "vid1", "question": "q?"})

        events = _parse_sse(resp.text)
        done = next(e for e in events if e["type"] == "done")

        assert len(done["citations"]) == 1
        assert done["citations"][0]["chunk_id"] == "cX"

    @patch("src.main.collection_exists", return_value=True)
    @patch("src.main.build_retriever")
    @patch("src.main.get_embeddings")
    def test_llm_exception_emits_error_sse_event(self, _mock_emb, mock_br, _mock_ce, client):
        """LLM failure inside event_stream emits an error SSE event instead of crashing."""
        from langchain_core.documents import Document
        from langchain_openai import ChatOpenAI

        doc = Document(
            page_content="transcript text",
            metadata={"chunk_id": "c1", "start_ts": 1.0, "end_ts": 5.0},
        )
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [doc]
        mock_br.return_value = mock_retriever

        async def raising_astream(self_llm, messages, stop=None, run_manager=None, **kwargs):
            raise RuntimeError("openai network error")
            yield  # pragma: no cover — makes this an async generator

        with patch.object(ChatOpenAI, "_astream", raising_astream):
            resp = client.post("/ask/stream", json={"video_id": "vid1", "question": "q?"})

        assert resp.status_code == 200
        events = _parse_sse(resp.text)
        error_events = [e for e in events if e.get("type") == "error"]
        assert len(error_events) == 1
        assert error_events[0]["code"] == "INTERNAL_ERROR"


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
