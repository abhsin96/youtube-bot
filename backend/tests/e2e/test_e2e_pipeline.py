"""E2E pipeline tests — no real OpenAI or YouTube API calls.

The mock OpenAI server (session-scoped) handles /v1/embeddings and
/v1/chat/completions deterministically.  Chroma is pre-populated from
fixture_data.py in a fresh temp directory per test.

Coverage
--------
- POST /query  happy path: answer + citations returned
- POST /query  conversation history flows through to the graph
- POST /query  unknown video → 404 VIDEO_NOT_INGESTED
- POST /query (stream=true)  emits token + done events
- POST /ingest  full pipeline (mocked transcript + mock embeddings)
"""

from __future__ import annotations

import contextlib
import json
from unittest.mock import patch

from fastapi.testclient import TestClient

from tests.e2e.fixture_data import FIXTURE_SEGMENTS, FIXTURE_VIDEO_ID
from tests.mock_openai.server import register_response

# ---------------------------------------------------------------------------
# /query — happy path
# ---------------------------------------------------------------------------


class TestQueryE2E:
    def test_query_returns_200_with_answer_and_citations(self, e2e_client: TestClient):
        register_response("list comprehension", "A list comprehension creates a list in one line.")
        resp = e2e_client.post(
            "/query",
            json={"video_id": FIXTURE_VIDEO_ID, "question": "What is a list comprehension?"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"]
        assert isinstance(body["citations"], list)
        assert isinstance(body["refused"], bool)

    def test_query_citations_have_required_fields(self, e2e_client: TestClient):
        resp = e2e_client.post(
            "/query",
            json={"video_id": FIXTURE_VIDEO_ID, "question": "What is a lambda function?"},
        )
        assert resp.status_code == 200
        for cit in resp.json()["citations"]:
            assert "chunk_id" in cit
            assert "start_ts" in cit
            assert "end_ts" in cit
            assert "text" in cit

    def test_query_registered_answer_is_returned(self, e2e_client: TestClient):
        register_response("decorator", "Decorators wrap a function using the @ symbol.")
        resp = e2e_client.post(
            "/query",
            json={"video_id": FIXTURE_VIDEO_ID, "question": "How do decorators work?"},
        )
        assert resp.status_code == 200
        assert "Decorators wrap" in resp.json()["answer"]

    def test_query_unknown_video_returns_404(self, e2e_client: TestClient):
        resp = e2e_client.post(
            "/query",
            json={"video_id": "not_a_real_video_id", "question": "anything?"},
        )
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "VIDEO_NOT_INGESTED"

    def test_query_with_conversation_history(self, e2e_client: TestClient):
        register_response("context manager", "Context managers use the with statement.")
        resp = e2e_client.post(
            "/query",
            json={
                "video_id": FIXTURE_VIDEO_ID,
                "question": "Can you give an example?",
                "conversation_history": [
                    {"role": "user", "content": "What is a context manager?"},
                    {
                        "role": "assistant",
                        "content": "A context manager handles setup and teardown.",
                    },
                ],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["answer"]

    def test_query_tokens_used_is_int_or_none(self, e2e_client: TestClient):
        resp = e2e_client.post(
            "/query",
            json={"video_id": FIXTURE_VIDEO_ID, "question": "What are f-strings?"},
        )
        assert resp.status_code == 200
        tokens = resp.json()["tokens_used"]
        assert tokens is None or isinstance(tokens, int)


# ---------------------------------------------------------------------------
# /query (stream=true) — happy path
# ---------------------------------------------------------------------------


def _parse_sse(body: str) -> list[dict]:
    events = []
    for line in body.splitlines():
        if line.startswith("data: "):
            with contextlib.suppress(json.JSONDecodeError):
                events.append(json.loads(line[6:]))
    return events


class TestQueryStreamE2E:
    def test_stream_returns_200_with_sse_content_type(self, e2e_client: TestClient):
        resp = e2e_client.post(
            "/query",
            json={"video_id": FIXTURE_VIDEO_ID, "question": "What are generators?", "stream": True},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    def test_stream_emits_done_event(self, e2e_client: TestClient):
        register_response("generator", "Generator expressions create lazy sequences.")
        resp = e2e_client.post(
            "/query",
            json={
                "video_id": FIXTURE_VIDEO_ID,
                "question": "Explain generator expressions.",
                "stream": True,
            },
        )
        events = _parse_sse(resp.text)
        types = {e.get("type") for e in events}
        assert "done" in types

    def test_stream_done_event_has_answer_and_citations(self, e2e_client: TestClient):
        resp = e2e_client.post(
            "/query",
            json={"video_id": FIXTURE_VIDEO_ID, "question": "What is unpacking?", "stream": True},
        )
        events = _parse_sse(resp.text)
        done = next(e for e in events if e.get("type") == "done")
        assert "answer" in done
        assert "citations" in done
        assert "refused" in done

    def test_stream_token_events_emitted(self, e2e_client: TestClient):
        register_response("type hint", "Type hints use the colon syntax for parameters.")
        resp = e2e_client.post(
            "/query",
            json={
                "video_id": FIXTURE_VIDEO_ID,
                "question": "What are type hints?",
                "stream": True,
            },
        )
        events = _parse_sse(resp.text)
        token_events = [e for e in events if e.get("type") == "token"]
        assert len(token_events) >= 1

    def test_stream_unknown_video_returns_404(self, e2e_client: TestClient):
        resp = e2e_client.post(
            "/query",
            json={"video_id": "nonexistent_xyz", "question": "q?", "stream": True},
        )
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "VIDEO_NOT_INGESTED"


# ---------------------------------------------------------------------------
# POST /ingest — full pipeline with mocked transcript API
# ---------------------------------------------------------------------------


class TestIngestE2E:
    def test_ingest_full_pipeline_returns_done(self, e2e_settings, chroma_client, mock_openai_url):
        from src.main import create_app

        client = TestClient(
            create_app(e2e_settings, chroma_client=chroma_client),
            raise_server_exceptions=False,
        )

        with patch("graphs.ingest_graph.fetch_transcript", return_value=FIXTURE_SEGMENTS):
            resp = client.post("/ingest", json={"video_id": "ingest_e2e_test_01"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "done"
        assert body["chunk_count"] > 0
        assert body["cached"] is False

    def test_ingest_idempotency_returns_cached(self, e2e_client: TestClient):
        resp = e2e_client.post("/ingest", json={"video_id": FIXTURE_VIDEO_ID})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "skipped"
        assert body["cached"] is True

    def test_ingest_force_reruns_pipeline(self, e2e_settings, chroma_client, mock_openai_url):
        from src.main import create_app

        # First ingest
        client = TestClient(
            create_app(e2e_settings, chroma_client=chroma_client),
            raise_server_exceptions=False,
        )
        with patch("graphs.ingest_graph.fetch_transcript", return_value=FIXTURE_SEGMENTS):
            client.post("/ingest", json={"video_id": "force_rerun_test_01"})
            # Force re-ingest
            resp = client.post("/ingest", json={"video_id": "force_rerun_test_01", "force": True})

        assert resp.status_code == 200
        assert resp.json()["status"] == "done"
