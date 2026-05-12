"""Tests for the standardised error envelope and CORS behaviour.

Every error response must have the shape::

    {"error": {"code": "<SCREAMING_SNAKE>", "message": "<string>"}}

CORS must allow requests originating from chrome-extension:// URIs.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from graphs.ask_graph import VIDEO_NOT_INGESTED
from src.config import Settings
from src.main import create_app

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _graph_error(code: str):
    return {
        "answer": "",
        "citations": [],
        "tokens_used": None,
        "refused": False,
        "error": code,
    }


@pytest.fixture()
def settings():
    return Settings(openai_api_key="sk-test", _env_file=None)


@pytest.fixture()
def client(settings):
    app = create_app(settings)
    return TestClient(app, raise_server_exceptions=False)


def _assert_envelope(body: dict) -> dict:
    assert "error" in body, f"missing 'error' key: {body}"
    err = body["error"]
    assert "code" in err, f"missing 'code' in error: {err}"
    assert "message" in err, f"missing 'message' in error: {err}"
    assert isinstance(err["code"], str) and err["code"]
    assert isinstance(err["message"], str)
    return err


# ---------------------------------------------------------------------------
# Error envelope — shape contract
# ---------------------------------------------------------------------------


class TestErrorEnvelopeShape:
    @patch("src.routers.query.build_ask_graph")
    @patch("src.routers.query.get_embeddings")
    def test_404_video_not_ingested_envelope(self, _mock_emb, mock_build, client):
        mock_build.return_value.invoke.return_value = _graph_error(VIDEO_NOT_INGESTED)
        resp = client.post("/query", json={"video_id": "missing", "question": "q?"})

        assert resp.status_code == 404
        err = _assert_envelope(resp.json())
        assert err["code"] == "VIDEO_NOT_INGESTED"

    @patch("src.routers.query.build_ask_graph")
    @patch("src.routers.query.get_embeddings")
    def test_500_internal_error_envelope(self, _mock_emb, mock_build, client):
        mock_build.return_value.invoke.return_value = _graph_error("something went wrong")
        resp = client.post("/query", json={"video_id": "vid1", "question": "q?"})

        assert resp.status_code == 500
        err = _assert_envelope(resp.json())
        assert err["code"] == "INTERNAL_ERROR"
        assert "something went wrong" in err["message"]

    @patch("src.routers.ingest.build_graph")
    @patch("src.routers.ingest.get_embeddings")
    def test_502_ingest_failed_envelope(self, _mock_emb, mock_bg, client):
        mock_bg.return_value.invoke.return_value = {
            "status": "error",
            "chunks": [],
            "error": "youtube fetch failed",
        }
        resp = client.post("/ingest", json={"video_id": "vid1"})

        assert resp.status_code == 502
        err = _assert_envelope(resp.json())
        assert err["code"] == "INGEST_FAILED"
        assert "youtube fetch failed" in err["message"]

    def test_422_validation_error_envelope(self, client):
        # Missing required field 'question'
        resp = client.post("/query", json={"video_id": "vid1"})

        assert resp.status_code == 422
        err = _assert_envelope(resp.json())
        assert err["code"] == "VALIDATION_ERROR"

    @patch("src.routers.query.collection_exists", return_value=False)
    @patch("src.routers.query.get_embeddings")
    def test_stream_404_envelope(self, _mock_emb, _mock_ce, client):
        resp = client.post("/query", json={"video_id": "missing", "question": "q?", "stream": True})

        assert resp.status_code == 404
        err = _assert_envelope(resp.json())
        assert err["code"] == "VIDEO_NOT_INGESTED"

    def test_405_method_not_allowed_envelope(self, client):
        resp = client.get("/query")

        assert resp.status_code == 405
        err = _assert_envelope(resp.json())
        assert err["code"] == "METHOD_NOT_ALLOWED"


# ---------------------------------------------------------------------------
# Global exception handler — unhandled exceptions never reach the client raw
# ---------------------------------------------------------------------------


class TestGlobalExceptionHandler:
    def test_unhandled_exception_returns_500_envelope(self, settings):
        app = create_app(settings)

        @app.get("/test-boom")
        async def boom():
            raise RuntimeError("totally unexpected")

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/test-boom")

        assert resp.status_code == 500
        err = _assert_envelope(resp.json())
        assert err["code"] == "INTERNAL_ERROR"

    def test_unhandled_exception_message_is_generic(self, settings):
        """Internal exception detail must not leak into the response body."""
        app = create_app(settings)
        secret_detail = "db-password-in-error-xyz"

        @app.get("/test-secret")
        async def secret_boom():
            raise RuntimeError(secret_detail)

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/test-secret")

        assert secret_detail not in resp.text


# ---------------------------------------------------------------------------
# CORS — chrome-extension origin
# ---------------------------------------------------------------------------


class TestCORSExtensionOrigin:
    EXT_ORIGIN = "chrome-extension://abcdefghijklmnopabcdefghijklmnop"

    def test_cors_preflight_allows_extension_origin(self, settings):
        app = create_app(settings)
        # raise_server_exceptions=False keeps the client from masking CORS errors
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.options(
            "/query",
            headers={
                "Origin": self.EXT_ORIGIN,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )

        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == self.EXT_ORIGIN

    def test_cors_simple_request_allows_extension_origin(self, settings):
        app = create_app(settings)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/health", headers={"Origin": self.EXT_ORIGIN})

        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == self.EXT_ORIGIN

    def test_cors_disallows_arbitrary_origin(self, settings):
        app = create_app(settings)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/health", headers={"Origin": "https://evil.example.com"})

        # Server responds, but must NOT echo the foreign origin back
        allow = resp.headers.get("access-control-allow-origin", "")
        assert allow != "https://evil.example.com"
