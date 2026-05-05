"""Tests for src/secrets.py and the /config/* endpoints.

The OS keyring is replaced with a deterministic in-memory backend for every
test so that the real system keychain is never touched and results are
reproducible across platforms.

Key-never-leaks invariant
-------------------------
The raw API key value must not appear in:
  - any HTTP response body
  - any log record emitted during a request
"""

from __future__ import annotations

import keyring as kr
import keyring.backend
import keyring.errors
import pytest
import structlog.testing
from fastapi.testclient import TestClient

from src.config import Settings
from src.main import create_app
from src.secrets import (
    clear_api_key,
    get_api_key,
    get_openai_key,
    set_api_key,
)

# ---------------------------------------------------------------------------
# In-memory keyring backend
# ---------------------------------------------------------------------------

_SENTINEL = object()


class InMemoryKeyring(keyring.backend.KeyringBackend):
    """Deterministic dict-backed keyring; no OS interaction."""

    priority = 999  # highest priority wins

    def __init__(self):
        self._store: dict[tuple[str, str], str] = {}

    def set_password(self, service: str, username: str, password: str) -> None:
        self._store[(service, username)] = password

    def get_password(self, service: str, username: str) -> str | None:
        return self._store.get((service, username))

    def delete_password(self, service: str, username: str) -> None:
        key = (service, username)
        if key not in self._store:
            raise keyring.errors.PasswordDeleteError(username)
        del self._store[key]


@pytest.fixture(autouse=True)
def _use_in_memory_keyring():
    """Replace the system keyring with an empty in-memory backend for each test."""
    backend = InMemoryKeyring()
    kr.set_keyring(backend)
    yield backend
    # Restore default resolution after the test
    kr.core._keyring_backend = None  # noqa: SLF001


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def settings():
    return Settings(openai_api_key="", _env_file=None)


@pytest.fixture()
def client(settings):
    app = create_app(settings)
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# secrets.py unit tests
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_get_returns_none_before_set(self):
        assert get_api_key() is None

    def test_set_then_get_returns_key(self):
        set_api_key("sk-test-123")
        assert get_api_key() == "sk-test-123"

    def test_clear_removes_key(self):
        set_api_key("sk-test-123")
        clear_api_key()
        assert get_api_key() is None

    def test_clear_when_nothing_set_is_noop(self):
        clear_api_key()  # must not raise
        assert get_api_key() is None

    def test_overwrite_replaces_key(self):
        set_api_key("sk-old")
        set_api_key("sk-new")
        assert get_api_key() == "sk-new"

    def test_get_openai_key_prefers_keyring_over_settings(self, settings):
        settings.openai_api_key = "sk-from-settings"
        set_api_key("sk-from-keyring")
        assert get_openai_key(settings) == "sk-from-keyring"

    def test_get_openai_key_falls_back_to_settings(self, settings):
        settings.openai_api_key = "sk-from-settings"
        assert get_openai_key(settings) == "sk-from-settings"

    def test_get_openai_key_returns_none_when_nothing_set(self, settings):
        settings.openai_api_key = ""
        assert get_openai_key(settings) is None


# ---------------------------------------------------------------------------
# /config/* endpoint tests
# ---------------------------------------------------------------------------


class TestConfigEndpoints:
    def test_status_has_key_false_when_no_key(self, client):
        resp = client.get("/config/status")
        assert resp.status_code == 200
        assert resp.json() == {"has_key": False}

    def test_post_api_key_stores_it(self, client):
        resp = client.post("/config/api-key", json={"api_key": "sk-stored"})
        assert resp.status_code == 204
        assert get_api_key() == "sk-stored"

    def test_status_has_key_true_after_post(self, client):
        client.post("/config/api-key", json={"api_key": "sk-stored"})
        resp = client.get("/config/status")
        assert resp.json() == {"has_key": True}

    def test_delete_api_key_clears_it(self, client):
        client.post("/config/api-key", json={"api_key": "sk-stored"})
        resp = client.delete("/config/api-key")
        assert resp.status_code == 204
        assert get_api_key() is None

    def test_status_has_key_false_after_delete(self, client):
        client.post("/config/api-key", json={"api_key": "sk-stored"})
        client.delete("/config/api-key")
        resp = client.get("/config/status")
        assert resp.json() == {"has_key": False}

    def test_post_missing_api_key_field_returns_422(self, client):
        resp = client.post("/config/api-key", json={})
        assert resp.status_code == 422

    def test_post_empty_api_key_returns_422(self, client):
        resp = client.post("/config/api-key", json={"api_key": ""})
        assert resp.status_code == 422

    def test_status_response_never_contains_key_value(self, client):
        secret = "sk-super-secret-12345"
        client.post("/config/api-key", json={"api_key": secret})
        resp = client.get("/config/status")
        assert secret not in resp.text

    def test_delete_response_never_contains_key_value(self, client):
        secret = "sk-super-secret-12345"
        client.post("/config/api-key", json={"api_key": secret})
        resp = client.delete("/config/api-key")
        assert secret not in resp.text


# ---------------------------------------------------------------------------
# /ask returns 400 when no key is configured
# ---------------------------------------------------------------------------


class TestMissingKeyReturns400:
    def test_ask_400_when_no_key(self, client):
        resp = client.post("/ask", json={"video_id": "vid1", "question": "q?"})
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "API_KEY_MISSING"

    def test_ask_stream_400_when_no_key(self, client):
        resp = client.post("/ask/stream", json={"video_id": "vid1", "question": "q?"})
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "API_KEY_MISSING"

    def test_ingest_400_when_no_key(self, client):
        resp = client.post("/ingest", json={"video_id": "vid1"})
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "API_KEY_MISSING"


# ---------------------------------------------------------------------------
# Key never appears in log output
# ---------------------------------------------------------------------------


class TestKeyNeverLeaksToLogs:
    def test_set_api_key_does_not_log_the_key(self):
        secret = "sk-never-log-this"
        with structlog.testing.capture_logs() as logs:
            set_api_key(secret)
        for record in logs:
            for v in record.values():
                assert secret not in str(v), f"Key leaked into log record: {record}"

    def test_get_api_key_does_not_log_the_key(self):
        secret = "sk-never-log-this"
        set_api_key(secret)
        with structlog.testing.capture_logs() as logs:
            get_api_key()
        for record in logs:
            for v in record.values():
                assert secret not in str(v), f"Key leaked into log record: {record}"

    def test_config_post_does_not_log_key(self, client):
        secret = "sk-never-in-logs"
        with structlog.testing.capture_logs() as logs:
            client.post("/config/api-key", json={"api_key": secret})
        for record in logs:
            for v in record.values():
                assert secret not in str(v), f"Key leaked into log record: {record}"
