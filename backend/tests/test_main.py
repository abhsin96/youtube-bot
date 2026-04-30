import pytest
from fastapi.testclient import TestClient

from src.config import Settings
from src.main import create_app


@pytest.fixture()
def client():
    settings = Settings(openai_api_key="sk-test", _env_file=None)
    app = create_app(settings)
    return TestClient(app)


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == "0.1.0"
    assert body["has_api_key"] is True
    assert isinstance(body["langsmith_enabled"], bool)


PREFLIGHT_ORIGINS = [
    "chrome-extension://abcdefghijklmnopabcdefghijklmnop",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost",
]


@pytest.mark.parametrize("origin", PREFLIGHT_ORIGINS)
def test_cors_preflight_allowed_origins(client, origin):
    response = client.options(
        "/health",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == origin
    allowed_methods = response.headers.get("access-control-allow-methods", "")
    for method in ("GET", "POST", "DELETE", "OPTIONS"):
        assert method in allowed_methods
