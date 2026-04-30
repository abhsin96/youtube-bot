from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.config import Settings, load_settings
from src.main import create_app


@pytest.fixture()
def client():
    settings = Settings(openai_api_key="sk-test", _env_file=None)
    app = create_app(settings)
    return TestClient(app)


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_cors_preflight_chrome_extension(client):
    origin = "chrome-extension://abcdefghijklmnopabcdefghijklmnop"
    response = client.options(
        "/health",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == origin


def test_missing_openai_key_raises_validation_error():
    with pytest.raises(ValidationError, match="openai_api_key"):
        Settings(_env_file=None)


def test_load_settings_exits_on_missing_key(capsys):
    side_effect = ValidationError.from_exception_data(
        "Settings",
        [{"type": "missing", "loc": ("openai_api_key",), "msg": "Field required", "input": {}}],
    )
    with (
        patch("src.config.Settings", side_effect=side_effect),
        pytest.raises(SystemExit) as exc_info,
    ):
        load_settings()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "OPENAI_API_KEY" in captured.err
