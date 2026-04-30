import re

import pytest
from fastapi.testclient import TestClient

from src.config import Settings
from src.main import create_app

UUID4_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


@pytest.fixture()
def client():
    settings = Settings(openai_api_key="sk-test", _env_file=None)
    return TestClient(create_app(settings))


def test_response_has_request_id_header(client):
    response = client.get("/health")
    assert "x-request-id" in response.headers
    assert UUID4_RE.match(response.headers["x-request-id"])


def test_client_supplied_request_id_is_echoed(client):
    custom_id = "my-trace-id-123"
    response = client.get("/health", headers={"X-Request-ID": custom_id})
    assert response.headers["x-request-id"] == custom_id


def test_each_request_gets_unique_id(client):
    ids = {client.get("/health").headers["x-request-id"] for _ in range(5)}
    assert len(ids) == 5
