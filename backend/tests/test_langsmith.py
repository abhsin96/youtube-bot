from unittest.mock import MagicMock, patch


def test_no_network_call_when_tracing_disabled(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")

    mock_send = MagicMock(side_effect=AssertionError("HTTP call made with tracing disabled"))

    with patch("httpx.Client.send", mock_send), patch("httpx.AsyncClient.send", mock_send):
        from src.chain import build_hello_chain

        chain = build_hello_chain()
        result = chain.invoke("world")

    assert result == "Hello, world!"
    mock_send.assert_not_called()
