from unittest.mock import MagicMock, patch


def test_no_network_call_when_tracing_disabled(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")

    mock_send = MagicMock(side_effect=AssertionError("HTTP call made with tracing disabled"))

    with patch("httpx.Client.send", mock_send), patch("httpx.AsyncClient.send", mock_send):
        from src.chain import build_rag_chain

        chain = build_rag_chain("gpt-4o-mini", "sk-test")

    assert callable(chain.invoke)
    mock_send.assert_not_called()
