from unittest.mock import MagicMock, patch


def test_no_network_call_when_tracing_disabled(monkeypatch):
    """Verify that LangSmith tracing is disabled when env vars are false.
    
    Since build_rag_chain was removed with the simple query path, this test
    now verifies that importing chain helpers doesn't trigger network calls.
    """
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")

    mock_send = MagicMock(side_effect=AssertionError("HTTP call made with tracing disabled"))

    with patch("httpx.Client.send", mock_send), patch("httpx.AsyncClient.send", mock_send):
        from src.chain import build_chat_prompt

        prompt = build_chat_prompt()

    assert prompt is not None
    mock_send.assert_not_called()
