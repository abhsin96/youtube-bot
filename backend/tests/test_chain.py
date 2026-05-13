from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder

from src.chain import (
    _HUMAN_TEMPLATE,
    RAGResult,
    _format_context,
    _seconds_to_mmss,
    answer_question,
    build_chat_prompt,
    build_rag_chain,
)

_DIM = 4
_VIDEO_ID = "vid1"
_QUESTION = "What is this video about?"


def _fake_embeddings():
    emb = MagicMock()
    emb.embed_query.return_value = [0.0] * _DIM
    emb.embed_documents.side_effect = lambda texts: [[0.0] * _DIM for _ in texts]
    return emb


def _doc(text: str, start_ts: float = 0.0) -> Document:
    return Document(
        page_content=text,
        metadata={"start_ts": start_ts, "end_ts": start_ts + 5.0, "chunk_id": "c1"},
    )


def _mock_retriever(docs: list[Document]):
    """Return a mock retriever whose invoke() returns *docs*."""
    r = MagicMock()
    r.invoke.return_value = docs
    return r


# --- _seconds_to_mmss ---


def test_seconds_to_mmss_whole_minutes():
    assert _seconds_to_mmss(60.0) == "[01:00]"


def test_seconds_to_mmss_zero():
    assert _seconds_to_mmss(0.0) == "[00:00]"


def test_seconds_to_mmss_sub_minute():
    assert _seconds_to_mmss(42.0) == "[00:42]"


def test_seconds_to_mmss_mixed():
    assert _seconds_to_mmss(83.9) == "[01:23]"


def test_seconds_to_mmss_large():
    assert _seconds_to_mmss(3723.0) == "[62:03]"


def test_seconds_to_mmss_int_input():
    assert _seconds_to_mmss(125) == "[02:05]"


# --- _format_context ---


def test_format_context_includes_text():
    docs = [_doc("hello world", start_ts=10.0)]
    ctx = _format_context(docs)
    assert "hello world" in ctx


def test_format_context_timestamp_is_mmss():
    docs = [_doc("hello", start_ts=83.0)]
    ctx = _format_context(docs)
    assert "[01:23]" in ctx


def test_format_context_timestamp_not_raw_seconds():
    docs = [_doc("hello", start_ts=42.0)]
    ctx = _format_context(docs)
    # Raw float (e.g. "42.0s") must not appear — only [mm:ss]
    assert "42.0" not in ctx
    assert "[00:42]" in ctx


def test_format_context_missing_timestamp_uses_placeholder():
    doc = Document(page_content="no ts", metadata={})
    ctx = _format_context([doc])
    assert "[??:??]" in ctx


def test_format_context_multiple_docs_separated():
    docs = [_doc("first", 0.0), _doc("second", 5.0)]
    ctx = _format_context(docs)
    assert "first" in ctx
    assert "second" in ctx


def test_format_context_empty():
    assert _format_context([]) == ""


# --- build_chat_prompt ---


def test_build_chat_prompt_returns_chat_prompt_template():
    from langchain_core.prompts import ChatPromptTemplate

    assert isinstance(build_chat_prompt(), ChatPromptTemplate)


def test_build_chat_prompt_has_three_messages():
    prompt = build_chat_prompt()
    assert len(prompt.messages) == 3


def test_build_chat_prompt_first_message_is_system():
    from langchain_core.prompts import SystemMessagePromptTemplate

    prompt = build_chat_prompt()
    assert isinstance(prompt.messages[0], SystemMessagePromptTemplate)


def test_build_chat_prompt_second_message_is_history_placeholder():
    prompt = build_chat_prompt()
    msg = prompt.messages[1]
    assert isinstance(msg, MessagesPlaceholder)
    assert msg.variable_name == "history"


def test_build_chat_prompt_third_message_is_human():
    from langchain_core.prompts import HumanMessagePromptTemplate

    prompt = build_chat_prompt()
    assert isinstance(prompt.messages[2], HumanMessagePromptTemplate)


def test_build_chat_prompt_human_contains_context_variable():
    prompt = build_chat_prompt()
    assert "context" in prompt.input_variables


def test_build_chat_prompt_human_contains_question_variable():
    prompt = build_chat_prompt()
    assert "question" in prompt.input_variables


def test_build_chat_prompt_system_contains_rules():
    prompt = build_chat_prompt()
    system_text = prompt.messages[0].prompt.template
    assert "ONLY" in system_text
    assert "[mm:ss]" in system_text


def test_build_chat_prompt_formats_with_empty_history():
    prompt = build_chat_prompt()
    messages = prompt.format_messages(context="[00:05] hello", question="what?", history=[])
    # system + human = 2 messages when history is empty
    assert len(messages) == 2


def test_build_chat_prompt_formats_with_history():
    prompt = build_chat_prompt()
    history = [HumanMessage(content="prev q"), AIMessage(content="prev a")]
    messages = prompt.format_messages(context="[00:05] hello", question="what?", history=history)
    # system + 2 history + human = 4
    assert len(messages) == 4


def test_build_chat_prompt_human_message_contains_excerpts_header():
    prompt = build_chat_prompt()
    messages = prompt.format_messages(context="[00:05] hello", question="why?", history=[])
    human_text = messages[-1].content
    assert "Relevant transcript excerpts" in human_text


def test_build_chat_prompt_human_message_contains_context():
    prompt = build_chat_prompt()
    messages = prompt.format_messages(
        context="[01:23] unique excerpt xyz", question="what?", history=[]
    )
    human_text = messages[-1].content
    assert "[01:23] unique excerpt xyz" in human_text


def test_build_chat_prompt_human_message_contains_question():
    prompt = build_chat_prompt()
    messages = prompt.format_messages(
        context="[00:00] text", question="my specific question", history=[]
    )
    human_text = messages[-1].content
    assert "my specific question" in human_text


def test_build_chat_prompt_history_messages_between_system_and_human():
    prompt = build_chat_prompt()
    history = [HumanMessage(content="old q"), AIMessage(content="old a")]
    messages = prompt.format_messages(context="[00:00] t", question="q?", history=history)
    from langchain_core.messages import AIMessage as AI
    from langchain_core.messages import HumanMessage as HM
    from langchain_core.messages import SystemMessage

    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HM)  # history[0]
    assert isinstance(messages[2], AI)  # history[1]
    assert isinstance(messages[3], HM)  # current question


def test_human_template_has_excerpts_then_question():
    # Context block must come before the Question: line
    ctx_pos = _HUMAN_TEMPLATE.index("{context}")
    q_pos = _HUMAN_TEMPLATE.index("{question}")
    assert ctx_pos < q_pos


# --- build_rag_chain ---


def test_build_rag_chain_returns_callable():
    chain = build_rag_chain("gpt-4o-mini", "sk-test")
    assert callable(chain.invoke)


def test_build_rag_chain_uses_temperature_0_2():
    with patch("src.chain.ChatOpenAI") as mock_cls:
        build_rag_chain("gpt-4o-mini", "sk-test")
    assert mock_cls.call_args.kwargs["temperature"] == 0.2


def test_build_rag_chain_streaming_enabled():
    with patch("src.chain.ChatOpenAI") as mock_cls:
        build_rag_chain("gpt-4o-mini", "sk-test")
    assert mock_cls.call_args.kwargs["streaming"] is True


def test_build_rag_chain_passes_chat_model():
    with patch("src.chain.ChatOpenAI") as mock_cls:
        build_rag_chain("gpt-4o-turbo", "sk-test")
    assert mock_cls.call_args.kwargs["model"] == "gpt-4o-turbo"


def test_build_rag_chain_passes_api_key():
    with patch("src.chain.ChatOpenAI") as mock_cls:
        build_rag_chain("gpt-4o-mini", "sk-mykey")
    assert mock_cls.call_args.kwargs["openai_api_key"] == "sk-mykey"


# --- answer_question ---


@pytest.fixture()
def mock_vector_store():
    """Return a MagicMock VectorStorePort whose build_retriever() returns a mock retriever."""
    store = MagicMock()
    retriever = MagicMock()
    retriever.invoke.return_value = []
    store.build_retriever.return_value = retriever
    return store


def test_answer_question_returns_rag_result(mock_vector_store):
    docs = [_doc("LangChain is a framework", start_ts=1.0)]
    mock_vector_store.build_retriever.return_value = _mock_retriever(docs)

    with patch("src.chain.build_rag_chain") as mock_chain_fn:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "LangChain is great."
        mock_chain_fn.return_value = mock_chain

        result = answer_question(
            _VIDEO_ID, _QUESTION, _fake_embeddings(), mock_vector_store, "gpt-4o-mini", "sk-test"
        )

    assert isinstance(result, RAGResult)
    assert result.answer == "LangChain is great."
    assert result.sources == docs


def test_answer_question_no_sources_returns_fallback(mock_vector_store):
    mock_vector_store.build_retriever.return_value = _mock_retriever([])
    result = answer_question(
        _VIDEO_ID, _QUESTION, _fake_embeddings(), mock_vector_store, "gpt-4o-mini", "sk-test"
    )
    assert "isn't available" in result.answer.lower()
    assert result.sources == []


def test_answer_question_passes_k_to_retriever(mock_vector_store):
    mock_vector_store.build_retriever.return_value = _mock_retriever([])
    answer_question(
        _VIDEO_ID, _QUESTION, _fake_embeddings(), mock_vector_store, "gpt-4o-mini", "sk-test", k=3
    )
    assert mock_vector_store.build_retriever.call_args.kwargs["k"] == 3


def test_answer_question_passes_question_to_retriever(mock_vector_store):
    retriever = _mock_retriever([])
    mock_vector_store.build_retriever.return_value = retriever
    answer_question(
        _VIDEO_ID, _QUESTION, _fake_embeddings(), mock_vector_store, "gpt-4o-mini", "sk-test"
    )
    retriever.invoke.assert_called_once_with(_QUESTION)


def test_answer_question_context_passed_to_chain(mock_vector_store):
    docs = [_doc("unique phrase xyz", start_ts=7.0)]
    mock_vector_store.build_retriever.return_value = _mock_retriever(docs)

    with patch("src.chain.build_rag_chain") as mock_chain_fn:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "ok"
        mock_chain_fn.return_value = mock_chain

        answer_question(
            _VIDEO_ID, _QUESTION, _fake_embeddings(), mock_vector_store, "gpt-4o-mini", "sk-test"
        )

    invoke_kwargs = mock_chain.invoke.call_args[0][0]
    assert "unique phrase xyz" in invoke_kwargs["context"]
    assert invoke_kwargs["question"] == _QUESTION


# --- history threading ---


def test_answer_question_passes_empty_history_by_default(mock_vector_store):
    mock_vector_store.build_retriever.return_value = _mock_retriever([])
    with patch("src.chain.build_rag_chain") as mock_chain_fn:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = (
            "I'm sorry, that information isn't available in the video transcript."
        )
        mock_chain_fn.return_value = mock_chain
        answer_question(
            _VIDEO_ID, _QUESTION, _fake_embeddings(), mock_vector_store, "gpt-4o-mini", "sk-test"
        )
    # no sources → chain never called; but if sources existed, history would be []
    # We verify the no-sources branch doesn't crash with missing history arg.


def test_answer_question_passes_history_to_chain(mock_vector_store):
    docs = [_doc("content", start_ts=5.0)]
    history = [HumanMessage(content="prev"), AIMessage(content="ans")]
    mock_vector_store.build_retriever.return_value = _mock_retriever(docs)

    with patch("src.chain.build_rag_chain") as mock_chain_fn:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "answer"
        mock_chain_fn.return_value = mock_chain
        answer_question(
            _VIDEO_ID,
            _QUESTION,
            _fake_embeddings(),
            mock_vector_store,
            "gpt-4o-mini",
            "sk-test",
            history=history,
        )

    invoke_kwargs = mock_chain.invoke.call_args[0][0]
    assert invoke_kwargs["history"] == history


def test_answer_question_none_history_becomes_empty_list(mock_vector_store):
    docs = [_doc("content", start_ts=5.0)]
    mock_vector_store.build_retriever.return_value = _mock_retriever(docs)

    with patch("src.chain.build_rag_chain") as mock_chain_fn:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "answer"
        mock_chain_fn.return_value = mock_chain
        answer_question(
            _VIDEO_ID,
            _QUESTION,
            _fake_embeddings(),
            mock_vector_store,
            "gpt-4o-mini",
            "sk-test",
            history=None,
        )

    invoke_kwargs = mock_chain.invoke.call_args[0][0]
    assert invoke_kwargs["history"] == []


# --- score_threshold ---


def test_answer_question_default_score_threshold_zero(mock_vector_store):
    mock_vector_store.build_retriever.return_value = _mock_retriever([])
    answer_question(
        _VIDEO_ID, _QUESTION, _fake_embeddings(), mock_vector_store, "gpt-4o-mini", "sk-test"
    )
    assert mock_vector_store.build_retriever.call_args.kwargs["score_threshold"] == 0.0


def test_answer_question_passes_score_threshold_to_retriever(mock_vector_store):
    mock_vector_store.build_retriever.return_value = _mock_retriever([])
    answer_question(
        _VIDEO_ID,
        _QUESTION,
        _fake_embeddings(),
        mock_vector_store,
        "gpt-4o-mini",
        "sk-test",
        score_threshold=0.25,
    )
    assert mock_vector_store.build_retriever.call_args.kwargs["score_threshold"] == 0.25


def test_answer_question_threshold_filters_propagate(mock_vector_store):
    """Docs that don't meet the threshold come back empty from the retriever."""
    mock_vector_store.build_retriever.return_value = _mock_retriever([])
    with patch("src.chain.build_rag_chain"):
        result = answer_question(
            _VIDEO_ID,
            _QUESTION,
            _fake_embeddings(),
            mock_vector_store,
            "gpt-4o-mini",
            "sk-test",
            score_threshold=0.9,
        )
    assert result.sources == []
    assert "isn't available" in result.answer.lower()
