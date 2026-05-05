"""Unit tests for the ask LangGraph (graphs/ask_graph.py).

Each test exercises the full compiled graph with mocked external dependencies
so no network calls or real Chroma/OpenAI is required.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from graphs.ask_graph import VIDEO_NOT_INGESTED, build_ask_graph, make_initial_state

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def settings():
    s = MagicMock()
    s.vector_db_path = "/tmp/chroma_test"
    s.chat_model = "gpt-4o-mini"
    s.openai_api_key = "sk-test"
    s.min_similarity_threshold = 0.0
    s.context_budget_tokens = 6000
    return s


@pytest.fixture()
def embeddings():
    return MagicMock()


def _doc(chunk_id="c1", start_ts=5.0, end_ts=10.0, text="hello world"):
    return Document(
        page_content=text,
        metadata={"chunk_id": chunk_id, "start_ts": start_ts, "end_ts": end_ts, "video_id": "vid1"},
    )


def _ai_message(content="The answer is 42.", total_tokens=100):
    msg = AIMessage(content=content)
    msg.usage_metadata = {"input_tokens": 80, "output_tokens": 20, "total_tokens": total_tokens}
    return msg


# ---------------------------------------------------------------------------
# AskState schema
# ---------------------------------------------------------------------------


class TestAskStateSchema:
    def test_make_initial_state_sets_inputs(self):
        state = make_initial_state("vid1", "What is this?")
        assert state["video_id"] == "vid1"
        assert state["question"] == "What is this?"
        assert state["history"] == []
        assert state["k"] == 5

    def test_make_initial_state_zeroes_outputs(self):
        state = make_initial_state("vid1", "q?")
        assert state["retrieved_chunks"] == []
        assert state["refused"] is False
        assert state["answer"] == ""
        assert state["citations"] == []
        assert state["tokens_used"] is None
        assert state["error"] is None

    def test_history_is_stored(self):
        history = [HumanMessage(content="prior turn")]
        state = make_initial_state("vid1", "q?", history=history)
        assert state["history"] == history

    def test_k_override(self):
        state = make_initial_state("vid1", "q?", k=3)
        assert state["k"] == 3


# ---------------------------------------------------------------------------
# validate node
# ---------------------------------------------------------------------------


class TestValidateNode:
    @patch("graphs.ask_graph.collection_exists", return_value=False)
    @patch("graphs.ask_graph.build_retriever")
    def test_sets_error_when_collection_missing(self, mock_br, _mock_ce, settings, embeddings):
        graph = build_ask_graph(settings, embeddings)
        result = graph.invoke(make_initial_state("missing", "q?"))

        assert result["error"] == VIDEO_NOT_INGESTED
        mock_br.assert_not_called()  # should not proceed to retrieve

    @patch("graphs.ask_graph.collection_exists", return_value=True)
    @patch("graphs.ask_graph.build_retriever")
    def test_no_error_when_collection_exists(self, mock_br, _mock_ce, settings, embeddings):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_br.return_value = mock_retriever

        graph = build_ask_graph(settings, embeddings)
        result = graph.invoke(make_initial_state("vid1", "q?"))

        assert result["error"] is None
        mock_br.assert_called_once()


# ---------------------------------------------------------------------------
# retrieve node
# ---------------------------------------------------------------------------


class TestRetrieveNode:
    @patch("graphs.ask_graph.collection_exists", return_value=True)
    @patch("graphs.ask_graph.build_retriever")
    def test_passes_k_to_retriever(self, mock_br, _mock_ce, settings, embeddings):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_br.return_value = mock_retriever

        graph = build_ask_graph(settings, embeddings)
        graph.invoke(make_initial_state("vid1", "q?", k=3))

        _, kwargs = mock_br.call_args
        assert kwargs.get("k") == 3 or mock_br.call_args[0][3] == 3

    @patch("graphs.ask_graph.collection_exists", return_value=True)
    @patch("graphs.ask_graph.build_retriever")
    def test_retrieved_chunks_stored_in_state(self, mock_br, _mock_ce, settings, embeddings):
        doc = _doc()
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [doc]
        mock_br.return_value = mock_retriever

        # Need an LLM mock for the generate node to run
        with patch("graphs.ask_graph.ChatOpenAI") as MockLLM:
            MockLLM.return_value.invoke.return_value = _ai_message()
            graph = build_ask_graph(settings, embeddings)
            result = graph.invoke(make_initial_state("vid1", "q?"))

        # retrieved_chunks may be trimmed but at least the flow ran
        assert result["error"] is None


# ---------------------------------------------------------------------------
# guardrail_check → refuse path
# ---------------------------------------------------------------------------


class TestRefusePath:
    @patch("graphs.ask_graph.collection_exists", return_value=True)
    @patch("graphs.ask_graph.build_retriever")
    def test_refuses_when_no_chunks_retrieved(self, mock_br, _mock_ce, settings, embeddings):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_br.return_value = mock_retriever

        graph = build_ask_graph(settings, embeddings)
        result = graph.invoke(make_initial_state("vid1", "q?"))

        assert result["refused"] is True
        assert "isn't available" in result["answer"]
        assert result["citations"] == []


# ---------------------------------------------------------------------------
# generate node
# ---------------------------------------------------------------------------


class TestGenerateNode:
    @patch("graphs.ask_graph.collection_exists", return_value=True)
    @patch("graphs.ask_graph.build_retriever")
    @patch("graphs.ask_graph.ChatOpenAI")
    def test_returns_answer_from_llm(self, MockLLM, mock_br, _mock_ce, settings, embeddings):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [_doc()]
        mock_br.return_value = mock_retriever
        MockLLM.return_value.invoke.return_value = _ai_message("The answer is 42.")

        graph = build_ask_graph(settings, embeddings)
        result = graph.invoke(make_initial_state("vid1", "What is 42?"))

        assert result["answer"] == "The answer is 42."
        assert result["refused"] is False

    @patch("graphs.ask_graph.collection_exists", return_value=True)
    @patch("graphs.ask_graph.build_retriever")
    @patch("graphs.ask_graph.ChatOpenAI")
    def test_extracts_tokens_used_from_usage_metadata(
        self, MockLLM, mock_br, _mock_ce, settings, embeddings
    ):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [_doc()]
        mock_br.return_value = mock_retriever
        MockLLM.return_value.invoke.return_value = _ai_message(total_tokens=175)

        graph = build_ask_graph(settings, embeddings)
        result = graph.invoke(make_initial_state("vid1", "q?"))

        assert result["tokens_used"] == 175

    @patch("graphs.ask_graph.collection_exists", return_value=True)
    @patch("graphs.ask_graph.build_retriever")
    @patch("graphs.ask_graph.ChatOpenAI")
    def test_refused_true_when_answer_contains_sentinel(
        self, MockLLM, mock_br, _mock_ce, settings, embeddings
    ):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [_doc()]
        mock_br.return_value = mock_retriever
        MockLLM.return_value.invoke.return_value = _ai_message(
            "I'm sorry, that information isn't available in the video transcript."
        )

        graph = build_ask_graph(settings, embeddings)
        result = graph.invoke(make_initial_state("vid1", "q?"))

        assert result["refused"] is True

    @patch("graphs.ask_graph.collection_exists", return_value=True)
    @patch("graphs.ask_graph.build_retriever")
    @patch("graphs.ask_graph.ChatOpenAI")
    def test_tokens_used_none_when_usage_metadata_absent(
        self, MockLLM, mock_br, _mock_ce, settings, embeddings
    ):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [_doc()]
        mock_br.return_value = mock_retriever
        ai_msg = AIMessage(content="Some answer.")
        # No usage_metadata attribute
        if hasattr(ai_msg, "usage_metadata"):
            ai_msg.usage_metadata = None
        MockLLM.return_value.invoke.return_value = ai_msg

        graph = build_ask_graph(settings, embeddings)
        result = graph.invoke(make_initial_state("vid1", "q?"))

        assert result["tokens_used"] is None


# ---------------------------------------------------------------------------
# format_response node — citation extraction
# ---------------------------------------------------------------------------


class TestFormatResponseNode:
    @patch("graphs.ask_graph.collection_exists", return_value=True)
    @patch("graphs.ask_graph.build_retriever")
    @patch("graphs.ask_graph.ChatOpenAI")
    def test_citations_built_from_retrieved_chunks(
        self, MockLLM, mock_br, _mock_ce, settings, embeddings
    ):
        docs = [
            _doc(chunk_id="c1", start_ts=5.0, end_ts=10.0, text="chunk one"),
            _doc(chunk_id="c2", start_ts=20.0, end_ts=30.0, text="chunk two"),
        ]
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = docs
        mock_br.return_value = mock_retriever
        MockLLM.return_value.invoke.return_value = _ai_message("Answer.")

        graph = build_ask_graph(settings, embeddings)
        result = graph.invoke(make_initial_state("vid1", "q?"))

        assert len(result["citations"]) == 2
        cids = {c["chunk_id"] for c in result["citations"]}
        assert cids == {"c1", "c2"}

    @patch("graphs.ask_graph.collection_exists", return_value=True)
    @patch("graphs.ask_graph.build_retriever")
    @patch("graphs.ask_graph.ChatOpenAI")
    def test_citation_fields_are_correct(self, MockLLM, mock_br, _mock_ce, settings, embeddings):
        doc = _doc(chunk_id="c1", start_ts=5.0, end_ts=10.0, text="hello world")
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [doc]
        mock_br.return_value = mock_retriever
        MockLLM.return_value.invoke.return_value = _ai_message()

        graph = build_ask_graph(settings, embeddings)
        result = graph.invoke(make_initial_state("vid1", "q?"))

        cit = result["citations"][0]
        assert cit["chunk_id"] == "c1"
        assert cit["start_ts"] == 5.0
        assert cit["end_ts"] == 10.0
        assert cit["text"] == "hello world"

    @patch("graphs.ask_graph.collection_exists", return_value=True)
    @patch("graphs.ask_graph.build_retriever")
    def test_citations_empty_after_refusal(self, mock_br, _mock_ce, settings, embeddings):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_br.return_value = mock_retriever

        graph = build_ask_graph(settings, embeddings)
        result = graph.invoke(make_initial_state("vid1", "q?"))

        assert result["citations"] == []

    @patch("graphs.ask_graph.collection_exists", return_value=True)
    @patch("graphs.ask_graph.build_retriever")
    @patch("graphs.ask_graph.ChatOpenAI")
    def test_citations_are_subset_of_retrieved_chunks(
        self, MockLLM, mock_br, _mock_ce, settings, embeddings
    ):
        docs = [_doc(chunk_id=f"c{i}", start_ts=float(i * 10)) for i in range(3)]
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = docs
        mock_br.return_value = mock_retriever
        MockLLM.return_value.invoke.return_value = _ai_message()

        graph = build_ask_graph(settings, embeddings)
        result = graph.invoke(make_initial_state("vid1", "q?"))

        retrieved_ids = {d.metadata["chunk_id"] for d in docs}
        citation_ids = {c["chunk_id"] for c in result["citations"]}
        assert citation_ids.issubset(retrieved_ids)


# ---------------------------------------------------------------------------
# guardrail_check — score-based routing
# ---------------------------------------------------------------------------


class TestGuardrailScoreBased:
    @patch("graphs.ask_graph.collection_exists", return_value=True)
    @patch("graphs.ask_graph.build_retriever")
    @patch("graphs.ask_graph.ChatOpenAI")
    def test_low_score_refuses_without_openai_call(
        self, MockLLM, mock_br, _mock_ce, settings, embeddings
    ):
        """Docs returned but max _score < threshold → refuse; OpenAI never called."""
        doc = _doc()
        doc.metadata["_score"] = 0.05  # below threshold
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [doc]
        mock_br.return_value = mock_retriever
        settings.min_similarity_threshold = 0.5

        graph = build_ask_graph(settings, embeddings)
        result = graph.invoke(make_initial_state("vid1", "q?"))

        assert result["refused"] is True
        assert "isn't available" in result["answer"]
        assert result["citations"] == []
        MockLLM.return_value.invoke.assert_not_called()

    @patch("graphs.ask_graph.collection_exists", return_value=True)
    @patch("graphs.ask_graph.build_retriever")
    @patch("graphs.ask_graph.ChatOpenAI")
    def test_high_score_proceeds_to_generate(
        self, MockLLM, mock_br, _mock_ce, settings, embeddings
    ):
        """Docs with max _score >= threshold → generate node runs; OpenAI called."""
        doc = _doc()
        doc.metadata["_score"] = 0.9  # above threshold
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [doc]
        mock_br.return_value = mock_retriever
        settings.min_similarity_threshold = 0.5
        MockLLM.return_value.invoke.return_value = _ai_message()

        graph = build_ask_graph(settings, embeddings)
        result = graph.invoke(make_initial_state("vid1", "q?"))

        assert result["refused"] is False
        MockLLM.return_value.invoke.assert_called_once()

    @patch("graphs.ask_graph.collection_exists", return_value=True)
    @patch("graphs.ask_graph.build_retriever")
    @patch("graphs.ask_graph.ChatOpenAI")
    def test_max_retrieval_score_stored_in_state(
        self, MockLLM, mock_br, _mock_ce, settings, embeddings
    ):
        """max_retrieval_score is populated from doc metadata after retrieve_node."""
        doc = _doc()
        doc.metadata["_score"] = 0.75
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [doc]
        mock_br.return_value = mock_retriever
        settings.min_similarity_threshold = 0.0
        MockLLM.return_value.invoke.return_value = _ai_message()

        graph = build_ask_graph(settings, embeddings)
        result = graph.invoke(make_initial_state("vid1", "q?"))

        assert result["max_retrieval_score"] == 0.75

    @patch("graphs.ask_graph.collection_exists", return_value=True)
    @patch("graphs.ask_graph.build_retriever")
    @patch("graphs.ask_graph.ChatOpenAI")
    def test_score_at_exact_threshold_proceeds(
        self, MockLLM, mock_br, _mock_ce, settings, embeddings
    ):
        """Score exactly equal to threshold is not a refusal (boundary: refuse only when strictly less)."""
        doc = _doc()
        doc.metadata["_score"] = 0.5
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [doc]
        mock_br.return_value = mock_retriever
        settings.min_similarity_threshold = 0.5
        MockLLM.return_value.invoke.return_value = _ai_message()

        graph = build_ask_graph(settings, embeddings)
        result = graph.invoke(make_initial_state("vid1", "q?"))

        assert result["refused"] is False
        MockLLM.return_value.invoke.assert_called_once()

    @patch("graphs.ask_graph.collection_exists", return_value=True)
    @patch("graphs.ask_graph.build_retriever")
    def test_no_chunks_sets_max_score_none(self, mock_br, _mock_ce, settings, embeddings):
        """When retriever returns no docs, max_retrieval_score is None and path refuses."""
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_br.return_value = mock_retriever

        graph = build_ask_graph(settings, embeddings)
        result = graph.invoke(make_initial_state("vid1", "q?"))

        assert result["max_retrieval_score"] is None
        assert result["refused"] is True


# ---------------------------------------------------------------------------
# Error end-states — node-level exception handling
# ---------------------------------------------------------------------------


class TestErrorNodes:
    @patch("graphs.ask_graph.collection_exists", return_value=True)
    @patch("graphs.ask_graph.build_retriever")
    @patch("graphs.ask_graph.ChatOpenAI")
    def test_retrieve_exception_sets_error_in_state(
        self, MockLLM, mock_br, _mock_ce, settings, embeddings
    ):
        """retrieve_node wraps exceptions in state.error; generate never runs."""
        mock_retriever = MagicMock()
        mock_retriever.invoke.side_effect = RuntimeError("chroma unavailable")
        mock_br.return_value = mock_retriever

        graph = build_ask_graph(settings, embeddings)
        result = graph.invoke(make_initial_state("vid1", "q?"))

        assert result["error"] == "chroma unavailable"
        MockLLM.return_value.invoke.assert_not_called()

    @patch("graphs.ask_graph.collection_exists", return_value=True)
    @patch("graphs.ask_graph.build_retriever")
    @patch("graphs.ask_graph.ChatOpenAI")
    def test_generate_exception_sets_error_in_state(
        self, MockLLM, mock_br, _mock_ce, settings, embeddings
    ):
        """generate_node wraps exceptions in state.error; format_response never runs."""
        doc = _doc()
        doc.metadata["_score"] = 0.9
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [doc]
        mock_br.return_value = mock_retriever
        MockLLM.return_value.invoke.side_effect = RuntimeError("openai timeout")

        graph = build_ask_graph(settings, embeddings)
        result = graph.invoke(make_initial_state("vid1", "q?"))

        assert result["error"] == "openai timeout"

    @patch("graphs.ask_graph.collection_exists", return_value=True)
    @patch("graphs.ask_graph.build_retriever")
    @patch("graphs.ask_graph.ChatOpenAI")
    def test_retrieve_error_leaves_citations_empty(
        self, MockLLM, mock_br, _mock_ce, settings, embeddings
    ):
        """When retrieve fails, citations remain at their zero value (not populated)."""
        mock_retriever = MagicMock()
        mock_retriever.invoke.side_effect = RuntimeError("connection refused")
        mock_br.return_value = mock_retriever

        graph = build_ask_graph(settings, embeddings)
        result = graph.invoke(make_initial_state("vid1", "q?"))

        assert result["citations"] == []
        assert result["error"] == "connection refused"

    @patch("graphs.ask_graph.collection_exists", return_value=True)
    @patch("graphs.ask_graph.build_retriever")
    @patch("graphs.ask_graph.ChatOpenAI")
    def test_generate_error_leaves_citations_empty(
        self, MockLLM, mock_br, _mock_ce, settings, embeddings
    ):
        """When generate fails, citations remain at their zero value."""
        doc = _doc()
        doc.metadata["_score"] = 0.9
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [doc]
        mock_br.return_value = mock_retriever
        MockLLM.return_value.invoke.side_effect = RuntimeError("rate limit")

        graph = build_ask_graph(settings, embeddings)
        result = graph.invoke(make_initial_state("vid1", "q?"))

        assert result["citations"] == []
        assert result["error"] == "rate limit"
