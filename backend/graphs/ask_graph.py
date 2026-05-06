"""
LangGraph RAG flow for the /ask endpoint.

Flow:
  validate → retrieve → detect_intent
                              ├─ creator question → creator_search ──┐
                              └─ video question → guardrail_check     │
                                                    ├─ no chunks → refuse → format_response → END
                                                    └─ chunks → generate ◄──────────────────────┘
                                                                   └─ format_response → END

State travels through every node; nodes return partial dicts that are merged
into the running AskState by LangGraph.
"""

from __future__ import annotations

import structlog
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langsmith.run_helpers import get_current_run_tree
from typing_extensions import TypedDict

from src.chain import _format_context, _load_system_prompt, _trim_to_budget, build_chat_prompt
from src.metadata_service import fetch_video_metadata
from src.retriever import build_retriever
from src.vector_store import collection_exists, get_channel_metadata
from src.web_search import search_creator_info

logger = structlog.get_logger(__name__)

REFUSAL_SENTINEL = "isn't available in the video transcript"
VIDEO_NOT_INGESTED = "VIDEO_NOT_INGESTED"


_CREATOR_KEYWORDS = frozenset({
    # explicit creator references
    "creator", "channel", "youtuber", "author",
    # "who is …" in all common forms
    "who is the", "who is this", "who is he", "who is she",
    "who are they", "who are these", "who are the",
    # how the creator was made / started
    "who made", "who created", "who runs",
    # pronouns pointing at the on-screen person
    "this guy", "this person", "this dude", "this woman", "this man",
    "the host", "the presenter", "the speaker",
    # social media & contact
    "instagram", "insta", "twitter", "tiktok", "facebook", "snapchat",
    "twitch", "linkedin", "social media", "social handle",
    "contact", "email", "reach out", "get in touch",
    "handle", "username", "account", "profile",
    # channel/career facts
    "subscriber", "other videos", "other content",
    "biography", "bio", "background", "famous", "known for",
    "started", "their channel", "this channel", "about the channel",
    "how many videos", "when did they", "their other",
})


class AskState(TypedDict):
    # ── Inputs ──────────────────────────────────────────────────────────────
    video_id: str
    question: str
    history: list  # list[BaseMessage]; empty list = no history
    k: int
    # ── Creator search ───────────────────────────────────────────────────────
    is_creator_question: bool      # detected by detect_intent_node
    channel_name: str | None       # resolved from stored metadata or oEmbed
    creator_info: str | None       # web search snippets for the creator
    # ── Intermediate ────────────────────────────────────────────────────────
    retrieved_chunks: list[Document]
    max_retrieval_score: float | None  # highest _score among retrieved docs
    # ── Outputs ─────────────────────────────────────────────────────────────
    refused: bool
    answer: str
    citations: list[dict]  # [{chunk_id, start_ts, end_ts, text}]
    tokens_used: int | None
    error: str | None  # VIDEO_NOT_INGESTED or None


def make_initial_state(
    video_id: str,
    question: str,
    history: list | None = None,
    k: int = 5,
) -> AskState:
    """Return a fully initialised AskState for graph.invoke()."""
    return {
        "video_id": video_id,
        "question": question,
        "history": history or [],
        "k": k,
        "is_creator_question": False,
        "channel_name": None,
        "creator_info": None,
        "retrieved_chunks": [],
        "max_retrieval_score": None,
        "refused": False,
        "answer": "",
        "citations": [],
        "tokens_used": None,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------


def build_ask_graph(settings, embeddings, *, api_key: str | None = None):
    """
    Compile the ask LangGraph, closing over *settings* and *embeddings*.

    Pass *api_key* to override ``settings.openai_api_key`` (e.g. when the key
    was retrieved from the OS keyring at request time).

    The LLM and prompt are instantiated once per compiled graph to amortise
    construction cost across repeated invocations.
    """
    vector_db_path = settings.vector_db_path
    chat_model_name = settings.chat_model
    openai_api_key = api_key or settings.openai_api_key
    score_threshold = settings.min_similarity_threshold
    context_budget_tokens = settings.context_budget_tokens
    openai_api_base: str = getattr(settings, "openai_api_base", "")

    llm_kwargs: dict = {
        "model": chat_model_name,
        "openai_api_key": openai_api_key,
        "temperature": 0.2,
        "streaming": False,
    }
    if openai_api_base:
        llm_kwargs["base_url"] = openai_api_base
    llm = ChatOpenAI(**llm_kwargs)
    prompt_template = build_chat_prompt()

    # ── Nodes ────────────────────────────────────────────────────────────────

    def validate_node(state: AskState) -> dict:
        """Abort with VIDEO_NOT_INGESTED if the Chroma collection is absent."""
        if not collection_exists(state["video_id"], vector_db_path):
            logger.info("collection not found", video_id=state["video_id"])
            return {"error": VIDEO_NOT_INGESTED}
        return {"error": None}

    def retrieve_node(state: AskState) -> dict:
        """Fetch the top-k relevant chunks from the vector store (no threshold filtering)."""
        try:
            # Remove score_threshold to get top-k results regardless of score
            retriever = build_retriever(
                state["video_id"],
                embeddings,
                vector_db_path,
                k=state.get("k", 5),
                score_threshold=0.0,  # No threshold - let LLM decide if answerable
            )
            chunks = retriever.invoke(state["question"])
            max_score: float | None = (
                max(doc.metadata.get("_score", 0.0) for doc in chunks) if chunks else None
            )
            logger.info(
                "chunks retrieved",
                video_id=state["video_id"],
                count=len(chunks),
                max_score=max_score,
            )
            return {"retrieved_chunks": chunks, "max_retrieval_score": max_score}
        except Exception as exc:
            logger.exception("retrieve_node_failed", video_id=state["video_id"])
            return {"error": str(exc)}

    def detect_intent_node(state: AskState) -> dict:
        """Classify question as creator-related or video-content-related.

        Uses keyword heuristics (no extra LLM call).  If creator intent is
        detected, also resolves the channel name from stored metadata so the
        subsequent creator_search_node can build a targeted query.
        """
        q = state["question"].lower()
        is_creator = any(kw in q for kw in _CREATOR_KEYWORDS)

        channel_name: str | None = None
        if is_creator:
            meta = get_channel_metadata(state["video_id"], vector_db_path)
            channel_name = meta.get("channel_name") or None
            logger.info(
                "creator_intent_detected",
                video_id=state["video_id"],
                channel_name=channel_name,
            )

        return {"is_creator_question": is_creator, "channel_name": channel_name}

    def creator_search_node(state: AskState) -> dict:
        """Search the web for creator/channel info and return formatted snippets.

        Falls back to the oEmbed API to resolve the channel name when it was
        not stored during ingest (e.g. for videos ingested before this feature).
        """
        channel_name = state.get("channel_name") or ""
        if not channel_name:
            meta = fetch_video_metadata(state["video_id"])
            channel_name = meta.get("channel_name", "")

        if not channel_name:
            logger.warning("creator_search_skipped_no_channel", video_id=state["video_id"])
            return {"creator_info": None}

        info = search_creator_info(channel_name, state["question"])
        return {"creator_info": info or None}

    def guardrail_check_node(state: AskState) -> dict:  # noqa: ARG001
        """Routing-only node — checks if we have any chunks to work with."""
        return {}

    def refuse_node(state: AskState) -> dict:  # noqa: ARG001
        """Emit the standard refusal when no context was found."""
        try:
            rt = get_current_run_tree()
            if rt is not None:
                rt.add_metadata(
                    {"refused": True, "max_retrieval_score": state.get("max_retrieval_score")}
                )
                rt.add_tags(["refused"])
                rt.patch()
        except Exception:
            pass
        return {
            "refused": True,
            "answer": "I'm sorry, that information isn't available in the video transcript.",
        }

    def generate_node(state: AskState) -> dict:
        """Trim context to token budget, call the LLM, capture usage metadata."""
        try:
            history = state.get("history") or []
            system = _load_system_prompt()
            chunks = _trim_to_budget(
                state["retrieved_chunks"],
                system,
                history,
                state["question"],
                chat_model_name,
                context_budget_tokens,
            )

            creator_info = state.get("creator_info") or ""

            if not chunks and not creator_info:
                return {
                    "retrieved_chunks": [],
                    "refused": True,
                    "answer": "I'm sorry, that information isn't available in the video transcript.",
                }

            context = _format_context(chunks) if chunks else ""
            if creator_info:
                separator = "\n\n" if context else ""
                context += f"{separator}[Creator/Channel Information from web search]\n{creator_info}"

            prompt_value = prompt_template.invoke(
                {"context": context, "question": state["question"], "history": history}
            )
            ai_message = llm.invoke(prompt_value)
            answer = ai_message.content if hasattr(ai_message, "content") else str(ai_message)

            # LangChain ≥ 0.2 surfaces token counts on AIMessage.usage_metadata
            tokens_used: int | None = None
            usage = getattr(ai_message, "usage_metadata", None)
            if isinstance(usage, dict):
                tokens_used = usage.get("total_tokens")

            refused = REFUSAL_SENTINEL.lower() in answer.lower()
            logger.info(
                "answer generated",
                video_id=state["video_id"],
                refused=refused,
                tokens=tokens_used,
            )
            return {
                "answer": answer,
                "refused": refused,
                "retrieved_chunks": chunks,  # trimmed — used for citation extraction
                "tokens_used": tokens_used,
            }
        except Exception as exc:
            logger.exception("generate_node_failed", video_id=state["video_id"])
            return {"error": str(exc)}

    def format_response_node(state: AskState) -> dict:
        """Build citations list from the (post-budget-trim) retrieved chunks."""
        if state.get("refused"):
            return {"citations": []}
        citations = [
            {
                "chunk_id": doc.metadata.get("chunk_id"),
                "start_ts": doc.metadata.get("start_ts"),
                "end_ts": doc.metadata.get("end_ts"),
                "text": doc.page_content,
            }
            for doc in state.get("retrieved_chunks", [])
        ]
        return {"citations": citations}

    # ── Routing ──────────────────────────────────────────────────────────────

    def route_after_validate(state: AskState) -> str:
        return END if state.get("error") else "retrieve"

    def route_after_retrieve(state: AskState) -> str:
        return END if state.get("error") else "detect_intent"

    def route_after_detect_intent(state: AskState) -> str:
        return "creator_search" if state.get("is_creator_question") else "guardrail_check"

    def route_after_generate(state: AskState) -> str:
        return END if state.get("error") else "format_response"

    def route_after_guardrail(state: AskState) -> str:
        chunks = state.get("retrieved_chunks") or []
        return "refuse" if not chunks else "generate"

    # ── Graph assembly ────────────────────────────────────────────────────────

    g = StateGraph(AskState)
    g.add_node("validate", validate_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("detect_intent", detect_intent_node)
    g.add_node("creator_search", creator_search_node)
    g.add_node("guardrail_check", guardrail_check_node)
    g.add_node("refuse", refuse_node)
    g.add_node("generate", generate_node)
    g.add_node("format_response", format_response_node)

    g.set_entry_point("validate")
    g.add_conditional_edges(
        "validate",
        route_after_validate,
        {END: END, "retrieve": "retrieve"},
    )
    g.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {END: END, "detect_intent": "detect_intent"},
    )
    g.add_conditional_edges(
        "detect_intent",
        route_after_detect_intent,
        {"creator_search": "creator_search", "guardrail_check": "guardrail_check"},
    )
    # Creator search always proceeds to generate (web results are the context).
    g.add_edge("creator_search", "generate")
    g.add_conditional_edges(
        "guardrail_check",
        route_after_guardrail,
        {"refuse": "refuse", "generate": "generate"},
    )
    g.add_edge("refuse", "format_response")
    g.add_conditional_edges(
        "generate",
        route_after_generate,
        {END: END, "format_response": "format_response"},
    )
    g.add_edge("format_response", END)

    return g.compile()
