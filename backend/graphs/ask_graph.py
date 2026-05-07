"""
LangGraph RAG flow for the /ask endpoint.

Flow:  validate → retrieve → generate → format_response → END

The LLM is bound to a ``search_web_for_creator`` tool.  It calls the tool
autonomously whenever the user's question requires information that is not in
the transcript (e.g. creator background, social media, subscriber count).
No keyword lists or intent classifiers — the model decides.

State travels through every node; nodes return partial dicts that are merged
into the running AskState by LangGraph.
"""

from __future__ import annotations

import chromadb
import structlog
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from src.chain import _CHAT_PROMPT, _format_context, _load_system_prompt, _trim_to_budget
from src.metadata_service import fetch_video_metadata
from src.retriever import build_retriever
from src.vector_store import collection_exists, get_channel_metadata
from src.web_search import search as web_search

logger = structlog.get_logger(__name__)

REFUSAL_SENTINEL = "isn't available in the video transcript"
VIDEO_NOT_INGESTED = "VIDEO_NOT_INGESTED"


# ---------------------------------------------------------------------------
# Tool — defined at module level so the docstring is stable for the LLM
# ---------------------------------------------------------------------------


@tool
def search_web_for_creator(query: str) -> str:
    """Search the internet for information about a YouTube creator or channel.

    Use this tool whenever the user asks about anything that would NOT be in
    the video transcript: the creator's name, biography, social media handles,
    Instagram, Twitter, TikTok, subscriber count, other videos, contact
    details, channel history, or any other information about who made the video.

    Args:
        query: A concise web search query. Include the creator or channel name
               if you know it (it is provided in the context as "Video Channel").
    """
    result = web_search(query)
    if not result:
        return "No results found."
    return result


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class AskState(TypedDict):
    # ── Inputs ──────────────────────────────────────────────────────────────
    video_id: str
    question: str
    history: list  # list[BaseMessage]; empty list = no history
    k: int
    # ── Intermediate ────────────────────────────────────────────────────────
    retrieved_chunks: list[Document]
    max_retrieval_score: float | None
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


def build_ask_graph(
    settings,
    embeddings,
    *,
    api_key: str | None = None,
    chroma_client: chromadb.ClientAPI | None = None,
):
    """
    Compile the ask LangGraph, closing over *settings*, *embeddings*, and *chroma_client*.

    The LLM is bound with a web-search tool so it can autonomously look up
    creator/channel information when the transcript context is insufficient.
    """
    chat_model_name = settings.chat_model
    openai_api_key = api_key or settings.openai_api_key
    context_budget_tokens = settings.context_budget_tokens
    openai_api_base: str = getattr(settings, "openai_api_base", "")
    score_threshold: float = settings.min_similarity_threshold

    llm_kwargs: dict = {
        "model": chat_model_name,
        "openai_api_key": openai_api_key,
        "temperature": 0.2,
        "streaming": True,
    }
    if openai_api_base:
        llm_kwargs["base_url"] = openai_api_base

    llm = ChatOpenAI(**llm_kwargs)
    llm_with_tools = llm.bind_tools([search_web_for_creator])
    # ── Nodes ────────────────────────────────────────────────────────────────

    def validate_node(state: AskState) -> dict:
        if not collection_exists(state["video_id"], chroma_client):
            logger.info("collection not found", video_id=state["video_id"])
            return {"error": VIDEO_NOT_INGESTED}
        return {"error": None}

    def retrieve_node(state: AskState) -> dict:
        try:
            retriever = build_retriever(
                state["video_id"],
                embeddings,
                chroma_client,
                k=state.get("k", 5),
                score_threshold=score_threshold,
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

    def generate_node(state: AskState) -> dict:
        """Call the LLM (with tool access) and handle any tool invocations."""
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

            # Build transcript context
            context = _format_context(chunks) if chunks else ""

            # Prepend the channel name so the LLM can build a good search query
            # when it decides to call the web search tool.
            channel_name = _resolve_channel_name(state["video_id"], chroma_client)
            if channel_name:
                prefix = f"[Video Channel: {channel_name}]\n\n"
                context = prefix + context if context else prefix.strip()

            if not context:
                # No transcript and no channel name — nothing to work with.
                return {
                    "retrieved_chunks": [],
                    "refused": True,
                    "answer": "I'm sorry, that information isn't available in the video transcript.",
                }

            prompt_value = _CHAT_PROMPT.invoke(
                {"context": context, "question": state["question"], "history": history}
            )
            messages = prompt_value.to_messages()

            # First LLM call — model may decide to invoke the search tool.
            ai_message = llm_with_tools.invoke(messages)

            # Execute any requested tool calls, then re-invoke for the final answer.
            tool_calls = getattr(ai_message, "tool_calls", None) or []
            if tool_calls:
                tool_messages: list = []
                for tc in tool_calls:
                    result = search_web_for_creator.invoke(tc["args"])
                    tool_messages.append(
                        ToolMessage(content=str(result), tool_call_id=tc["id"])
                    )
                    logger.info(
                        "tool_called",
                        video_id=state["video_id"],
                        tool=tc["name"],
                        query=tc["args"].get("query", ""),
                    )
                ai_message = llm.invoke(messages + [ai_message] + tool_messages)

            answer = ai_message.content if hasattr(ai_message, "content") else str(ai_message)

            tokens_used: int | None = None
            usage = getattr(ai_message, "usage_metadata", None)
            if isinstance(usage, dict):
                tokens_used = usage.get("total_tokens")

            refused = REFUSAL_SENTINEL.lower() in answer.lower()
            logger.info(
                "answer generated",
                video_id=state["video_id"],
                refused=refused,
                tool_used=bool(tool_calls),
                tokens=tokens_used,
            )
            return {
                "answer": answer,
                "refused": refused,
                "retrieved_chunks": chunks,
                "tokens_used": tokens_used,
            }
        except Exception as exc:
            logger.exception("generate_node_failed", video_id=state["video_id"])
            return {"error": str(exc)}

    def format_response_node(state: AskState) -> dict:
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
        return END if state.get("error") else "generate"

    def route_after_generate(state: AskState) -> str:
        return END if state.get("error") else "format_response"

    # ── Graph assembly ────────────────────────────────────────────────────────

    g = StateGraph(AskState)
    g.add_node("validate", validate_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.add_node("format_response", format_response_node)

    g.set_entry_point("validate")
    g.add_conditional_edges("validate", route_after_validate, {END: END, "retrieve": "retrieve"})
    g.add_conditional_edges("retrieve", route_after_retrieve, {END: END, "generate": "generate"})
    g.add_conditional_edges(
        "generate", route_after_generate, {END: END, "format_response": "format_response"}
    )
    g.add_edge("format_response", END)

    return g.compile()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_channel_name(video_id: str, chroma_client) -> str:
    """Return channel name from stored metadata, falling back to oEmbed API."""
    meta = get_channel_metadata(video_id, chroma_client)
    name = meta.get("channel_name", "")
    if not name:
        name = fetch_video_metadata(video_id).get("channel_name", "")
    return name
