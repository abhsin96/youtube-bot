import functools
from pathlib import Path

import structlog
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.tokens import count_tokens

logger = structlog.get_logger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# Human turn template: chunks block (with inline [mm:ss] timestamps) then the question.
# Keeping context in the human turn (not the system message) means:
#   - the system prompt stays version-stable across different context sizes
#   - history messages slot naturally between system and the current retrieval context
_HUMAN_TEMPLATE = """\
Relevant transcript excerpts:
{context}

Question: {question}"""


@functools.lru_cache(maxsize=1)
def _load_system_prompt() -> str:
    return (_PROMPTS_DIR / "system_prompt.txt").read_text(encoding="utf-8")


def _seconds_to_mmss(seconds: float) -> str:
    """Convert a float seconds value to [mm:ss] notation."""
    total = int(seconds)
    return f"[{total // 60:02d}:{total % 60:02d}]"


def _format_context(docs: list[Document]) -> str:
    parts = []
    for doc in docs:
        raw_ts = doc.metadata.get("start_ts")
        ts = _seconds_to_mmss(raw_ts) if isinstance(raw_ts, (int, float)) else "[??:??]"
        parts.append(f"{ts} {doc.page_content}")
    return "\n\n".join(parts)


def _count_prompt_tokens(
    sources: list[Document],
    system: str,
    history: list[BaseMessage],
    question: str,
    model: str,
) -> int:
    """Estimate total prompt tokens across all four sections."""
    history_text = "\n".join(str(m.content) for m in history)
    return (
        count_tokens(system, model)
        + count_tokens(history_text, model)
        + count_tokens(_format_context(sources), model)
        + count_tokens(question, model)
    )


def _trim_to_budget(
    sources: list[Document],
    system: str,
    history: list[BaseMessage],
    question: str,
    model: str,
    budget: int,
) -> list[Document]:
    """Drop lowest-score chunks (tail of the sorted list) until the prompt fits budget.

    *sources* must already be sorted highest-relevance-first (as VideoRetriever returns).
    Drops one chunk at a time from the end so high-score evidence is preserved.
    """
    trimmed = list(sources)
    while trimmed and _count_prompt_tokens(trimmed, system, history, question, model) > budget:
        dropped = trimmed.pop()
        logger.debug(
            "context budget: dropped chunk",
            chunk_id=dropped.metadata.get("chunk_id"),
            remaining=len(trimmed),
        )
    if len(trimmed) < len(sources):
        logger.info(
            "context budget enforced",
            original=len(sources),
            kept=len(trimmed),
            budget=budget,
        )
    return trimmed


def build_chat_prompt() -> ChatPromptTemplate:
    """Return a ChatPromptTemplate with four slots:

        [system]   rules loaded from prompts/system_prompt.txt
        [history]  MessagesPlaceholder — zero or more prior (human, ai) turns
        [human]    retrieved chunks block  +  current question

    Variables required at invoke time: context (str), question (str), history (list).
    """
    system = _load_system_prompt()
    return ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder(variable_name="history"),
            ("human", _HUMAN_TEMPLATE),
        ]
    )


_CHAT_PROMPT = build_chat_prompt()
