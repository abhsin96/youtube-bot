from dataclasses import dataclass, field
from pathlib import Path

import structlog
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from src.retriever import build_retriever

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


def _load_system_prompt() -> str:
    return (_PROMPTS_DIR / "system_prompt.txt").read_text(encoding="utf-8")


def _seconds_to_mmss(seconds: float) -> str:
    """Convert a float seconds value to [mm:ss] notation."""
    total = int(seconds)
    return f"[{total // 60:02d}:{total % 60:02d}]"


@dataclass
class RAGResult:
    answer: str
    sources: list[Document] = field(default_factory=list)


def _format_context(docs: list[Document]) -> str:
    parts = []
    for doc in docs:
        raw_ts = doc.metadata.get("start_ts")
        ts = _seconds_to_mmss(raw_ts) if isinstance(raw_ts, (int, float)) else "[??:??]"
        parts.append(f"{ts} {doc.page_content}")
    return "\n\n".join(parts)


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


def build_rag_chain(chat_model: str, openai_api_key: str):
    """Return an LCEL chain: {context, question, history} → answer str."""
    llm = ChatOpenAI(model=chat_model, openai_api_key=openai_api_key)
    return build_chat_prompt() | llm | StrOutputParser()


def answer_question(
    video_id: str,
    question: str,
    embeddings: Embeddings,
    vector_db_path,
    chat_model: str,
    openai_api_key: str,
    k: int = 5,
    score_threshold: float = 0.0,
    history: list[BaseMessage] | None = None,
) -> RAGResult:
    """Retrieve relevant chunks then call the LLM; return answer + sources."""
    retriever = build_retriever(
        video_id, embeddings, vector_db_path, k=k, score_threshold=score_threshold
    )
    sources = retriever.invoke(question)

    if not sources:
        logger.info("no sources found", video_id=video_id, question=question)
        return RAGResult(
            answer="I'm sorry, that information isn't available in the video transcript.",
            sources=[],
        )

    context = _format_context(sources)
    chain = build_rag_chain(chat_model, openai_api_key)
    answer = chain.invoke(
        {
            "context": context,
            "question": question,
            "history": history or [],
        }
    )

    logger.info("question answered", video_id=video_id, sources=len(sources))
    return RAGResult(answer=answer, sources=sources)
