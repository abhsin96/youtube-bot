from dataclasses import dataclass, field
from pathlib import Path

import structlog
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.retriever import build_retriever

logger = structlog.get_logger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _load_system_prompt() -> str:
    return (_PROMPTS_DIR / "system_prompt.txt").read_text(encoding="utf-8")


def _seconds_to_mmss(seconds: float) -> str:
    """Convert a float seconds value to [mm:ss] notation."""
    total = int(seconds)
    return f"[{total // 60:02d}:{total % 60:02d}]"


_HUMAN = "{question}"


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


def build_rag_chain(chat_model: str, openai_api_key: str):
    """Return an LCEL chain: question str → answer str (stateless)."""
    llm = ChatOpenAI(model=chat_model, openai_api_key=openai_api_key)
    system = _load_system_prompt()
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", _HUMAN)])
    return prompt | llm | StrOutputParser()


def answer_question(
    video_id: str,
    question: str,
    embeddings: Embeddings,
    vector_db_path,
    chat_model: str,
    openai_api_key: str,
    k: int = 5,
    score_threshold: float = 0.0,
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
    answer = chain.invoke({"context": context, "question": question})

    logger.info("question answered", video_id=video_id, sources=len(sources))
    return RAGResult(answer=answer, sources=sources)
