from dataclasses import dataclass, field

import structlog
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.vector_store import query

logger = structlog.get_logger(__name__)

_SYSTEM = """You are a helpful assistant that answers questions about a YouTube video \
based solely on the transcript excerpts provided below.

Rules:
- Answer only from the provided excerpts. Do not use outside knowledge.
- If the excerpts do not contain enough information, say so clearly.
- Be concise. Cite timestamps (start_ts) when relevant.

Transcript excerpts:
{context}"""

_HUMAN = "{question}"


@dataclass
class RAGResult:
    answer: str
    sources: list[Document] = field(default_factory=list)


def _format_context(docs: list[Document]) -> str:
    parts = []
    for doc in docs:
        ts = doc.metadata.get("start_ts", "?")
        parts.append(f"[{ts}s] {doc.page_content}")
    return "\n\n".join(parts)


def build_rag_chain(chat_model: str, openai_api_key: str):
    """Return an LCEL chain: question str → answer str (stateless)."""
    llm = ChatOpenAI(model=chat_model, openai_api_key=openai_api_key)
    prompt = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", _HUMAN)])
    return prompt | llm | StrOutputParser()


def answer_question(
    video_id: str,
    question: str,
    embeddings: Embeddings,
    vector_db_path,
    chat_model: str,
    openai_api_key: str,
    k: int = 5,
) -> RAGResult:
    """Retrieve relevant chunks then call the LLM; return answer + sources."""
    query_vec = embeddings.embed_query(question)
    sources = query(video_id, query_vec, embeddings, vector_db_path, k=k)

    if not sources:
        logger.info("no sources found", video_id=video_id, question=question)
        return RAGResult(
            answer="I couldn't find relevant information in the video transcript.",
            sources=[],
        )

    context = _format_context(sources)
    chain = build_rag_chain(chat_model, openai_api_key)
    answer = chain.invoke({"context": context, "question": question})

    logger.info(
        "question answered",
        video_id=video_id,
        sources=len(sources),
    )
    return RAGResult(answer=answer, sources=sources)
