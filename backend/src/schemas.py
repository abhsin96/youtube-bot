from typing import Literal

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    video_id: str = Field(..., min_length=1, description="YouTube video ID")
    force: bool = Field(False, description="Re-ingest even if collection already exists")
    stream: bool = Field(False, description="Return SSE progress stream instead of a JSON body")


class IngestResponse(BaseModel):
    status: str
    chunk_count: int
    cached: bool


class QuestionResponse(BaseModel):
    answer: str
    sources: list[dict]


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"] = Field(..., description="Speaker role")
    content: str = Field(..., min_length=1, description="Message text")


class QueryRequest(BaseModel):
    video_id: str = Field(..., min_length=1, description="YouTube video ID")
    question: str = Field(..., min_length=1, description="Question about the video")
    stream: bool = Field(False, description="Enable streaming response")
    conversation_history: list[ConversationTurn] = Field(
        default_factory=list, description="Prior turn messages"
    )
    k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")
    thread_id: str | None = Field(None, description="Thread ID for conversation tracking")


class CitationSchema(BaseModel):
    chunk_id: str | None = None
    start_ts: float | None = None
    end_ts: float | None = None
    text: str


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationSchema]
    tokens_used: int | None = None
    refused: bool = False
    thread_id: str


class ApiKeyRequest(BaseModel):
    api_key: str = Field(..., min_length=1, description="OpenAI API key")


class ConfigStatusResponse(BaseModel):
    has_key: bool
