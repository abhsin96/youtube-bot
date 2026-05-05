"""Lightweight OpenAI mock server for E2E tests.

Exposes:
  POST /v1/embeddings         — deterministic 1536-dim unit vectors keyed by text hash
  POST /v1/chat/completions   — configurable response registry; supports streaming

Usage in tests
--------------
from tests.mock_openai.server import register_response, clear_responses

register_response("list comprehension", "A list comprehension creates a list in one line.")
# ... run test ...
clear_responses()          # or rely on the autouse mock_responses fixture
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import threading
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="mock-openai")

_EMBED_DIM = 1536
_registry_lock = threading.Lock()
_CHAT_REGISTRY: dict[str, str] = {}
_DEFAULT_ANSWER = "Based on the video transcript, I can provide the following answer."


# ---------------------------------------------------------------------------
# Public test helpers
# ---------------------------------------------------------------------------


def register_response(snippet: str, answer: str) -> None:
    """Register a canned answer returned when *snippet* appears in the user message."""
    with _registry_lock:
        _CHAT_REGISTRY[snippet] = answer


def clear_responses() -> None:
    """Remove all registered canned answers (reset to default)."""
    with _registry_lock:
        _CHAT_REGISTRY.clear()


# ---------------------------------------------------------------------------
# Deterministic embedding
# ---------------------------------------------------------------------------


def _deterministic_vector(text: str | list[int]) -> list[float]:
    """Return a reproducible 1536-dim unit vector for *text* (sha256-seeded RNG).

    Args:
        text: Either a string or a list of token IDs
    """
    # Handle both string and tokenized input
    text_str = str(text) if isinstance(text, list) else text

    seed = int(hashlib.sha256(text_str.encode()).hexdigest()[:16], 16)
    rng = random.Random(seed)
    vec = [rng.gauss(0, 1) for _ in range(_EMBED_DIM)]
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


# ---------------------------------------------------------------------------
# /v1/embeddings
# ---------------------------------------------------------------------------


@app.post("/v1/embeddings")
@app.post("/embeddings")
async def embeddings(request: Request) -> JSONResponse:
    body = await request.json()
    raw = body.get("input", [])
    inputs: list[str | list[int]] = raw if isinstance(raw, list) else [raw]
    data = [
        {"object": "embedding", "index": i, "embedding": _deterministic_vector(text)}
        for i, text in enumerate(inputs)
    ]
    # Calculate token count - handle both string and list inputs
    token_count = sum(len(t.split()) if isinstance(t, str) else len(t) for t in inputs)
    return JSONResponse(
        {
            "object": "list",
            "data": data,
            "model": body.get("model", "text-embedding-3-small"),
            "usage": {"prompt_tokens": token_count, "total_tokens": token_count},
        }
    )


# ---------------------------------------------------------------------------
# /v1/chat/completions
# ---------------------------------------------------------------------------


def _pick_answer(messages: list[dict[str, Any]]) -> str:
    user_content = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_content = str(msg.get("content", ""))
            break
    with _registry_lock:
        for snippet, answer in _CHAT_REGISTRY.items():
            if snippet.lower() in user_content.lower():
                return answer
    return _DEFAULT_ANSWER


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages: list[dict[str, Any]] = body.get("messages", [])
    answer = _pick_answer(messages)
    model = body.get("model", "gpt-4o-mini")
    stream = body.get("stream", False)
    created = int(time.time())
    completion_id = f"chatcmpl-mock-{created}"
    prompt_tokens = sum(len(str(m.get("content", "")).split()) for m in messages)
    completion_tokens = len(answer.split())

    if stream:

        async def _sse_stream():
            words = answer.split()
            for i, word in enumerate(words):
                content = word if i == 0 else f" {word}"
                chunk: dict[str, Any] = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            final: dict[str, Any] = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_sse_stream(), media_type="text/event-stream")

    return JSONResponse(
        {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    )
