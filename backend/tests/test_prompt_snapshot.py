"""
Snapshot test for the fully assembled ChatPromptTemplate.

Fixed inputs (context, question, history) are rendered through build_chat_prompt()
and compared against a stored golden file.  Any change to the system prompt file,
the human template, placeholder wiring, or timestamp formatting will fail this test.

Updating the snapshot
---------------------
When a prompt change is intentional, regenerate the snapshot and commit it:

    UPDATE_SNAPSHOTS=1 pytest tests/test_prompt_snapshot.py

The diff in the golden file is then the code-review artifact for the prompt change.
"""

import os
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.chain import build_chat_prompt

_SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
_SNAPSHOT_FILE = _SNAPSHOT_DIR / "prompt_assembled.txt"

# ---------------------------------------------------------------------------
# Fixed inputs — must never change between runs without updating the snapshot
# ---------------------------------------------------------------------------

_CONTEXT = (
    "[00:05] The mitochondria is the powerhouse of the cell.\n\n"
    "[01:23] ATP synthesis occurs across the inner mitochondrial membrane."
)
_QUESTION = "What does the speaker say about energy production?"
_HISTORY = [
    HumanMessage(content="What is the video about?"),
    AIMessage(content="It covers cell biology basics."),
]


# ---------------------------------------------------------------------------
# Rendering helper
# ---------------------------------------------------------------------------

_ROLE_MAP = {
    "SystemMessage": "system",
    "HumanMessage": "human",
    "AIMessage": "ai",
}


def _render(context: str, question: str, history: list) -> str:
    """Render the prompt to a human-readable string suitable for snapshot storage."""
    prompt = build_chat_prompt()
    messages = prompt.format_messages(context=context, question=question, history=history)
    parts = []
    for msg in messages:
        role = _ROLE_MAP.get(type(msg).__name__, type(msg).__name__.lower())
        parts.append(f"=== {role} ===\n{msg.content}")
    return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Snapshot management
# ---------------------------------------------------------------------------


def _should_update() -> bool:
    return os.getenv("UPDATE_SNAPSHOTS", "").strip().lower() in ("1", "true", "yes")


def _write_snapshot(text: str) -> None:
    _SNAPSHOT_DIR.mkdir(exist_ok=True)
    _SNAPSHOT_FILE.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_prompt_snapshot_matches_golden():
    """Full assembled prompt must match the stored snapshot."""
    rendered = _render(_CONTEXT, _QUESTION, _HISTORY)

    if _should_update() or not _SNAPSHOT_FILE.exists():
        _write_snapshot(rendered)
        if _should_update():
            pytest.skip(
                "Snapshot updated — review the diff in tests/snapshots/prompt_assembled.txt"
            )
        return  # first-time generation; snapshot now exists

    stored = _SNAPSHOT_FILE.read_text(encoding="utf-8")
    assert rendered == stored, (
        "Assembled prompt has drifted from the stored snapshot.\n"
        "If this change is intentional run:\n\n"
        "    UPDATE_SNAPSHOTS=1 pytest tests/test_prompt_snapshot.py\n\n"
        "then commit tests/snapshots/prompt_assembled.txt alongside the code change.\n"
        "\n--- stored (first 300 chars) ---\n"
        f"{stored[:300]!r}"
        "\n\n--- rendered (first 300 chars) ---\n"
        f"{rendered[:300]!r}"
    )


def test_prompt_snapshot_contains_system_rules():
    """Sanity: system section must include the key rules even without the snapshot file."""
    rendered = _render(_CONTEXT, _QUESTION, history=[])
    assert "=== system ===" in rendered
    assert "ONLY" in rendered
    assert "[mm:ss]" in rendered
    assert "[hh:mm:ss]" in rendered


def test_prompt_timestamp_format_wording():
    """Exact canonical wording for timestamp format must be present in the system prompt."""
    rendered = _render(_CONTEXT, _QUESTION, history=[])
    assert "[mm:ss] for videos under 1 hour" in rendered
    assert "[hh:mm:ss] otherwise" in rendered
    assert "Never use other formats" in rendered


def test_prompt_snapshot_contains_context_verbatim():
    rendered = _render(_CONTEXT, _QUESTION, history=[])
    assert _CONTEXT in rendered


def test_prompt_snapshot_contains_question_verbatim():
    rendered = _render(_CONTEXT, _QUESTION, history=[])
    assert _QUESTION in rendered


def test_prompt_snapshot_contains_history_turns():
    rendered = _render(_CONTEXT, _QUESTION, _HISTORY)
    assert "What is the video about?" in rendered
    assert "It covers cell biology basics." in rendered


def test_prompt_snapshot_message_order():
    """system → history human → history ai → current human."""
    rendered = _render(_CONTEXT, _QUESTION, _HISTORY)
    pos_system = rendered.index("=== system ===")
    pos_human_hist = rendered.index("What is the video about?")
    pos_ai_hist = rendered.index("It covers cell biology basics.")
    pos_question = rendered.index(_QUESTION)
    assert pos_system < pos_human_hist < pos_ai_hist < pos_question


def test_prompt_snapshot_context_before_question_in_human_turn():
    rendered = _render(_CONTEXT, _QUESTION, history=[])
    assert rendered.index(_CONTEXT) < rendered.index(_QUESTION)


def test_prompt_snapshot_no_raw_seconds_in_context():
    """Timestamps must be [mm:ss] not raw floats."""
    rendered = _render(_CONTEXT, _QUESTION, history=[])
    # _CONTEXT already uses [mm:ss]; ensure no "0.0s" style leaks through _format_context
    assert "0.0s" not in rendered
    assert "1.0s" not in rendered
