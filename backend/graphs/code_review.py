"""
LangGraph code-review workflow.

Graph shape:
    load_files → review_file → (loop back if files remain) → summarise → END

Each iteration of review_file processes one source file and appends a
FileFinding to the state.  Once all files are reviewed the graph transitions
to summarise, which asks the LLM to produce a ranked, actionable report.

Usage:
    python graphs/code_review.py [file_or_glob ...]

    Defaults to reviewing every *.py file under src/.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from typing import Annotated, TypedDict

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_settings  # noqa: E402

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class FileFinding(TypedDict):
    path: str
    issues: str  # raw LLM output for this file


def _merge_findings(a: list[FileFinding], b: list[FileFinding]) -> list[FileFinding]:
    return a + b


class ReviewState(TypedDict):
    files: list[str]  # absolute paths still to review
    findings: Annotated[list[FileFinding], _merge_findings]
    summary: str
    model: str
    openai_api_key: str


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

_REVIEW_SYSTEM = """\
You are a senior Python engineer doing a focused code review.
Examine the file below and list ONLY real problems — bugs, security issues,
missing error handling at I/O boundaries, and type-safety gaps.
Skip style nits.  Format each finding as:

  [SEVERITY] line N — description

Severities: CRITICAL | HIGH | MEDIUM | LOW.
If the file has no issues, respond with exactly: "No issues found."
"""

_SUMMARY_SYSTEM = """\
You are a tech lead writing the final summary of a code review.
Given per-file findings, produce a concise, prioritised report:

1. List CRITICAL and HIGH issues first with the file and line reference.
2. Group MEDIUM/LOW issues briefly.
3. End with an overall health verdict: GOOD / NEEDS WORK / CRITICAL.

Be direct.  No waffle.
"""


def _llm(state: ReviewState) -> ChatOpenAI:
    return ChatOpenAI(
        model=state["model"],
        openai_api_key=state["openai_api_key"],
        temperature=0,
    )


def load_files(state: ReviewState) -> dict:
    """No-op at graph entry — files are already in state."""
    logger.info("code review started", file_count=len(state["files"]))
    return {}


def review_file(state: ReviewState) -> dict:
    """Pop the first pending file, review it, append finding."""
    remaining = list(state["files"])
    path = remaining.pop(0)

    try:
        source = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        return {
            "files": remaining,
            "findings": [{"path": path, "issues": f"Could not read file: {exc}"}],
        }

    prompt = f"File: {path}\n\n```python\n{source}\n```"
    llm = _llm(state)
    response = llm.invoke([SystemMessage(content=_REVIEW_SYSTEM), HumanMessage(content=prompt)])
    issues = response.content.strip()
    logger.info("file reviewed", path=path, has_issues=issues != "No issues found.")
    return {
        "files": remaining,
        "findings": [{"path": path, "issues": issues}],
    }


def summarise(state: ReviewState) -> dict:
    """Collapse all per-file findings into a ranked final report."""
    lines = []
    for f in state["findings"]:
        lines.append(f"### {f['path']}\n{f['issues']}")
    combined = "\n\n".join(lines)

    llm = _llm(state)
    response = llm.invoke(
        [
            SystemMessage(content=_SUMMARY_SYSTEM),
            HumanMessage(content=combined),
        ]
    )
    summary = response.content.strip()
    logger.info("review summary generated")
    return {"summary": summary}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def _route_after_review(state: ReviewState) -> str:
    return "review_file" if state["files"] else "summarise"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def build_graph():
    g = StateGraph(ReviewState)
    g.add_node("load_files", load_files)
    g.add_node("review_file", review_file)
    g.add_node("summarise", summarise)

    g.set_entry_point("load_files")
    g.add_conditional_edges("load_files", _route_after_review)
    g.add_conditional_edges("review_file", _route_after_review)
    g.add_edge("summarise", END)

    return g.compile()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _collect_files(patterns: list[str]) -> list[str]:
    root = Path(__file__).resolve().parent.parent
    if not patterns:
        patterns = ["src/*.py"]
    paths: list[Path] = []
    for pat in patterns:
        matches = sorted(root.glob(pat)) if not Path(pat).is_absolute() else [Path(pat)]
        paths.extend(m for m in matches if m.suffix == ".py" and m.name != "__init__.py")
    return [str(p) for p in paths]


def main(file_patterns: list[str] | None = None) -> str:
    settings = load_settings()
    files = _collect_files(file_patterns or [])
    if not files:
        logger.error("no_python_files_found")
        sys.exit(1)

    graph = build_graph()
    result = graph.invoke(
        {
            "files": files,
            "findings": [],
            "summary": "",
            "model": settings.chat_model,
            "openai_api_key": settings.openai_api_key,
        }
    )

    summary = result["summary"]
    print("\n" + "=" * 72)
    print("CODE REVIEW REPORT")
    print("=" * 72)
    print(summary)
    print("=" * 72)

    print("\n--- Per-file findings ---")
    for finding in result["findings"]:
        print(f"\n{finding['path']}")
        print(textwrap.indent(finding["issues"], "  "))

    return summary


if __name__ == "__main__":
    main(sys.argv[1:] if len(sys.argv) > 1 else None)
