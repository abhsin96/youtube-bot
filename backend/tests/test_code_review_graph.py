from unittest.mock import MagicMock, patch

from graphs.code_review import (
    ReviewState,
    _collect_files,
    _route_after_review,
    build_graph,
    load_files,
    review_file,
    summarise,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state(**overrides) -> ReviewState:
    base: ReviewState = {
        "files": [],
        "findings": [],
        "summary": "",
        "model": "gpt-4o-mini",
        "openai_api_key": "sk-test",
    }
    base.update(overrides)
    return base


def _mock_llm(content: str):
    response = MagicMock()
    response.content = content
    llm = MagicMock()
    llm.invoke.return_value = response
    return llm


# ---------------------------------------------------------------------------
# load_files
# ---------------------------------------------------------------------------


def test_load_files_returns_empty_delta():
    result = load_files(_state(files=["a.py", "b.py"]))
    assert result == {}


# ---------------------------------------------------------------------------
# _route_after_review
# ---------------------------------------------------------------------------


def test_route_continues_when_files_remain():
    state = _state(files=["remaining.py"])
    assert _route_after_review(state) == "review_file"


def test_route_goes_to_summarise_when_no_files():
    state = _state(files=[])
    assert _route_after_review(state) == "summarise"


# ---------------------------------------------------------------------------
# review_file
# ---------------------------------------------------------------------------


def test_review_file_pops_first_file(tmp_path):
    f = tmp_path / "module.py"
    f.write_text("x = 1\n")
    state = _state(files=[str(f), "other.py"])

    with patch("graphs.code_review._llm", return_value=_mock_llm("No issues found.")):
        delta = review_file(state)

    assert str(f) not in delta["files"]
    assert "other.py" in delta["files"]


def test_review_file_appends_finding(tmp_path):
    f = tmp_path / "module.py"
    f.write_text("pass\n")
    state = _state(files=[str(f)])

    with patch("graphs.code_review._llm", return_value=_mock_llm("[HIGH] line 1 — bad")):
        delta = review_file(state)

    assert len(delta["findings"]) == 1
    assert delta["findings"][0]["path"] == str(f)
    assert "[HIGH]" in delta["findings"][0]["issues"]


def test_review_file_handles_missing_file():
    state = _state(files=["/nonexistent/path.py"])
    delta = review_file(state)
    assert "Could not read file" in delta["findings"][0]["issues"]


def test_review_file_passes_source_to_llm(tmp_path):
    f = tmp_path / "mod.py"
    f.write_text("def foo(): pass\n")
    state = _state(files=[str(f)])

    mock = _mock_llm("No issues found.")
    with patch("graphs.code_review._llm", return_value=mock):
        review_file(state)

    call_args = mock.invoke.call_args[0][0]
    assert any("def foo" in str(m.content) for m in call_args)


# ---------------------------------------------------------------------------
# summarise
# ---------------------------------------------------------------------------


def test_summarise_sets_summary():
    state = _state(
        findings=[
            {"path": "a.py", "issues": "[HIGH] line 1 — bug"},
            {"path": "b.py", "issues": "No issues found."},
        ]
    )
    with patch("graphs.code_review._llm", return_value=_mock_llm("Overall: NEEDS WORK")):
        delta = summarise(state)

    assert delta["summary"] == "Overall: NEEDS WORK"


def test_summarise_includes_all_findings_in_prompt():
    state = _state(
        findings=[
            {"path": "x.py", "issues": "unique-string-alpha"},
            {"path": "y.py", "issues": "unique-string-beta"},
        ]
    )
    mock = _mock_llm("verdict")
    with patch("graphs.code_review._llm", return_value=mock):
        summarise(state)

    prompt_text = str(mock.invoke.call_args)
    assert "unique-string-alpha" in prompt_text
    assert "unique-string-beta" in prompt_text


# ---------------------------------------------------------------------------
# Full graph (mocked LLM)
# ---------------------------------------------------------------------------


def test_graph_processes_all_files(tmp_path):
    files = []
    for i in range(3):
        f = tmp_path / f"mod{i}.py"
        f.write_text(f"x = {i}\n")
        files.append(str(f))

    mock = _mock_llm("No issues found.")
    with patch("graphs.code_review._llm", return_value=mock):
        graph = build_graph()
        result = graph.invoke(_state(files=files))

    assert len(result["findings"]) == 3
    assert result["summary"] == "No issues found."


def test_graph_summary_populated(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("pass\n")

    with patch("graphs.code_review._llm", return_value=_mock_llm("GOOD")):
        result = build_graph().invoke(_state(files=[str(f)]))

    assert result["summary"] == "GOOD"


def test_graph_empty_file_list_skips_review():
    with patch("graphs.code_review._llm", return_value=_mock_llm("nothing")):
        result = build_graph().invoke(_state(files=[]))

    assert result["findings"] == []
    assert result["summary"] == "nothing"


# ---------------------------------------------------------------------------
# _collect_files
# ---------------------------------------------------------------------------


def test_collect_files_default_finds_src_py():
    files = _collect_files([])
    assert any("src/" in f or "src\\" in f for f in files)
    assert all(f.endswith(".py") for f in files)
    assert not any("__init__.py" in f for f in files)


def test_collect_files_custom_glob(tmp_path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.txt").write_text("")
    files = _collect_files([str(tmp_path / "*.py")])
    # absolute path glob — just verify extension filter
    assert all(f.endswith(".py") for f in files)
