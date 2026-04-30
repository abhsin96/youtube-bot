from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.config import Settings, load_settings


def make(**overrides) -> Settings:
    return Settings(openai_api_key="sk-test", _env_file=None, **overrides)


# --- defaults ---


def test_defaults():
    s = make()
    assert s.embed_model == "text-embedding-3-small"
    assert s.chat_model == "gpt-4o-mini"
    assert s.vector_db_path == Path("chroma_db")
    assert s.min_similarity_threshold == 0.25
    assert s.max_history_turns == 10
    assert s.langsmith_project == "youtube-extention"
    assert s.langsmith_tracing == "false"


# --- required field ---


def test_missing_openai_key_raises():
    with pytest.raises(ValidationError, match="openai_api_key"):
        Settings(_env_file=None)


def test_empty_openai_key_raises():
    with pytest.raises(ValidationError, match="OPENAI_API_KEY"):
        Settings(openai_api_key="  ", _env_file=None)


# --- overrides ---


def test_custom_models():
    s = make(embed_model="text-embedding-ada-002", chat_model="gpt-4o")
    assert s.embed_model == "text-embedding-ada-002"
    assert s.chat_model == "gpt-4o"


def test_custom_retrieval_params():
    s = make(min_similarity_threshold=0.5, max_history_turns=5)
    assert s.min_similarity_threshold == 0.5
    assert s.max_history_turns == 5


def test_vector_db_path_as_string():
    s = make(vector_db_path="/tmp/mydb")
    assert s.vector_db_path == Path("/tmp/mydb")


# --- validators ---


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_similarity_threshold_out_of_range(bad):
    with pytest.raises(ValidationError, match="MIN_SIMILARITY_THRESHOLD"):
        make(min_similarity_threshold=bad)


def test_max_history_turns_below_one():
    with pytest.raises(ValidationError, match="MAX_HISTORY_TURNS"):
        make(max_history_turns=0)


# --- load_settings error path ---


def test_load_settings_exits_on_missing_key(capsys):
    side_effect = ValidationError.from_exception_data(
        "Settings",
        [{"type": "missing", "loc": ("openai_api_key",), "msg": "Field required", "input": {}}],
    )
    with (
        patch("src.config.Settings", side_effect=side_effect),
        pytest.raises(SystemExit) as exc_info,
    ):
        load_settings()

    assert exc_info.value.code == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().err
