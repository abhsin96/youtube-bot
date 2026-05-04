import sys
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- required ---
    openai_api_key: str

    # --- LangSmith ---
    langsmith_api_key: str = ""
    langsmith_project: str = "youtube-extention"
    langsmith_tracing: str = "false"

    # --- models ---
    embed_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"

    # --- retrieval ---
    vector_db_path: Path = Path("chroma_db")
    min_similarity_threshold: float = 0.25
    max_history_turns: int = 10
    context_budget_tokens: int = 6000

    # --- server ---
    version: str = "0.1.0"
    log_level: str = "INFO"
    json_logs: bool = True
    allowed_origins: list[str] = ["http://localhost:3000"]

    @field_validator("openai_api_key")
    @classmethod
    def require_openai_key(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("OPENAI_API_KEY must not be empty")
        return v

    @field_validator("min_similarity_threshold")
    @classmethod
    def validate_similarity(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("MIN_SIMILARITY_THRESHOLD must be between 0 and 1")
        return v

    @field_validator("max_history_turns")
    @classmethod
    def validate_history_turns(cls, v: int) -> int:
        if v < 1:
            raise ValueError("MAX_HISTORY_TURNS must be >= 1")
        return v


def load_settings() -> Settings:
    try:
        return Settings()
    except Exception as exc:
        # extract field names and surface them as uppercase env var names
        missing_vars = [
            loc.upper()
            for error in (getattr(exc, "errors", lambda: [])())
            for loc in [str(error.get("loc", ["unknown"])[0])]
        ]
        detail = ", ".join(missing_vars) if missing_vars else str(exc)
        print(f"[startup error] Missing required environment variables: {detail}", file=sys.stderr)
        print("  Check your .env file or environment variables.", file=sys.stderr)
        sys.exit(1)
