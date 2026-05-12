import sys

import structlog
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = structlog.get_logger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- API key (optional at startup; may be provided later via /config/api-key) ---
    openai_api_key: str = ""

    # --- LangSmith ---
    langsmith_api_key: str = ""
    langsmith_project: str = "youtube-extention"
    langsmith_tracing: str = "false"

    # --- OpenAI base URL override (empty = use default api.openai.com) ---
    openai_api_base: str = ""

    # --- models ---
    embed_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"

    # --- Chroma server ---
    chroma_host: str = "localhost"
    chroma_port: int = 8001

    # --- retrieval ---
    min_similarity_threshold: float = 0.25
    max_history_turns: int = 10
    context_budget_tokens: int = 6000

    # --- Redis (empty string = use the in-memory thread store) ---
    redis_url: str = ""

    # --- server ---
    version: str = "0.1.0"
    log_level: str = "INFO"
    json_logs: bool = True
    allowed_origins: list[str] = ["http://localhost:3000"]

    @field_validator("openai_api_key")
    @classmethod
    def strip_openai_key(cls, v: str) -> str:
        # Key is optional at startup; it may be supplied via POST /config/api-key.
        return v.strip()

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

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        upper = v.upper()
        if upper not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(
                f"LOG_LEVEL must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL — got {v!r}"
            )
        return upper

    @model_validator(mode="after")
    def warn_langsmith_misconfigured(self) -> "Settings":
        if self.langsmith_tracing.lower() == "true" and not self.langsmith_api_key:
            logger.warning(
                "langsmith_tracing_enabled_without_key",
                hint="Set LANGSMITH_API_KEY to enable tracing.",
            )
        return self


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
        logger.error(
            "startup_error",
            missing_vars=detail,
            hint="Check your .env file or environment variables.",
        )
        sys.exit(1)
