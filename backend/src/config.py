import sys

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str
    langsmith_api_key: str = ""
    langsmith_project: str = "youtube-extention"
    langsmith_tracing: str = "false"

    log_level: str = "INFO"
    allowed_origins: list[str] = ["http://localhost:3000"]

    @field_validator("openai_api_key")
    @classmethod
    def require_openai_key(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("OPENAI_API_KEY must not be empty")
        return v


def load_settings() -> Settings:
    try:
        return Settings()
    except Exception as exc:
        missing = [
            line.strip()
            for line in str(exc).splitlines()
            if "openai_api_key" in line.lower() or "missing" in line.lower()
        ]
        detail = "; ".join(missing) if missing else str(exc)
        print(f"[startup error] Invalid configuration: {detail}", file=sys.stderr)
        print("  Set OPENAI_API_KEY in your environment or backend/.env file.", file=sys.stderr)
        sys.exit(1)
