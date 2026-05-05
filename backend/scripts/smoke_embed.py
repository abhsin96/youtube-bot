"""
Smoke-test: embed a few chunks and verify a LangSmith trace is created.

Usage (from backend/):
    python scripts/smoke_embed.py

Exits 0 on success, 1 on failure.
"""

import sys
import time

sys.path.insert(0, ".")

# Load .env into os.environ BEFORE langsmith/langchain initialise so the
# tracing flag and API key are visible to the SDK at import time.
from dotenv import load_dotenv  # noqa: E402

load_dotenv(".env")

from langchain_core.documents import Document  # noqa: E402
from langsmith import Client  # noqa: E402
from langsmith.run_helpers import tracing_context  # noqa: E402

from src.config import load_settings  # noqa: E402
from src.embeddings import embed_chunks, get_embeddings  # noqa: E402

SAMPLE_DOCS = [
    Document(page_content="LangSmith captures traces for observability."),
    Document(page_content="Exponential backoff protects against rate limits."),
    Document(page_content="Chroma persists vectors across server restarts."),
]


def main() -> None:
    settings = load_settings()

    if not settings.langsmith_api_key:
        print("LANGSMITH_API_KEY is not set.")
        sys.exit(1)

    embeddings = get_embeddings(settings)
    print(f"Embedding {len(SAMPLE_DOCS)} chunks with {settings.embed_model} …")

    t0 = time.monotonic()
    with tracing_context(enabled=True, project_name=settings.langsmith_project):
        pairs = embed_chunks(SAMPLE_DOCS, embeddings)
    elapsed = time.monotonic() - t0
    print(f"  Done in {elapsed:.2f}s — {len(pairs)} (doc, vector) pairs returned.")

    print("Waiting for trace to flush …")
    time.sleep(5)

    client = Client(api_key=settings.langsmith_api_key)

    # Create project if it doesn't exist yet
    try:
        client.read_project(project_name=settings.langsmith_project)
    except Exception:
        print(f"  Project '{settings.langsmith_project}' not found — trace may still be flushing.")
        print("  Check https://smith.langchain.com manually.")
        sys.exit(1)

    runs = list(
        client.list_runs(
            project_name=settings.langsmith_project,
            filter='eq(name, "embed_chunks")',
            limit=5,
        )
    )

    if not runs:
        print("No 'embed_chunks' runs found — trace may still be flushing.")
        print(f"Check project '{settings.langsmith_project}' at https://smith.langchain.com")
        sys.exit(1)

    latest = runs[0]
    print("\nTrace confirmed:")
    print(f"  run_name : {latest.name}")
    print(f"  run_id   : {latest.id}")
    print(f"  status   : {latest.status}")
    print(f"  project  : {settings.langsmith_project}")


if __name__ == "__main__":
    main()
