# ADR 002 — ChromaDB Client-Server Mode

**Status:** Accepted  
**Date:** 2026-05-07

## Context

The backend originally used `chromadb.PersistentClient` (embedded mode), writing directly to a `chroma_db/` directory on the local filesystem.  This worked well during prototyping but created two problems:

1. **Single-process coupling.** The embedded client holds an exclusive lock on the database directory.  Running multiple API workers (uvicorn `--workers N`) caused lock contention and silent data corruption.
2. **Deployment inflexibility.** The database lived inside the application container, making it difficult to scale the API tier horizontally or persist data independently of the container lifecycle.

## Decision

Migrate to **ChromaDB client-server mode** (`chromadb.HttpClient`) with Chroma running as a dedicated service.

* `src/vector_store.py` exposes `get_chroma_client(host, port)` which returns an `HttpClient`.
* All public functions (`add_documents`, `query`, `collection_exists`, `delete_collection`, `save_channel_metadata`, `get_channel_metadata`) accept `client: chromadb.ClientAPI` instead of a filesystem path.
* `src/config.py` exposes `chroma_host` and `chroma_port` (defaulting to `localhost:8001`).
* `docker-compose.yml` defines the `chroma` service (port 8001 → container 8000) and the `api` service.
* The Chroma client is initialised lazily in the FastAPI `lifespan` event so that `create_app()` at module import time does not attempt a network connection.

## Consequences

**Good:**
* Multiple API workers can now connect to the same Chroma instance without lock contention.
* The vector store survives API container restarts without data loss.
* Tests inject `chromadb.EphemeralClient()` directly, keeping the test suite fully offline.

**Trade-offs:**
* Adds a network hop between the API and the vector store (negligible latency on localhost/Docker bridge).
* Requires the Chroma service to be running before the API starts in production (handled by `depends_on` in Compose).
* `chromadb.HttpClient` raises immediately on creation if the server is unreachable, so the lazy lifespan pattern is load-bearing for the test suite.
