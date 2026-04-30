# ADR 001 — Vector Store: Chroma (with FAISS swap path)

**Status:** Accepted  
**Date:** 2026-04-30

---

## Decision

Use **Chroma** (`langchain-chroma`) as the vector store with a `PersistentClient`
backed by `VECTOR_DB_PATH`.

---

## Context

The project needs per-video vector collections that survive server restarts,
support metadata filtering, and require no external infrastructure in development.
Two candidates were evaluated: Chroma and FAISS.

| Criterion | Chroma | FAISS |
|---|---|---|
| Persistence | Built-in (`PersistentClient`) | Manual (save/load `.index` + `.pkl`) |
| Named collections | Yes (`collection_name=`) | No — one index per file |
| Metadata filtering | Yes (`where=` clause) | No — post-filter only |
| External process | No | No |
| Production scaling | Chroma server / Cloud | Faiss-CPU on any machine |
| LangChain integration | `langchain-chroma` | `langchain-community` FAISS wrapper |

Chroma wins on persistence and metadata ergonomics for the current scope.
FAISS wins on raw ANN throughput and zero Python dependencies beyond `faiss-cpu`.

---

## Swap path: Chroma → FAISS

All vector-store logic is isolated behind four functions in `src/vector_store.py`.
Replacing Chroma with FAISS requires changes in **that file only**.

### Interface changes

| Function | Chroma today | FAISS equivalent |
|---|---|---|
| `add_documents` | `Chroma.add_documents(docs)` | `FAISS.from_documents(docs, emb)` then `index.save_local(path)` |
| `query` | `Chroma.similarity_search_by_vector(vec, k)` | `FAISS.load_local(path, emb).similarity_search_by_vector(vec, k)` |
| `collection_exists` | `client.list_collections()` name check | Check `Path(f"{path}/{video_id}.faiss").exists()` |
| `delete_collection` | `client.delete_collection(name)` | `Path(f"{path}/{video_id}.faiss").unlink()` + `.pkl` sidecar |

### Persistence

Chroma stores all collections in one SQLite+binary tree under `VECTOR_DB_PATH/`.
FAISS requires one `.faiss` + one `.pkl` file pair per video:

```
VECTOR_DB_PATH/
  video_dQw4w9WgXcQ.faiss
  video_dQw4w9WgXcQ.pkl
  video_abc123.faiss
  video_abc123.pkl
```

The `_collection_name` / `_sanitize_video_id` helpers can be reused as-is to
derive filenames.

### Embedding pass-through

Both backends accept an `Embeddings` instance.  No change is needed at the call
site — `add_documents`, `query` already receive the embedding object as a
parameter.  The only difference is when the embedding is called:

- **Chroma** — `embedding.embed_documents(texts)` on `add_documents`;
  `embedding.embed_query(text)` internally when using `similarity_search`.
  Our `query` function bypasses this by accepting a pre-computed vector and
  calling `similarity_search_by_vector` directly — **this works identically
  with FAISS**.

- **FAISS** — `FAISS.from_documents` calls `embed_documents` once at index
  build time.  `load_local` requires the same embedding object to be passed
  again so it can embed the query vector at search time (or use
  `similarity_search_by_vector` with a pre-computed vector, same as Chroma).

### What does NOT change

- `src/tokens.py`, `src/chunker.py`, `src/document_converter.py` — unaffected.
- `src/config.py` — `VECTOR_DB_PATH` stays; no new settings needed.
- All callers of `add_documents` / `query` / `collection_exists` /
  `delete_collection` — the function signatures are unchanged.
- All existing tests in `test_vector_store.py` — the fake-embedding pattern and
  `tmp_path` fixture work identically with FAISS.

### Migration steps

1. `pip install faiss-cpu` (add to `requirements.txt`; remove `langchain-chroma`).
2. Replace `src/vector_store.py` implementation (≈ 85 lines).
3. Delete the existing `VECTOR_DB_PATH` directory (different on-disk format).
4. Run `make test` — all tests should pass without changes.
