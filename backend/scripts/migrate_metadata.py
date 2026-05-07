"""One-shot migration: move channel metadata from JSON sidecar files into Chroma collection metadata.

Usage:
    python scripts/migrate_metadata.py [--db-path PATH] [--dry-run]

Default db-path is the value of VECTOR_DB_PATH in .env, falling back to ./chroma_db.
"""

import argparse
import json
import sys
from pathlib import Path

# Make sure the backend package root is on the path when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb
from chromadb.config import Settings as ChromaSettings

_NO_TELEMETRY = ChromaSettings(anonymized_telemetry=False)


def _collection_name_from_sidecar(stem: str) -> str:
    """Convert a sidecar filename stem like 'abc_123_meta' → collection name 'video_abc_123'."""
    # Strip trailing _meta suffix
    if stem.endswith("_meta"):
        stem = stem[: -len("_meta")]
    return f"video_{stem}"[:63]


def migrate(db_path: Path, *, dry_run: bool) -> None:
    sidecars = list(db_path.glob("*_meta.json"))
    if not sidecars:
        print(f"No sidecar files found in {db_path}. Nothing to migrate.")
        return

    client = chromadb.PersistentClient(path=str(db_path), settings=_NO_TELEMETRY)
    existing = {c.name for c in client.list_collections()}

    migrated = 0
    skipped = 0
    for sidecar in sidecars:
        collection_name = _collection_name_from_sidecar(sidecar.stem)
        if collection_name not in existing:
            print(f"  SKIP  {sidecar.name} — collection '{collection_name}' not found")
            skipped += 1
            continue

        try:
            channel_meta = json.loads(sidecar.read_text())
        except Exception as exc:
            print(f"  ERROR {sidecar.name} — failed to read JSON: {exc}")
            skipped += 1
            continue

        collection = client.get_collection(collection_name)
        existing_meta = collection.metadata or {}
        non_hnsw = {k: v for k, v in existing_meta.items() if not k.startswith("hnsw:")}
        updated = {**non_hnsw, **channel_meta}

        if dry_run:
            print(f"  DRY   {sidecar.name} → {collection_name}: {channel_meta}")
        else:
            collection.modify(metadata=updated)
            sidecar.unlink()
            print(f"  OK    {sidecar.name} → {collection_name}: {channel_meta}")
        migrated += 1

    print(f"\nDone. migrated={migrated}, skipped={skipped}, dry_run={dry_run}")


def _default_db_path() -> Path:
    env_file = Path(__file__).resolve().parents[1] / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("VECTOR_DB_PATH="):
                return Path(line.split("=", 1)[1].strip())
    return Path(__file__).resolve().parents[1] / "chroma_db"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=None, help="Path to the Chroma DB directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    args = parser.parse_args()

    db_path = args.db_path or _default_db_path()
    if not db_path.exists():
        print(f"DB path does not exist: {db_path}")
        sys.exit(1)

    print(f"Migrating channel metadata from JSON sidecars in: {db_path}")
    if args.dry_run:
        print("(dry-run mode — no changes will be made)\n")
    migrate(db_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
