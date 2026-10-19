"""Export one user's consistent memory snapshot to an atomic JSON artifact.

Usage: python -m autopersona_memory.export_snapshot ROOT USER_ID OUTPUT
The output contains memory contents. Stdout contains only counts and a digest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .models import MemoryBundle
from .store import MEMORY_CLASSES, JsonlMemoryStore, MemoryStoreError, memory_snapshot_fingerprint


def export_snapshot(root: str | Path, user_id: str, destination: str | Path) -> dict:
    root = Path(root).resolve()
    destination = Path(destination).resolve()
    if not root.is_dir():
        raise ValueError("Memory root must be an existing directory")
    if destination == root or root in destination.parents:
        raise ValueError("Export destination must be outside the memory store")
    # Hash and serialize exactly the same captured snapshot, never a second read.
    snapshot = JsonlMemoryStore(root).snapshot(user_id)
    counts = {bank: len(rows) for bank, rows in snapshot.to_dict().items()}
    summary = {"schema_version": 1, "fingerprint": memory_snapshot_fingerprint(snapshot), "counts": counts}
    content = json.dumps({**summary, "memories": snapshot.to_dict()}, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    JsonlMemoryStore._atomic_write(destination, content)
    return summary


def verify_export(destination: str | Path) -> dict:
    """Validate an exported snapshot without opening the source memory store.

    This is deliberately content-only: an operator can verify an artifact before
    sharing or restoring it, even when the original store is offline.
    """
    path = Path(destination)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read snapshot export: {path}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Unsupported snapshot export schema")
    if set(payload) != {"schema_version", "fingerprint", "counts", "memories"}:
        raise ValueError("Snapshot export contains unsupported fields")
    memories = payload.get("memories")
    counts = payload.get("counts")
    if not isinstance(memories, dict) or not isinstance(counts, dict):
        raise ValueError("Snapshot export is missing memories or counts")
    if set(memories) != set(MEMORY_CLASSES) or any(not isinstance(rows, list) for rows in memories.values()):
        raise ValueError("Snapshot export memories must contain lists")
    calculated_counts = {bank: len(rows) for bank, rows in memories.items()}
    if counts != calculated_counts:
        raise ValueError("Snapshot export counts do not match memories")
    try:
        bundle = MemoryBundle(**{
            bank: [MEMORY_CLASSES[bank].from_dict(row) for row in rows]
            for bank, rows in memories.items()
        })
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("Snapshot export contains invalid memory records") from error
    fingerprint = memory_snapshot_fingerprint(bundle)
    if payload.get("fingerprint") != fingerprint:
        raise ValueError("Snapshot export fingerprint does not match memories")
    return {"schema_version": 1, "fingerprint": fingerprint, "counts": calculated_counts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?")
    parser.add_argument("user_id", nargs="?")
    parser.add_argument("destination", nargs="?")
    parser.add_argument("--verify", metavar="EXPORT", help="validate an existing export without reading a store")
    args = parser.parse_args()
    try:
        if args.verify:
            if any(value is not None for value in (args.root, args.user_id, args.destination)):
                parser.error("--verify does not accept root, user_id, or destination")
            summary = verify_export(args.verify)
        else:
            if not all(value is not None for value in (args.root, args.user_id, args.destination)):
                parser.error("root, user_id, and destination are required unless --verify is used")
            summary = export_snapshot(args.root, args.user_id, args.destination)
    except (MemoryStoreError, ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
