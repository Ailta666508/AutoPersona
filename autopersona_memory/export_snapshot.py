"""Export one user's consistent memory snapshot to an atomic JSON artifact.

Usage: python -m autopersona_memory.export_snapshot ROOT USER_ID OUTPUT
The output contains memory contents. Stdout contains only counts and a digest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .store import JsonlMemoryStore, MemoryStoreError, memory_snapshot_fingerprint


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("user_id")
    parser.add_argument("destination")
    args = parser.parse_args()
    try:
        summary = export_snapshot(args.root, args.user_id, args.destination)
    except (MemoryStoreError, ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
