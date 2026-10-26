"""Export one user's consistent memory snapshot to an atomic JSON artifact.

Usage: python -m autopersona_memory.export_snapshot ROOT USER_ID OUTPUT
The output contains memory contents. Stdout contains only counts and a digest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

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
    fingerprint = payload.get("fingerprint")
    if not isinstance(fingerprint, str) or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None:
        raise ValueError("Snapshot export fingerprint must be a lowercase SHA-256 digest")
    if set(memories) != set(MEMORY_CLASSES) or any(not isinstance(rows, list) for rows in memories.values()):
        raise ValueError("Snapshot export memories must contain lists")
    calculated_counts = {bank: len(rows) for bank, rows in memories.items()}
    if counts != calculated_counts or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in counts.values()
    ):
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


def restore_snapshot(root: str | Path, user_id: str, source: str | Path) -> dict:
    """Restore a verified export into one user scope.

    The complete artifact is validated before any bank is replaced. Each bank
    then uses the store's atomic replacement and locking path; a failed bank
    write cannot truncate its previous file.
    """
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise ValueError("Memory root must be an existing directory")
    source_path = Path(source)
    source_resolved = source_path.resolve()
    if source_resolved == root_path or root_path in source_resolved.parents:
        raise ValueError("Restore source must be outside the memory store")
    summary = verify_export(source_path)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    bundle = MemoryBundle(**{
        bank: [MEMORY_CLASSES[bank].from_dict(row) for row in payload["memories"][bank]]
        for bank in MEMORY_CLASSES
    })
    store = JsonlMemoryStore(root_path)
    previous = {bank: store.list(user_id, bank) for bank in MEMORY_CLASSES}
    replaced: list[str] = []
    try:
        for bank in MEMORY_CLASSES:
            store.replace(user_id, bank, getattr(bundle, bank))
            replaced.append(bank)
    except Exception as error:
        rollback_errors: list[Exception] = []
        for bank in reversed(replaced):
            try:
                store.replace(user_id, bank, previous[bank])
            except Exception as rollback_error:
                rollback_errors.append(rollback_error)
        if rollback_errors:
            raise MemoryStoreError(
                f"Snapshot restore failed and rollback was incomplete for {len(rollback_errors)} bank(s)"
            ) from rollback_errors[0]
        raise MemoryStoreError("Snapshot restore failed; previous banks were restored") from error
    return summary


def preview_restore(root: str | Path, user_id: str, source: str | Path) -> dict:
    """Compare a verified export with the current user snapshot without writing."""
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise ValueError("Memory root must be an existing directory")
    source_path = Path(source)
    source_resolved = source_path.resolve()
    if source_resolved == root_path or root_path in source_resolved.parents:
        raise ValueError("Restore source must be outside the memory store")
    incoming = verify_export(source_path)
    current = JsonlMemoryStore(root_path).snapshot(user_id)
    current_counts = {bank: len(rows) for bank, rows in current.to_dict().items()}
    current_fingerprint = memory_snapshot_fingerprint(current)
    incoming_counts = incoming["counts"]
    changes = {
        bank: {
            "current": current_counts[bank],
            "incoming": incoming_counts[bank],
            "delta": incoming_counts[bank] - current_counts[bank],
        }
        for bank in MEMORY_CLASSES
    }
    return {
        "schema_version": 1,
        "current": {"fingerprint": current_fingerprint, "counts": current_counts},
        "incoming": incoming,
        "changes": changes,
        "already_matches": current_fingerprint == incoming["fingerprint"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?")
    parser.add_argument("user_id", nargs="?")
    parser.add_argument("destination", nargs="?")
    parser.add_argument("--verify", metavar="EXPORT", help="validate an existing export without reading a store")
    parser.add_argument("--restore", metavar="EXPORT", help="restore a verified export into root/user_id")
    parser.add_argument("--restore-preview", metavar="EXPORT", help="compare a verified export with root/user_id without writing")
    args = parser.parse_args()
    try:
        if sum(value is not None for value in (args.verify, args.restore, args.restore_preview)) > 1:
            parser.error("--verify, --restore, and --restore-preview are mutually exclusive")
        if args.verify:
            if any(value is not None for value in (args.root, args.user_id, args.destination)):
                parser.error("--verify does not accept root, user_id, or destination")
            summary = verify_export(args.verify)
        elif args.restore:
            if args.root is None or args.user_id is None or args.destination is not None:
                parser.error("--restore requires root and user_id, and no destination positional argument")
            summary = restore_snapshot(args.root, args.user_id, args.restore)
        elif args.restore_preview:
            if args.root is None or args.user_id is None or args.destination is not None:
                parser.error("--restore-preview requires root and user_id, and no destination positional argument")
            summary = preview_restore(args.root, args.user_id, args.restore_preview)
        else:
            if not all(value is not None for value in (args.root, args.user_id, args.destination)):
                parser.error("root, user_id, and destination are required unless --verify is used")
            summary = export_snapshot(args.root, args.user_id, args.destination)
    except (MemoryStoreError, ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
