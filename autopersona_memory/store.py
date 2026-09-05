from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Iterable

from .models import Memory, MemoryType, PersonaMemory, TrajectoryMemory, WorkspaceMemory


MEMORY_CLASSES = {
    "trajectory": TrajectoryMemory,
    "workspace": WorkspaceMemory,
    "persona": PersonaMemory,
}


class MemoryStoreError(RuntimeError):
    """Raised when persisted memory cannot be read or written safely."""


class MemoryStoreCorruptionError(MemoryStoreError):
    """Raised when a JSONL memory file contains a malformed record."""


class JsonlMemoryStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        # CRUD operations use read-modify-replace sequences. Keep those sequences
        # indivisible so concurrent agent tasks cannot silently overwrite one
        # another inside the same runtime.
        self._lock = threading.RLock()
        for memory_type in MEMORY_CLASSES:
            (self.root / memory_type).mkdir(parents=True, exist_ok=True)

    def list(self, user_id: str, memory_type: MemoryType) -> list[Memory]:
        with self._lock:
            path = self._path(user_id, memory_type)
            if not path.exists():
                return []
            memory_class = MEMORY_CLASSES[memory_type]
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError as error:
                raise MemoryStoreError(f"Unable to read memory file: {path}") from error

            memories = []
            for line_number, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                try:
                    memories.append(memory_class.from_dict(json.loads(line)))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                    raise MemoryStoreCorruptionError(
                        f"Invalid {memory_type} memory record at {path}:{line_number}"
                    ) from error
            return memories

    def add(self, user_id: str, memory_type: MemoryType, memory: Memory) -> None:
        with self._lock:
            path = self._path(user_id, memory_type)
            if path.exists():
                self.list(user_id, memory_type)
            try:
                existing = path.read_text(encoding="utf-8") if path.exists() else ""
            except OSError as error:
                raise MemoryStoreError(f"Unable to read memory file: {path}") from error
            row = json.dumps(memory.to_dict(), ensure_ascii=False, sort_keys=True)
            separator = "" if not existing or existing.endswith("\n") else "\n"
            self._atomic_write(path, f"{existing}{separator}{row}\n")

    def replace(
        self,
        user_id: str,
        memory_type: MemoryType,
        memories: Iterable[Memory],
    ) -> None:
        with self._lock:
            path = self._path(user_id, memory_type)
            rows = [json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) for item in memories]
            self._atomic_write(path, "\n".join(rows) + ("\n" if rows else ""))

    def update(
        self,
        user_id: str,
        memory_type: MemoryType,
        index: int,
        memory: Memory,
    ) -> None:
        with self._lock:
            memories = self.list(user_id, memory_type)
            memories[index] = memory
            self.replace(user_id, memory_type, memories)

    def delete(self, user_id: str, memory_type: MemoryType, index: int) -> None:
        with self._lock:
            memories = self.list(user_id, memory_type)
            del memories[index]
            self.replace(user_id, memory_type, memories)

    def _path(self, user_id: str, memory_type: MemoryType) -> Path:
        safe_user = re.sub(r"[^A-Za-z0-9_.-]+", "_", user_id)
        return self.root / memory_type / f"{safe_user}.jsonl"

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        """Replace ``path`` only after a complete same-directory write and fsync."""
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
        except OSError as error:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise MemoryStoreError(f"Unable to atomically write memory file: {path}") from error
