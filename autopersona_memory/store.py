from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Iterable, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows falls back to the process lock.
    fcntl = None

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
            with self._file_lock(path):
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
            with self._file_lock(path):
                self._replace_unlocked(path, memories)

    def update(
        self,
        user_id: str,
        memory_type: MemoryType,
        index: int,
        memory: Memory,
    ) -> None:
        with self._lock:
            path = self._path(user_id, memory_type)
            with self._file_lock(path):
                memories = self.list(user_id, memory_type)
                memories[index] = memory
                self._replace_unlocked(path, memories)

    def delete(self, user_id: str, memory_type: MemoryType, index: int) -> None:
        with self._lock:
            path = self._path(user_id, memory_type)
            with self._file_lock(path):
                memories = self.list(user_id, memory_type)
                del memories[index]
                self._replace_unlocked(path, memories)

    def _path(self, user_id: str, memory_type: MemoryType) -> Path:
        if memory_type not in MEMORY_CLASSES:
            raise ValueError(f"Unsupported memory type: {memory_type}")
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        safe_user = re.sub(r"[^A-Za-z0-9_.-]+", "_", user_id)
        return self.root / memory_type / f"{safe_user}.jsonl"

    @staticmethod
    @contextmanager
    def _file_lock(path: Path) -> Iterator[IO[str]]:
        """Coordinate writers that use separate store instances on Unix hosts."""
        lock_path = path.with_name(f".{path.name}.lock")
        try:
            stream = lock_path.open("a", encoding="utf-8")
        except OSError as error:
            raise MemoryStoreError(f"Unable to open memory lock file: {lock_path}") from error
        try:
            if fcntl is not None:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            yield stream
        except OSError as error:
            raise MemoryStoreError(f"Unable to lock memory file: {path}") from error
        finally:
            if fcntl is not None:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            stream.close()

    @staticmethod
    def _replace_unlocked(path: Path, memories: Iterable[Memory]) -> None:
        rows = [json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) for item in memories]
        JsonlMemoryStore._atomic_write(path, "\n".join(rows) + ("\n" if rows else ""))

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        """Replace ``path`` only after syncing the file and its directory entry."""
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
            temporary_path = None
            JsonlMemoryStore._fsync_directory(path.parent)
        except OSError as error:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise MemoryStoreError(f"Unable to atomically write memory file: {path}") from error

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        """Persist the rename metadata on platforms that support directory fsync."""
        if os.name == "nt":  # pragma: no cover - directory descriptors are Unix-specific.
            return
        descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
