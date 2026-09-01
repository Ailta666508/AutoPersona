from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from .models import Memory, MemoryType, PersonaMemory, TrajectoryMemory, WorkspaceMemory


MEMORY_CLASSES = {
    "trajectory": TrajectoryMemory,
    "workspace": WorkspaceMemory,
    "persona": PersonaMemory,
}


class JsonlMemoryStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        for memory_type in MEMORY_CLASSES:
            (self.root / memory_type).mkdir(parents=True, exist_ok=True)

    def list(self, user_id: str, memory_type: MemoryType) -> list[Memory]:
        path = self._path(user_id, memory_type)
        if not path.exists():
            return []
        memory_class = MEMORY_CLASSES[memory_type]
        return [
            memory_class.from_dict(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def add(self, user_id: str, memory_type: MemoryType, memory: Memory) -> None:
        path = self._path(user_id, memory_type)
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(memory.to_dict(), ensure_ascii=False, sort_keys=True))
            stream.write("\n")

    def replace(
        self,
        user_id: str,
        memory_type: MemoryType,
        memories: Iterable[Memory],
    ) -> None:
        path = self._path(user_id, memory_type)
        rows = [json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) for item in memories]
        path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

    def update(
        self,
        user_id: str,
        memory_type: MemoryType,
        index: int,
        memory: Memory,
    ) -> None:
        memories = self.list(user_id, memory_type)
        memories[index] = memory
        self.replace(user_id, memory_type, memories)

    def delete(self, user_id: str, memory_type: MemoryType, index: int) -> None:
        memories = self.list(user_id, memory_type)
        del memories[index]
        self.replace(user_id, memory_type, memories)

    def _path(self, user_id: str, memory_type: MemoryType) -> Path:
        safe_user = re.sub(r"[^A-Za-z0-9_.-]+", "_", user_id)
        return self.root / memory_type / f"{safe_user}.jsonl"
