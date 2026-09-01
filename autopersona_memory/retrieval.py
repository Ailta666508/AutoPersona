from __future__ import annotations

import json
import math
from typing import Callable, Sequence

from .models import Memory, MemoryBundle, MemoryType, SearchRequest
from .store import JsonlMemoryStore


Embedder = Callable[[str], Sequence[float]]


class MemoryRetriever:
    def __init__(self, store: JsonlMemoryStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder

    def retrieve(
        self,
        user_id: str,
        searches: list[SearchRequest],
        top_k: int = 3,
    ) -> MemoryBundle:
        selected: dict[MemoryType, list[Memory]] = {
            "trajectory": [],
            "workspace": [],
            "persona": [],
        }
        for search in searches:
            candidates = self.store.list(user_id, search.memory)
            query_vector = self.embedder(search.query)
            scored = [
                (_cosine(query_vector, self.embedder(_text(item))), item)
                for item in candidates
            ]
            scored.sort(key=lambda row: row[0], reverse=True)
            for score, item in scored[:top_k]:
                if score > 0 and item not in selected[search.memory]:
                    selected[search.memory].append(item)
        return MemoryBundle(
            trajectory=list(selected["trajectory"]),
            workspace=list(selected["workspace"]),
            persona=list(selected["persona"]),
        )


def _text(memory: Memory) -> str:
    return json.dumps(memory.to_dict(), ensure_ascii=False, sort_keys=True)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)
