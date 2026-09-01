from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..models import PersonaRequest, RawTaskRecord


@dataclass(frozen=True)
class Persona2WebCase:
    user_id: str
    history: list[dict[str, Any]]
    query: str
    task: dict[str, Any]


def load_persona2web(path: str | Path) -> Persona2WebCase:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Persona2WebCase(
        user_id=str(data.get("user_id", "default")),
        history=list(data.get("history", []) or []),
        query=str(data.get("query", "")),
        task=dict(data.get("task", {}) or {}),
    )


def history_to_raw_tasks(case: Persona2WebCase) -> list[RawTaskRecord]:
    return [
        RawTaskRecord(
            user_id=case.user_id,
            query=str(item.get("query", "")),
            trace=list(item.get("trace", item.get("events", [])) or []),
            outcome=item.get("outcome", item.get("result")),
            insight=str(item.get("insight", "")),
            task=item.get("task"),
            interaction=list(item.get("interaction", item.get("events", [])) or []),
            state=item.get("state"),
            persona=list(item.get("persona", []) or []),
        )
        for item in case.history
    ]


def query_to_persona_request(case: Persona2WebCase) -> PersonaRequest:
    return PersonaRequest(
        user_id=case.user_id,
        query=case.query,
        dag=case.task,
    )
