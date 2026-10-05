from __future__ import annotations

from typing import Any, Protocol

from .models import (
    ClarificationRequest,
    MemoryBundle,
    PersonaDecision,
    PersonaRequest,
    SearchRequest,
)
from .retrieval import MemoryRetriever


class SearchModel(Protocol):
    def __call__(self, query: str) -> list[dict[str, str]]:
        ...


class DecisionModel(Protocol):
    def __call__(
        self,
        task: dict[str, Any],
        memory_list: dict[str, Any],
    ) -> dict[str, Any]:
        ...


class PersonaAgent:
    def __init__(
        self,
        retriever: MemoryRetriever,
        search_model: SearchModel,
        decision_model: DecisionModel,
        *,
        top_k: int = 3,
        min_similarity: float = 0.0,
    ):
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if not isinstance(min_similarity, (int, float)) or isinstance(min_similarity, bool):
            raise ValueError("min_similarity must be a finite number between 0 and 1")
        if not 0.0 <= float(min_similarity) <= 1.0:
            raise ValueError("min_similarity must be a finite number between 0 and 1")
        self.retriever = retriever
        self.search_model = search_model
        self.decision_model = decision_model
        self.top_k = top_k
        self.min_similarity = float(min_similarity)

    def decide(self, request: PersonaRequest) -> PersonaDecision:
        searches = [SearchRequest(**item) for item in self.search_model(request.query)]
        memories = self.retriever.retrieve(
            request.user_id,
            searches,
            top_k=self.top_k,
            min_similarity=self.min_similarity,
        )
        output = self.decision_model(
            {"query": request.query, "dag": request.dag},
            memories.to_dict(),
        )
        action = str(output.get("action", "final"))
        if action == "clarify":
            question = str(output.get("question", "")).strip()
            if not question:
                raise ValueError("clarification requires a question")
            return PersonaDecision(
                action="clarify",
                memories=memories,
                clarification=ClarificationRequest(question=question),
            )
        return PersonaDecision(
            action="final",
            memories=memories,
            answer=(str(output["answer"]) if output.get("answer") is not None else None),
        )
