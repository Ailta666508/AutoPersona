from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .models import ClarificationRequest, MemoryBundle, PersonaRequest
from .persona_agent import PersonaAgent


@dataclass(frozen=True)
class TaskNode:
    name: str
    instruction: str
    dependencies: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ExecutionState:
    status: str
    node: str | None = None
    clarification: ClarificationRequest | None = None
    outputs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionResult:
    state: ExecutionState
    output: Any = None


Runner = Callable[[str, MemoryBundle, dict[str, Any], list[str]], Any]


class MemoryAwareExecutor:
    def __init__(self, persona_agent: PersonaAgent):
        self.persona_agent = persona_agent

    def execute(
        self,
        user_id: str,
        query: str,
        nodes: list[TaskNode],
        runner: Runner,
        additional_tools: list[str] | None = None,
    ) -> ExecutionResult:
        ordered = _topological_order(nodes)
        outputs: dict[str, Any] = {}
        dag = {
            "nodes": [
                {
                    "name": node.name,
                    "instruction": node.instruction,
                    "dependencies": node.dependencies,
                }
                for node in ordered
            ]
        }
        tools = list(additional_tools or [])
        for node in ordered:
            predecessors = {name: outputs[name] for name in node.dependencies}
            request = PersonaRequest(
                user_id=user_id,
                query=f"{query}\n\nCurrent node: {node.instruction}",
                dag=dag,
                additional_tools=tools,
            )
            decision = self.persona_agent.decide(request)
            if decision.action == "clarify":
                return ExecutionResult(
                    state=ExecutionState(
                        status="paused",
                        node=node.name,
                        clarification=decision.clarification,
                        outputs=outputs,
                    )
                )
            outputs[node.name] = runner(
                node.instruction,
                decision.memories,
                predecessors,
                tools,
            )
        return ExecutionResult(
            state=ExecutionState(status="completed", outputs=outputs),
            output=outputs[ordered[-1].name] if ordered else None,
        )


def _topological_order(nodes: list[TaskNode]) -> list[TaskNode]:
    by_name = {node.name: node for node in nodes}
    pending = {node.name: set(node.dependencies) for node in nodes}
    ordered: list[TaskNode] = []
    while pending:
        ready = [name for name, dependencies in pending.items() if not dependencies]
        if not ready:
            raise ValueError("task graph contains a cycle or missing dependency")
        for name in ready:
            ordered.append(by_name[name])
            del pending[name]
            for dependencies in pending.values():
                dependencies.discard(name)
    return ordered
