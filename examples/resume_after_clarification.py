"""Pause a memory-aware task graph, then resume it with a clarification answer."""

from pathlib import Path
from tempfile import TemporaryDirectory

from autopersona_memory import (
    JsonlMemoryStore,
    MemoryAwareExecutor,
    MemoryRetriever,
    PersonaAgent,
    TaskNode,
)


def embed(text: str) -> list[float]:
    return [1.0]


def main() -> None:
    with TemporaryDirectory() as directory:
        retriever = MemoryRetriever(JsonlMemoryStore(Path(directory) / "memory"), embed)

        def decide(task, memories):
            query = task["query"]
            if "Current node: Draft summary" in query and "Clarification answer:" not in query:
                return {"action": "clarify", "question": "Which citation style should I use?"}
            return {"action": "final", "answer": "continue"}

        executor = MemoryAwareExecutor(
            PersonaAgent(retriever, lambda query: [], decide)
        )
        nodes = [
            TaskNode("collect", "Collect sources"),
            TaskNode("draft", "Draft summary", ["collect"]),
        ]
        runner = lambda instruction, memories, predecessors, tools: {
            "instruction": instruction,
            "predecessors": predecessors,
        }

        paused = executor.execute("demo-user", "Prepare notes", nodes, runner)
        print(paused.state.status, paused.state.clarification.question)
        resumed = executor.resume(
            "demo-user",
            "Prepare notes",
            nodes,
            runner,
            paused.state,
            "Use APA style",
        )
        print(resumed.state.status, resumed.output)


if __name__ == "__main__":
    main()
