"""Run a deterministic, API-free AutoPersona memory and clarification demo."""

from pathlib import Path
from tempfile import TemporaryDirectory

from autopersona_memory import (
    JsonlMemoryStore,
    MemoryAwareExecutor,
    MemoryRetriever,
    MemoryUpdater,
    PersonaAgent,
    RawTaskRecord,
    TaskNode,
    ingest_task_history,
)


def embed(text: str) -> list[float]:
    lowered = text.lower()
    return [float(term in lowered) for term in ("paper", "open-source", "budget")] + [1.0]


def add_new_memory(memory_type, new_memory, candidates):
    return {"operation": "add", "memory": new_memory}


def main() -> None:
    with TemporaryDirectory() as directory:
        store = JsonlMemoryStore(Path(directory) / "memory")
        updater = MemoryUpdater(store, embed, add_new_memory)
        ingest_task_history(
            store,
            updater,
            RawTaskRecord(
                user_id="demo-user",
                query="Find papers with available implementations",
                outcome="completed",
                persona=[{
                    "topic": "paper search",
                    "preference": "Prefer open-source implementations",
                    "strategy": "Check code availability and reproduction cost",
                }],
            ),
        )

        retriever = MemoryRetriever(store, embed)
        search_model = lambda query: [{"memory": "persona", "query": query}]

        def decision_model(task, memories):
            if memories["persona"]:
                return {"action": "final", "answer": "Apply the recorded research preference."}
            return {"action": "clarify", "question": "Should the result include open-source code?"}

        agent = PersonaAgent(retriever, search_model, decision_model)
        executor = MemoryAwareExecutor(agent)
        result = executor.execute(
            user_id="demo-user",
            query="Recommend a paper",
            nodes=[TaskNode(name="search", instruction="Search for related papers")],
            runner=lambda instruction, memories, predecessors, tools: {
                "instruction": instruction,
                "persona": [item.to_dict() for item in memories.persona],
            },
        )
        print(result.state.status)
        print(result.output)


if __name__ == "__main__":
    main()
