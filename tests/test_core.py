import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from autopersona_memory import (
    ClarificationEvaluationCase,
    ExecutionState,
    JsonlMemoryStore,
    MemoryAwareExecutor,
    MemoryRetriever,
    MemoryStoreCorruptionError,
    MemoryStoreError,
    MemoryUpdater,
    MvpMetrics,
    PersonaAgent,
    PersonaMemory,
    PersonaRequest,
    RawTaskRecord,
    SearchRequest,
    TaskNode,
    TrajectoryMemory,
    WorkspaceMemory,
    evaluate_clarification_policy,
    ingest_task_history,
)
from autopersona_memory.adapters import (
    history_to_raw_tasks,
    load_persona2web,
    query_to_persona_request,
)


def embed(text):
    lowered = text.lower()
    return [float(word in lowered) for word in ("paper", "python", "dark")]


class StoreAndMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = JsonlMemoryStore(Path(self.temp.name) / "memory")

    def tearDown(self):
        self.temp.cleanup()

    def test_jsonl_store_crud_and_user_path_sanitization(self):
        first = PersonaMemory("paper", "Open source", "Check code")
        second = PersonaMemory("paper", "Recent", "Check date")
        user_id = "../../user name"
        self.store.add(user_id, "persona", first)
        self.assertEqual(self.store.list(user_id, "persona"), [first])
        self.store.update(user_id, "persona", 0, second)
        self.assertEqual(self.store.list(user_id, "persona"), [second])
        self.store.delete(user_id, "persona", 0)
        self.assertEqual(self.store.list(user_id, "persona"), [])
        self.assertFalse((Path(self.temp.name) / "user name.jsonl").exists())

    def test_corrupt_jsonl_reports_file_and_line_without_partial_results(self):
        path = self.store._path("alice", "persona")
        path.write_text(
            '{"preference": "valid", "strategy": "keep", "topic": "paper"}\n'
            '{"preference": "truncated"\n',
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            MemoryStoreCorruptionError,
            rf"Invalid persona memory record at {path}:2",
        ):
            self.store.list("alice", "persona")

    def test_failed_atomic_replace_preserves_existing_memories(self):
        original = PersonaMemory("paper", "Open source", "Check code")
        replacement = PersonaMemory("paper", "Recent", "Check date")
        self.store.add("alice", "persona", original)
        path = self.store._path("alice", "persona")
        before = path.read_bytes()

        with patch("autopersona_memory.store.os.replace", side_effect=OSError("disk error")):
            with self.assertRaisesRegex(MemoryStoreError, "Unable to atomically write"):
                self.store.replace("alice", "persona", [replacement])

        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(self.store.list("alice", "persona"), [original])
        self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_concurrent_adds_do_not_lose_memory_records(self):
        def add_record(index: int) -> None:
            self.store.add(
                "alice",
                "persona",
                PersonaMemory("paper", f"preference-{index}", f"strategy-{index}"),
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(add_record, range(24)))

        memories = self.store.list("alice", "persona")
        self.assertEqual(len(memories), 24)
        self.assertEqual({memory.preference for memory in memories}, {f"preference-{i}" for i in range(24)})

    def test_ingestion_builds_all_three_memory_layers(self):
        updater = MemoryUpdater(
            self.store,
            embed,
            lambda memory_type, new, candidates: {"operation": "add", "memory": new},
        )
        record = RawTaskRecord(
            user_id="alice",
            query="Find a paper",
            trace=["search"],
            outcome="done",
            task={"status": "complete"},
            persona=[{"topic": "paper", "preference": "Open source", "strategy": "Check code"}],
        )
        result = ingest_task_history(self.store, updater, record)
        self.assertEqual(result.trajectory.query, "Find a paper")
        self.assertEqual(len(self.store.list("alice", "trajectory")), 1)
        self.assertEqual(len(self.store.list("alice", "workspace")), 1)
        self.assertEqual(len(self.store.list("alice", "persona")), 1)

    def test_updater_applies_candidate_relative_update(self):
        self.store.add("alice", "persona", PersonaMemory("python", "tabs", "Use tabs"))
        self.store.add("alice", "persona", PersonaMemory("paper", "old", "old strategy"))
        updated = {"topic": "paper", "preference": "open source", "strategy": "check code"}
        updater = MemoryUpdater(
            self.store,
            embed,
            lambda memory_type, new, candidates: {"operation": "update", "index": 0, "memory": updated},
            top_k=1,
        )
        updater.update("alice", "persona", PersonaMemory.from_dict(updated))
        memories = self.store.list("alice", "persona")
        self.assertEqual(memories[0].preference, "tabs")
        self.assertEqual(memories[1].preference, "open source")

    def test_retrieval_is_separated_by_memory_type(self):
        self.store.add("alice", "trajectory", TrajectoryMemory(query="Python task"))
        self.store.add("alice", "workspace", WorkspaceMemory(task="dark theme"))
        self.store.add("alice", "persona", PersonaMemory("paper", "Open source", "Check code"))
        bundle = MemoryRetriever(self.store, embed).retrieve(
            "alice",
            [SearchRequest("trajectory", "python"), SearchRequest("persona", "paper")],
        )
        self.assertEqual(len(bundle.trajectory), 1)
        self.assertEqual(len(bundle.persona), 1)
        self.assertEqual(bundle.workspace, [])


class AgentAndExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        store = JsonlMemoryStore(Path(self.temp.name) / "memory")
        store.add("alice", "persona", PersonaMemory("paper", "Open source", "Check code"))
        retriever = MemoryRetriever(store, embed)
        self.search = lambda query: [{"memory": "persona", "query": "paper"}]
        self.retriever = retriever

    def tearDown(self):
        self.temp.cleanup()

    def test_persona_agent_returns_final_with_retrieved_evidence(self):
        agent = PersonaAgent(
            self.retriever,
            self.search,
            lambda task, memories: {"action": "final", "answer": memories["persona"][0]["preference"]},
        )
        decision = agent.decide(PersonaRequest("alice", "Recommend a paper"))
        self.assertEqual(decision.action, "final")
        self.assertEqual(decision.answer, "Open source")
        self.assertEqual(len(decision.memories.persona), 1)

    def test_persona_agent_requires_nonempty_clarification(self):
        agent = PersonaAgent(
            self.retriever,
            self.search,
            lambda task, memories: {"action": "clarify", "question": ""},
        )
        with self.assertRaisesRegex(ValueError, "requires a question"):
            agent.decide(PersonaRequest("alice", "Recommend a paper"))

    def test_executor_orders_dependencies_and_passes_predecessors(self):
        agent = PersonaAgent(
            self.retriever,
            self.search,
            lambda task, memories: {"action": "final", "answer": "continue"},
        )
        seen = []
        result = MemoryAwareExecutor(agent).execute(
            "alice",
            "Prepare research notes",
            [
                TaskNode("write", "Write notes", ["search"]),
                TaskNode("search", "Search papers"),
            ],
            lambda instruction, memories, predecessors, tools: seen.append((instruction, predecessors)) or instruction,
        )
        self.assertEqual(result.state.status, "completed")
        self.assertEqual(seen[0], ("Search papers", {}))
        self.assertEqual(seen[1][1], {"search": "Search papers"})
        self.assertEqual(result.output, "Write notes")

    def test_executor_pauses_before_running_node(self):
        agent = PersonaAgent(
            self.retriever,
            self.search,
            lambda task, memories: {"action": "clarify", "question": "Which budget?"},
        )
        calls = []
        result = MemoryAwareExecutor(agent).execute(
            "alice", "Book travel", [TaskNode("book", "Book hotel")],
            lambda *args: calls.append(args),
        )
        self.assertEqual(result.state.status, "paused")
        self.assertEqual(result.state.node, "book")
        self.assertEqual(result.state.clarification.question, "Which budget?")
        self.assertEqual(calls, [])

    def test_executor_rejects_cycles_and_missing_dependencies(self):
        agent = PersonaAgent(self.retriever, self.search, lambda task, memories: {"action": "final"})
        executor = MemoryAwareExecutor(agent)
        with self.assertRaisesRegex(ValueError, "cycle or missing dependency"):
            executor.execute(
                "alice", "task", [TaskNode("a", "A", ["b"]), TaskNode("b", "B", ["a"])],
                lambda *args: None,
            )
        with self.assertRaisesRegex(ValueError, "cycle or missing dependency"):
            executor.execute("alice", "task", [TaskNode("a", "A", ["unknown"])], lambda *args: None)

    def test_executor_resumes_without_repeating_completed_nodes(self):
        def decide(task, memories):
            query = task["query"]
            if "Current node: Write notes" in query and "Clarification answer:" not in query:
                return {"action": "clarify", "question": "Which citation style?"}
            return {"action": "final", "answer": "continue"}

        agent = PersonaAgent(self.retriever, self.search, decide)
        executor = MemoryAwareExecutor(agent)
        nodes = [
            TaskNode("search", "Search papers"),
            TaskNode("write", "Write notes", ["search"]),
        ]
        calls = []
        first = executor.execute(
            "alice",
            "Prepare research notes",
            nodes,
            lambda instruction, memories, predecessors, tools: calls.append(instruction)
            or instruction,
        )

        self.assertEqual(first.state.status, "paused")
        self.assertEqual(first.state.outputs, {"search": "Search papers"})
        resumed = executor.resume(
            "alice",
            "Prepare research notes",
            nodes,
            lambda instruction, memories, predecessors, tools: calls.append(instruction)
            or {"instruction": instruction, "predecessors": predecessors},
            first.state,
            "Use APA style",
        )

        self.assertEqual(resumed.state.status, "completed")
        self.assertEqual(calls, ["Search papers", "Write notes"])
        self.assertEqual(resumed.output["predecessors"], {"search": "Search papers"})

    def test_executor_rejects_invalid_resume_state(self):
        agent = PersonaAgent(self.retriever, self.search, lambda task, memories: {"action": "final"})
        executor = MemoryAwareExecutor(agent)
        nodes = [TaskNode("search", "Search papers")]
        with self.assertRaisesRegex(ValueError, "paused execution state"):
            executor.resume(
                "alice",
                "query",
                nodes,
                lambda *args: None,
                ExecutionState(status="completed"),
                "answer",
            )
class AdapterAndMetricsTests(unittest.TestCase):
    def test_persona2web_adapter_preserves_history_and_request(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.json"
            path.write_text(json.dumps({
                "user_id": "u1",
                "history": [{"query": "Old", "events": ["click"], "result": "done"}],
                "query": "New task",
                "task": {"nodes": ["search"]},
            }))
            case = load_persona2web(path)
        records = history_to_raw_tasks(case)
        request = query_to_persona_request(case)
        self.assertEqual(records[0].trace, ["click"])
        self.assertEqual(records[0].outcome, "done")
        self.assertEqual(request.user_id, "u1")
        self.assertEqual(request.dag, {"nodes": ["search"]})

    def test_metrics_report_accuracy_and_clarification_counts(self):
        metrics = MvpMetrics()
        metrics.record(True, "necessary", 12.0)
        metrics.record(False, "unnecessary", 8.0)
        self.assertEqual(metrics.snapshot(), {
            "accuracy": 0.5,
            "necessary_clarifications": 1.0,
            "unnecessary_clarifications": 1.0,
            "elapsed_ms": 20.0,
        })


class ClarificationEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        store = JsonlMemoryStore(Path(self.temp.name) / "memory")
        store.add("known", "persona", PersonaMemory("paper", "Open source", "Check code"))
        retriever = MemoryRetriever(store, embed)
        self.agent = PersonaAgent(
            retriever,
            lambda query: [{"memory": "persona", "query": query}],
            lambda task, memories: (
                {"action": "final", "answer": "use preference"}
                if memories["persona"]
                else {"action": "clarify", "question": "Which constraints matter?"}
            ),
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_evaluator_reports_decisions_and_retrieved_evidence(self):
        report = evaluate_clarification_policy(
            self.agent,
            [
                ClarificationEvaluationCase(
                    "known preference", PersonaRequest("known", "paper"), "final"
                ),
                ClarificationEvaluationCase(
                    "missing preference", PersonaRequest("new", "paper"), "clarify"
                ),
            ],
        )

        self.assertEqual(report.accuracy, 1.0)
        self.assertEqual(report.necessary_clarifications, 1)
        self.assertEqual(report.unnecessary_clarifications, 0)
        self.assertEqual(report.missed_clarifications, 0)
        self.assertEqual(report.results[0].retrieved_counts["persona"], 1)
        self.assertEqual(report.results[1].retrieved_counts["persona"], 0)

    def test_evaluator_rejects_an_empty_case_set(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            evaluate_clarification_policy(self.agent, [])


if __name__ == "__main__":
    unittest.main()
