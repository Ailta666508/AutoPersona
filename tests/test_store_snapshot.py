import tempfile
import unittest
from pathlib import Path

from autopersona_memory import JsonlMemoryStore, MemoryStoreCorruptionError
from autopersona_memory.models import PersonaMemory, TrajectoryMemory, WorkspaceMemory


class StoreSnapshotTests(unittest.TestCase):
    def test_snapshot_returns_all_banks_in_persisted_order(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JsonlMemoryStore(directory)
            trajectory = TrajectoryMemory("query", insight="learned")
            workspace = WorkspaceMemory(task="draft")
            personas = [
                PersonaMemory("style", "brief", "summarize"),
                PersonaMemory("format", "markdown", "use headings"),
            ]
            store.add("alice", "trajectory", trajectory)
            store.add("alice", "workspace", workspace)
            store.add_many("alice", "persona", personas)

            snapshot = store.snapshot("alice")

            self.assertEqual(snapshot.trajectory, [trajectory])
            self.assertEqual(snapshot.workspace, [workspace])
            self.assertEqual(snapshot.persona, personas)
            self.assertEqual(store.snapshot("bob").to_dict(), {
                "trajectory": [], "workspace": [], "persona": [],
            })

    def test_corrupt_bank_fails_snapshot_without_modifying_any_bank(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JsonlMemoryStore(directory)
            memory = PersonaMemory("style", "brief", "summarize")
            store.add("alice", "persona", memory)
            corrupt = store._path("alice", "workspace")
            corrupt.write_bytes(b"\xff")
            before = {path: path.read_bytes() for path in Path(directory).rglob("*.jsonl")}

            with self.assertRaises(MemoryStoreCorruptionError):
                store.snapshot("alice")

            self.assertEqual(
                {path: path.read_bytes() for path in Path(directory).rglob("*.jsonl")},
                before,
            )
