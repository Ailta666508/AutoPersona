import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autopersona_memory import JsonlMemoryStore, PersonaMemory, TrajectoryMemory, WorkspaceMemory, MemoryStoreError
from autopersona_memory.export_snapshot import export_snapshot, restore_snapshot, verify_export


class ExportSnapshotTests(unittest.TestCase):
    def test_cli_exports_only_selected_user_with_verifiable_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "store"
            store = JsonlMemoryStore(root)
            store.add("alice", "persona", PersonaMemory("style", "简洁", "short"))
            store.add("bob", "workspace", WorkspaceMemory(task="not alice"))
            target = Path(directory) / "exports" / "snapshot.json"
            result = subprocess.run([sys.executable, "-m", "autopersona_memory.export_snapshot", str(root), "alice", str(target)], capture_output=True, text=True, check=True)
            payload = json.loads(target.read_text())
            canonical = json.dumps(payload["memories"], ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
            self.assertEqual(payload["fingerprint"], hashlib.sha256(canonical.encode()).hexdigest())
            self.assertEqual(payload["counts"], {"persona": 1, "workspace": 0, "trajectory": 0})
            self.assertNotIn("memories", json.loads(result.stdout))
            self.assertNotIn("not alice", target.read_text())
            self.assertEqual(payload["schema_version"], 1)

    def test_replace_failure_and_corruption_preserve_existing_export(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JsonlMemoryStore(Path(directory) / "store")
            target = Path(directory) / "snapshot.json"
            target.write_text("previous export")
            with patch("autopersona_memory.store.os.replace", side_effect=OSError("synthetic failure")):
                with self.assertRaises(MemoryStoreError):
                    export_snapshot(store.root, "alice", target)
            self.assertEqual(target.read_text(), "previous export")
            self.assertEqual(list(target.parent.glob("*.tmp")), [])
            store._path("alice", "persona").write_text("{bad json")
            with self.assertRaises(MemoryStoreError):
                export_snapshot(store.root, "alice", target)
            self.assertEqual(target.read_text(), "previous export")

    def test_rejects_store_overwrite_and_missing_root(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JsonlMemoryStore(Path(directory) / "store")
            with self.assertRaisesRegex(ValueError, "outside"):
                export_snapshot(store.root, "alice", store._path("alice", "persona"))
            with self.assertRaisesRegex(ValueError, "existing"):
                export_snapshot(Path(directory) / "absent", "alice", Path(directory) / "out.json")
            self.assertFalse((Path(directory) / "absent").exists())

    def test_hash_and_content_use_one_snapshot_read(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JsonlMemoryStore(Path(directory) / "store")
            bundle = store.snapshot("alice")
            with patch.object(JsonlMemoryStore, "snapshot", return_value=bundle) as capture:
                export_snapshot(store.root, "alice", Path(directory) / "export.json")
                capture.assert_called_once_with("alice")

    def test_verifies_export_without_source_store_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "store"
            store = JsonlMemoryStore(root)
            store.add("alice", "persona", PersonaMemory("style", "brief", "short"))
            target = Path(directory) / "snapshot.json"
            summary = export_snapshot(root, "alice", target)
            self.assertEqual(verify_export(target), summary)
            root.rename(Path(directory) / "source-is-offline")
            result = subprocess.run([sys.executable, "-m", "autopersona_memory.export_snapshot", "--verify", str(target)], capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(result.stdout), summary)
            payload = json.loads(target.read_text())
            payload["memories"]["persona"][0]["preference"] = "modified"
            target.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                verify_export(target)

    def test_rejects_ambiguous_export_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "snapshot.json"
            target.write_text(json.dumps({"schema_version": 1, "fingerprint": "", "counts": {}, "memories": {}, "extra": True}))
            with self.assertRaisesRegex(ValueError, "unsupported fields"):
                verify_export(target)

    def test_rejects_noncanonical_digest_and_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "snapshot.json"
            base = {"schema_version": 1, "fingerprint": "not-a-digest", "counts": {"persona": 0, "workspace": 0, "trajectory": 0}, "memories": {"persona": [], "workspace": [], "trajectory": []}}
            target.write_text(json.dumps(base))
            with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
                verify_export(target)
            base["fingerprint"] = "0" * 64
            base["counts"]["persona"] = True
            target.write_text(json.dumps(base))
            with self.assertRaisesRegex(ValueError, "counts"):
                verify_export(target)

    def test_restore_validates_before_replacing_all_banks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "store"
            store = JsonlMemoryStore(root)
            store.add("alice", "persona", PersonaMemory("style", "old", "keep"))
            export_path = Path(directory) / "snapshot.json"
            source = JsonlMemoryStore(Path(directory) / "source")
            source.add("alice", "persona", PersonaMemory("style", "new", "restore"))
            source.add("alice", "workspace", WorkspaceMemory(task="restored"))
            export_snapshot(source.root, "alice", export_path)
            self.assertEqual(restore_snapshot(root, "alice", export_path)["counts"]["workspace"], 1)
            restored = JsonlMemoryStore(root).snapshot("alice")
            self.assertEqual(restored.persona[0].preference, "new")
            self.assertEqual(restored.workspace[0].task, "restored")
            payload = json.loads(export_path.read_text())
            payload["memories"]["persona"][0]["preference"] = "tampered"
            export_path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                restore_snapshot(root, "alice", export_path)
            self.assertEqual(JsonlMemoryStore(root).snapshot("alice").persona[0].preference, "new")

    def test_restore_rolls_back_banks_when_a_later_replace_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "store"
            store = JsonlMemoryStore(root)
            store.add("alice", "persona", PersonaMemory("style", "old", "keep"))
            export_path = Path(directory) / "snapshot.json"
            source = JsonlMemoryStore(Path(directory) / "source")
            source.add("alice", "persona", PersonaMemory("style", "new", "restore"))
            export_snapshot(source.root, "alice", export_path)
            original_replace = JsonlMemoryStore.replace
            calls = []
            def fail_workspace(self, user_id, memory_type, memories):
                calls.append(memory_type)
                if memory_type == "workspace":
                    raise OSError("synthetic disk failure")
                return original_replace(self, user_id, memory_type, memories)
            with patch.object(JsonlMemoryStore, "replace", fail_workspace):
                with self.assertRaisesRegex(MemoryStoreError, "previous banks were restored"):
                    restore_snapshot(root, "alice", export_path)
            self.assertEqual(JsonlMemoryStore(root).snapshot("alice").persona[0].preference, "old")

    def test_restore_rejects_an_export_inside_the_memory_store(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "store"
            store = JsonlMemoryStore(root)
            export_path = root / "snapshot.json"
            export_snapshot(root, "alice", Path(directory) / "outside.json")
            export_path.write_text(Path(directory, "outside.json").read_text())
            with self.assertRaisesRegex(ValueError, "outside"):
                restore_snapshot(root, "alice", export_path)

    def test_restore_attempts_every_rollback_after_one_rollback_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "store"
            store = JsonlMemoryStore(root)
            store.add("alice", "trajectory", TrajectoryMemory(query="old"))
            store.add("alice", "persona", PersonaMemory("style", "old", "keep"))
            store.add("alice", "workspace", WorkspaceMemory(task="old"))
            source = JsonlMemoryStore(Path(directory) / "source")
            source.add("alice", "trajectory", TrajectoryMemory(query="new"))
            source.add("alice", "persona", PersonaMemory("style", "new", "restore"))
            source.add("alice", "workspace", WorkspaceMemory(task="new"))
            export_path = Path(directory) / "snapshot.json"
            export_snapshot(source.root, "alice", export_path)
            original_replace = JsonlMemoryStore.replace
            calls: list[str] = []

            def fail_forward_and_one_rollback(self, user_id, memory_type, memories):
                calls.append(memory_type)
                if memory_type == "persona" and calls.count("persona") == 1:
                    raise OSError("synthetic forward failure")
                if memory_type == "workspace" and calls.count("workspace") == 2:
                    raise OSError("synthetic rollback failure")
                return original_replace(self, user_id, memory_type, memories)

            with patch.object(JsonlMemoryStore, "replace", fail_forward_and_one_rollback):
                with self.assertRaisesRegex(MemoryStoreError, "incomplete for 1 bank"):
                    restore_snapshot(root, "alice", export_path)
            self.assertEqual(calls[-2:], ["workspace", "trajectory"])
            self.assertEqual(JsonlMemoryStore(root).snapshot("alice").trajectory[0].query, "old")
            self.assertEqual(JsonlMemoryStore(root).snapshot("alice").persona[0].preference, "old")
