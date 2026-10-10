import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autopersona_memory import JsonlMemoryStore, PersonaMemory, WorkspaceMemory, MemoryStoreError
from autopersona_memory.export_snapshot import export_snapshot


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
