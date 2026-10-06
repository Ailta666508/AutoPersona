import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from autopersona_memory.store import JsonlMemoryStore, MemoryStoreError
from autopersona_memory.models import PersonaMemory, WorkspaceMemory

class BatchStoreTests(unittest.TestCase):
    def test_batch_preserves_order_and_user_isolation(self):
        with tempfile.TemporaryDirectory() as d:
            store = JsonlMemoryStore(d)
            first = PersonaMemory('a', 'b', 'c')
            store.add('alice', 'persona', first)
            with patch.object(JsonlMemoryStore, '_atomic_write', wraps=store._atomic_write) as write:
                store.add_many('alice', 'persona', (PersonaMemory(str(i), 'x', 'y') for i in range(3)))
                self.assertEqual(write.call_count, 1)
            self.assertEqual(len(store.list('alice', 'persona')), 4)
            self.assertEqual(store.list('alice', 'persona')[0].to_dict(), first.to_dict())
            self.assertEqual(store.list('bob', 'persona'), [])

    def test_iteration_or_replace_failure_preserves_existing_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            store = JsonlMemoryStore(d)
            record = PersonaMemory('a', 'b', 'c')
            store.add('alice', 'persona', record)
            path = Path(d) / 'persona/alice.jsonl'
            before = path.read_bytes()
            def broken():
                yield record
                raise RuntimeError('synthetic iterator failure')
            with self.assertRaises(RuntimeError):
                store.add_many('alice', 'persona', broken())
            self.assertEqual(path.read_bytes(), before)
            with patch('autopersona_memory.store.os.replace', side_effect=OSError('synthetic')):
                with self.assertRaises(MemoryStoreError):
                    store.add_many('alice', 'persona', [record, record])
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(path.parent.glob('*.tmp')), [])

    def test_wrong_bank_and_empty_batch_do_not_write(self):
        with tempfile.TemporaryDirectory() as d:
            store = JsonlMemoryStore(d)
            with self.assertRaises(TypeError):
                store.add_many('alice', 'workspace', [PersonaMemory('a', 'b', 'c')])
            store.add_many('alice', 'persona', [])
            self.assertEqual(list(Path(d).rglob('*.jsonl')), [])
