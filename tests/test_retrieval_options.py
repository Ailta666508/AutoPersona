import tempfile
import unittest
from unittest.mock import Mock
from autopersona_memory import JsonlMemoryStore, MemoryRetriever, PersonaMemory, SearchRequest


class RetrievalOptionsTests(unittest.TestCase):
    def test_threshold_filters_weak_matches_and_preserves_defaults(self):
        with tempfile.TemporaryDirectory() as root:
            store = JsonlMemoryStore(root)
            strong = PersonaMemory("strong", "p", "s")
            weak = PersonaMemory("weak", "p", "s")
            store.replace("alice", "persona", [strong, weak])
            retriever = MemoryRetriever(store, lambda text: [1, 0] if "weak" not in text else [1, 1])
            query = [SearchRequest("persona", "query")]
            self.assertEqual(retriever.retrieve("alice", query).persona, [strong, weak])
            self.assertEqual(retriever.retrieve("alice", query, min_similarity=0.9).persona, [strong])

    def test_cache_is_per_call_and_reads_each_bank_once(self):
        with tempfile.TemporaryDirectory() as root:
            store = JsonlMemoryStore(root)
            memory = PersonaMemory("t", "p", "s")
            store.add("alice", "persona", memory)
            embed = Mock(return_value=[1, 0])
            store.list = Mock(wraps=store.list)
            retriever = MemoryRetriever(store, embed)
            searches = [SearchRequest("persona", "query")] * 3
            self.assertEqual(retriever.retrieve("alice", searches).persona, [memory])
            self.assertEqual(embed.call_count, 2)
            self.assertEqual(store.list.call_count, 1)
            store.replace("alice", "persona", [])
            self.assertEqual(retriever.retrieve("alice", searches).persona, [])
            self.assertEqual(retriever.retrieve("bob", searches).persona, [])

    def test_rejects_invalid_options(self):
        with tempfile.TemporaryDirectory() as root:
            retriever = MemoryRetriever(JsonlMemoryStore(root), lambda text: [1])
            for value in (True, 1.5, 0, -1):
                with self.assertRaises(ValueError):
                    retriever.retrieve("alice", [], top_k=value)
            for value in (-0.1, 1.1, float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    retriever.retrieve("alice", [], min_similarity=value)
