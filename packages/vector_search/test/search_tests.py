"""
search_tests.py
───────────────
Tests for result-dict construction in search.search.

Qdrant and the embedder are stubbed so these tests exercise only the mapping
from query-response points to result dicts — no running Qdrant or model needed.
"""

import unittest
from types import SimpleNamespace

from packages.vector_search.src import search as s
from packages.vector_search.src.models import PayloadFields


class _FakeEmbedder:
    def embed(self, texts):
        return [[0.1, 0.2] for _ in texts]


class _FakeQdrant:
    def __init__(self, points):
        self._points = points

    def query_points(self, **kwargs):
        return SimpleNamespace(points=self._points)


class Search(unittest.TestCase):
    def test_maps_payload_fields_into_result_dicts(self):
        points = [
            SimpleNamespace(
                score=0.87654,
                payload={
                    "document_id": "Aave",
                    "chunk_index": 1,
                    "total_chunks": 2,
                    "text": "lending protocol",
                    "source": "record.json",
                },
            )
        ]
        results = s.search("q", "col", _FakeQdrant(points), _FakeEmbedder(), top_k=1)

        self.assertEqual(
            results,
            [
                {
                    "score": 0.8765,  # rounded to 4 places
                    "document_id": "Aave",
                    "chunk_index": 1,
                    "total_chunks": 2,
                    "text": "lending protocol",
                    "source": "record.json",
                }
            ],
        )

    def test_missing_payload_does_not_raise(self):
        # A point with payload=None must yield a dict of Nones, not AttributeError.
        points = [
            SimpleNamespace(score=0.9, payload=None),
            SimpleNamespace(score=0.8, payload={"document_id": "Aave"}),
        ]
        results = s.search("q", "col", _FakeQdrant(points), _FakeEmbedder(), top_k=2)

        self.assertEqual(results[0]["score"], 0.9)
        self.assertIsNone(results[0]["document_id"])
        self.assertIsNone(results[0]["text"])
        self.assertEqual(results[1]["document_id"], "Aave")

    def test_maps_configured_payload_fields_to_generic_results(self):
        points = [
            SimpleNamespace(
                score=0.9,
                payload={"name": "Aave", "origin": "record.json"},
            )
        ]

        results = s.search(
            "q",
            "col",
            _FakeQdrant(points),
            _FakeEmbedder(),
            payload_fields=PayloadFields(document_id="name", source="origin"),
        )

        self.assertEqual(results[0]["document_id"], "Aave")
        self.assertEqual(results[0]["source"], "record.json")


if __name__ == "__main__":
    unittest.main()
