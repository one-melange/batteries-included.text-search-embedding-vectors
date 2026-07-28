"""
qdrant_store_tests.py
─────────────────────
Tests for point construction, deterministic ids, content hashing, existing-hash
scanning, and dimension-mismatch collection recreation in qdrant_store.
"""

import unittest
from types import SimpleNamespace

from packages.arc.src.vector_search.models import Chunk, PayloadFields
from packages.arc.src.vector_search.qdrant_store import (
    chunks_to_points,
    content_hash,
    document_key,
    load_existing_hashes,
    point_id,
    setup_collection,
    upsert_points,
)


def _chunk(name: str = "Aave", chunk_index: int = 0, text: str = "some text") -> Chunk:
    return Chunk(
        document_id=name,
        text=text,
        chunk_index=chunk_index,
        total_chunks=1,
        source="f.json",
        content_hash=content_hash(text),
    )


class FakeQdrant:
    """Minimal in-memory stand-in for the QdrantClient calls qdrant_store makes."""

    def __init__(self, sizes: dict | None = None, scroll_points: list | None = None):
        self.sizes = dict(sizes or {})  # collection name -> vector size
        self.created: list[tuple[str, int]] = []
        self.deleted: list[str] = []
        self._scroll_points = scroll_points or []

    def get_collection(self, name):
        if name not in self.sizes:
            raise ValueError(f"collection {name} not found")
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(vectors=SimpleNamespace(size=self.sizes[name]))
            )
        )

    def create_collection(self, collection_name, vectors_config):
        self.sizes[collection_name] = vectors_config.size
        self.created.append((collection_name, vectors_config.size))

    def delete_collection(self, collection_name):
        self.sizes.pop(collection_name, None)
        self.deleted.append(collection_name)

    def scroll(self, collection_name, **kwargs):
        # Single page; second call would signal completion via offset=None.
        return self._scroll_points, None


class ChunksToPoints(unittest.TestCase):
    def test_zips_chunks_with_embeddings_into_payloads(self):
        chunks = [_chunk("Aave"), _chunk("Compound")]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]

        points = chunks_to_points(chunks, embeddings)

        self.assertEqual(len(points), 2)
        self.assertEqual(points[0].vector, [0.1, 0.2])
        self.assertEqual(points[0].payload["document_id"], "Aave")
        self.assertEqual(points[1].payload["document_id"], "Compound")
        # Source file is carried into the payload for provenance.
        self.assertEqual(points[0].payload["source"], "f.json")
        # The content hash is stored so unchanged assets can be skipped later.
        self.assertEqual(points[0].payload["content_hash"], content_hash("some text"))

    def test_ids_are_deterministic_per_chunk(self):
        # The same (source, document_id, chunk_index) must map to the same id
        # across runs so re-upserting replaces in place instead of duplicating.
        p1 = chunks_to_points([_chunk("Aave", 0)], [[0.1]])
        p2 = chunks_to_points([_chunk("Aave", 0)], [[0.9]])
        self.assertEqual(p1[0].id, p2[0].id)
        # A different chunk index yields a different id.
        self.assertNotEqual(
            point_id("f.json", "Aave", 0), point_id("f.json", "Aave", 1)
        )

    def test_raises_on_length_mismatch(self):
        # zip(strict=True) must raise rather than silently truncate, which would
        # misalign chunks with the wrong embeddings.
        with self.assertRaises(ValueError):
            chunks_to_points([_chunk(), _chunk()], [[0.1, 0.2]])
        with self.assertRaises(ValueError):
            chunks_to_points([_chunk()], [[0.1, 0.2], [0.3, 0.4]])

    def test_supports_adapter_specific_identity_field_names(self):
        fields = PayloadFields(document_id="name", source="origin")

        points = chunks_to_points([_chunk()], [[0.1]], fields)

        self.assertEqual(points[0].payload["name"], "Aave")
        self.assertEqual(points[0].payload["origin"], "f.json")
        self.assertNotIn("document_id", points[0].payload)


class SetupCollection(unittest.TestCase):
    def test_creates_when_missing(self):
        q = FakeQdrant()
        setup_collection(q, "descriptions_clean", 1024)
        self.assertEqual(q.created, [("descriptions_clean", 1024)])
        self.assertEqual(q.deleted, [])

    def test_reuses_when_dim_matches(self):
        q = FakeQdrant(sizes={"descriptions_clean": 1024})
        setup_collection(q, "descriptions_clean", 1024)
        self.assertEqual(q.created, [])  # untouched
        self.assertEqual(q.deleted, [])

    def test_recreates_on_dim_mismatch(self):
        # A model/backend switch changes vector_dim; the stale collection must be
        # dropped and recreated rather than left to fail every upsert.
        q = FakeQdrant(sizes={"descriptions_clean": 1024})
        setup_collection(q, "descriptions_clean", 384)
        self.assertEqual(q.deleted, ["descriptions_clean"])
        self.assertEqual(q.created, [("descriptions_clean", 384)])


class LoadExistingHashes(unittest.TestCase):
    def test_builds_document_key_to_hash_map(self):
        points = [
            SimpleNamespace(
                payload={
                    "document_id": "Aave",
                    "source": "a.json",
                    "content_hash": "h1",
                }
            ),
            SimpleNamespace(
                payload={
                    "document_id": "Compound",
                    "source": "b.json",
                    "content_hash": "h2",
                }
            ),
            # A pre-hashing point (no content_hash) is ignored, not crashed on.
            SimpleNamespace(payload={"document_id": "Old", "source": "c.json"}),
        ]
        q = FakeQdrant(scroll_points=points)

        hashes = load_existing_hashes(q, "descriptions_clean")

        self.assertEqual(hashes[document_key("a.json", "Aave")], "h1")
        self.assertEqual(hashes[document_key("b.json", "Compound")], "h2")
        self.assertNotIn(document_key("c.json", "Old"), hashes)

    def test_missing_collection_yields_empty(self):
        # scroll raising (unknown collection) is treated as "nothing indexed yet".
        class Boom(FakeQdrant):
            def scroll(self, *a, **k):
                raise ValueError("no such collection")

        self.assertEqual(load_existing_hashes(Boom(), "descriptions_clean"), {})


class UpsertPoints(unittest.TestCase):
    def test_splits_qdrant_requests_at_its_own_batch_size(self):
        class UpsertRecorder:
            def __init__(self):
                self.batch_sizes = []

            def upsert(self, collection_name, points):
                self.batch_sizes.append(len(points))

        qdrant = UpsertRecorder()
        points = chunks_to_points(
            [_chunk(f"Asset {i}") for i in range(5)],
            [[float(i)] for i in range(5)],
        )

        upsert_points(qdrant, "descriptions_clean", points, batch_size=2)

        self.assertEqual(qdrant.batch_sizes, [2, 2, 1])


if __name__ == "__main__":
    unittest.main()
