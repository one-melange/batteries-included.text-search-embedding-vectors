"""Tests for generic chunking and incremental document indexing."""

import unittest
from unittest import mock

from packages.vector_search.src import embed_pipeline as p
from packages.vector_search.src.config import MAX_TOKENS_IN_CHUNK
from packages.vector_search.src.embedder import LazyEmbedder
from packages.vector_search.src.models import Document
from packages.vector_search.src.qdrant_store import content_hash, document_key


def _document(
    document_id: str,
    text: str,
    source: str = "source.json",
) -> Document:
    return Document(document_id=document_id, text=text, source=source)


class FakeEmbedder:
    vector_dim = 3

    def __init__(self):
        self.batches = []

    def embed(self, texts):
        self.batches.append(list(texts))
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeQdrant:
    def __init__(self):
        self.upserted = []
        self.upsert_batches = []
        self.deletes = []

    def upsert(self, collection_name, points):
        self.upsert_batches.append(len(points))
        self.upserted.extend(points)

    def delete(self, collection_name, points_selector):
        self.deletes.append(points_selector)


class ChunkText(unittest.TestCase):
    def test_raises_when_overlap_not_less_than_max_tokens(self):
        enc = p._load_tokenizer()
        long_text = "token " * 1000
        with self.assertRaises(ValueError):
            p.chunk_text(long_text, enc, max_tokens=10, overlap=10)
        with self.assertRaises(ValueError):
            p.chunk_text(long_text, enc, max_tokens=10, overlap=20)


class ChunkDocument(unittest.TestCase):
    def test_short_document_is_one_chunk(self):
        enc = p._load_tokenizer()
        chunks = p.chunk_document(_document("doc-1", "a short blurb"), enc)

        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.document_id, "doc-1")
        self.assertEqual(chunk.text, "a short blurb")
        self.assertEqual(chunk.chunk_index, 0)
        self.assertEqual(chunk.total_chunks, 1)
        self.assertEqual(chunk.source, "source.json")
        self.assertEqual(chunk.content_hash, content_hash("a short blurb"))

    def test_long_document_splits_within_token_budget(self):
        enc = p._load_tokenizer()
        text = "A decentralized non-custodial liquidity protocol. " * 200
        chunks = p.chunk_document(_document("doc-1", text), enc)

        self.assertGreater(len(chunks), 1)
        expected_hash = content_hash(text)
        for index, chunk in enumerate(chunks):
            self.assertEqual(chunk.chunk_index, index)
            self.assertEqual(chunk.total_chunks, len(chunks))
            self.assertEqual(chunk.content_hash, expected_hash)
            self.assertLessEqual(len(enc.encode(chunk.text)), MAX_TOKENS_IN_CHUNK)

    def test_metadata_is_copied_to_every_chunk(self):
        document = Document(
            document_id="doc-1",
            text="text",
            source="source",
            metadata={"category": "reference"},
        )

        chunks = p.chunk_document(document, p._load_tokenizer())

        self.assertEqual(chunks[0].metadata, {"category": "reference"})

    def test_iter_chunks_streams_documents(self):
        chunks = list(
            p.iter_chunks(
                [
                    _document("doc-1", "first"),
                    _document("doc-2", "second"),
                ]
            )
        )

        self.assertEqual([chunk.document_id for chunk in chunks], ["doc-1", "doc-2"])
        self.assertEqual([chunk.text for chunk in chunks], ["first", "second"])


class StreamEmbedUpsert(unittest.TestCase):
    collection = "documents"

    def test_fresh_run_embeds_every_document(self):
        documents = [_document("one", "first"), _document("two", "second")]
        qdrant, embedder = FakeQdrant(), FakeEmbedder()

        counts = p._stream_embed_upsert(
            qdrant,
            embedder,
            documents,
            self.collection,
            existing_hashes={},
        )

        self.assertEqual(counts["embedded_documents"], 2)
        self.assertEqual(counts["skipped_documents"], 0)
        self.assertEqual(counts["embedded_chunks"], 2)
        self.assertEqual(counts["deleted_documents"], 0)
        self.assertEqual(len(qdrant.upserted), 2)
        self.assertEqual(qdrant.deletes, [])

    def test_pipeline_and_qdrant_batches_are_independent(self):
        documents = [
            _document(f"doc-{index}", f"text {index}") for index in range(5)
        ]
        qdrant, embedder = FakeQdrant(), FakeEmbedder()

        with (
            mock.patch.object(p, "PIPELINE_CHUNK_BATCH_SIZE", 2),
            mock.patch.object(p, "QDRANT_UPSERT_BATCH_SIZE", 4),
        ):
            p._stream_embed_upsert(
                qdrant,
                embedder,
                documents,
                self.collection,
                existing_hashes={},
            )

        self.assertEqual([len(batch) for batch in embedder.batches], [2, 2, 1])
        self.assertEqual(qdrant.upsert_batches, [4, 1])

    def test_unchanged_documents_are_skipped(self):
        documents = [_document("one", "first"), _document("two", "second")]
        existing = {
            document_key(document.source, document.document_id): content_hash(
                document.text
            )
            for document in documents
        }
        qdrant, embedder = FakeQdrant(), FakeEmbedder()

        counts = p._stream_embed_upsert(
            qdrant,
            embedder,
            documents,
            self.collection,
            existing,
        )

        self.assertEqual(counts["skipped_documents"], 2)
        self.assertEqual(counts["embedded_documents"], 0)
        self.assertEqual(qdrant.upserted, [])
        self.assertEqual(qdrant.deletes, [])

    def test_changed_document_is_deleted_then_reembedded(self):
        document = _document("one", "new text")
        existing = {
            document_key(document.source, document.document_id): content_hash(
                "old text"
            )
        }
        qdrant, embedder = FakeQdrant(), FakeEmbedder()

        counts = p._stream_embed_upsert(
            qdrant,
            embedder,
            [document],
            self.collection,
            existing,
        )

        self.assertEqual(counts["embedded_documents"], 1)
        self.assertEqual(counts["deleted_documents"], 0)
        self.assertEqual(len(qdrant.deletes), 1)
        self.assertEqual(len(qdrant.upserted), 1)

    def test_document_absent_from_source_is_pruned(self):
        current = _document("one", "first")
        existing = {
            document_key(current.source, current.document_id): content_hash(
                current.text
            ),
            document_key("gone", "missing"): "deadbeef",
        }
        qdrant, embedder = FakeQdrant(), FakeEmbedder()

        counts = p._stream_embed_upsert(
            qdrant,
            embedder,
            [current],
            self.collection,
            existing,
        )

        self.assertEqual(counts["skipped_documents"], 1)
        self.assertEqual(counts["deleted_documents"], 1)
        self.assertEqual(len(qdrant.deletes), 1)


class RunPipeline(unittest.TestCase):
    def test_unchanged_run_does_not_load_embedding_backend(self):
        document = _document("one", "text")
        existing = {
            document_key(document.source, document.document_id): content_hash(
                document.text
            )
        }
        backend_factory = mock.Mock(return_value=FakeEmbedder())
        lazy_embedder = LazyEmbedder(backend_factory, vector_dim=3)
        monitor = mock.MagicMock()
        monitor.__enter__.return_value = monitor
        monitor.__exit__.return_value = False
        monitor.result.return_value = {
            "mem_peak_mb": 1,
            "mem_avg_mb": 1,
            "cpu_peak_pct": 0,
            "elapsed_seconds": 0,
        }

        with (
            mock.patch.object(p, "ResourceMonitor", return_value=monitor),
            mock.patch.object(p, "QdrantClient", return_value=FakeQdrant()),
            mock.patch.object(p, "_assert_qdrant_running"),
            mock.patch.object(p, "get_lazy_embedder", return_value=lazy_embedder),
            mock.patch.object(p, "setup_collection") as setup_collection,
            mock.patch.object(p, "load_existing_hashes", return_value=existing),
        ):
            _, returned_embedder = p.run_pipeline([document], "documents")

        setup_collection.assert_called_once_with(
            mock.ANY,
            "documents",
            lazy_embedder.vector_dim,
        )
        self.assertIs(returned_embedder, lazy_embedder)
        self.assertFalse(lazy_embedder.is_loaded)
        backend_factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
