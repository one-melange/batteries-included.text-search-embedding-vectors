"""Tests that embedding backends use their own independent batch controls."""

import unittest
from types import SimpleNamespace
from unittest import mock

from packages.vector_search.src import embedder as e


class _Array:
    def __init__(self, value):
        self.value = value

    def tolist(self):
        return self.value


class EmbedderBatching(unittest.TestCase):
    def test_openai_uses_request_batch_size(self):
        calls = []

        class Embeddings:
            def create(self, model, input):
                calls.append(list(input))
                return SimpleNamespace(
                    data=[
                        SimpleNamespace(embedding=[float(i)]) for i in range(len(input))
                    ]
                )

        embedder = object.__new__(e.OpenAIEmbedder)
        embedder._client = SimpleNamespace(embeddings=Embeddings())

        with mock.patch.object(e, "OPENAI_REQUEST_BATCH_SIZE", 2):
            vectors = embedder.embed(["a", "b", "c", "d", "e"])

        self.assertEqual([len(batch) for batch in calls], [2, 2, 1])
        self.assertEqual(len(vectors), 5)

    def test_openai_forwards_dimensions_only_when_explicit(self):
        kwargs_seen = []

        class Embeddings:
            def create(self, model, input, **kwargs):
                kwargs_seen.append((model, kwargs))
                return SimpleNamespace(
                    data=[SimpleNamespace(embedding=[0.0]) for _ in input]
                )

        embedder = object.__new__(e.OpenAIEmbedder)
        embedder._client = SimpleNamespace(embeddings=Embeddings())

        # Default (derived) dimension: `dimensions` must not be forwarded, since
        # it is a v3-only parameter that would break ada-002.
        with (
            mock.patch.object(e, "_OPENAI_DIM_EXPLICIT", False),
            mock.patch.object(e, "_OPENAI_MODEL", "text-embedding-3-small"),
        ):
            embedder.embed(["a"])
        self.assertEqual(kwargs_seen[-1], ("text-embedding-3-small", {}))

        # Explicit override: `dimensions` is forwarded so returned vectors match.
        with (
            mock.patch.object(e, "_OPENAI_DIM_EXPLICIT", True),
            mock.patch.object(e, "_OPENAI_DIM", 1024),
            mock.patch.object(e, "_OPENAI_MODEL", "text-embedding-3-large"),
        ):
            embedder.embed(["a"])
        self.assertEqual(
            kwargs_seen[-1], ("text-embedding-3-large", {"dimensions": 1024})
        )

    def test_sentence_transformers_uses_local_inference_batch_size(self):
        class Model:
            def __init__(self):
                self.batch_size = None

            def encode(self, texts, batch_size, **kwargs):
                self.batch_size = batch_size
                return _Array([[0.1] for _ in texts])

        embedder = object.__new__(e.LocalEmbedder)
        embedder._model = Model()

        with mock.patch.object(e, "LOCAL_INFERENCE_BATCH_SIZE", 7):
            embedder.embed(["a", "b"])

        self.assertEqual(embedder._model.batch_size, 7)

    def test_fastembed_uses_local_inference_batch_size(self):
        class Model:
            def __init__(self):
                self.batch_size = None

            def embed(self, texts, batch_size):
                self.batch_size = batch_size
                return iter([_Array([0.1]) for _ in texts])

        embedder = object.__new__(e.FastEmbedEmbedder)
        embedder._model = Model()

        with mock.patch.object(e, "LOCAL_INFERENCE_BATCH_SIZE", 11):
            vectors = embedder.embed(["a", "b"])

        self.assertEqual(embedder._model.batch_size, 11)
        self.assertEqual(vectors, [[0.1], [0.1]])


class LazyEmbedder(unittest.TestCase):
    def test_vector_dimension_does_not_load_backend(self):
        factory = mock.Mock(return_value=SimpleNamespace(vector_dim=3))
        embedder = e.LazyEmbedder(factory, vector_dim=3)

        self.assertEqual(embedder.vector_dim, 3)
        self.assertFalse(embedder.is_loaded)
        factory.assert_not_called()

    def test_first_embed_loads_backend_once(self):
        backend = mock.Mock(vector_dim=3)
        backend.embed.side_effect = lambda texts: [[0.1, 0.2, 0.3] for _ in texts]
        factory = mock.Mock(return_value=backend)
        embedder = e.LazyEmbedder(factory, vector_dim=3)

        embedder.embed(["first"])
        embedder.embed(["second"])

        factory.assert_called_once_with()
        self.assertTrue(embedder.is_loaded)

    def test_rejects_dimension_mismatch_when_backend_loads(self):
        backend = mock.Mock(vector_dim=4)
        embedder = e.LazyEmbedder(lambda: backend, vector_dim=3)

        with self.assertRaisesRegex(RuntimeError, "expected 3, got 4"):
            embedder.embed(["text"])

    def test_factory_function_defers_real_backend(self):
        backend = mock.Mock(vector_dim=1024)
        backend.embed.return_value = [[0.1]]
        with (
            mock.patch.object(e, "_configured_vector_dim", return_value=1024),
            mock.patch.object(e, "get_embedder", return_value=backend) as factory,
        ):
            embedder = e.get_lazy_embedder()
            factory.assert_not_called()
            embedder.embed(["text"])

        factory.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
