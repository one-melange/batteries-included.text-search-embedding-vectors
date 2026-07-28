"""
config_tests.py
───────────────
Tests that Qdrant connection config is read from the environment, so the same
code targets the local Docker container or a Qdrant Cloud cluster without edits.

config reads the environment at import time, so each test reloads the module
under a patched os.environ and reloads it once more afterwards to restore the
process-wide defaults for other tests.
"""

import importlib
import os
import unittest
from unittest import mock

from packages.vector_search.src import config


class QdrantConnectionConfig(unittest.TestCase):
    def tearDown(self):
        # Restore module state from the ambient (unpatched) environment.
        importlib.reload(config)

    def test_defaults_to_local_docker_without_env(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            reloaded = importlib.reload(config)
        self.assertEqual(reloaded.QDRANT_URL, "http://localhost:6333")
        self.assertIsNone(reloaded.QDRANT_API_KEY)

    def test_reads_cluster_url_and_api_key_from_env(self):
        env = {
            "QDRANT_URL": "https://cluster.cloud.qdrant.io:6333",
            "QDRANT_API_KEY": "secret-key",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            reloaded = importlib.reload(config)
        self.assertEqual(reloaded.QDRANT_URL, "https://cluster.cloud.qdrant.io:6333")
        self.assertEqual(reloaded.QDRANT_API_KEY, "secret-key")

    def test_batching_concerns_have_independent_settings(self):
        env = {
            "VECTOR_SEARCH_PIPELINE_CHUNK_BATCH_SIZE": "400",
            "VECTOR_SEARCH_LOCAL_INFERENCE_BATCH_SIZE": "16",
            "VECTOR_SEARCH_OPENAI_REQUEST_BATCH_SIZE": "80",
            "VECTOR_SEARCH_QDRANT_UPSERT_BATCH_SIZE": "200",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            reloaded = importlib.reload(config)

        self.assertEqual(reloaded.PIPELINE_CHUNK_BATCH_SIZE, 400)
        self.assertEqual(reloaded.LOCAL_INFERENCE_BATCH_SIZE, 16)
        self.assertEqual(reloaded.OPENAI_REQUEST_BATCH_SIZE, 80)
        self.assertEqual(reloaded.QDRANT_UPSERT_BATCH_SIZE, 200)

    def test_legacy_embed_batch_size_remains_a_fallback(self):
        with mock.patch.dict(
            os.environ, {"VECTOR_SEARCH_EMBED_BATCH_SIZE": "24"}, clear=True
        ):
            reloaded = importlib.reload(config)

        self.assertEqual(reloaded.PIPELINE_CHUNK_BATCH_SIZE, 24)
        self.assertEqual(reloaded.LOCAL_INFERENCE_BATCH_SIZE, 24)
        self.assertEqual(reloaded.OPENAI_REQUEST_BATCH_SIZE, 24)
        self.assertEqual(reloaded.QDRANT_UPSERT_BATCH_SIZE, 200)

    def test_rejects_non_positive_batch_size(self):
        with mock.patch.dict(
            os.environ,
            {"VECTOR_SEARCH_LOCAL_INFERENCE_BATCH_SIZE": "0"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "must be greater than zero"):
                importlib.reload(config)


if __name__ == "__main__":
    unittest.main()
