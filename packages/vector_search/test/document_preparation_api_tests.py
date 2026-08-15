"""HTTP and WebSocket tests for the preparation API."""

import asyncio
import threading
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from packages.vector_search.src.document_preparation.api import app
from packages.vector_search.src.document_preparation.models import PreparationOptions


class FakeService:
    def options(self):
        return PreparationOptions(tokenizers=[], chunkers=[])

    def validate_registry(self, request):
        if request.pairs[0].tokenizer == "missing":
            raise ValueError("Unsupported tokenizer: missing")

    async def compare(self, request, emit):
        await emit(
            {
                "type": "pair.tokenized",
                "run_id": request.run_id,
                "pair_id": request.pairs[0].id,
                "result": {"token_count": 2},
            }
        )
        await emit(
            {
                "type": "pair.chunked",
                "run_id": request.run_id,
                "pair_id": request.pairs[0].id,
                "result": {"chunk_count": 1},
            }
        )


class ReplacementService(FakeService):
    def __init__(self):
        self.started = threading.Event()
        self.cancelled = threading.Event()

    async def compare(self, request, emit):
        if request.run_id == "first":
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
        await emit({"type": "run.completed", "run_id": request.run_id})


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.patch = mock.patch(
            "packages.vector_search.src.document_preparation.api.service",
            new=FakeService(),
        )
        self.patch.start()
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        self.patch.stop()

    def payload(self, tokenizer="tiktoken"):
        return {
            "run_id": "run-1",
            "document": "hello world",
            "pairs": [
                {"id": "pair-1", "tokenizer": tokenizer, "chunker": "langchain"}
            ],
            "chunk_size": 32,
            "chunk_overlap": 4,
        }

    def test_options_endpoint_is_model_free(self):
        response = self.client.get("/api/document-preparation/options")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["maximum_comparisons"], 4)

    def test_websocket_streams_two_pair_stages(self):
        with self.client.websocket_connect("/ws/document-preparation") as websocket:
            websocket.send_json(self.payload())
            self.assertEqual(websocket.receive_json()["type"], "pair.tokenized")
            self.assertEqual(websocket.receive_json()["type"], "pair.chunked")

    def test_websocket_returns_validation_errors_without_disconnect(self):
        with self.client.websocket_connect("/ws/document-preparation") as websocket:
            websocket.send_json(self.payload(tokenizer="missing"))
            error = websocket.receive_json()
            self.assertEqual(error["type"], "run.failed")
            self.assertIn("Unsupported tokenizer", error["message"])

            websocket.send_json(self.payload())
            self.assertEqual(websocket.receive_json()["type"], "pair.tokenized")

    def test_new_request_cancels_the_active_run(self):
        replacement_service = ReplacementService()
        with mock.patch(
            "packages.vector_search.src.document_preparation.api.service",
            new=replacement_service,
        ):
            with self.client.websocket_connect("/ws/document-preparation") as websocket:
                first = self.payload()
                first["run_id"] = "first"
                websocket.send_json(first)
                self.assertTrue(replacement_service.started.wait(timeout=1))

                second = self.payload()
                second["run_id"] = "second"
                websocket.send_json(second)
                completed = websocket.receive_json()

        self.assertEqual(completed["run_id"], "second")
        self.assertTrue(replacement_service.cancelled.is_set())


if __name__ == "__main__":
    unittest.main()
