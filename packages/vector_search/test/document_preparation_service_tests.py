"""Tests for staged, concurrent comparison orchestration."""

import asyncio
import unittest
from unittest import mock

from packages.vector_search.src.document_preparation.models import (
    ComparisonRequest,
    PairSelection,
    RawChunk,
    TokenPiece,
)
from packages.vector_search.src.document_preparation.service import ComparisonService
from packages.vector_search.src.document_preparation.tokenizers import TokenizerAdapter


class FakeTokenizer(TokenizerAdapter):
    id = "tiktoken"
    label = "Fake"
    detail = "Fake"
    model = "fake-model"

    def encode_ids(self, text):
        return [ord(character) for character in text]

    def decode_ids(self, token_ids):
        return "".join(chr(token_id) for token_id in token_ids)

    def _pieces(self, text, token_ids):
        del text
        return [
            TokenPiece(index=index, id=value, piece=chr(value), decoded=chr(value))
            for index, value in enumerate(token_ids)
        ]


class FakeChunker:
    id = "langchain"

    def chunk(self, text, tokenizer, chunk_size, overlap):
        del tokenizer, chunk_size, overlap
        midpoint = len(text) // 2
        return [
            RawChunk(text=text[:midpoint], start=0, end=midpoint),
            RawChunk(text=text[midpoint:], start=midpoint, end=len(text)),
        ]


class FailingChunker:
    id = "semchunk"

    def chunk(self, text, tokenizer, chunk_size, overlap):
        del text, tokenizer, chunk_size, overlap
        raise RuntimeError("chunker exploded")


class ComparisonServiceTests(unittest.IsolatedAsyncioTestCase):
    def request(self):
        return ComparisonRequest(
            run_id="run-1",
            document="abcdefghij",
            pairs=[PairSelection(id="pair-1", tokenizer="tiktoken", chunker="langchain")],
            chunk_size=8,
            chunk_overlap=1,
        )

    async def test_emits_tokenization_before_chunks_for_each_pair(self):
        events = []

        async def emit(event):
            events.append(event)

        with (
            mock.patch(
                "packages.vector_search.src.document_preparation.service.get_tokenizer",
                return_value=FakeTokenizer(),
            ),
            mock.patch(
                "packages.vector_search.src.document_preparation.service.get_chunker",
                return_value=FakeChunker(),
            ),
        ):
            await ComparisonService().compare(self.request(), emit)

        self.assertEqual(
            [event["type"] for event in events],
            ["run.started", "pair.tokenized", "pair.chunked", "run.completed"],
        )
        tokenized = events[1]["result"]
        chunked = events[2]["result"]
        self.assertEqual(tokenized["decoded_text"], "abcdefghij")
        self.assertEqual(chunked["chunk_count"], 2)
        self.assertEqual(chunked["chunks"][0]["decoded_text"], "abcde")

    async def test_one_pair_failure_does_not_cancel_successful_pair(self):
        request = self.request().model_copy(
            update={
                "pairs": [
                    PairSelection(id="good", tokenizer="tiktoken", chunker="langchain"),
                    PairSelection(id="bad", tokenizer="tiktoken", chunker="semchunk"),
                ]
            }
        )
        events = []

        async def emit(event):
            await asyncio.sleep(0)
            events.append(event)

        def chunker(chunker_id):
            return FakeChunker() if chunker_id == "langchain" else FailingChunker()

        with (
            mock.patch(
                "packages.vector_search.src.document_preparation.service.get_tokenizer",
                return_value=FakeTokenizer(),
            ),
            mock.patch(
                "packages.vector_search.src.document_preparation.service.get_chunker",
                side_effect=chunker,
            ),
        ):
            await ComparisonService().compare(request, emit)

        self.assertIn("pair.chunked", [event["type"] for event in events])
        failed = next(event for event in events if event["type"] == "pair.failed")
        self.assertEqual(failed["pair_id"], "bad")
        completed = events[-1]
        self.assertEqual((completed["succeeded"], completed["failed"]), (1, 1))

    def test_rejects_ids_outside_the_registry(self):
        service = ComparisonService()
        request = self.request().model_copy(
            update={
                "pairs": [
                    PairSelection(id="pair", tokenizer="missing", chunker="langchain")
                ]
            }
        )
        with self.assertRaisesRegex(ValueError, "Unsupported tokenizer"):
            service.validate_registry(request)


if __name__ == "__main__":
    unittest.main()
