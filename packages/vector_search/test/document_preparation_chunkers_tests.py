"""Contract tests for every chunker adapter and all sixteen pair registrations."""

import unittest
from unittest import mock

from packages.vector_search.src.document_preparation import chunkers as module
from packages.vector_search.src.document_preparation.models import RawChunk, TokenPiece
from packages.vector_search.src.document_preparation.tokenizers import (
    TOKENIZER_TYPES,
    TokenizerAdapter,
)


class CharacterTokenizer(TokenizerAdapter):
    id = "characters"
    label = "Characters"
    detail = "test tokenizer"
    model = "test"

    def encode_ids(self, text):
        return [ord(character) for character in text]

    def decode_ids(self, token_ids):
        return "".join(chr(token_id) for token_id in token_ids)

    def _pieces(self, text, token_ids):
        del text
        return [
            TokenPiece(index=index, id=token_id, piece=chr(token_id), decoded=chr(token_id))
            for index, token_id in enumerate(token_ids)
        ]


class _ChonkieResult:
    def __init__(self, text, start, end):
        self.text = text
        self.start_index = start
        self.end_index = end


class _Chonkie:
    def chunk(self, text):
        midpoint = text.index("Second")
        return [
            _ChonkieResult(text[:midpoint], 0, midpoint),
            _ChonkieResult(text[midpoint:], midpoint, len(text)),
        ]


class ChunkerAdapterTests(unittest.TestCase):
    text = "First sentence has detail. Second sentence changes the topic completely."
    tokenizer = CharacterTokenizer()

    def test_registry_builds_the_full_sixteen_pair_matrix(self):
        self.assertEqual(
            set(module.CHUNKER_TYPES),
            {"langchain", "semchunk", "chonkie", "llamaindex"},
        )
        matrix = {
            (tokenizer, chunker)
            for tokenizer in TOKENIZER_TYPES
            for chunker in module.CHUNKER_TYPES
        }
        self.assertEqual(len(matrix), 16)

    def test_langchain_semchunk_and_llamaindex_return_bounded_text(self):
        for adapter in (
            module.LangChainChunker(),
            module.SemchunkChunker(),
            module.LlamaIndexChunker(),
        ):
            with self.subTest(adapter=adapter.id):
                chunks = adapter.chunk(self.text, self.tokenizer, 28, 4)
                self.assertGreater(len(chunks), 1)
                self.assertTrue(all(chunk.text for chunk in chunks))
                self.assertTrue(
                    all(self.tokenizer.count(chunk.text) <= 28 for chunk in chunks)
                )

    def test_chonkie_retains_boundaries_and_normalizes_selected_budget(self):
        with mock.patch.object(module, "_chonkie_semantic_chunker", return_value=_Chonkie()):
            chunks = module.ChonkieChunker().chunk(
                self.text, self.tokenizer, chunk_size=24, overlap=4
            )

        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(self.tokenizer.count(chunk.text) <= 24 for chunk in chunks))
        self.assertTrue(all(chunk.start is None for chunk in chunks))

    def test_offset_locator_handles_repeated_and_missing_chunks(self):
        chunks = module._locate_chunks("one two one", ["one", "one", "missing"])
        self.assertEqual((chunks[0].start, chunks[0].end), (0, 3))
        self.assertEqual((chunks[1].start, chunks[1].end), (8, 11))
        self.assertIsNone(chunks[2].start)

    def test_normalizer_returns_an_empty_chunk_for_no_semantic_output(self):
        self.assertEqual(
            module._normalize_semantic_chunks([], self.tokenizer, 20, 2),
            [RawChunk(text="")],
        )


if __name__ == "__main__":
    unittest.main()
