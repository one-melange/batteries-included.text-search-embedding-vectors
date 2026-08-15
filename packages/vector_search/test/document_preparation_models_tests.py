"""Validation tests for comparison request contracts."""

import unittest

from pydantic import ValidationError

from packages.vector_search.src.document_preparation.models import (
    ComparisonRequest,
    PairSelection,
)


def _pair(pair_id="pair-1", tokenizer="tiktoken", chunker="langchain"):
    return PairSelection(id=pair_id, tokenizer=tokenizer, chunker=chunker)


class ComparisonRequestTests(unittest.TestCase):
    def test_accepts_one_to_four_unique_pairs(self):
        request = ComparisonRequest(
            document="Useful document text",
            pairs=[
                _pair("one"),
                _pair("two", chunker="semchunk"),
                _pair("three", tokenizer="quicktok"),
                _pair("four", tokenizer="sentencepiece"),
            ],
        )

        self.assertEqual(len(request.pairs), 4)
        self.assertLess(request.chunk_overlap, request.chunk_size)

    def test_rejects_more_than_four_pairs(self):
        with self.assertRaises(ValidationError):
            ComparisonRequest(
                document="text",
                pairs=[_pair(str(index), chunker=str(index)) for index in range(5)],
            )

    def test_rejects_duplicate_combinations_and_ids(self):
        with self.assertRaisesRegex(ValidationError, "pair ids must be unique"):
            ComparisonRequest(
                document="text",
                pairs=[_pair("same"), _pair("same", chunker="semchunk")],
            )
        with self.assertRaisesRegex(
            ValidationError, "tokenizer/chunker combinations must be unique"
        ):
            ComparisonRequest(
                document="text",
                pairs=[_pair("one"), _pair("two")],
            )

    def test_rejects_empty_document_and_invalid_overlap(self):
        with self.assertRaisesRegex(ValidationError, "non-whitespace"):
            ComparisonRequest(document="  \n", pairs=[_pair()])
        with self.assertRaisesRegex(ValidationError, "smaller than chunk_size"):
            ComparisonRequest(
                document="text",
                pairs=[_pair()],
                chunk_size=32,
                chunk_overlap=32,
            )


if __name__ == "__main__":
    unittest.main()
