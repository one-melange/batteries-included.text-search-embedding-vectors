"""Contract tests for normalized tokenizer output."""

import unittest

from packages.vector_search.src.document_preparation import tokenizers as module


class _Encoding:
    ids = [10, 11]
    tokens = ["hello", "##s"]
    offsets = [(0, 5), (5, 6)]


class _HuggingFaceTokenizer:
    def encode(self, text, add_special_tokens=False):
        self.last_text = text
        self.add_special_tokens = add_special_tokens
        return _Encoding()

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return {10: "hello", 11: "s"}.get(token_ids[0], "hellos") if len(token_ids) == 1 else "hellos"


class _SentencePieceProcessor:
    def encode(self, text, out_type=int):
        del text, out_type
        return [3, 4]

    def decode(self, token_ids):
        return "".join({3: "hello", 4: " world"}[item] for item in token_ids)

    def id_to_piece(self, token_ids):
        del token_ids
        return ["▁hello", "▁world"]


class TokenizerAdapterTests(unittest.TestCase):
    def test_tiktoken_and_quicktok_match_and_decode_complete_text(self):
        text = "Hello, world! 👋\n"
        tiktoken_result = module.TikTokenAdapter().tokenize(text)
        quicktok_result = module.QuickTokAdapter().tokenize(text)

        self.assertEqual(
            [token.id for token in tiktoken_result.tokens],
            [token.id for token in quicktok_result.tokens],
        )
        self.assertEqual(tiktoken_result.decoded_text, text)
        self.assertEqual(quicktok_result.decoded_text, text)
        self.assertEqual(tiktoken_result.token_count, len(tiktoken_result.tokens))

    def test_huggingface_adapter_preserves_pieces_and_offsets(self):
        adapter = object.__new__(module.HuggingFaceAdapter)
        adapter._tokenizer = _HuggingFaceTokenizer()

        result = adapter.tokenize("hellos")

        self.assertEqual(result.decoded_text, "hellos")
        self.assertEqual([token.piece for token in result.tokens], ["hello", "##s"])
        self.assertEqual((result.tokens[1].start, result.tokens[1].end), (5, 6))

    def test_sentencepiece_adapter_exposes_model_pieces_and_decoded_text(self):
        adapter = object.__new__(module.SentencePieceAdapter)
        adapter._processor = _SentencePieceProcessor()

        result = adapter.tokenize("hello world")

        self.assertEqual(result.decoded_text, "hello world")
        self.assertEqual([token.piece for token in result.tokens], ["▁hello", "▁world"])
        self.assertEqual([token.decoded for token in result.tokens], ["hello", " world"])

    def test_registry_contains_the_four_matrix_rows(self):
        self.assertEqual(
            set(module.TOKENIZER_TYPES),
            {"tiktoken", "huggingface", "quicktok", "sentencepiece"},
        )
        self.assertEqual(len(module.tokenizer_options()), 4)
        with self.assertRaisesRegex(ValueError, "Unsupported tokenizer"):
            module.get_tokenizer("missing")


if __name__ == "__main__":
    unittest.main()
