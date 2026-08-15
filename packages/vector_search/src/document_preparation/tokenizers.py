"""Adapters that normalize the four tokenizer families in the matrix."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from functools import lru_cache
from time import perf_counter
from typing import Any

from packages.vector_search.src.document_preparation.models import (
    Option,
    TokenizationResult,
    TokenPiece,
)

TIKTOKEN_MODEL = os.environ.get("DOCUMENT_PREPARATION_TIKTOKEN_MODEL", "cl100k_base")
QUICKTOK_MODEL = os.environ.get("DOCUMENT_PREPARATION_QUICKTOK_MODEL", "cl100k_base")
HF_MODEL = os.environ.get("DOCUMENT_PREPARATION_HF_MODEL", "bert-base-uncased")
SENTENCEPIECE_REPOSITORY = os.environ.get(
    "DOCUMENT_PREPARATION_SENTENCEPIECE_REPOSITORY", "google-t5/t5-small"
)
SENTENCEPIECE_FILENAME = os.environ.get(
    "DOCUMENT_PREPARATION_SENTENCEPIECE_FILENAME", "spiece.model"
)


def _decoded_bytes(value: bytes) -> str:
    return value.decode("utf-8", errors="replace")


class TokenizerAdapter(ABC):
    """Small common API needed by both visualization and chunk libraries."""

    id: str
    label: str
    detail: str
    model: str

    @abstractmethod
    def encode_ids(self, text: str) -> list[int]:
        """Encode text to integer token ids."""

    @abstractmethod
    def decode_ids(self, token_ids: list[int]) -> str:
        """Decode a complete sequence of token ids."""

    @abstractmethod
    def _pieces(self, text: str, token_ids: list[int]) -> list[TokenPiece]:
        """Return readable pieces and optional character offsets."""

    def count(self, text: str) -> int:
        return len(self.encode_ids(text))

    def tokenize(self, text: str) -> TokenizationResult:
        started = perf_counter()
        token_ids = self.encode_ids(text)
        tokens = self._pieces(text, token_ids)
        return TokenizationResult(
            tokenizer_id=self.id,
            model=self.model,
            token_count=len(token_ids),
            decoded_text=self.decode_ids(token_ids),
            tokens=tokens,
            duration_ms=round((perf_counter() - started) * 1000, 3),
        )


class TikTokenAdapter(TokenizerAdapter):
    id = "tiktoken"
    label = "tiktoken"
    detail = "OpenAI byte-pair encoding"
    model = TIKTOKEN_MODEL

    def __init__(self) -> None:
        import tiktoken

        self._encoding = tiktoken.get_encoding(self.model)

    def encode_ids(self, text: str) -> list[int]:
        return self._encoding.encode(text, disallowed_special=())

    def decode_ids(self, token_ids: list[int]) -> str:
        return self._encoding.decode(token_ids)

    def _pieces(self, text: str, token_ids: list[int]) -> list[TokenPiece]:
        del text
        return [
            TokenPiece(
                index=index,
                id=token_id,
                piece=_decoded_bytes(self._encoding.decode_single_token_bytes(token_id)),
                decoded=_decoded_bytes(
                    self._encoding.decode_single_token_bytes(token_id)
                ),
            )
            for index, token_id in enumerate(token_ids)
        ]


class QuickTokAdapter(TokenizerAdapter):
    id = "quicktok"
    label = "quicktok"
    detail = "Native C++ byte-pair encoding"
    model = QUICKTOK_MODEL

    def __init__(self) -> None:
        import quicktok

        self._encoding = quicktok.get_encoding(self.model)

    def encode_ids(self, text: str) -> list[int]:
        return list(self._encoding.encode(text))

    def decode_ids(self, token_ids: list[int]) -> str:
        return self._encoding.decode(token_ids)

    def _pieces(self, text: str, token_ids: list[int]) -> list[TokenPiece]:
        del text
        pieces: list[TokenPiece] = []
        for index, token_id in enumerate(token_ids):
            decoded = self._encoding.decode([token_id])
            pieces.append(
                TokenPiece(
                    index=index,
                    id=token_id,
                    piece=decoded,
                    decoded=decoded,
                )
            )
        return pieces


class HuggingFaceAdapter(TokenizerAdapter):
    id = "huggingface"
    label = "HF tokenizers"
    detail = "Hugging Face Rust tokenizer"
    model = HF_MODEL

    def __init__(self) -> None:
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_pretrained(self.model)

    def encode_ids(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False).ids

    def decode_ids(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=False)

    def _pieces(self, text: str, token_ids: list[int]) -> list[TokenPiece]:
        encoding = self._tokenizer.encode(text, add_special_tokens=False)
        return [
            TokenPiece(
                index=index,
                id=token_id,
                piece=encoding.tokens[index],
                decoded=self._tokenizer.decode(
                    [token_id], skip_special_tokens=False
                ),
                start=encoding.offsets[index][0],
                end=encoding.offsets[index][1],
            )
            for index, token_id in enumerate(token_ids)
        ]


class SentencePieceAdapter(TokenizerAdapter):
    id = "sentencepiece"
    label = "SentencePiece"
    detail = "Google language-independent subword tokenizer"
    model = f"{SENTENCEPIECE_REPOSITORY}/{SENTENCEPIECE_FILENAME}"

    def __init__(self) -> None:
        from huggingface_hub import hf_hub_download
        import sentencepiece as spm

        configured_path = os.environ.get("DOCUMENT_PREPARATION_SENTENCEPIECE_MODEL")
        model_path = configured_path or hf_hub_download(
            repo_id=SENTENCEPIECE_REPOSITORY,
            filename=SENTENCEPIECE_FILENAME,
        )
        self.model = configured_path or self.model
        self._processor = spm.SentencePieceProcessor(model_file=model_path)

    def encode_ids(self, text: str) -> list[int]:
        return list(self._processor.encode(text, out_type=int))

    def decode_ids(self, token_ids: list[int]) -> str:
        return self._processor.decode(token_ids)

    def _pieces(self, text: str, token_ids: list[int]) -> list[TokenPiece]:
        del text
        string_pieces = self._processor.id_to_piece(token_ids)
        return [
            TokenPiece(
                index=index,
                id=token_id,
                piece=string_pieces[index],
                decoded=self._processor.decode([token_id]),
            )
            for index, token_id in enumerate(token_ids)
        ]


TOKENIZER_TYPES: dict[str, type[TokenizerAdapter]] = {
    TikTokenAdapter.id: TikTokenAdapter,
    HuggingFaceAdapter.id: HuggingFaceAdapter,
    QuickTokAdapter.id: QuickTokAdapter,
    SentencePieceAdapter.id: SentencePieceAdapter,
}


@lru_cache(maxsize=len(TOKENIZER_TYPES))
def get_tokenizer(tokenizer_id: str) -> TokenizerAdapter:
    """Load each tokenizer at most once per server process."""

    try:
        adapter_type = TOKENIZER_TYPES[tokenizer_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported tokenizer: {tokenizer_id}") from exc
    return adapter_type()


def tokenizer_options() -> list[Option]:
    """List tokenizers without downloading their model artifacts."""

    return [
        Option(
            id=adapter.id,
            label=adapter.label,
            detail=adapter.detail,
            model=adapter.model,
        )
        for adapter in TOKENIZER_TYPES.values()
    ]


def clear_tokenizer_cache() -> None:
    """Test hook for replacing model-backed adapters safely."""

    get_tokenizer.cache_clear()
