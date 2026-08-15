"""Typed contracts shared by the preparation API and frontend."""

from __future__ import annotations

from statistics import mean
from typing import Self
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

MAX_COMPARISONS = 4
MAX_DOCUMENT_CHARACTERS = 100_000


class PairSelection(BaseModel):
    """One tokenizer/chunker combination selected by the user."""

    id: str = Field(min_length=1, max_length=80)
    tokenizer: str = Field(min_length=1, max_length=80)
    chunker: str = Field(min_length=1, max_length=80)


class ComparisonRequest(BaseModel):
    """A complete comparison run submitted over the WebSocket."""

    run_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    document: str = Field(max_length=MAX_DOCUMENT_CHARACTERS)
    pairs: list[PairSelection] = Field(min_length=1, max_length=MAX_COMPARISONS)
    chunk_size: int = Field(default=256, ge=8, le=4096)
    chunk_overlap: int = Field(default=32, ge=0, le=2048)

    @model_validator(mode="after")
    def validate_comparison(self) -> Self:
        if not self.document.strip():
            raise ValueError("document must contain non-whitespace text")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        pair_ids = [pair.id for pair in self.pairs]
        if len(pair_ids) != len(set(pair_ids)):
            raise ValueError("pair ids must be unique")

        combinations = [(pair.tokenizer, pair.chunker) for pair in self.pairs]
        if len(combinations) != len(set(combinations)):
            raise ValueError("tokenizer/chunker combinations must be unique")
        return self


class TokenPiece(BaseModel):
    """A token with both its machine id and human-readable representation."""

    index: int
    id: int
    piece: str
    decoded: str
    start: int | None = None
    end: int | None = None


class TokenizationResult(BaseModel):
    """Normalized output produced by any tokenizer adapter."""

    tokenizer_id: str
    model: str
    token_count: int
    decoded_text: str
    tokens: list[TokenPiece]
    duration_ms: float = 0


class RawChunk(BaseModel):
    """Library-independent chunk before token annotations are attached."""

    text: str
    start: int | None = None
    end: int | None = None


class PreparedChunk(BaseModel):
    """Chunk rendered in one comparison card."""

    index: int
    text: str
    decoded_text: str
    start: int | None = None
    end: int | None = None
    token_count: int
    tokens: list[TokenPiece]


class ChunkingResult(BaseModel):
    """Normalized chunks and compact statistics for one pair."""

    chunker_id: str
    chunk_count: int
    chunks: list[PreparedChunk]
    duration_ms: float
    minimum_tokens: int
    maximum_tokens: int
    average_tokens: float

    @classmethod
    def from_chunks(
        cls,
        *,
        chunker_id: str,
        chunks: list[PreparedChunk],
        duration_ms: float,
    ) -> "ChunkingResult":
        counts = [chunk.token_count for chunk in chunks]
        return cls(
            chunker_id=chunker_id,
            chunk_count=len(chunks),
            chunks=chunks,
            duration_ms=duration_ms,
            minimum_tokens=min(counts, default=0),
            maximum_tokens=max(counts, default=0),
            average_tokens=round(mean(counts), 2) if counts else 0,
        )


class Option(BaseModel):
    """One selectable implementation exposed by the options endpoint."""

    id: str
    label: str
    detail: str
    model: str | None = None


class PreparationOptions(BaseModel):
    """Frontend configuration returned without loading any model."""

    tokenizers: list[Option]
    chunkers: list[Option]
    maximum_comparisons: int = MAX_COMPARISONS
    maximum_document_characters: int = MAX_DOCUMENT_CHARACTERS
    default_chunk_size: int = 256
    default_chunk_overlap: int = 32
