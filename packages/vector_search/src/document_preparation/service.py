"""Concurrent two-stage orchestration for document preparation comparisons."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from time import perf_counter
from typing import Any

from packages.vector_search.src.document_preparation.chunkers import (
    CHUNKER_TYPES,
    get_chunker,
)
from packages.vector_search.src.document_preparation.models import (
    ChunkingResult,
    ComparisonRequest,
    PairSelection,
    PreparedChunk,
    PreparationOptions,
)
from packages.vector_search.src.document_preparation.tokenizers import (
    TOKENIZER_TYPES,
    get_tokenizer,
)
from packages.vector_search.src.document_preparation.chunkers import chunker_options
from packages.vector_search.src.document_preparation.tokenizers import tokenizer_options

Event = dict[str, Any]
EventSink = Callable[[Event], Awaitable[None]]


class ComparisonService:
    """Run selected pairs concurrently and stream independent results."""

    def options(self) -> PreparationOptions:
        return PreparationOptions(
            tokenizers=tokenizer_options(),
            chunkers=chunker_options(),
        )

    def validate_registry(self, request: ComparisonRequest) -> None:
        for pair in request.pairs:
            if pair.tokenizer not in TOKENIZER_TYPES:
                raise ValueError(f"Unsupported tokenizer: {pair.tokenizer}")
            if pair.chunker not in CHUNKER_TYPES:
                raise ValueError(f"Unsupported chunker: {pair.chunker}")

    async def compare(self, request: ComparisonRequest, emit: EventSink) -> None:
        self.validate_registry(request)
        await emit(
            {
                "type": "run.started",
                "run_id": request.run_id,
                "pair_count": len(request.pairs),
            }
        )

        outcomes = await asyncio.gather(
            *(self._run_pair(request, pair, emit) for pair in request.pairs)
        )
        await emit(
            {
                "type": "run.completed",
                "run_id": request.run_id,
                "succeeded": outcomes.count(True),
                "failed": outcomes.count(False),
            }
        )

    async def _run_pair(
        self,
        request: ComparisonRequest,
        pair: PairSelection,
        emit: EventSink,
    ) -> bool:
        try:
            tokenizer = await asyncio.to_thread(get_tokenizer, pair.tokenizer)
            tokenization = await asyncio.to_thread(tokenizer.tokenize, request.document)
            await emit(
                {
                    "type": "pair.tokenized",
                    "run_id": request.run_id,
                    "pair_id": pair.id,
                    "result": tokenization.model_dump(mode="json"),
                }
            )

            chunker = get_chunker(pair.chunker)
            started = perf_counter()
            raw_chunks = await asyncio.to_thread(
                chunker.chunk,
                request.document,
                tokenizer,
                request.chunk_size,
                request.chunk_overlap,
            )
            annotated = await asyncio.gather(
                *(
                    asyncio.to_thread(tokenizer.tokenize, raw_chunk.text)
                    for raw_chunk in raw_chunks
                )
            )
            chunks = [
                PreparedChunk(
                    index=index,
                    text=raw_chunk.text,
                    decoded_text=tokenized.decoded_text,
                    start=raw_chunk.start,
                    end=raw_chunk.end,
                    token_count=tokenized.token_count,
                    tokens=tokenized.tokens,
                )
                for index, (raw_chunk, tokenized) in enumerate(
                    zip(raw_chunks, annotated, strict=True)
                )
            ]
            result = ChunkingResult.from_chunks(
                chunker_id=pair.chunker,
                chunks=chunks,
                duration_ms=round((perf_counter() - started) * 1000, 3),
            )
            await emit(
                {
                    "type": "pair.chunked",
                    "run_id": request.run_id,
                    "pair_id": pair.id,
                    "result": result.model_dump(mode="json"),
                }
            )
            return True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await emit(
                {
                    "type": "pair.failed",
                    "run_id": request.run_id,
                    "pair_id": pair.id,
                    "message": str(exc) or exc.__class__.__name__,
                }
            )
            return False
