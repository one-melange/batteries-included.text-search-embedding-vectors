"""Adapters for the four chunking libraries in the comparison matrix."""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from functools import lru_cache

from packages.vector_search.src.document_preparation.models import Option, RawChunk
from packages.vector_search.src.document_preparation.tokenizers import TokenizerAdapter

CHONKIE_EMBEDDING_MODEL = os.environ.get(
    "DOCUMENT_PREPARATION_CHONKIE_MODEL", "minishlab/potion-base-32M"
)

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")


def _split_sentences(text: str) -> list[str]:
    """Split sentences without LlamaIndex's optional NLTK data files.

    LlamaIndex accepts a caller-supplied sentence tokenizer. Keeping it local
    makes the Node Parser deterministic and avoids a runtime corpus download.
    """

    return [part for part in _SENTENCE_BOUNDARY.split(text) if part]


def _locate_chunks(text: str, chunks: list[str]) -> list[RawChunk]:
    """Best-effort source offsets for libraries that return text only."""

    cursor = 0
    located: list[RawChunk] = []
    for chunk in chunks:
        start = text.find(chunk, cursor)
        if start < 0:
            start = text.find(chunk)
        if start < 0:
            located.append(RawChunk(text=chunk))
            continue
        end = start + len(chunk)
        located.append(RawChunk(text=chunk, start=start, end=end))
        cursor = max(cursor, end)
    return located


def _normalize_semantic_chunks(
    chunks: list[RawChunk],
    tokenizer: TokenizerAdapter,
    chunk_size: int,
    overlap: int,
) -> list[RawChunk]:
    """Enforce the selected tokenizer's budget on Chonkie semantic output.

    Chonkie's semantic splitter deliberately uses the tokenizer belonging to
    its embedding model. The comparison adapter retains those semantic
    boundaries, then token-windows an oversized result and carries the selected
    overlap into the next chunk. Normalized chunks have no reliable contiguous
    source offsets because overlap duplicates source text.
    """

    normalized: list[RawChunk] = []
    previous_ids: list[int] = []
    for raw in chunks:
        current_ids = tokenizer.encode_ids(raw.text)
        if not current_ids:
            continue

        prefix = previous_ids[-overlap:] if normalized and overlap else []
        position = 0
        while position < len(current_ids):
            capacity = chunk_size - len(prefix)
            selected = prefix + current_ids[position : position + capacity]
            normalized.append(RawChunk(text=tokenizer.decode_ids(selected)))
            position += capacity
            previous_ids = selected
            prefix = previous_ids[-overlap:] if overlap else []

    return normalized or [RawChunk(text="")]


class ChunkerAdapter(ABC):
    """Common entry point for a tokenizer/chunker pairing."""

    id: str
    label: str
    detail: str

    @abstractmethod
    def chunk(
        self,
        text: str,
        tokenizer: TokenizerAdapter,
        chunk_size: int,
        overlap: int,
    ) -> list[RawChunk]:
        """Split text and return chunks in document order."""


class LangChainChunker(ChunkerAdapter):
    id = "langchain"
    label = "LangChain"
    detail = "RecursiveCharacterTextSplitter"

    def chunk(
        self,
        text: str,
        tokenizer: TokenizerAdapter,
        chunk_size: int,
        overlap: int,
    ) -> list[RawChunk]:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=tokenizer.count,
            add_start_index=True,
            keep_separator=True,
        )
        documents = splitter.create_documents([text])
        return [
            RawChunk(
                text=document.page_content,
                start=document.metadata.get("start_index"),
                end=(
                    document.metadata["start_index"] + len(document.page_content)
                    if "start_index" in document.metadata
                    else None
                ),
            )
            for document in documents
        ] or [RawChunk(text="", start=0, end=0)]


class SemchunkChunker(ChunkerAdapter):
    id = "semchunk"
    label = "semchunk"
    detail = "Semantic heuristic chunking"

    def chunk(
        self,
        text: str,
        tokenizer: TokenizerAdapter,
        chunk_size: int,
        overlap: int,
    ) -> list[RawChunk]:
        import semchunk

        chunker = semchunk.chunkerify(tokenizer.count, chunk_size)
        chunk_texts, offsets = chunker(
            text,
            offsets=True,
            overlap=overlap or None,
        )
        return [
            RawChunk(text=chunk, start=offset[0], end=offset[1])
            for chunk, offset in zip(chunk_texts, offsets, strict=True)
        ] or [RawChunk(text="", start=0, end=0)]


@lru_cache(maxsize=8)
def _chonkie_semantic_chunker(chunk_size: int):
    from chonkie import SemanticChunker

    return SemanticChunker(
        embedding_model=CHONKIE_EMBEDDING_MODEL,
        chunk_size=chunk_size,
    )


class ChonkieChunker(ChunkerAdapter):
    id = "chonkie"
    label = "Chonkie"
    detail = "SemanticChunker with selected-tokenizer budget normalization"

    def chunk(
        self,
        text: str,
        tokenizer: TokenizerAdapter,
        chunk_size: int,
        overlap: int,
    ) -> list[RawChunk]:
        chunker = _chonkie_semantic_chunker(chunk_size)
        chonkie_chunks = chunker.chunk(text)
        raw = [
            RawChunk(
                text=chunk.text,
                start=getattr(chunk, "start_index", None),
                end=getattr(chunk, "end_index", None),
            )
            for chunk in chonkie_chunks
        ]
        return _normalize_semantic_chunks(raw, tokenizer, chunk_size, overlap)


class LlamaIndexChunker(ChunkerAdapter):
    id = "llamaindex"
    label = "LlamaIndex"
    detail = "SentenceSplitter node parser"

    def chunk(
        self,
        text: str,
        tokenizer: TokenizerAdapter,
        chunk_size: int,
        overlap: int,
    ) -> list[RawChunk]:
        from llama_index.core.node_parser import SentenceSplitter

        splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            tokenizer=tokenizer.encode_ids,
            chunking_tokenizer_fn=_split_sentences,
        )
        return _locate_chunks(text, splitter.split_text(text)) or [
            RawChunk(text="", start=0, end=0)
        ]


CHUNKER_TYPES: dict[str, type[ChunkerAdapter]] = {
    LangChainChunker.id: LangChainChunker,
    SemchunkChunker.id: SemchunkChunker,
    ChonkieChunker.id: ChonkieChunker,
    LlamaIndexChunker.id: LlamaIndexChunker,
}


@lru_cache(maxsize=len(CHUNKER_TYPES))
def get_chunker(chunker_id: str) -> ChunkerAdapter:
    try:
        adapter_type = CHUNKER_TYPES[chunker_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported chunker: {chunker_id}") from exc
    return adapter_type()


def chunker_options() -> list[Option]:
    return [
        Option(id=adapter.id, label=adapter.label, detail=adapter.detail)
        for adapter in CHUNKER_TYPES.values()
    ]


def clear_chunker_cache() -> None:
    get_chunker.cache_clear()
    _chonkie_semantic_chunker.cache_clear()
