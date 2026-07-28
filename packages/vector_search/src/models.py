"""Data contracts shared by the generic vector-search modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Source-independent input accepted by the embedding pipeline.

    Attributes:
        document_id: Stable identifier unique within ``source``.
        text: Complete text selected by the source adapter for embedding.
        source: Stable provenance identifier, such as a repository-relative path,
            object-store key, or database namespace.
        metadata: Additional JSON-compatible payload values copied to every
            indexed chunk. Standard point fields take precedence on key
            collisions.
    """

    document_id: str
    text: str
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """Token-bounded unit passed to an embedder and stored as one Qdrant point.

    Attributes:
        document_id: Identifier copied from the parent document.
        text: Slice that is actually embedded.
        chunk_index: Zero-based position within the parent document.
        total_chunks: Number of chunks produced for the parent document.
        source: Provenance identifier copied from the parent document.
        content_hash: Hash of the complete parent text, shared by all its chunks.
        metadata: Adapter-supplied payload values copied from the parent document.
    """

    document_id: str
    text: str
    chunk_index: int
    total_chunks: int
    source: str
    content_hash: str
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class PayloadFields:
    """Map generic identity fields onto an existing Qdrant payload schema.

    Attributes:
        document_id: Payload key containing ``Document.document_id``.
        source: Payload key containing ``Document.source``.

    Adapters can customize these names to preserve compatibility with an
    established collection without leaking its domain vocabulary into the
    generic pipeline.
    """

    document_id: str = "document_id"
    source: str = "source"


DEFAULT_PAYLOAD_FIELDS = PayloadFields()
