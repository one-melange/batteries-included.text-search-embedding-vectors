"""Qdrant collection, point, and incremental-index helpers."""

from __future__ import annotations

import hashlib
import logging
import uuid

from packages.arc.src.vector_search.config import QDRANT_UPSERT_BATCH_SIZE
from packages.arc.src.vector_search.models import (
    DEFAULT_PAYLOAD_FIELDS,
    Chunk,
    PayloadFields,
)
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

log = logging.getLogger(__name__)

# Fixed namespace for deterministic point IDs. Deriving a point's UUID from
# (source, document_id, chunk_index) makes upsert idempotent: re-running the
# pipeline replaces a chunk's point in place instead of appending a duplicate
# under a fresh random UUID. The namespace value itself is arbitrary but must
# never change, or a later run would generate different IDs and leave the old
# points alongside their replacements.
_POINT_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")


DocumentKey = tuple[str, str]


def document_key(source: str, document_id: str) -> DocumentKey:
    """Build the compound identity used by incremental indexing.

    Args:
        source: Stable source identifier supplied by the document adapter.
        document_id: Identifier unique within that source.

    Returns:
        A ``(source, document_id)`` tuple. Using a tuple avoids delimiter
        ambiguity when either component contains punctuation.
    """
    return source, document_id


def content_hash(text: str) -> str:
    """Calculate the deterministic hash used for change detection.

    Args:
        text: Complete document text. ``None`` is tolerated at runtime and
            treated as an empty string for backward compatibility.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def point_id(source: str, document_id: str, chunk_index: int) -> str:
    """Build the deterministic Qdrant id for one document chunk.

    UUID5 makes repeated indexing idempotent: the same identity and chunk index
    overwrite the same point, while distinct chunks receive distinct ids. The
    namespace and input serialization must remain stable to preserve that
    property for existing collections.

    Args:
        source: Stable source identifier.
        document_id: Identifier unique within the source.
        chunk_index: Zero-based chunk position.

    Returns:
        UUID string accepted by Qdrant as a point id.
    """
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{source}::{document_id}::{chunk_index}"))


def _collection_vector_size(qdrant: QdrantClient, name: str) -> int | None:
    """Read the size of an unnamed, single-vector collection.

    Args:
        qdrant: Client connected to the target Qdrant instance.
        name: Collection to inspect.

    Returns:
        Configured vector size, or ``None`` when the collection is missing, the
        request fails, or the collection uses an unsupported configuration such
        as named vectors.
    """
    try:
        info = qdrant.get_collection(name)
    except Exception:
        return None
    try:
        vectors = info.config.params.vectors
        # Unnamed single-vector collections expose ``.size`` directly; named-vector
        # configs are a dict. This pipeline only uses the unnamed form.
        return getattr(vectors, "size", None)
    except AttributeError:
        return None


def setup_collection(qdrant: QdrantClient, name: str, vector_dim: int) -> None:
    """
    Ensure a Qdrant collection exists with the given vector configuration,
    recreating it if its stored dimensionality no longer matches.

    Behaviour:
      * Missing collection or unreadable size → attempt to create it.
      * Exists with matching dim      → leave untouched (fast path).
      * Exists with a *different* dim → delete and recreate. This self-heals a
        model/backend switch (e.g. bge-large 1024 → bge-base 768). Without it,
        upserts would fail with a dimension-mismatch error against the stale
        collection, and a colleague would have to drop it by hand.

    Collections use cosine distance.

    Args:
        qdrant:     An initialised QdrantClient connected to your Qdrant
                    instance.
        name:       Name of the collection to create.  Must be a non-empty
                    string.  Qdrant collection names are case-sensitive.
        vector_dim: The dimensionality of vectors that will be stored in this
                    collection. Must match the active embedder.

    Returns:
        None.

    Raises:
        qdrant_client.http.exceptions.UnexpectedResponse: If the Qdrant
            instance returns an error (e.g. invalid config, auth failure).
    """
    existing_size = _collection_vector_size(qdrant, name)

    if existing_size is None:
        qdrant.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )
        log.info("Created collection '%s'  (dim=%d, distance=Cosine)", name, vector_dim)
        return

    if existing_size != vector_dim:
        log.warning(
            "Collection '%s' has dim=%d but the embedder produces dim=%d — "
            "recreating it (all previously stored vectors are dropped).",
            name,
            existing_size,
            vector_dim,
        )
        qdrant.delete_collection(collection_name=name)
        qdrant.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )
        log.info("Recreated collection '%s'  (dim=%d, distance=Cosine)", name, vector_dim)
        return

    log.info("Collection '%s' already exists (dim=%d) — reusing", name, vector_dim)


def load_existing_hashes(
    qdrant: QdrantClient,
    collection: str,
    payload_fields: PayloadFields = DEFAULT_PAYLOAD_FIELDS,
) -> dict[DocumentKey, str]:
    """
    Return stored content hashes keyed by document identity.

    Scrolls the collection, without vectors, and reads the payload fields needed
    to decide whether a document's text changed.

    Points without a ``content_hash`` payload are absent from the map, so their
    documents are treated as changed.

    Args:
        qdrant:     An initialised QdrantClient.
        collection: Collection to scan.
        payload_fields: Payload keys containing document identity and source.

    Returns:
        Mapping of ``document_key(source, document_id)`` to stored content hash.
        If scrolling fails, returns any hashes accumulated before the failure
        (usually an empty mapping).
    """
    hashes: dict[DocumentKey, str] = {}
    offset = None
    try:
        while True:
            points, offset = qdrant.scroll(
                collection_name=collection,
                with_payload=[
                    payload_fields.document_id,
                    payload_fields.source,
                    "content_hash",
                ],
                with_vectors=False,
                limit=1000,
                offset=offset,
            )
            for p in points:
                payload = p.payload or {}
                h = payload.get("content_hash")
                document_id = payload.get(payload_fields.document_id)
                source = payload.get(payload_fields.source)
                if h and document_id is not None and source is not None:
                    hashes[document_key(source, document_id)] = h
            if offset is None:
                break
    except Exception:
        # Treat a failed scan as having no additional known hashes. The pipeline
        # re-embeds entries absent from the partial result.
        return hashes

    log.info("Loaded %d existing document hashes from '%s'", len(hashes), collection)
    return hashes


def delete_document_points(
    qdrant: QdrantClient,
    collection: str,
    source: str,
    document_id: str,
    payload_fields: PayloadFields = DEFAULT_PAYLOAD_FIELDS,
) -> None:
    """
    Delete every point belonging to one document.

    Called for changed documents and documents removed from the source.
    Deterministic point IDs already make same-index chunks upsert-in-place, but a
    shorter new document produces fewer chunks. Deleting first prevents orphaned
    higher-index points from lingering.

    Args:
        qdrant: Client connected to the target Qdrant instance.
        collection: Collection containing the document.
        source: Stored source identifier.
        document_id: Stored document identifier.
        payload_fields: Payload keys used for document identity and source.

    Returns:
        None.

    Raises:
        Qdrant client errors are propagated to the caller.
    """
    qdrant.delete(
        collection_name=collection,
        points_selector=FilterSelector(
            filter=Filter(
                must=[
                    FieldCondition(
                        key=payload_fields.source, match=MatchValue(value=source)
                    ),
                    FieldCondition(
                        key=payload_fields.document_id,
                        match=MatchValue(value=document_id),
                    ),
                ]
            )
        ),
    )


def chunks_to_points(
    chunks: list[Chunk],
    embeddings: list[list[float]],
    payload_fields: PayloadFields = DEFAULT_PAYLOAD_FIELDS,
) -> list[PointStruct]:
    """
    Zip Chunk objects with their embedding vectors to produce Qdrant PointStructs.

    Each PointStruct carries:
      - A deterministic UUID5 id derived from (source, document_id,
        chunk_index), so re-running the pipeline upserts each chunk in place
        rather than appending a duplicate under a fresh random id.
      - The embedding vector (used for similarity search).
      - A payload dict with all metadata needed to reconstruct the source
        context after a search hit is returned, plus a ``content_hash`` used to
        skip unchanged documents on subsequent runs.

    Payload fields stored per point
    ────────────────────────────────
    document id  str  — Stored under ``payload_fields.document_id``.
    text         str  — The chunk text that was actually embedded.  Returned
                        verbatim in search results so callers do not need to
                        re-fetch the source file.
    chunk_index  int  — Zero-based position of this chunk within its document.
                        Useful for reconstructing the full text in order.
    total_chunks int  — Total number of chunks the document was split into. A
                        value of 1 means the full document fit within
                        MAX_TOKENS_IN_CHUNK and was not split.
    source       str  — Stored under ``payload_fields.source``.
    content_hash str  — SHA-256 of the full document text, shared by all
                        chunks for incremental indexing.

    Args:
        chunks:     Ordered list of Chunk instances.
        embeddings: Ordered list of embedding vectors in the same order as
                    ``chunks``.  Must be the same length as ``chunks``.
        payload_fields: Payload keys under which document identity and source
                        are stored. Chunk metadata is copied first, then standard
                        fields overwrite any colliding metadata keys.

    Returns:
        A list of PointStruct objects ready to pass to ``upsert_points``.

    Raises:
        ValueError: If ``chunks`` and ``embeddings`` have different lengths.
                    ``zip(strict=True)`` raises immediately rather than silently
                    truncating to the shorter list, which would misalign chunks
                    with the wrong embeddings.
    """
    return [
        PointStruct(
            id=point_id(chunk.source, chunk.document_id, chunk.chunk_index),
            vector=embedding,
            payload={
                **chunk.metadata,
                payload_fields.document_id: chunk.document_id,
                "text": chunk.text,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
                payload_fields.source: chunk.source,
                "content_hash": chunk.content_hash,
            },
        )
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]


def upsert_points(
    qdrant: QdrantClient,
    collection: str,
    points: list[PointStruct],
    batch_size: int = QDRANT_UPSERT_BATCH_SIZE,
) -> None:
    """
    Write all PointStructs into a Qdrant collection, split into batches.

    Splitting requests bounds request size and client/server memory use.

    Each batch is an upsert (insert-or-replace). ``chunks_to_points`` assigns
    deterministic UUIDs, so re-running the pipeline updates matching chunks in
    place rather than creating duplicates.

    Args:
        qdrant:      An initialised QdrantClient connected to your Qdrant
                     instance.
        collection:  Name of the target collection.  Must already exist —
                     call ``setup_collection`` first.
        points:      List of PointStruct objects to write.  Can be empty; an
                     empty list is a no-op.
        batch_size:  Maximum points in each upsert request. Defaults to
                     ``QDRANT_UPSERT_BATCH_SIZE``.

    Returns:
        None.

    Raises:
        qdrant_client.http.exceptions.UnexpectedResponse: If Qdrant rejects
            the request (e.g. vector dimension mismatch, missing collection).
    """
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        qdrant.upsert(collection_name=collection, points=batch)
        log.info(
            "  Upserted %d / %d points → '%s'",
            min(i + batch_size, len(points)),
            len(points),
            collection,
        )
