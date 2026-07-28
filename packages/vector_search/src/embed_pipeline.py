"""Generic streaming pipeline from documents to a Qdrant collection.

Document discovery, parsing, and text selection belong to source adapters
outside this package. This module accepts an iterable of ``Document`` objects
and handles chunking, incremental embedding, upserting, and pruning.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator

import tiktoken
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from packages.arc.src.vector_search.config import (
    CHUNK_OVERLAP,
    MAX_TOKENS_IN_CHUNK,
    MEM_CEILING_MB,
    PIPELINE_CHUNK_BATCH_SIZE,
    QDRANT_API_KEY,
    QDRANT_UPSERT_BATCH_SIZE,
    QDRANT_URL,
    TOKENIZER_MODEL,
)
from packages.arc.src.vector_search.embedder import Embedder, get_lazy_embedder
from packages.arc.src.vector_search.models import (
    DEFAULT_PAYLOAD_FIELDS,
    Chunk,
    Document,
    PayloadFields,
)
from packages.arc.src.vector_search.qdrant_store import (
    DocumentKey,
    chunks_to_points,
    content_hash,
    delete_document_points,
    document_key,
    load_existing_hashes,
    setup_collection,
    upsert_points,
)
from packages.general_utilities.resource_monitor import ResourceMonitor

log = logging.getLogger(__name__)


def _load_tokenizer() -> tiktoken.Encoding:
    """Load the tokenizer used to calculate chunk boundaries.

    Chunking deliberately uses one stable tokenizer independently of the active
    embedding backend. This keeps stored chunk boundaries deterministic when a
    caller changes between remote and local embedders, although a local model
    may count the resulting text differently.

    Returns:
        The tiktoken encoding registered for ``TOKENIZER_MODEL``.

    Raises:
        KeyError: If tiktoken does not recognize ``TOKENIZER_MODEL``.
    """
    return tiktoken.encoding_for_model(TOKENIZER_MODEL)


def chunk_text(
    text: str,
    enc: tiktoken.Encoding,
    max_tokens: int = MAX_TOKENS_IN_CHUNK,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping, token-bounded windows.

    The full text is encoded once. Text at or below ``max_tokens`` is returned
    unchanged as a single chunk. Longer text is traversed with windows of
    ``max_tokens`` and a stride of ``max_tokens - overlap``; decoding each
    window preserves context on both sides of a boundary.

    Args:
        text: Text to split.
        enc: Tokenizer used for both encoding and decoding.
        max_tokens: Maximum number of encoded tokens in each chunk.
        overlap: Number of tokens repeated between adjacent chunks. Must be
            strictly less than ``max_tokens``.

    Returns:
        Chunks in source order. The list always contains at least one element;
        empty input produces ``[""]``.

    Raises:
        ValueError: If ``overlap`` is greater than or equal to ``max_tokens``,
            which would make the window stride non-positive.
    """
    if overlap >= max_tokens:
        raise ValueError(
            f"overlap ({overlap}) must be less than max_tokens ({max_tokens}); "
            "otherwise the chunking stride is non-positive and would loop forever"
        )

    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunks.append(enc.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += max_tokens - overlap
    return chunks


def chunk_document(
    document: Document,
    enc: tiktoken.Encoding,
    hash_value: str | None = None,
) -> list[Chunk]:
    """Convert one document into indexable chunks.

    Every chunk retains the document identity, source, metadata, ordinal
    position, total chunk count, and the hash of the *complete* document text.
    The shared hash allows a later pipeline run to decide whether the document
    changed without reconstructing it from stored chunks.

    Args:
        document: Source-independent document to chunk.
        enc: Tokenizer used to enforce the chunk-size limit.
        hash_value: Optional precomputed hash of ``document.text``. The
            incremental pipeline supplies this after its skip check to avoid
            hashing the same text twice.

    Returns:
        Chunks in source order. A short or empty document produces one chunk.

    Raises:
        ValueError: If the configured overlap is not smaller than the chunk
            token limit.
    """
    document_hash = (
        hash_value if hash_value is not None else content_hash(document.text)
    )
    texts = chunk_text(document.text, enc)
    if len(texts) > 1:
        log.debug("  %r → %d chunks", document.document_id, len(texts))
    return [
        Chunk(
            document_id=document.document_id,
            text=text,
            chunk_index=index,
            total_chunks=len(texts),
            source=document.source,
            content_hash=document_hash,
            metadata=document.metadata,
        )
        for index, text in enumerate(texts)
    ]


def iter_chunks(
    documents: Iterable[Document],
    enc: tiktoken.Encoding | None = None,
) -> Iterator[Chunk]:
    """Stream chunks from a document iterable.

    Each document is fully chunked and yielded before the next document is
    requested, so the function does not materialize the complete source.

    Args:
        documents: Documents in the order they should be indexed.
        enc: Optional tokenizer to reuse. When omitted, the configured tokenizer
            is loaded once when iteration begins.

    Yields:
        Each document's chunks in document order and chunk-index order.

    Raises:
        Any exception raised by the document iterable, tokenizer, or
        ``chunk_document`` is propagated.
    """
    enc = enc or _load_tokenizer()
    for document in documents:
        yield from chunk_document(document, enc)


def _assert_qdrant_running(qdrant: QdrantClient) -> None:
    """Force a connection before model loading or document iteration begins.

    ``QdrantClient`` connects lazily. Calling ``get_collections`` here makes
    connection, authentication, and endpoint failures surface before the
    pipeline performs expensive embedding work.

    Args:
        qdrant: Configured Qdrant client.

    Returns:
        None.

    Raises:
        RuntimeError: If the preflight request fails. The original client
            exception is retained as the cause.
    """
    try:
        qdrant.get_collections()
    except Exception as exc:
        raise RuntimeError(f"Cannot reach Qdrant at {QDRANT_URL}.") from exc


def _embed_batch(
    embedder: Embedder,
    batch: list[Chunk],
    payload_fields: PayloadFields,
) -> list[PointStruct]:
    """Embed a chunk batch and convert the vectors to Qdrant points.

    Chunks are submitted in ascending text-length order so local inference
    microbatches contain similarly sized inputs and waste less work on padding.
    Because Qdrant payloads must remain aligned with their original chunks, the
    returned vectors are permuted back to caller order before point construction.

    Args:
        embedder: Backend used to encode chunk text.
        batch: Chunks to encode. An empty batch is accepted.
        payload_fields: Payload keys used for document identity and source.

    Returns:
        One Qdrant point per input chunk, in the same order as ``batch``.

    Raises:
        RuntimeError: If the backend returns a different number of vectors than
            the number of input chunks.
        ValueError: If point construction detects a chunk/vector length mismatch.
        Any backend exception is propagated unchanged.
    """
    if not batch:
        return []

    order = sorted(range(len(batch)), key=lambda index: len(batch[index].text))
    sorted_vectors = embedder.embed([batch[index].text for index in order])
    if len(sorted_vectors) != len(batch):
        raise RuntimeError(
            f"Embedding backend returned {len(sorted_vectors)} vectors for "
            f"{len(batch)} chunks; refusing to upsert misaligned points."
        )

    ranks = [0] * len(order)
    for slot, original_index in enumerate(order):
        ranks[original_index] = slot
    vectors = [sorted_vectors[rank] for rank in ranks]
    return chunks_to_points(batch, vectors, payload_fields)


def _upsert_full_batches(
    qdrant: QdrantClient,
    collection: str,
    points: list[PointStruct],
) -> None:
    """Upload every complete Qdrant-sized prefix of a mutable point buffer.

    Uploaded points are deleted from ``points`` in place. A remainder smaller
    than ``QDRANT_UPSERT_BATCH_SIZE`` stays buffered so a later embedding batch
    can fill the request rather than sending many small upserts.

    Args:
        qdrant: Client receiving the points.
        collection: Existing target collection.
        points: Mutable pending-point buffer.

    Returns:
        None. ``points`` is mutated in place.

    Raises:
        Qdrant client errors raised by ``upsert_points`` are propagated.
    """
    while len(points) >= QDRANT_UPSERT_BATCH_SIZE:
        upsert_points(
            qdrant,
            collection,
            points[:QDRANT_UPSERT_BATCH_SIZE],
            batch_size=QDRANT_UPSERT_BATCH_SIZE,
        )
        del points[:QDRANT_UPSERT_BATCH_SIZE]


def _stream_embed_upsert(
    qdrant: QdrantClient,
    embedder: Embedder,
    documents: Iterable[Document],
    collection: str,
    existing_hashes: dict[DocumentKey, str],
    payload_fields: PayloadFields = DEFAULT_PAYLOAD_FIELDS,
) -> dict[str, int]:
    """Stream, compare, chunk, embed, upsert, and prune documents.

    For every document, the pipeline compares the hash of its complete text with
    the hash already stored in Qdrant:

    1. Matching hashes skip chunking and embedding entirely.
    2. Changed documents have all old points deleted before replacement, which
       prevents stale high-index chunks when a document becomes shorter.
    3. New and changed chunks accumulate to ``PIPELINE_CHUNK_BATCH_SIZE`` for
       embedding. Resulting points independently accumulate into
       ``QDRANT_UPSERT_BATCH_SIZE`` requests.
    4. After the source is exhausted, stored document identities that were not
       observed are deleted.

    Because of step 4, ``documents`` must represent the complete source of truth
    for ``collection``. Passing a partial source intentionally treats omitted
    documents as deletions.

    Args:
        qdrant: Client connected to the target Qdrant instance.
        embedder: Backend used for new and changed documents.
        documents: Complete iterable of documents that should remain indexed.
        collection: Existing target collection.
        existing_hashes: Hashes loaded from the collection, keyed by
            ``(source, document_id)``.
        payload_fields: Payload keys used for document identity and source.

    Returns:
        Counts for ``embedded_documents``, ``skipped_documents``,
        ``embedded_chunks``, and ``deleted_documents``.

    Raises:
        RuntimeError: If an embedding batch returns the wrong number of vectors.
        Any document-source, tokenizer, embedding, or Qdrant exception is
        propagated.
    """
    enc = _load_tokenizer()
    chunk_batch: list[Chunk] = []
    pending_points: list[PointStruct] = []
    embedded_documents = 0
    skipped_documents = 0
    embedded_chunks = 0
    processed_keys: set[DocumentKey] = set()

    for document in documents:
        key = document_key(document.source, document.document_id)
        processed_keys.add(key)
        current_hash = content_hash(document.text)
        previous_hash = existing_hashes.get(key)

        if previous_hash == current_hash:
            skipped_documents += 1
            continue

        if previous_hash is not None:
            delete_document_points(
                qdrant,
                collection,
                document.source,
                document.document_id,
                payload_fields,
            )

        chunk_batch.extend(chunk_document(document, enc, current_hash))
        embedded_documents += 1

        if len(chunk_batch) >= PIPELINE_CHUNK_BATCH_SIZE:
            pending_points.extend(
                _embed_batch(embedder, chunk_batch, payload_fields)
            )
            embedded_chunks += len(chunk_batch)
            chunk_batch.clear()
            _upsert_full_batches(qdrant, collection, pending_points)

    pending_points.extend(_embed_batch(embedder, chunk_batch, payload_fields))
    embedded_chunks += len(chunk_batch)
    _upsert_full_batches(qdrant, collection, pending_points)
    upsert_points(
        qdrant,
        collection,
        pending_points,
        batch_size=QDRANT_UPSERT_BATCH_SIZE,
    )

    deleted_documents = 0
    for source, document_id in existing_hashes.keys() - processed_keys:
        delete_document_points(
            qdrant,
            collection,
            source,
            document_id,
            payload_fields,
        )
        deleted_documents += 1

    return {
        "embedded_documents": embedded_documents,
        "skipped_documents": skipped_documents,
        "embedded_chunks": embedded_chunks,
        "deleted_documents": deleted_documents,
    }


def _log_run_summary(resources: dict, counts: dict[str, int]) -> None:
    """Log work counts and resource measurements for one completed run.

    Args:
        resources: Result returned by ``ResourceMonitor.result``. It must contain
            peak/average memory, peak CPU, and elapsed-time values.
        counts: Counters returned by ``_stream_embed_upsert``.

    A warning is also emitted when ``MEM_CEILING_MB`` is configured and measured
    peak RSS exceeds that soft limit.

    Returns:
        None.
    """
    log.info(
        "Pipeline complete ✓  embedded_documents=%d skipped_documents=%d "
        "deleted_documents=%d embedded_chunks=%d "
        "| peak_rss=%.0fMB avg_rss=%.0fMB cpu_peak=%.0f%% elapsed=%.1fs",
        counts["embedded_documents"],
        counts["skipped_documents"],
        counts["deleted_documents"],
        counts["embedded_chunks"],
        resources["mem_peak_mb"],
        resources["mem_avg_mb"],
        resources["cpu_peak_pct"],
        resources["elapsed_seconds"],
    )
    if MEM_CEILING_MB and resources["mem_peak_mb"] > MEM_CEILING_MB:
        log.warning(
            "Peak RSS %.0fMB exceeded VECTOR_SEARCH_MEM_CEILING_MB=%d — consider "
            "lowering VECTOR_SEARCH_LOCAL_INFERENCE_BATCH_SIZE or using a "
            "smaller model.",
            resources["mem_peak_mb"],
            MEM_CEILING_MB,
        )


def run_pipeline(
    documents: Iterable[Document],
    collection: str,
    payload_fields: PayloadFields = DEFAULT_PAYLOAD_FIELDS,
) -> tuple[QdrantClient, Embedder]:
    """Execute the complete incremental indexing pipeline.

    The orchestration order is significant:

    1. Preflight Qdrant before loading a model or consuming the source.
    2. Construct a lazy embedder and ensure the collection uses its dimension.
       A dimension mismatch recreates the collection.
    3. Load stored hashes using the configured payload field names.
    4. Stream the complete source through skip, replace, upsert, and prune logic.
    5. Log resource usage and return the live client and embedder so callers may
       issue queries without rebuilding either object.

    The real embedding backend is initialized only when at least one document
    needs embedding. A fully unchanged run can finish without loading model
    weights or constructing the OpenAI client.

    Args:
        documents: Complete source of documents that should exist in the
            collection after the run. Omitting previously indexed documents
            deletes them.
        collection: Qdrant collection to create, reuse, or recreate.
        payload_fields: Payload keys used to persist document identity and
            source. Query consumers must use the same mapping.

    Returns:
        ``(qdrant, embedder)`` configured for the indexed collection.

    Raises:
        RuntimeError: If Qdrant preflight fails, or if an embedder returns the
            wrong number of vectors.
        ImportError: If the selected embedding backend dependency is unavailable.
        ValueError: If chunk configuration or source documents are invalid.
        Any source, embedding-backend, or Qdrant client exception is propagated.
    """
    with ResourceMonitor() as monitor:
        qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        _assert_qdrant_running(qdrant)

        embedder = get_lazy_embedder()
        setup_collection(qdrant, collection, embedder.vector_dim)
        existing_hashes = load_existing_hashes(
            qdrant,
            collection,
            payload_fields,
        )
        counts = _stream_embed_upsert(
            qdrant,
            embedder,
            documents,
            collection,
            existing_hashes,
            payload_fields,
        )

    _log_run_summary(monitor.result(), counts)
    return qdrant, embedder
