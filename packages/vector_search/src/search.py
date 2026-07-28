"""Embed queries and search a Qdrant vector collection.

The query embedder must use the same model and configuration as the index.
Matching dimensions alone do not guarantee a compatible embedding space.
"""

from __future__ import annotations

import logging

from qdrant_client import QdrantClient

from packages.vector_search.src.embedder import Embedder
from packages.vector_search.src.models import (
    DEFAULT_PAYLOAD_FIELDS,
    PayloadFields,
)

log = logging.getLogger(__name__)


def search(
    query: str,
    collection: str,
    qdrant: QdrantClient,
    embedder: Embedder,
    top_k: int = 5,
    payload_fields: PayloadFields = DEFAULT_PAYLOAD_FIELDS,
) -> list[dict]:
    """
    Embed a natural-language query and return the most similar chunks from a
    Qdrant collection.

    The query is embedded and passed to Qdrant's ``query_points`` API. The
    collection's configured distance metric determines scoring.

    Result structure
    ────────────────
    Each result dict contains the following keys:

        score        float  Qdrant score rounded to four decimal places.
                            Higher is more similar for the cosine collection
                            created by this package.

        document_id  str    Identity supplied by the document source.

        chunk_index  int    Zero-based position of this chunk within its
                            document. If ``total_chunks`` is 1 the chunk
                            represents the full description.

        total_chunks int    Total number of chunks the source document was split
                            into.  Together with ``chunk_index`` this allows
                            callers to determine whether a match is from the
                            beginning, middle, or end of a long description.

        text         str    The actual chunk text that was embedded.  This is
                            the content to display or pass to an LLM for
                            further processing.

        source       str    Source identifier supplied with the document.

    Args:
        query:      Natural-language search string.  It is embedded as-is, so
                    phrase it similarly to how the stored descriptions are
                    written for best results.  Typically 1–3 sentences.
        collection: Name of a Qdrant collection built with the same embedding
                    model and configuration.
        qdrant:     An initialised QdrantClient connected to your Qdrant
                    instance.
        embedder:   Embedder configured identically to the one used to build
                    the collection.
        top_k:      Maximum number of results to return.  Qdrant will return
                    fewer results if the collection contains fewer than
                    ``top_k`` points.  Default is 5.
        payload_fields: Payload keys used by the collection for document
                    identity and source.

    Returns:
        A list of up to ``top_k`` result dicts, sorted by descending score
        (most similar first).

    Raises:
        Embedding and Qdrant client errors are propagated to the caller.

    Example
    ───────
    >>> results = search("leveraged yield strategy", "documents",
    ...                  qdrant, embedder, top_k=3)
    >>> for r in results:
    ...     print(r["score"], r["document_id"])
    """
    q_vector = embedder.embed([query])[0]

    # ``query_points`` returns a response wrapper; scored hits are in ``.points``.
    hits = qdrant.query_points(
        collection_name=collection,
        query=q_vector,
        limit=top_k,
        with_payload=True,
    ).points

    results = []
    for hit in hits:
        # Qdrant can return payload=None for points indexed without metadata.
        payload = hit.payload or {}
        results.append(
            {
                "score":        round(hit.score, 4),
                "document_id":  payload.get(payload_fields.document_id),
                "chunk_index":  payload.get("chunk_index"),
                "total_chunks": payload.get("total_chunks"),
                "text":         payload.get("text"),
                "source":       payload.get(payload_fields.source),
            }
        )
    return results


def print_results(results: list[dict], collection: str) -> None:
    """
    Pretty-print a list of search result dicts to stdout.

    Displays one result per line with its cosine score, document id, chunk
    position, and the first 160 characters of the matched text.  Useful for
    interactive testing from the CLI or a REPL.

    Args:
        results:    Result dicts returned by ``search()`` for points carrying
                    this package's standard payload fields. An empty list
                    prints only the collection header.
        collection: Collection name used as the section header in the output.

    Returns:
        None. Output is written to stdout.

    Raises:
        KeyError: If a result is missing a standard result field.
        TypeError: If score or chunk-position values cannot be formatted.
    """
    print(f"\n── {collection} ─────────────────────────────")
    for r in results:
        chunk_label = (
            f"chunk {r['chunk_index'] + 1}/{r['total_chunks']}"
            if r["total_chunks"] > 1
            else "single chunk"
        )
        print(f"  [{r['score']:.4f}] {r['document_id']}  ({chunk_label})")
        text    = r["text"] or ""
        preview = text[:160].replace("\n", " ")
        print(f"    {preview}{'…' if len(text) > 160 else ''}")
