"""Throwaway driver: embed 100 generated fake documents into Qdrant, then search.

Run from the repo root with the light local backend:
    VECTOR_SEARCH_LOCAL_MODEL=BAAI/bge-small-en-v1.5 uv run python <this file>
"""

from __future__ import annotations

import logging
import random

from packages.vector_search.src.embed_pipeline import run_pipeline
from packages.vector_search.src.models import Document
from packages.vector_search.src.search import search, print_results

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# Deterministic generation (no reliance on wall-clock/random seeds elsewhere).
rng = random.Random(1234)

TOPICS = [
    ("Payment processing", "Handles card authorization, capture, refunds, and "
     "reconciliation across multiple gateways with idempotent retries."),
    ("Vector databases", "Store high-dimensional embeddings and serve approximate "
     "nearest-neighbour queries with HNSW indexes and payload filtering."),
    ("Kubernetes networking", "Pod-to-pod routing, services, ingress controllers, "
     "and network policies that isolate workloads across namespaces."),
    ("Coffee roasting", "Green beans develop flavour through Maillard reactions and "
     "first crack; roast profiles balance acidity, body, and sweetness."),
    ("Tax accounting", "Depreciation schedules, deferred liabilities, and quarterly "
     "estimated payments that reconcile against the general ledger."),
    ("Marine biology", "Coral reefs host symbiotic algae; bleaching events follow "
     "sustained temperature anomalies that disrupt the symbiosis."),
    ("Distributed consensus", "Raft and Paxos elect a leader and replicate a log so "
     "a cluster agrees on ordered state despite node failures."),
    ("Home baking", "Gluten development, hydration ratios, and fermentation time "
     "determine the crumb structure of a sourdough loaf."),
    ("Wildfire ecology", "Some pine cones are serotinous and only release seeds "
     "after fire, so periodic burns regenerate the forest canopy."),
    ("GPU scheduling", "Warps execute in lockstep; occupancy and memory coalescing "
     "govern how efficiently kernels saturate streaming multiprocessors."),
]

FILLER = (
    "This section elaborates on the practical trade-offs, common failure modes, "
    "operational tuning, and the way the concepts connect to adjacent systems in "
    "a production setting. Concrete examples illustrate the boundaries."
)


def make_document(i: int) -> Document:
    """Build one fake Document of at most 500 words."""
    title, seed = TOPICS[i % len(TOPICS)]
    variant = i // len(TOPICS)
    words: list[str] = f"{title} note {i}. {seed}".split()
    # Pad with filler sentences up to a random length, hard-capped under 500 words.
    target = rng.randint(120, 460)
    while len(words) < target:
        words.extend(FILLER.split())
    words = words[:target]  # guarantee <= 500 words
    text = " ".join(words)
    return Document(
        document_id=f"doc-{i:03d}",
        text=text,
        source=f"faux/{title.lower().replace(' ', '_')}.md",
        metadata={"topic": title, "variant": variant, "word_count": len(words)},
    )


def main() -> None:
    documents = [make_document(i) for i in range(100)]
    max_words = max(d.metadata["word_count"] for d in documents)
    print(f"Generated {len(documents)} documents; max words in any doc = {max_words}")
    assert max_words <= 500, "a document exceeded 500 words"

    collection = "demo_documents"
    qdrant, embedder = run_pipeline(documents, collection)

    info = qdrant.get_collection(collection)
    print("\n================ RESULT ================")
    print(f"collection      : {collection}")
    print(f"vector_dim      : {embedder.vector_dim}")
    print(f"points in store : {info.points_count}")

    for query in ["how do transactions get authorized and refunded?",
                  "leader election in a cluster of unreliable machines",
                  "why do forests need fire to regrow"]:
        print(f"\nQUERY: {query}")
        results = search(query, collection, qdrant, embedder, top_k=3)
        print_results(results, collection)


if __name__ == "__main__":
    main()
