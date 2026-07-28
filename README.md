# tutorials.text-search-embedding-vectors

Learn how to create a document search pipeline with embedding vectors.

A small, self-contained tutorial that turns arbitrary documents into a
searchable [Qdrant](https://qdrant.tech/) collection: chunk text, embed it
(OpenAI or a local model), upsert the vectors, and run semantic search — with
incremental re-indexing so unchanged documents are skipped on later runs.

## How it works

```
documents ──▶ chunk ──▶ embed ──▶ upsert to Qdrant ──▶ search
             (tiktoken)  (OpenAI / fastembed / sentence-transformers)
```

- **Incremental**: each document's text is hashed; unchanged documents are
  skipped, changed ones are re-embedded, and documents dropped from the source
  are pruned from the collection.
- **Backend-agnostic**: uses OpenAI when `OPENAI_API_KEY` is set, otherwise a
  local model (fastembed/ONNX by default, no API key required).
- **Deterministic point IDs**: re-running upserts each chunk in place instead of
  creating duplicates.

## Layout

```
packages/
  vector_search/
    src/
      config.py         # env-backed settings
      models.py         # Document / Chunk / PayloadFields
      embedder.py       # OpenAI / fastembed / sentence-transformers backends
      qdrant_store.py   # collection setup, points, hashing, upsert
      embed_pipeline.py # run_pipeline(): chunk → embed → upsert → prune
      search.py         # search() / print_results()
    test/               # unittest-style suites (run with pytest)
  utilities/
    src/resource_monitor.py  # psutil CPU/RSS sampler used by the pipeline
scripts/
  run_demo.py           # embeds 100 generated docs, then searches
docker-compose.yaml     # local Qdrant service
.env.example            # every configurable environment variable, documented
```

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python package/venv manager)
- Python 3.14 (pinned in `.python-version`; `requires-python = ">=3.12"`)
- Docker (for the local Qdrant instance)

## Setup

```bash
# Install dependencies into a 3.14 virtualenv (.venv) from uv.lock
uv sync

# Start the local Qdrant instance (http://localhost:6333)
docker compose up -d qdrant
```

Configuration is via environment variables — see [.env.example](.env.example)
for every setting and its default. Nothing needs editing to run locally.

## Run the tests

```bash
uv run pytest packages/vector_search/test -v
```

All suites are `unittest`-based and require no running Qdrant or model — Qdrant
and the embedder are stubbed. (`pyproject.toml` configures pytest to discover the
`*_tests.py` files and put the repo root on the import path.)

## Run the demo

Embeds 100 generated documents into Qdrant and runs a few semantic queries. The
local `bge-small` model (~130 MB, downloaded on first use) keeps it fast and
key-free:

```bash
PYTHONPATH=. VECTOR_SEARCH_LOCAL_MODEL=BAAI/bge-small-en-v1.5 \
  uv run python scripts/run_demo.py
```

`PYTHONPATH=.` makes the `packages.*` imports resolve when running a script
directly (pytest gets this from `pyproject.toml`).

## Use it in code

```python
from packages.vector_search.src.embed_pipeline import run_pipeline
from packages.vector_search.src.models import Document
from packages.vector_search.src.search import search, print_results

docs = [
    Document(document_id="doc-1", text="...", source="notes/doc-1.md"),
    # ...
]

# Chunk, embed, and upsert; returns a live client + embedder for querying.
qdrant, embedder = run_pipeline(docs, collection="my_docs")

results = search("a natural-language question", "my_docs", qdrant, embedder, top_k=5)
print_results(results, "my_docs")
```

`run_pipeline` treats `docs` as the complete source of truth for the collection:
omitting a previously indexed document deletes it. Query with the **same** model
and configuration used to build the collection — matching vector dimensions
alone does not guarantee a compatible embedding space.

## Stopping Qdrant

```bash
docker compose down       # stop the container (data persists in the volume)
docker compose down -v    # also delete the volume (wipes all collections)
```
