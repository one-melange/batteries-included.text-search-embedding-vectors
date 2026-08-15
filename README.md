# batteries-included.text-search-embedding-vectors

Create a document search pipeline with embedding vectors and compare how
tokenizer/chunker choices prepare documents before indexing.

The goal is to be up and running in under 30 minutes, hence why this is part of my "batteries-included" series.

This is a small, self-contained projects that turns arbitrary documents into a
searchable [Qdrant](https://qdrant.tech/) collection: chunk text, embed it
(OpenAI or a local model), upsert the vectors, and run semantic search — with
incremental re-indexing so unchanged documents are skipped on later runs.

There are also plenty of knobs you can adjust and play with, helping you see how your search quality and relevance is affected. The SolidJS comparison UI makes the preparation stage visible: it streams decoded tokens first, then the resulting chunks.

## How it works

```
documents ──▶ chunk ──▶ embed ──▶ upsert to Qdrant ──▶ search
             (tiktoken)  (OpenAI / fastembed / sentence-transformers)

document ──▶ tokenize ──▶ chunk ──▶ compare decoded results in the browser
              × 4         × 4
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
      document_preparation/ # tokenizer/chunker adapters + WebSocket API
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
frontend/               # SolidJS Document Preparation application
docker-compose.yaml     # local Qdrant service
.env.example            # every configurable environment variable, documented
```

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python package/venv manager)
- Python 3.14 (pinned in `.python-version`; `requires-python = ">=3.12"`)
- Docker (for the local Qdrant instance)
- Node.js 24 and pnpm 11 (for the comparison frontend)

## Setup

```bash
# Install dependencies into a 3.14 virtualenv (.venv) from uv.lock
uv sync

# Install the SolidJS frontend
cd frontend
pnpm install --frozen-lockfile
cd ..

# Start the local Qdrant instance (http://localhost:6333)
docker compose up -d qdrant
```

Configuration is via environment variables — see [.env.example](.env.example)
for every setting and its default. Nothing needs editing to run locally.

## Run the tests

```bash
uv run pytest packages/vector_search/test -v

cd frontend
pnpm test
pnpm build
```

All suites are `unittest`-based and require no running Qdrant or model — Qdrant
and the embedder are stubbed. (`pyproject.toml` configures pytest to discover the
`*_tests.py` files and put the repo root on the import path.)

The run is **hermetic** — no network access needed. The chunker's `tiktoken`
`cl100k_base` vocabulary is vendored under
`packages/vector_search/test/_fixtures/tiktoken_cache`, and `conftest.py` points
`TIKTOKEN_CACHE_DIR` at it (set the env var yourself to override).

## Document Preparation UI

The `/document-preparation` page compares up to four pairs at once from this
16-pair matrix:

| Tokenizers | Chunkers |
| --- | --- |
| tiktoken (`cl100k_base`) | LangChain `RecursiveCharacterTextSplitter` |
| Hugging Face Tokenizers (`bert-base-uncased`) | semchunk heuristic chunker |
| quicktok (`cl100k_base`) | Chonkie `SemanticChunker` |
| SentencePiece (`google-t5/t5-small`) | LlamaIndex `SentenceSplitter` |

Start the Python API and frontend in separate terminals:

```bash
uv run uvicorn \
  packages.vector_search.src.document_preparation.api:app \
  --reload
```

```bash
cd frontend
pnpm dev
```

Open `http://localhost:5173/document-preparation`. Paste text or load a `.txt`
or `.md` file, select one to four unique pairs, and run the comparison. Each
card updates twice: decoded document tokens appear as soon as tokenization
finishes, followed by decoded tokens grouped inside every chunk.

Hugging Face, SentencePiece, and Chonkie's local semantic model download their
artifacts on first use and reuse the local cache afterward. No document text is
persisted by the API. See `.env.example` to change the representative models,
provide a local SentencePiece model, or configure browser origins.

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
