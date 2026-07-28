"""
config.py
─────────
Environment-backed settings shared by the vector-search modules.

The default OpenAI model is defined here. Backend selection and effective
vector dimensionality are resolved in ``embedder.py``.

Constants
─────────
QDRANT_URL
    HTTP address of the Qdrant instance.  Read from the ``QDRANT_URL``
    environment variable, defaulting to ``http://localhost:6333`` for the
    local Docker container (``docker compose up -d qdrant``).  To target a
    Qdrant Cloud cluster, set ``QDRANT_URL`` to the cluster URL from the
    console, e.g. ``https://<cluster>.cloud.qdrant.io:6333``.

QDRANT_API_KEY
    Optional API key, read from the ``QDRANT_API_KEY`` environment variable.
    Leave unset (``None``) for the local Docker container, which needs no auth;
    set it to the cluster key when targeting Qdrant Cloud.  Embeddings are
    still computed client-side either way — only the connection target changes.

TOKENIZER_MODEL
    The model name passed to ``tiktoken.encoding_for_model`` when chunking.
    Chunking uses this tokenizer even when a local embedding model is active;
    local backends may tokenize the resulting text differently.

MAX_TOKENS_IN_CHUNK
    Maximum number of tokens allowed in a single chunk.  Texts longer than
    this will be split with CHUNK_OVERLAP tokens of context carried forward.
    The limit is measured with ``TOKENIZER_MODEL`` and is not a guarantee about
    a differently tokenized local model's input length.

CHUNK_OVERLAP
    Number of tokens repeated at the start of each new chunk from the tail of
    the previous one.  Overlap preserves cross-boundary context so that a
    sentence split across two chunks can still be retrieved by either half.

PIPELINE_CHUNK_BATCH_SIZE
    How many chunks the streaming pipeline materialises before asking the
    selected backend to embed them. This bounds Python-side chunk/vector/point
    buffering, but does not control a backend's inference/request batch size.

LOCAL_INFERENCE_BATCH_SIZE
    Microbatch size used by the local ONNX and sentence-transformers backends.
    This is the primary control for native inference workspace memory.

OPENAI_REQUEST_BATCH_SIZE
    Number of texts sent in one OpenAI embeddings API request. It is independent
    of local inference and Qdrant upload batching.

EMBED_THREADS
    Cap on the number of threads the local ONNX embedding backend (fastembed)
    may use, read from VECTOR_SEARCH_EMBED_THREADS.  ``None`` (the default, when
    the env var is unset or 0) lets the backend decide.

MEM_CEILING_MB
    Optional soft ceiling, read from VECTOR_SEARCH_MEM_CEILING_MB.  When set and
    the ResourceMonitor's measured peak RSS exceeds it, the pipeline logs a
    warning so runaway memory is visible rather than silent.  ``None`` disables
    the check.

QDRANT_UPSERT_BATCH_SIZE
    How many Qdrant PointStructs are written in a single upsert call.  Larger
    batches are more efficient but increase peak memory use. Points from
    multiple pipeline batches are accumulated to fill each request.

VECTOR_SEARCH_EMBED_BATCH_SIZE
    Deprecated compatibility fallback for the pipeline, local inference, and
    OpenAI request batch sizes. Any explicit new setting takes precedence.

"""

import os

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")  # None for local Docker (no auth)

# Default used when VECTOR_SEARCH_OPENAI_MODEL is unset.
DEFAULT_OPENAI_EMBED_MODEL = "text-embedding-3-small"

TOKENIZER_MODEL = DEFAULT_OPENAI_EMBED_MODEL  # used by tiktoken for chunk sizing
MAX_TOKENS_IN_CHUNK = 512
CHUNK_OVERLAP = 50


def _positive_int_env(name: str, default: str, fallback: str | None = None) -> int:
    """Read and validate a positive integer environment setting.

    Args:
        name: Environment variable to read.
        default: String value used when neither ``name`` nor ``fallback`` is
            configured.
        fallback: Optional compatibility value used only when ``name`` is absent.

    Returns:
        Parsed positive integer.

    Raises:
        ValueError: If the selected value is not an integer or is not positive.
    """
    raw = os.environ.get(name, fallback if fallback is not None else default)
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero, got {value}")
    return value


# Before these concerns were separated, one value controlled pipeline buffering,
# local inference, and OpenAI requests. Keep it as a fallback so existing shell
# configurations remain compatible while allowing each new setting to override it.
_legacy_embed_batch_size = os.environ.get("VECTOR_SEARCH_EMBED_BATCH_SIZE")

PIPELINE_CHUNK_BATCH_SIZE = _positive_int_env(
    "VECTOR_SEARCH_PIPELINE_CHUNK_BATCH_SIZE", "100", _legacy_embed_batch_size
)
LOCAL_INFERENCE_BATCH_SIZE = _positive_int_env(
    "VECTOR_SEARCH_LOCAL_INFERENCE_BATCH_SIZE", "16", _legacy_embed_batch_size
)
OPENAI_REQUEST_BATCH_SIZE = _positive_int_env(
    "VECTOR_SEARCH_OPENAI_REQUEST_BATCH_SIZE", "100", _legacy_embed_batch_size
)
QDRANT_UPSERT_BATCH_SIZE = _positive_int_env(
    "VECTOR_SEARCH_QDRANT_UPSERT_BATCH_SIZE", "200"
)

# 0 / unset → let the backend choose its own thread count.
EMBED_THREADS = int(os.environ.get("VECTOR_SEARCH_EMBED_THREADS", "0")) or None

# Optional soft peak-RSS ceiling; unset → no warning.
_mem_ceiling = os.environ.get("VECTOR_SEARCH_MEM_CEILING_MB")
MEM_CEILING_MB = int(_mem_ceiling) if _mem_ceiling else None
