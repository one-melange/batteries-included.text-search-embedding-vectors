"""Embedding backends and environment-based backend selection.

``get_embedder`` selects OpenAI when ``OPENAI_API_KEY`` is set. Otherwise it
uses the configured local runtime: fastembed/ONNX by default, or
sentence-transformers when ``VECTOR_SEARCH_LOCAL_BACKEND`` requests it.
When the configured dimension is available as metadata, ``get_lazy_embedder``
defers client/model construction until the first call to ``embed``.

Indexing and querying must use the same model and configuration, not merely
vectors with the same dimension, because different models occupy different
embedding spaces.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
import time

from packages.arc.src.vector_search.config import (
    DEFAULT_OPENAI_EMBED_MODEL,
    EMBED_THREADS,
    LOCAL_INFERENCE_BATCH_SIZE,
    OPENAI_REQUEST_BATCH_SIZE,
)

log = logging.getLogger(__name__)

# ── Constants shared between the backends ─────────────────────────────────────

# Resolve the collection dimension from known model metadata or an explicit
# override. The override is also sent to OpenAI for models that support shortened
# embeddings.
_OPENAI_MODEL = os.getenv("VECTOR_SEARCH_OPENAI_MODEL", DEFAULT_OPENAI_EMBED_MODEL)
_KNOWN_OPENAI_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def _resolve_openai_dim() -> tuple[int, bool]:
    """
    Determine the OpenAI embedding dimension and whether it was set explicitly.

    Returns a ``(dim, is_explicit_override)`` pair.  ``is_explicit_override`` is
    True only when VECTOR_SEARCH_OPENAI_DIM was provided, in which case the
    dimension is passed through to the embeddings API so the returned vectors
    actually match (OpenAI v3 models support shortening via the ``dimensions``
    request parameter).

    Returns:
        The effective vector dimension and whether it came from
        ``VECTOR_SEARCH_OPENAI_DIM``.

    Raises:
        ValueError: If ``VECTOR_SEARCH_OPENAI_DIM`` is set but is not a positive
            integer.
    """
    override = os.getenv("VECTOR_SEARCH_OPENAI_DIM")
    if override:
        try:
            dim = int(override)
            if dim <= 0:
                raise ValueError("Dimension must be a positive integer.")
            return dim, True
        except ValueError as err:
            raise ValueError(
                f"Invalid VECTOR_SEARCH_OPENAI_DIM {override!r}: must be a positive integer."
            ) from err

    known = _KNOWN_OPENAI_DIMS.get(_OPENAI_MODEL.lower())
    if known is None:
        log.warning(
            "Unknown OpenAI embedding model %r — assuming dim=1536. Set "
            "VECTOR_SEARCH_OPENAI_DIM to the correct value if this is wrong.",
            _OPENAI_MODEL,
        )
        return 1536, False

    return known, False


_OPENAI_DIM, _OPENAI_DIM_EXPLICIT = _resolve_openai_dim()

# The local model is shared by both local runtimes. Its dimension is read from
# known metadata for lazy setup, then verified when the backend loads.
_LOCAL_MODEL = os.getenv("VECTOR_SEARCH_LOCAL_MODEL", "BAAI/bge-large-en-v1.5")
_KNOWN_SENTENCE_TRANSFORMER_DIMS = {
    "baai/bge-large-en-v1.5": 1024,
    "baai/bge-base-en-v1.5": 768,
    "baai/bge-small-en-v1.5": 384,
}

# Which local runtime to use when no OPENAI_API_KEY is set. "fastembed" (default)
# runs the model via ONNX without torch. Set
# VECTOR_SEARCH_LOCAL_BACKEND=sentence-transformers to use the torch backend.
_LOCAL_BACKEND = os.getenv("VECTOR_SEARCH_LOCAL_BACKEND", "fastembed").strip().lower()


# ─────────────────────────────────────────────────────────────────────────────
# Abstract interface
# ─────────────────────────────────────────────────────────────────────────────


class Embedder(ABC):
    """Common contract implemented by every embedding backend.

    Pipeline and search code depend only on ``vector_dim`` and ``embed``.
    Implementations must return exactly one vector per input text, preserve input
    order, and keep the vector dimension stable for their lifetime.
    """

    @property
    @abstractmethod
    def vector_dim(self) -> int:
        """
        The fixed dimensionality of every vector produced by this backend.

        This value is used when creating a Qdrant collection.  It must be
        consistent across every call to ``embed`` made against a given
        collection — mixing embedders with different dimensionalities will
        cause Qdrant to reject upsert or search requests.

        Returns:
            Number of float values in every vector returned by ``embed``.
        """

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Produce one embedding vector per input string.

        Args:
            texts: Strings accepted by the selected backend. Pipeline inputs
                have already been bounded by ``MAX_TOKENS_IN_CHUNK``.

        Returns:
            A list of float vectors in the same order as ``texts``.  Every
            vector has length equal to ``self.vector_dim``.

        Raises:
            Backend-specific validation, authentication, network, or inference
            errors are propagated to the caller.
        """


class LazyEmbedder(Embedder):
    """Proxy that defers construction of a heavyweight embedding backend.

    Collection setup can read ``vector_dim`` without loading model weights or
    constructing a remote API client. The first call to ``embed`` creates the
    real backend and verifies that its reported dimension still matches the
    metadata used during collection setup.
    """

    def __init__(self, factory: Callable[[], Embedder], vector_dim: int) -> None:
        """Initialize a lazy backend proxy.

        Args:
            factory: Zero-argument callable that constructs the real backend.
            vector_dim: Expected dimension known before backend construction.
        """
        self._factory = factory
        self._vector_dim = vector_dim
        self._instance: Embedder | None = None

    @property
    def vector_dim(self) -> int:
        """Return the configured dimension without loading the backend.

        Returns:
            Expected vector dimension supplied at construction.
        """
        return self._vector_dim

    @property
    def is_loaded(self) -> bool:
        """Return whether the heavyweight backend has been constructed.

        Returns:
            ``True`` after the first successful backend construction.
        """
        return self._instance is not None

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Load the real backend on demand and delegate embedding.

        Args:
            texts: Input strings passed unchanged to the real backend.

        Returns:
            Vectors returned by the real backend in input order.

        Raises:
            RuntimeError: If the constructed backend reports a dimension that
                differs from the dimension supplied to this proxy.
            Any backend-construction or embedding exception is propagated.
        """
        if self._instance is None:
            instance = self._factory()
            if instance.vector_dim != self._vector_dim:
                raise RuntimeError(
                    "Embedding backend dimension changed during lazy load: "
                    f"expected {self._vector_dim}, got {instance.vector_dim}"
                )
            self._instance = instance
        return self._instance.embed(texts)


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI backend
# ─────────────────────────────────────────────────────────────────────────────


class OpenAIEmbedder(Embedder):
    """
    Embedding backend that calls the OpenAI /v1/embeddings endpoint.

    Model: VECTOR_SEARCH_OPENAI_MODEL (default text-embedding-3-small)
    Dimensions: derived from the model (3-small=1536, 3-large=3072), or
                VECTOR_SEARCH_OPENAI_DIM when set explicitly.
    The factory reads ``OPENAI_API_KEY`` and passes it to the constructor.

    Texts are submitted in ``OPENAI_REQUEST_BATCH_SIZE`` batches. Progress is
    logged after each batch.
    """

    def __init__(self, api_key: str) -> None:
        """
        Initialise the OpenAI client.

        Args:
            api_key: An OpenAI API key.

        Raises:
            ImportError: If the ``openai`` package is not installed.
        """
        from openai import OpenAI  # deferred so the module loads without openai

        self._client = OpenAI(api_key=api_key)
        log.info(
            "OpenAIEmbedder initialised  (model: %s, dim: %d)",
            _OPENAI_MODEL,
            _OPENAI_DIM,
        )

    @property
    def vector_dim(self) -> int:
        """Return the configured OpenAI output dimension.

        Returns:
            Effective dimension resolved from model metadata or the explicit
            dimension override.
        """
        return _OPENAI_DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Send ``texts`` to the OpenAI embeddings endpoint in batches.

        Args:
            texts: Strings to embed.

        Returns:
            One ``self.vector_dim``-dimensional float vector per input string,
            in input order.

        Raises:
            openai.AuthenticationError: If the API key is invalid or revoked.
            openai.RateLimitError: If the account's RPM/TPM quota is exceeded.
        """
        all_embeddings: list[list[float]] = []

        # Only forward `dimensions` when explicitly overridden: it is a v3-only
        # parameter, so passing it unconditionally would break ada-002.
        extra = {"dimensions": _OPENAI_DIM} if _OPENAI_DIM_EXPLICIT else {}

        for i in range(0, len(texts), OPENAI_REQUEST_BATCH_SIZE):
            batch = texts[i : i + OPENAI_REQUEST_BATCH_SIZE]
            response = self._client.embeddings.create(
                model=_OPENAI_MODEL, input=batch, **extra
            )
            all_embeddings.extend(r.embedding for r in response.data)
            log.info(
                "  [OpenAI] Embedded %d / %d",
                min(i + OPENAI_REQUEST_BATCH_SIZE, len(texts)),
                len(texts),
            )

        return all_embeddings


# ─────────────────────────────────────────────────────────────────────────────
# Local sentence-transformers backend
# ─────────────────────────────────────────────────────────────────────────────


class LocalEmbedder(Embedder):
    """
    Run ``VECTOR_SEARCH_LOCAL_MODEL`` locally through sentence-transformers.

    The model defaults to ``BAAI/bge-large-en-v1.5`` and is downloaded on first
    use if it is not already cached. Its output dimension is read from the
    loaded model.

    Requires: pip install sentence-transformers torch
    """

    def __init__(self) -> None:
        """
        Load the model from the local HuggingFace cache, downloading it first
        if necessary.

        Raises:
            ImportError: If ``sentence-transformers`` is not installed.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for the local embedding fallback.\n"
                "Install it with:  pip install sentence-transformers torch"
            ) from exc

        log.info("Loading local model %s (first run downloads ~1.3 GB)…", _LOCAL_MODEL)
        self._model = SentenceTransformer(_LOCAL_MODEL)
        self._dim = self._model.get_sentence_embedding_dimension()
        log.info("LocalEmbedder ready  (model: %s, dim: %d)", _LOCAL_MODEL, self._dim)

    @property
    def vector_dim(self) -> int:
        """Return the dimension reported by the loaded sentence-transformer.

        Returns:
            Length of every vector produced by ``embed``.
        """
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Encode ``texts`` using the locally loaded sentence-transformers model.

        sentence-transformers handles device selection and processes inputs in
        ``LOCAL_INFERENCE_BATCH_SIZE`` batches.

        Args:
            texts: Strings to embed.

        Returns:
            One ``self.vector_dim``-length vector per input string, in input order.

        Raises:
            Tokenization and inference errors from sentence-transformers are
            propagated.
        """
        log.info("  [Local] Encoding %d texts…", len(texts))
        vectors = self._model.encode(
            texts,
            batch_size=LOCAL_INFERENCE_BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return vectors.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Local ONNX backend (fastembed) — the light default local path
# ─────────────────────────────────────────────────────────────────────────────


class FastEmbedEmbedder(Embedder):
    """
    Embedding backend that runs a BAAI/bge model via ``fastembed`` (ONNX).

    This is the default local backend when ``OPENAI_API_KEY`` is unset. It uses
    ONNX Runtime rather than torch and returns the vectors produced by
    ``VECTOR_SEARCH_LOCAL_MODEL``.

    Model: VECTOR_SEARCH_LOCAL_MODEL (default BAAI/bge-large-en-v1.5)
    Requires: pip install fastembed

    ``VECTOR_SEARCH_EMBED_THREADS`` optionally caps ONNX Runtime threads; when
    unset or zero, the runtime chooses its own thread count.
    """

    def __init__(self, model_name: str = _LOCAL_MODEL) -> None:
        """
        Load the ONNX model, downloading it into fastembed's cache on first use.

        Args:
            model_name: fastembed model identifier to load.

        Raises:
            ImportError: If ``fastembed`` is not installed.
        """
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise ImportError(
                "fastembed is required for the local ONNX embedding backend.\n"
                "Install it with:  pip install fastembed\n"
                "(or set VECTOR_SEARCH_LOCAL_BACKEND=sentence-transformers to use "
                "the torch backend instead)."
            ) from exc

        log.info(
            "Loading local ONNX model %s via fastembed (first run downloads weights)…",
            model_name,
        )
        # threads=None lets ONNX Runtime choose its own thread count.
        self._model = TextEmbedding(model_name=model_name, threads=EMBED_THREADS)
        # Probe the true output dimensionality once, rather than hard-coding it —
        # this keeps the backend correct for any bge model the env selects.
        probe = next(iter(self._model.embed(["dimension probe"])))
        self._dim = len(probe)
        log.info(
            "FastEmbedEmbedder ready  (model: %s, dim: %d, threads: %s)",
            model_name,
            self._dim,
            EMBED_THREADS if EMBED_THREADS else "auto",
        )

    @property
    def vector_dim(self) -> int:
        """Return the dimension measured from the model's probe vector.

        Returns:
            Length of every vector produced by ``embed``.
        """
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed ``texts`` via ONNX. ``fastembed.embed`` yields vectors lazily, so
        this method materialises the returned vectors before returning.

        Args:
            texts: Strings to encode. fastembed processes them in
                ``LOCAL_INFERENCE_BATCH_SIZE`` microbatches.

        Returns:
            One float vector per input string, in input order.

        Raises:
            Tokenization and ONNX inference errors from fastembed are propagated.
        """
        t = time.perf_counter()
        out = [
            vector.tolist()
            for vector in self._model.embed(
                texts, batch_size=LOCAL_INFERENCE_BATCH_SIZE
            )
        ]
        elapsed = time.perf_counter() - t
        log.info("  [ONNX] %d chunks in %.1fs (%.1f/s)",
                 len(texts), elapsed, len(texts) / elapsed)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────


def _get_local_embedder() -> Embedder:
    """Construct the configured local embedding backend.

    ``VECTOR_SEARCH_LOCAL_BACKEND=sentence-transformers`` selects the torch
    implementation. ``fastembed`` is the default and also the fallback for an
    unrecognized value, after logging a warning.

    Returns:
        A loaded ``FastEmbedEmbedder`` or ``LocalEmbedder``.

    Raises:
        ImportError: If the selected backend dependency is unavailable.
        Model loading and download errors are propagated.
    """
    if _LOCAL_BACKEND == "sentence-transformers":
        return LocalEmbedder()
    if _LOCAL_BACKEND != "fastembed":
        log.warning(
            "Unknown VECTOR_SEARCH_LOCAL_BACKEND=%r; falling back to fastembed.",
            _LOCAL_BACKEND,
        )
    return FastEmbedEmbedder()


def get_embedder() -> Embedder:
    """
    Select and return the appropriate embedding backend based on the environment.

    Selection logic
    ───────────────
    1. If the ``OPENAI_API_KEY`` environment variable is set and non-empty,
       return an ``OpenAIEmbedder`` backed by that key.
    2. Otherwise, log a WARNING explaining the fallback, then return the local
       backend selected by VECTOR_SEARCH_LOCAL_BACKEND — ``FastEmbedEmbedder``
       (ONNX, default) or ``LocalEmbedder`` (sentence-transformers/torch) —
       running VECTOR_SEARCH_LOCAL_MODEL (default BAAI/bge-large-en-v1.5).

    Returns:
        An ``Embedder`` instance ready to call ``embed()``.

    Raises:
        ImportError: If the local path is taken but the selected backend's
                     dependency (``fastembed`` or ``sentence-transformers``) is
                     not installed.

    Example
    ───────
    >>> embedder = get_embedder()
    >>> vectors = embedder.embed(["leveraged yield strategy"])
    >>> len(vectors[0]) == embedder.vector_dim
    True
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        return OpenAIEmbedder(api_key=api_key)

    log.warning(
        "OPENAI_API_KEY is not set — falling back to the local %s backend "
        "(model %s).  Set OPENAI_API_KEY to use the OpenAI backend instead.  "
        "Note: if you switch backends/models on an existing Qdrant collection its "
        "vector dimensionality may differ (OpenAI=1536, bge-large=1024, "
        "bge-base=768, bge-small=384); setup_collection recreates the collection "
        "automatically when the dimension changes.",
        _LOCAL_BACKEND,
        _LOCAL_MODEL,
    )
    return _get_local_embedder()


def _configured_vector_dim() -> int | None:
    """Resolve vector dimension from configuration metadata when possible.

    OpenAI dimensions come from the selected model or explicit override.
    Known sentence-transformer models use the local lookup table. fastembed
    exposes model dimensions through its registry without opening an inference
    session.

    Returns:
        Configured dimension, or ``None`` for an unknown sentence-transformer
        model whose projection size must be read after loading.

    Raises:
        ImportError: If fastembed is selected but unavailable.
        Errors from fastembed's model registry are propagated.
    """
    if os.getenv("OPENAI_API_KEY"):
        return _OPENAI_DIM

    if _LOCAL_BACKEND == "sentence-transformers":
        return _KNOWN_SENTENCE_TRANSFORMER_DIMS.get(_LOCAL_MODEL.lower())

    # FastEmbed's registry contains dimensions as metadata. Importing the
    # runtime is much cheaper than constructing TextEmbedding, which opens the
    # ONNX model and allocates its long-lived inference session.
    from fastembed import TextEmbedding

    return TextEmbedding.get_embedding_size(_LOCAL_MODEL)


def get_lazy_embedder() -> LazyEmbedder:
    """Return an embedder suitable for collection setup before inference.

    When dimension metadata is available, the returned proxy does not construct
    the backend until its first ``embed`` call. For an arbitrary
    sentence-transformer model with an unknown dimension, this function loads the
    backend immediately, captures its dimension, and wraps that instance.

    Returns:
        A ``LazyEmbedder`` whose ``vector_dim`` is immediately available.

    Raises:
        ImportError: If the selected backend dependency is unavailable.
        Backend metadata or model-loading errors are propagated.
    """
    vector_dim = _configured_vector_dim()
    if vector_dim is not None:
        log.info(
            "Configured embedding dimension is %d; deferring backend load until needed",
            vector_dim,
        )
        return LazyEmbedder(get_embedder, vector_dim)

    # An arbitrary sentence-transformers model may have a projection dimension
    # that cannot be inferred safely from its name. Preserve support for those
    # configurations by loading eagerly only in this uncommon fallback path.
    instance = get_embedder()
    return LazyEmbedder(lambda: instance, instance.vector_dim)
