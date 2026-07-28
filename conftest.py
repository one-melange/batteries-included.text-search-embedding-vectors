"""Global pytest configuration.

Makes the chunker tests hermetic: they call
``tiktoken.encoding_for_model("text-embedding-3-small")`` which, on a cold
machine, downloads the ``cl100k_base`` BPE vocabulary from OpenAI's public
blobstore. We vendor that single ~1.6 MB file under
``packages/vector_search/test/_fixtures/tiktoken_cache`` (named by the sha1 of
the download URL, which is exactly how tiktoken looks it up) and point
``TIKTOKEN_CACHE_DIR`` at it, so the test run needs no network.

``setdefault`` means a developer who has already set ``TIKTOKEN_CACHE_DIR`` (or
wants a different cache) keeps their value; we only supply the vendored default.
"""

import os
from pathlib import Path

_VENDORED_TIKTOKEN_CACHE = (
    Path(__file__).parent
    / "packages"
    / "vector_search"
    / "test"
    / "_fixtures"
    / "tiktoken_cache"
)

os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(_VENDORED_TIKTOKEN_CACHE))
