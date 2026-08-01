from __future__ import annotations

import hashlib
from pathlib import Path


def hash_text(text: str, length: int = 16) -> str:
    """Return a short SHA-256 hash for prompt/provenance references."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def hash_file(path: str | Path, length: int = 16) -> str:
    """Return a short SHA-256 hash for a file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:length]
