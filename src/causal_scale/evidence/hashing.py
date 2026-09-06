from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_json_bytes(value)).hexdigest()


def row_ids_sha256(values: np.ndarray) -> str:
    ids = np.asarray(values)
    if ids.ndim != 1 or len(ids) == 0 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("row IDs must be a nonempty integer vector")
    if (ids < 0).any() or len(np.unique(ids)) != len(ids):
        raise ValueError("row IDs must be nonnegative and unique")
    canonical = np.sort(ids.astype("<i8", copy=False), kind="mergesort")
    digest = hashlib.sha256()
    digest.update(len(canonical).to_bytes(8, "little"))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def tree_manifest(root: str | Path, patterns: tuple[str, ...] = ("*",)) -> dict[str, str]:
    base = Path(root)
    files: set[Path] = set()
    for pattern in patterns:
        files.update(
            path
            for path in base.rglob(pattern)
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        )
    return {path.relative_to(base).as_posix(): sha256_file(path) for path in sorted(files)}
