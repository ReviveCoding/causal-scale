from __future__ import annotations

import hashlib

import numpy as np


def stable_random_scores(source_row_ids: np.ndarray, seed: int) -> np.ndarray:
    ids = np.asarray(source_row_ids)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("source_row_ids must be an integer vector")
    result = np.empty(len(ids), dtype=np.uint64)
    for index, row_id in enumerate(ids):
        payload = f"{seed}:{int(row_id)}".encode("ascii")
        result[index] = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    return result.astype(np.float64) / np.float64(np.iinfo(np.uint64).max)
