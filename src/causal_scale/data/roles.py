from __future__ import annotations

import numpy as np

from causal_scale.evidence.hashing import row_ids_sha256

TOTAL_ROWS = 13_979_592


def splitmix64_priorities(source_row_ids: np.ndarray) -> np.ndarray:
    values = np.asarray(source_row_ids, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


def priority_window(
    total_rows: int, rows: int, offset: int, chunk_rows: int = 250_000
) -> np.ndarray:
    if min(total_rows, rows, chunk_rows) <= 0 or offset < 0 or offset + rows > total_rows:
        raise ValueError("invalid deterministic priority window")
    target = rows + offset
    selected_ids = np.empty(0, dtype=np.int64)
    selected_priorities = np.empty(0, dtype=np.uint64)
    for start in range(0, total_rows, chunk_rows):
        ids = np.arange(start, min(start + chunk_rows, total_rows), dtype=np.int64)
        priorities = splitmix64_priorities(ids)
        selected_ids = np.concatenate((selected_ids, ids))
        selected_priorities = np.concatenate((selected_priorities, priorities))
        if len(selected_ids) > target:
            keep = np.argpartition(selected_priorities, target - 1)[:target]
            selected_ids, selected_priorities = selected_ids[keep], selected_priorities[keep]
    order = np.argsort(selected_priorities, kind="stable")
    return np.sort(selected_ids[order][offset:target], kind="mergesort")


def priority_rank_prefix(total_rows: int, target: int, chunk_rows: int = 250_000) -> np.ndarray:
    """Return source IDs in SplitMix64 priority order through ``target``."""
    if target <= 0 or target > total_rows:
        raise ValueError("invalid priority prefix")
    selected_ids = np.empty(0, dtype=np.int64)
    selected_priorities = np.empty(0, dtype=np.uint64)
    for start in range(0, total_rows, chunk_rows):
        ids = np.arange(start, min(start + chunk_rows, total_rows), dtype=np.int64)
        priorities = splitmix64_priorities(ids)
        selected_ids = np.concatenate((selected_ids, ids))
        selected_priorities = np.concatenate((selected_priorities, priorities))
        if len(selected_ids) > target:
            keep = np.argpartition(selected_priorities, target - 1)[:target]
            selected_ids, selected_priorities = selected_ids[keep], selected_priorities[keep]
    return selected_ids[np.argsort(selected_priorities, kind="stable")]


HISTORICAL_WINDOWS = {
    "development_0_250k": (0, 250_000),
    "development_250k_500k": (250_000, 250_000),
    "historical_250k": (500_000, 250_000),
    "historical_locked_2m": (750_000, 2_000_000),
}
PROSPECTIVE_WINDOW = (2_750_000, 2_000_000)


def role_manifest(total_rows: int = TOTAL_ROWS, include_ids: bool = False) -> dict[str, object]:
    roles: dict[str, object] = {}
    prior: list[np.ndarray] = []
    prefix = priority_rank_prefix(total_rows, PROSPECTIVE_WINDOW[0] + PROSPECTIVE_WINDOW[1])
    for name, (offset, rows) in HISTORICAL_WINDOWS.items():
        ids = np.sort(prefix[offset : offset + rows], kind="mergesort")
        roles[name] = {"offset": offset, "rows": rows, "row_ids_sha256": row_ids_sha256(ids)}
        prior.append(ids)
        if include_ids:
            roles[name]["source_row_ids"] = ids.tolist()  # type: ignore[index]
    offset, rows = PROSPECTIVE_WINDOW
    prospective = np.sort(prefix[offset : offset + rows], kind="mergesort")
    combined_prior = np.concatenate(prior)
    roles["prospective_2m"] = {
        "offset": offset,
        "rows": rows,
        "row_ids_sha256": row_ids_sha256(prospective),
        "intersection_with_all_prior": int(
            np.intersect1d(combined_prior, prospective, assume_unique=True).size
        ),
    }
    return {
        "schema": "causal-scale-row-roles-v1",
        "algorithm": "splitmix64_priority_window_v1",
        "total_rows": total_rows,
        "roles": roles,
    }
