from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BootstrapEstimate:
    value: float
    lower: float
    upper: float
    standard_error: float
    draws: np.ndarray


def paired_cluster_bootstrap(
    values: np.ndarray,
    clusters: np.ndarray,
    replicates: int = 4000,
    seed: int = 20261620,
    alpha: float = 0.05,
) -> BootstrapEstimate:
    value_matrix = np.asarray(values, dtype=float)
    if value_matrix.ndim == 1:
        value_matrix = value_matrix[:, None]
    cluster_codes, uniques = pd.factorize(np.asarray(clusters), sort=True)
    if len(value_matrix) != len(cluster_codes) or len(uniques) < 2:
        raise ValueError("bootstrap inputs require aligned rows and at least two clusters")
    cluster_count = len(uniques)
    cluster_sizes = np.bincount(cluster_codes, minlength=cluster_count).astype(float)
    cluster_sums = np.column_stack(
        [
            np.bincount(cluster_codes, weights=value_matrix[:, col], minlength=cluster_count)
            for col in range(value_matrix.shape[1])
        ]
    )
    rng = np.random.default_rng(seed)
    draws = np.empty((replicates, value_matrix.shape[1]), dtype=float)
    # Preserve the RNG stream and reduction order while amortizing Python calls.
    # Bound gathered arrays to approximately 8 MiB, or one replicate if larger.
    bytes_per_replicate = cluster_count * (value_matrix.shape[1] + 2) * 8
    batch_size = max(1, min(256, (8 * 1024 * 1024) // bytes_per_replicate))
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        sampled = rng.integers(0, cluster_count, size=(stop - start, cluster_count))
        draws[start:stop] = (
            cluster_sums[sampled].sum(axis=1) / cluster_sizes[sampled].sum(axis=1)[:, None]
        )
    primary = draws[:, 0]
    return BootstrapEstimate(
        float(value_matrix[:, 0].mean()),
        float(np.quantile(primary, alpha / 2)),
        float(np.quantile(primary, 1 - alpha / 2)),
        float(primary.std(ddof=1)),
        draws,
    )
