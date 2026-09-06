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
    x = np.asarray(values, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    cluster_codes, uniques = pd.factorize(np.asarray(clusters), sort=True)
    if len(x) != len(cluster_codes) or len(uniques) < 2:
        raise ValueError("bootstrap inputs require aligned rows and at least two clusters")
    counts = np.bincount(cluster_codes, minlength=len(uniques)).astype(float)
    sums = np.column_stack(
        [
            np.bincount(cluster_codes, weights=x[:, col], minlength=len(uniques))
            for col in range(x.shape[1])
        ]
    )
    rng = np.random.default_rng(seed)
    draws = np.empty((replicates, x.shape[1]), dtype=float)
    for replicate in range(replicates):
        sampled = rng.integers(0, len(uniques), size=len(uniques))
        draws[replicate] = sums[sampled].sum(axis=0) / counts[sampled].sum()
    primary = draws[:, 0]
    return BootstrapEstimate(
        float(x[:, 0].mean()),
        float(np.quantile(primary, alpha / 2)),
        float(np.quantile(primary, 1 - alpha / 2)),
        float(primary.std(ddof=1)),
        draws,
    )
