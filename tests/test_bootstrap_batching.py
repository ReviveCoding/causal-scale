from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_scale.evaluation.bootstrap import paired_cluster_bootstrap


@pytest.mark.parametrize("columns", [1, 3])
@pytest.mark.parametrize("replicates", [17, 257, 513])
def test_batched_draws_match_sequential_rng_and_reductions(columns, replicates):
    rng = np.random.default_rng(91)
    values = rng.normal(size=(73, columns))
    clusters = np.arange(73) % 7
    codes, unique = pd.factorize(clusters, sort=True)
    sizes = np.bincount(codes, minlength=len(unique)).astype(float)
    sums = np.column_stack(
        [
            np.bincount(codes, weights=values[:, column], minlength=len(unique))
            for column in range(columns)
        ]
    )
    reference_rng = np.random.default_rng(123)
    expected = np.empty((replicates, columns))
    for replicate in range(replicates):
        selected = reference_rng.integers(0, len(unique), size=len(unique))
        expected[replicate] = sums[selected].sum(axis=0) / sizes[selected].sum()

    result = paired_cluster_bootstrap(
        values[:, 0] if columns == 1 else values, clusters, replicates, seed=123
    )
    np.testing.assert_array_equal(result.draws, expected)
    assert result.lower == np.quantile(expected[:, 0], 0.025)
    assert result.upper == np.quantile(expected[:, 0], 0.975)
    assert result.standard_error == expected[:, 0].std(ddof=1)
