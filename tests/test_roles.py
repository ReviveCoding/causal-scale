from __future__ import annotations

import numpy as np

from causal_scale.data.roles import priority_window
from causal_scale.evidence.hashing import row_ids_sha256


def test_priority_windows_are_disjoint_and_repeatable() -> None:
    first = priority_window(10_000, 1_000, 0)
    second = priority_window(10_000, 1_000, 1_000)
    assert not np.intersect1d(first, second).size
    np.testing.assert_array_equal(first, priority_window(10_000, 1_000, 0))
    assert row_ids_sha256(first) == row_ids_sha256(first[::-1])
