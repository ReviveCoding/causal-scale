from __future__ import annotations

import numpy as np

from causal_scale.baselines.random_policy import stable_random_scores
from causal_scale.policy.engine import capacity_policy


def test_numeric_tie_keys_do_not_sort_lexicographically() -> None:
    scores = np.ones(3)
    policy = capacity_policy(scores, 2 / 3, tie_key=np.array([10, 2, 3]), positive_only=False)
    np.testing.assert_array_equal(policy.membership, [0, 1, 1])


def test_historical_position_matches_stable_position_order() -> None:
    scores = np.array([1.0, 1.0, 0.2, -1.0])
    policy = capacity_policy(scores, 0.5, historical_position=np.arange(4))
    np.testing.assert_array_equal(policy.membership, [1, 1, 0, 0])


def test_random_policy_is_deterministic_and_id_keyed() -> None:
    ids = np.array([9, 1, 100], dtype=np.int64)
    first = stable_random_scores(ids, 7)
    second = stable_random_scores(ids, 7)
    np.testing.assert_array_equal(first, second)
    assert len(np.unique(first)) == len(first)
