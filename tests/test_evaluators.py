from __future__ import annotations

import numpy as np

from causal_scale.evaluation.aipw import aipw_policy_scores, dr_effect_score
from causal_scale.evaluation.ipw import rct_ipw_incremental_scores


def test_dr_score_constant_truth_in_expectation() -> None:
    y = np.array([0, 1, 0, 1], dtype=float)
    t = np.array([0, 0, 1, 1])
    m0 = np.full(4, 0.5)
    m1 = np.full(4, 0.5)
    gamma = dr_effect_score(y, t, m0, m1, 0.5)
    assert abs(gamma.mean()) < 1e-12


def test_never_treat_has_zero_incremental_value() -> None:
    y = np.array([0, 1, 1, 0], dtype=float)
    t = np.array([0, 1, 0, 1])
    policy = np.zeros(4)
    m0 = np.full(4, 0.4)
    m1 = np.full(4, 0.6)
    np.testing.assert_array_equal(aipw_policy_scores(y, t, policy, m0, m1, 0.5), np.zeros(4))
    np.testing.assert_array_equal(rct_ipw_incremental_scores(y, t, policy, 0.5), np.zeros(4))


def test_propensity_edges_block() -> None:
    inputs = (np.ones(2), np.array([0, 1]), np.ones(2), np.zeros(2), np.ones(2))
    for propensity in (0.0, 1.0):
        try:
            dr_effect_score(*inputs[:2], inputs[3], inputs[4], propensity)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid propensity admitted")
