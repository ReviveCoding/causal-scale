from __future__ import annotations

import numpy as np

from causal_scale.calibration.affine import PositiveSlopeAffineCalibrator


def test_affine_slope_is_positive_and_ranking_preserved() -> None:
    raw = np.array([-2.0, -1.0, 0.0, 1.0])
    gamma = -raw
    calibrator = PositiveSlopeAffineCalibrator().fit(raw, gamma)
    transformed = calibrator.transform(raw)
    assert calibrator.slope_ > 0
    np.testing.assert_array_equal(np.argsort(raw), np.argsort(transformed))
