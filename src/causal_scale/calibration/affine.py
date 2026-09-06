from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PositiveSlopeAffineCalibrator:
    minimum_slope: float = 1e-8

    def fit(
        self, prediction: np.ndarray, pseudo_outcome: np.ndarray
    ) -> PositiveSlopeAffineCalibrator:
        x, y = np.asarray(prediction, dtype=float), np.asarray(pseudo_outcome, dtype=float)
        if (
            x.shape != y.shape
            or x.ndim != 1
            or len(x) < 2
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
        ):
            raise ValueError("calibration inputs must be aligned finite vectors")
        centered = x - x.mean()
        denom = float(np.dot(centered, centered))
        unconstrained = (
            float(np.dot(centered, y - y.mean()) / denom) if denom > 0 else self.minimum_slope
        )
        self.slope_ = max(unconstrained, self.minimum_slope)
        self.intercept_ = float(y.mean() - self.slope_ * x.mean())
        return self

    def transform(self, prediction: np.ndarray) -> np.ndarray:
        return self.intercept_ + self.slope_ * np.asarray(prediction, dtype=float)

    def to_dict(self) -> dict[str, float]:
        return {
            "intercept": self.intercept_,
            "slope": self.slope_,
            "minimum_slope": self.minimum_slope,
        }
