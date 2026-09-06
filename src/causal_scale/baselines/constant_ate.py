from __future__ import annotations

import numpy as np
import pandas as pd

from causal_scale.models.base import CausalEstimator, validate_training_frame


class ConstantATE(CausalEstimator):
    model_id = "CONSTANT_ATE"

    def fit(self, frame: pd.DataFrame, **kwargs: object) -> ConstantATE:
        _, treatment, outcome = validate_training_frame(frame)
        self.ate_ = float(outcome[treatment == 1].mean() - outcome[treatment == 0].mean())
        self.mu0_ = float(outcome[treatment == 0].mean())
        self.mu1_ = float(outcome[treatment == 1].mean())
        return self

    def predict_effect(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), self.ate_)

    def predict_outcomes(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return np.full(len(frame), self.mu0_), np.full(len(frame), self.mu1_)
