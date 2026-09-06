from __future__ import annotations

import numpy as np
import pandas as pd

from causal_scale.gpu.xgboost_backend import classifier, regressor
from causal_scale.models.base import FEATURES, CausalEstimator, validate_training_frame


class XGBXLearner(CausalEstimator):
    model_id = "X_LEARNER"

    def __init__(self, seed: int = 20260717, device: str = "cpu", **params: object) -> None:
        self.seed, self.device, self.params = seed, device, params

    def fit(self, frame: pd.DataFrame, **kwargs: object) -> XGBXLearner:
        x, t, y = validate_training_frame(frame)
        self.mu0_ = classifier(self.seed, self.device, **self.params).fit(x[t == 0], y[t == 0])
        self.mu1_ = classifier(self.seed + 1, self.device, **self.params).fit(x[t == 1], y[t == 1])
        d1 = y[t == 1] - self.mu0_.predict_proba(x[t == 1])[:, 1]
        d0 = self.mu1_.predict_proba(x[t == 0])[:, 1] - y[t == 0]
        self.tau1_ = regressor(self.seed + 2, self.device, **self.params).fit(x[t == 1], d1)
        self.tau0_ = regressor(self.seed + 3, self.device, **self.params).fit(x[t == 0], d0)
        self.propensity_ = float(np.clip(t.mean(), 0.02, 0.98))
        return self

    def predict_outcomes(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = frame.loc[:, FEATURES].to_numpy(np.float32)
        return self.mu0_.predict_proba(x)[:, 1], self.mu1_.predict_proba(x)[:, 1]

    def predict_effect(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame.loc[:, FEATURES].to_numpy(np.float32)
        return self.propensity_ * self.tau0_.predict(x) + (
            1 - self.propensity_
        ) * self.tau1_.predict(x)
