from __future__ import annotations

import numpy as np
import pandas as pd

from causal_scale.gpu.xgboost_backend import classifier
from causal_scale.models.base import FEATURES, CausalEstimator, validate_training_frame


class XGBTlearner(CausalEstimator):
    model_id = "GPU_T"

    def __init__(self, seed: int = 20260717, device: str = "cpu", **params: object) -> None:
        self.seed, self.device, self.params = seed, device, params

    def fit(self, frame: pd.DataFrame, **kwargs: object) -> XGBTlearner:
        x, treatment, outcome = validate_training_frame(frame)
        self.model0_ = classifier(self.seed, self.device, **self.params).fit(
            x[treatment == 0], outcome[treatment == 0]
        )
        self.model1_ = classifier(self.seed + 1, self.device, **self.params).fit(
            x[treatment == 1], outcome[treatment == 1]
        )
        self.propensity_ = float(treatment.mean())
        return self

    def predict_outcomes(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = frame.loc[:, FEATURES].to_numpy(np.float32)
        return self.model0_.predict_proba(x)[:, 1], self.model1_.predict_proba(x)[:, 1]

    def predict_effect(self, frame: pd.DataFrame) -> np.ndarray:
        mu0, mu1 = self.predict_outcomes(frame)
        return mu1 - mu0
