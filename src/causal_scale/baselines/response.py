from __future__ import annotations

import numpy as np
import pandas as pd

from causal_scale.gpu.xgboost_backend import classifier
from causal_scale.models.base import FEATURES, CausalEstimator, validate_training_frame


class ResponseModel(CausalEstimator):
    def __init__(
        self,
        model_id: str,
        treated_only: bool,
        seed: int = 20260717,
        device: str = "cpu",
        **params: object,
    ) -> None:
        self.model_id, self.treated_only, self.seed, self.device, self.params = (
            model_id,
            treated_only,
            seed,
            device,
            params,
        )

    def fit(self, frame: pd.DataFrame, **kwargs: object) -> ResponseModel:
        x, treatment, outcome = validate_training_frame(frame)
        mask = treatment == 1 if self.treated_only else np.ones(len(frame), dtype=bool)
        self.model_ = classifier(self.seed, self.device, **self.params).fit(x[mask], outcome[mask])
        self.mean_ = float(outcome[mask].mean())
        return self

    def predict_score(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(frame.loc[:, FEATURES].to_numpy(np.float32))[:, 1]

    def predict_effect(self, frame: pd.DataFrame) -> np.ndarray:
        raise TypeError(f"{self.model_id} is a non-causal response score, not a treatment effect")

    def predict_outcomes(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        score = self.predict_score(frame)
        return score, score
