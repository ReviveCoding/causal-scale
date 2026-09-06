from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from causal_scale.gpu.xgboost_backend import classifier, regressor
from causal_scale.models.base import FEATURES, CausalEstimator, validate_training_frame


class XGBRLearner(CausalEstimator):
    model_id = "R_LEARNER"

    def __init__(
        self, seed: int = 20260717, device: str = "cpu", folds: int = 3, **params: object
    ) -> None:
        self.seed, self.device, self.folds, self.params = seed, device, folds, params

    def fit(self, frame: pd.DataFrame, **kwargs: object) -> XGBRLearner:
        x, t, y = validate_training_frame(frame)
        self.propensity_ = float(np.clip(t.mean(), 0.02, 0.98))
        m_oof = np.empty(len(frame), dtype=float)
        splitter = KFold(self.folds, shuffle=True, random_state=self.seed)
        for fold, (train, hold) in enumerate(splitter.split(x)):
            nuisance = classifier(self.seed + fold, self.device, **self.params).fit(
                x[train], y[train]
            )
            m_oof[hold] = nuisance.predict_proba(x[hold])[:, 1]
        residual_t = t.astype(float) - self.propensity_
        keep = np.abs(residual_t) >= 0.02
        pseudo = (y - m_oof) / residual_t
        weights = residual_t**2
        self.effect_model_ = regressor(self.seed + 100, self.device, **self.params).fit(
            x[keep], pseudo[keep], sample_weight=weights[keep]
        )
        self.mu0_ = classifier(self.seed + 200, self.device, **self.params).fit(
            x[t == 0], y[t == 0]
        )
        self.mu1_ = classifier(self.seed + 201, self.device, **self.params).fit(
            x[t == 1], y[t == 1]
        )
        return self

    def predict_effect(self, frame: pd.DataFrame) -> np.ndarray:
        return self.effect_model_.predict(frame.loc[:, FEATURES].to_numpy(np.float32))

    def predict_outcomes(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = frame.loc[:, FEATURES].to_numpy(np.float32)
        return self.mu0_.predict_proba(x)[:, 1], self.mu1_.predict_proba(x)[:, 1]
