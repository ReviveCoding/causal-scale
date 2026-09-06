from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from causal_scale.calibration.metrics import causal_calibration
from causal_scale.evaluation.aipw import aipw_policy_scores, dr_effect_score
from causal_scale.models.historical_adapter import HistoricalPortfolioAdapter
from causal_scale.policy.engine import capacity_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=Path(".local-data/marts/historical_locked_2m.parquet")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("evidence/retrospective/ablation/ablation.csv")
    )
    args = parser.parse_args()
    frame = pd.read_parquet(args.data)
    adapter = HistoricalPortfolioAdapter()
    mu0, mu1 = adapter.predict_outcomes(frame)
    gamma = dr_effect_score(
        frame.visit.to_numpy(), frame.treatment.to_numpy(), mu0, mu1, adapter.propensity
    )
    calibrators = joblib.load(".local-models/frozen/common_calibrators.joblib")
    c1 = adapter.predict_effect("HIST_C1", frame)
    c3 = adapter.predict_effect("HIST_C3", frame)
    c5 = adapter.predict_effect("HIST_C5", frame)
    blend = 0.7 * c1 + 0.3 * c5
    e7_definition = adapter.portfolio["E7"]
    scores = {
        "HIST_C1": c1,
        "HIST_C3": c3,
        "HIST_C3_COMMON_CAL": calibrators["HIST_C3"].transform(c3),
        "HIST_C5": c5,
        "HIST_PRE_FINAL_BLEND": blend,
        "HIST_E7": float(e7_definition["affine_intercept"])
        + float(e7_definition["affine_slope"]) * blend,
    }
    records, policies = [], {}
    for model, score in scores.items():
        policy = capacity_policy(score, 0.2, historical_position=np.arange(len(frame))).membership
        policies[model] = policy
        value = aipw_policy_scores(
            frame.visit.to_numpy(), frame.treatment.to_numpy(), policy, mu0, mu1, adapter.propensity
        ).mean()
        records.append(
            {
                "model": model,
                "policy_value": value,
                "calibration_rmse": causal_calibration(score, gamma)[1],
                "disagreement_vs_hist_c3": float(
                    np.mean(policy != policies.get("HIST_C3", policy))
                ),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
