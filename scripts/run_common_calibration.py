from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fit_baselines import partition_development

from causal_scale.calibration.affine import PositiveSlopeAffineCalibrator
from causal_scale.calibration.metrics import calibration_comparison
from causal_scale.evaluation.aipw import dr_effect_score
from causal_scale.evidence.hashing import sha256_file
from causal_scale.evidence.manifest import atomic_json
from causal_scale.models.historical_adapter import HistoricalPortfolioAdapter
from causal_scale.policy.engine import capacity_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--development", type=Path, default=Path(".local-data/marts/development.parquet")
    )
    parser.add_argument("--models", type=Path, default=Path(".local-models/frozen"))
    parser.add_argument("--output", type=Path, default=Path("evidence/models/common_calibration"))
    args = parser.parse_args()
    _, calibration, _ = partition_development(pd.read_parquet(args.development))
    historical = HistoricalPortfolioAdapter()
    mu0, mu1 = historical.predict_outcomes(calibration)
    gamma = dr_effect_score(
        calibration.visit.to_numpy(),
        calibration.treatment.to_numpy(),
        mu0,
        mu1,
        historical.propensity,
    )
    models: dict[str, object] = {name: historical for name in ("HIST_C1", "HIST_C3", "HIST_E7")}
    for path in args.models.glob("*.joblib"):
        model = joblib.load(path)
        model_id = getattr(model, "model_id", "")
        if model_id and not model_id.startswith(("RESPONSE_", "INDEPENDENT_")):
            models[model_id] = model
    calibrators, rows = {}, []
    for model_id, model in models.items():
        raw = (
            historical.predict_effect(model_id, calibration)
            if model is historical
            else model.predict_effect(calibration)
        )
        calibrator = PositiveSlopeAffineCalibrator().fit(raw, gamma)
        transformed = calibrator.transform(raw)
        raw_policy = capacity_policy(raw, 0.2, historical_position=np.arange(len(raw))).membership
        calibrated_policy = capacity_policy(
            transformed, 0.2, historical_position=np.arange(len(raw))
        ).membership
        calibrators[model_id] = calibrator
        rows.append(
            {
                "model": model_id,
                **calibrator.to_dict(),
                **calibration_comparison(raw, transformed, gamma, raw_policy, calibrated_policy),
            }
        )
    artifact = args.models / "common_calibrators.joblib"
    joblib.dump(calibrators, artifact, compress=3)
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output / "calibration_metrics.csv", index=False)
    report = {
        "status": "PASS",
        "rows": len(calibration),
        "artifact": str(artifact),
        "artifact_sha256": sha256_file(artifact),
        "models": rows,
        "response_scores_excluded": True,
    }
    atomic_json(args.output / "metrics.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
