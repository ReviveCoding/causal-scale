from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from causal_scale.baselines.random_policy import stable_random_scores
from causal_scale.calibration.metrics import causal_calibration
from causal_scale.evaluation.aipw import aipw_policy_scores, dr_effect_score
from causal_scale.evaluation.ipw import rct_ipw_incremental_scores
from causal_scale.evaluation.ranking import auuc, qini
from causal_scale.evidence.manifest import atomic_json
from causal_scale.models.historical_adapter import HistoricalPortfolioAdapter
from causal_scale.policy.engine import capacity_policy

CAPACITIES = (0.05, 0.10, 0.20, 0.30, 0.50)


def model_scores(
    models: Path, frame: pd.DataFrame, historical: HistoricalPortfolioAdapter
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    result = {
        name: historical.predict_effect(name, frame) for name in ("HIST_C1", "HIST_C3", "HIST_E7")
    }
    devices = {name: "historical_cpu" for name in result}
    for path in sorted(models.glob("*.joblib")):
        model = joblib.load(path)
        model_id = getattr(model, "model_id", "")
        if not model_id or model_id == "INDEPENDENT_AIPW_NUISANCE":
            continue
        if hasattr(model, "predict_score"):
            result[model_id] = np.asarray(model.predict_score(frame), dtype=float)
        else:
            result[model_id] = np.asarray(model.predict_effect(frame), dtype=float)
        devices[model_id] = str(getattr(model, "device", "cpu"))
    result["RANDOM"] = stable_random_scores(frame.source_row_id.to_numpy(), 20260717)
    devices["RANDOM"] = "deterministic_hash"
    return result, devices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=Path(".local-data/marts/historical_locked_2m.parquet")
    )
    parser.add_argument("--models", type=Path, default=Path(".local-models/frozen"))
    parser.add_argument(
        "--output", type=Path, default=Path("evidence/retrospective/model_benchmark")
    )
    args = parser.parse_args()
    started = time.perf_counter()
    frame = pd.read_parquet(args.data)
    historical = HistoricalPortfolioAdapter()
    mu0, mu1 = historical.predict_outcomes(frame)
    propensity = historical.propensity
    nuisance = joblib.load(args.models / "independent_aipw_nuisance.joblib")
    independent_mu0, independent_mu1 = nuisance.predict_outcomes(frame)
    gamma = dr_effect_score(
        frame.visit.to_numpy(), frame.treatment.to_numpy(), mu0, mu1, propensity
    )
    scores, devices = model_scores(args.models, frame, historical)
    rows = []
    score_output = pd.DataFrame({"source_row_id": frame.source_row_id})
    for model_id, score in scores.items():
        score_output[model_id] = score
        is_response = model_id.startswith("RESPONSE_") or model_id == "RANDOM"
        calibration_rmse = None if is_response else causal_calibration(score, gamma)[1]
        ranking_auuc = auuc(score, frame.visit.to_numpy(), frame.treatment.to_numpy(), propensity)
        ranking_qini = qini(score, frame.visit.to_numpy(), frame.treatment.to_numpy(), propensity)
        for capacity in CAPACITIES:
            policy_score = score
            positive_only = not is_response
            if model_id == "CONSTANT_ATE" and np.all(score > 0):
                policy_score = stable_random_scores(frame.source_row_id.to_numpy(), 20260717)
                positive_only = False
            policy = capacity_policy(
                policy_score,
                capacity,
                historical_position=np.arange(len(frame)),
                positive_only=positive_only,
            ).membership
            for evaluator, values in {
                "HISTORICAL_AIPW": aipw_policy_scores(
                    frame.visit.to_numpy(), frame.treatment.to_numpy(), policy, mu0, mu1, propensity
                ),
                "RCT_IPW_HT": rct_ipw_incremental_scores(
                    frame.visit.to_numpy(), frame.treatment.to_numpy(), policy, propensity
                ),
                "INDEPENDENT_AIPW": aipw_policy_scores(
                    frame.visit.to_numpy(),
                    frame.treatment.to_numpy(),
                    policy,
                    independent_mu0,
                    independent_mu1,
                    propensity,
                ),
            }.items():
                rows.append(
                    {
                        "model": model_id,
                        "model_family": "noncausal" if is_response else "causal",
                        "device": devices[model_id],
                        "raw_or_calibrated": "RAW",
                        "capacity": capacity,
                        "evaluator": evaluator,
                        "policy_value": float(values.mean()),
                        "CI_lower": None,
                        "CI_upper": None,
                        "calibration_RMSE": calibration_rmse,
                        "AUUC": ranking_auuc,
                        "Qini": ranking_qini,
                        "evidence_class": "RETROSPECTIVE",
                        "treated_fraction": float(policy.mean()),
                    }
                )
    table = pd.DataFrame(rows)
    baseline = (
        table[(table.model == "HIST_C3") & (table.capacity == 0.2)]
        .set_index("evaluator")
        .policy_value
    )
    table["delta_vs_hist_c3"] = table.apply(
        lambda row: row.policy_value - baseline[row.evaluator] if row.capacity == 0.2 else None,
        axis=1,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / "model_benchmark.csv", index=False)
    score_output.to_parquet(args.output / "scores.parquet", index=False, compression="zstd")
    metrics = {
        "status": "PASS",
        "rows": len(frame),
        "capacity_primary": 0.2,
        "elapsed_seconds": time.perf_counter() - started,
        "historical_e7_value": float(
            table.query(
                "model == 'HIST_E7' and capacity == 0.2 and evaluator == 'HISTORICAL_AIPW'"
            ).policy_value.iloc[0]
        ),
        "historical_c3_value": float(
            table.query(
                "model == 'HIST_C3' and capacity == 0.2 and evaluator == 'HISTORICAL_AIPW'"
            ).policy_value.iloc[0]
        ),
    }
    atomic_json(args.output / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
