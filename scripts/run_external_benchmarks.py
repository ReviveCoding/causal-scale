from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from causal_scale.baselines import ConstantATE, XGBRLearner, XGBTlearner, XGBXLearner
from causal_scale.evidence.hashing import sha256_file
from causal_scale.evidence.manifest import atomic_json
from causal_scale.gpu.preflight import xgboost_cuda_smoke

SEED = 20260717
TWINS_FILES = {
    "X": "twin_pairs_X_3years_samesex.csv",
    "T": "twin_pairs_T_3years_samesex.csv",
    "Y": "twin_pairs_Y_3years_samesex.csv",
}
TWINS_SOURCE = "https://github.com/AMLab-Amsterdam/CEVAE/tree/master/datasets/TWINS"


def prepare_twins(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    paths = {name: root / filename for name, filename in TWINS_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "DATA_BLOCKED_MANUAL_PROVIDER_ACTION: download the three AMLab CEVAE Twins CSVs: "
            + ", ".join(missing)
        )
    x = pd.read_csv(paths["X"])
    outcomes = pd.read_csv(paths["Y"])
    if len(x) != 71_345 or len(outcomes) != len(x):
        raise ValueError("Twins source row identity mismatch")
    excluded = {column for column in x if column.lower().startswith("unnamed")}
    excluded.update(column for column in x if column.startswith("infant_id"))
    feature_columns = [column for column in x if column not in excluded][:12]
    features = x.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    features = features.fillna(features.median()).fillna(0.0)
    frame = features.rename(
        columns=dict(zip(feature_columns, (f"f{i}" for i in range(12)), strict=True))
    )
    y0 = outcomes["mort_0"].to_numpy(dtype=np.float32)
    y1 = outcomes["mort_1"].to_numpy(dtype=np.float32)
    rng = np.random.default_rng(SEED)
    treatment = rng.binomial(1, 0.5, len(frame)).astype(np.int8)
    frame["treatment"] = treatment
    frame["visit"] = np.where(treatment == 1, y1, y0)
    permutation = rng.permutation(len(frame))
    split = int(0.8 * len(frame))
    return (
        frame.iloc[permutation[:split]].copy(),
        frame.iloc[permutation[split:]].copy(),
        y0[permutation[split:]],
        y1[permutation[split:]],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(".local-data/external/twins"))
    parser.add_argument("--output", type=Path, default=Path("evidence/external/benchmark"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        train, test, y0, y1 = prepare_twins(args.data)
    except FileNotFoundError as error:
        report = {"status": "DATA_BLOCKED_MANUAL_PROVIDER_ACTION", "error": str(error)}
        atomic_json(args.output / "metrics.json", report)
        print(json.dumps(report, indent=2))
        return
    gpu = xgboost_cuda_smoke()
    device = "cuda" if gpu["status"] == "GPU_XGBOOST_AVAILABLE" else "cpu"
    params = {"n_estimators": 200, "max_depth": 5, "n_jobs": 8}
    models = [
        ConstantATE(),
        XGBTlearner(device=device, **params),
        XGBXLearner(device=device, **params),
        XGBRLearner(device=device, **params),
    ]
    truth = y1 - y0
    oracle_order = np.argsort(-truth, kind="mergesort")
    oracle_membership = np.zeros(len(test), dtype=bool)
    oracle_membership[oracle_order[: int(0.2 * len(test))]] = True
    rows = []
    for model in models:
        model.fit(train)
        prediction = model.predict_effect(test)
        selected = np.argsort(-prediction, kind="mergesort")[: int(0.2 * len(test))]
        rows.append(
            {
                "dataset": "twins",
                "model": model.model_id,
                "replication": 0,
                "PEHE": float(np.sqrt(np.mean((prediction - truth) ** 2))),
                "ATE_error": float(abs(prediction.mean() - truth.mean())),
                "AUUC": None,
                "Qini": None,
                "metric_validity": "KNOWN_PAIRED_POTENTIAL_OUTCOMES",
                "oracle_policy_value": float(truth[oracle_membership].mean()),
                "estimated_policy_value_oracle_truth": float(truth[selected].mean()),
                "policy_regret": float(truth[oracle_membership].mean() - truth[selected].mean()),
                "device": getattr(model, "device", "cpu"),
            }
        )
    supplemental = [
        {
            "dataset": "ihdp",
            "model": None,
            "replication": None,
            "PEHE": None,
            "ATE_error": None,
            "AUUC": None,
            "Qini": None,
            "metric_validity": "SKIPPED_RESOURCE_LIMIT",
        },
        {
            "dataset": "hillstrom",
            "model": None,
            "replication": None,
            "PEHE": None,
            "ATE_error": None,
            "AUUC": None,
            "Qini": None,
            "metric_validity": "OPTIONAL_NOT_RUN",
        },
    ]
    pd.DataFrame([*rows, *supplemental]).to_csv(args.output / "external_benchmark.csv", index=False)
    report = {
        "status": "PASS",
        "twins": {
            "source": TWINS_SOURCE,
            "source_files": {
                name: {
                    "path": str(args.data / filename),
                    "sha256": sha256_file(args.data / filename),
                }
                for name, filename in TWINS_FILES.items()
            },
            "rows": 71_345,
            "train_rows": len(train),
            "test_rows": len(test),
            "treatment_definition": "randomly reveal heavier twin (1) or lighter twin (0)",
            "outcome": "mortality within three years",
            "results": rows,
        },
        "ihdp": "SKIPPED_RESOURCE_LIMIT",
        "hillstrom": "OPTIONAL_NOT_RUN",
        "gpu_qualification": gpu["status"],
    }
    atomic_json(args.output / "metrics.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
