from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from causal_scale.evidence.hashing import sha256_file
from causal_scale.evidence.manifest import atomic_json

REPORTS = Path("reports")
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"


def read_json(path: str) -> dict[str, Any] | None:
    target = Path(path)
    return json.loads(target.read_text(encoding="utf-8")) if target.is_file() else None


def read_csv(path: str) -> pd.DataFrame | None:
    target = Path(path)
    return pd.read_csv(target) if target.is_file() else None


def save_figure(name: str, title: str) -> None:
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=160)
    plt.close()


def unavailable(name: str, title: str, reason: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.axis("off")
    plt.text(0.5, 0.55, title, ha="center", va="center", fontsize=15)
    plt.text(0.5, 0.4, reason, ha="center", va="center", fontsize=10, wrap=True)
    save_figure(name, title)


def architecture() -> None:
    plt.figure(figsize=(12, 4))
    plt.axis("off")
    labels = [
        "Authenticated\nsource",
        "Canonical IDs\n& row roles",
        "Frozen models\n& calibration",
        "pandas / Spark\npolicy + evaluators",
        "Cryptographic\nevidence + reports",
    ]
    for index, label in enumerate(labels):
        x = 0.02 + index * 0.195
        plt.gca().add_patch(
            plt.Rectangle((x, 0.35), 0.16, 0.3, fill=True, color="#dceaf7", ec="#24557a")
        )
        plt.text(x + 0.08, 0.5, label, ha="center", va="center", fontsize=9)
        if index < len(labels) - 1:
            plt.arrow(x + 0.16, 0.5, 0.03, 0, head_width=0.025, length_includes_head=True)
    save_figure("01_architecture.png", "CAUSAL-SCALE evidence-preserving architecture")


def plot_reports() -> None:
    architecture()
    benchmark = read_csv("evidence/retrospective/model_benchmark/model_benchmark.csv")
    if benchmark is not None:
        primary = benchmark.query("capacity == 0.2 and evaluator == 'HISTORICAL_AIPW'")
        plt.figure(figsize=(11, 5))
        plt.bar(primary.model, primary.policy_value)
        plt.xticks(rotation=55, ha="right")
        plt.ylabel("AIPW incremental policy value")
        save_figure("02_retrospective_policy_value.png", "Retrospective policy value at 20%")
        response = primary[primary.model.isin(["RESPONSE_MARGINAL", "RESPONSE_TREATED", "HIST_E7"])]
        plt.figure(figsize=(7, 4))
        plt.bar(response.model, response.policy_value)
        plt.xticks(rotation=25, ha="right")
        plt.ylabel("AIPW incremental policy value")
        save_figure("07_response_vs_uplift.png", "Response targeting versus uplift targeting")
    else:
        unavailable(
            "02_retrospective_policy_value.png",
            "Retrospective policy value",
            "Evidence unavailable",
        )
        unavailable("07_response_vs_uplift.png", "Response versus uplift", "Evidence unavailable")
    capacity = read_csv("evidence/retrospective/capacity_frontier/capacity_frontier.csv")
    if capacity is not None:
        plt.figure(figsize=(7, 4))
        for column in ("HIST_E7", "HIST_C3"):
            plt.plot(capacity.capacity, capacity[column], marker="o", label=column)
        plt.xlabel("Capacity")
        plt.ylabel("Policy value")
        plt.legend()
        save_figure("03_capacity_frontier.png", "Capacity versus policy value")
    else:
        unavailable("03_capacity_frontier.png", "Capacity frontier", "Evidence unavailable")
    historical = read_json("evidence/historical_reference/historical_metrics.json")
    if historical:
        interval = historical["value_difference_two_sided_95pct"]
        center = historical["value_difference"]
        plt.figure(figsize=(6, 3.5))
        plt.errorbar(
            [center],
            [0],
            xerr=[[center - interval["lower"]], [interval["upper"] - center]],
            fmt="o",
        )
        plt.axvline(0, color="black", linewidth=0.8)
        plt.yticks([0], ["Historical E7-C3"])
        plt.xlabel("AIPW value difference")
        save_figure("04_e7_c3_uncertainty.png", "E7 minus C3 with paired uncertainty")
    ablation = read_csv("evidence/retrospective/ablation/ablation.csv")
    if ablation is not None:
        plt.figure(figsize=(9, 4))
        plt.bar(ablation.model, ablation.policy_value)
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("Policy value")
        save_figure("05_orbit_ablation.png", "ORBIT component ablation")
    else:
        unavailable("05_orbit_ablation.png", "ORBIT ablation", "Evidence unavailable")
    calibration = read_csv("evidence/models/common_calibration/calibration_metrics.csv")
    if calibration is not None:
        causal = calibration.dropna(subset=["raw_calibration_rmse"])
        x = np.arange(len(causal))
        plt.figure(figsize=(10, 4))
        plt.bar(x - 0.2, causal.raw_calibration_rmse, 0.4, label="Raw")
        plt.bar(x + 0.2, causal.calibrated_calibration_rmse, 0.4, label="Common cal")
        plt.xticks(x, causal.model, rotation=35, ha="right")
        plt.ylabel("Calibration RMSE")
        plt.legend()
        save_figure("06_causal_calibration.png", "Raw and common causal calibration")
    backends = read_csv("evidence/systems/spark_backends/systems_benchmark.csv")
    if backends is not None:
        plt.figure(figsize=(8, 4))
        plt.bar(backends.backend, backends.runtime)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Runtime (seconds)")
        save_figure("08_spark_backends.png", "Spark scoring backend runtime")
    else:
        unavailable("08_spark_backends.png", "Spark backends", "Evidence unavailable")
    scale = read_csv("evidence/systems/scale/systems_benchmark.csv")
    if scale is not None:
        plt.figure(figsize=(7, 4))
        plt.plot(scale.rows, scale.runtime, marker="o")
        plt.xlabel("Rows")
        plt.ylabel("Runtime (seconds)")
        save_figure("09_rows_vs_runtime.png", "Feature-only rows versus runtime")
        plt.figure(figsize=(7, 4))
        plt.plot(scale.rows, scale.rows_per_sec, marker="o")
        plt.xlabel("Rows")
        plt.ylabel("Rows per second")
        save_figure("10_scoring_throughput.png", "Scoring-only throughput")
    else:
        unavailable("09_rows_vs_runtime.png", "Rows versus runtime", "Evidence unavailable")
        unavailable("10_scoring_throughput.png", "Scoring throughput", "Evidence unavailable")
    unavailable(
        "11_end_to_end_throughput.png",
        "Outcome-dependent end-to-end throughput",
        "Optional post-prospective systems-only benchmark was not run.",
    )
    local = read_csv("evidence/systems/local_scaling/systems_benchmark.csv")
    if local is not None:
        plt.figure(figsize=(7, 4))
        plt.plot(local.threads_or_workers, local.runtime, marker="o")
        plt.xlabel("Same-machine local workers")
        plt.ylabel("Runtime (seconds)")
        save_figure("12_local_scaling.png", "Local Spark resource scaling")
        plt.figure(figsize=(7, 4))
        plt.plot(local.threads_or_workers, local.parallel_efficiency, marker="o")
        plt.xlabel("Same-machine local workers")
        plt.ylabel("Parallel efficiency")
        save_figure("13_parallel_efficiency.png", "Local parallel efficiency")
    else:
        unavailable("12_local_scaling.png", "Local scaling", "Evidence unavailable")
        unavailable("13_parallel_efficiency.png", "Parallel efficiency", "Evidence unavailable")
    gpu = read_json("evidence/gpu/controlled_benchmark/metrics.json")
    if gpu:
        gpu_row = gpu["benchmark"][0]
        speedup = float(gpu_row["cpu_gpu_speedup"])
        cuda_train = float(gpu_row["train_time"])
        plt.figure(figsize=(6, 4))
        plt.bar(["CPU", "CUDA"], [cuda_train * speedup, cuda_train])
        plt.ylabel("Training seconds")
        save_figure("14_gpu_cpu_cuda.png", "Controlled CPU versus CUDA training")
    prospective = read_json("evidence/prospective/prospective_results.json")
    if historical and prospective:
        plt.figure(figsize=(7, 4))
        labels = ["Historical E7", "Historical C3", "Prospective E7", "Prospective C3"]
        values = [
            historical["candidate_value"],
            historical["baseline_value"],
            prospective["candidate_value"],
            prospective["baseline_value"],
        ]
        plt.bar(labels, values)
        plt.xticks(rotation=25, ha="right")
        plt.ylabel("AIPW policy value")
        save_figure("15_historical_prospective.png", "Historical and prospective E7/C3")
    else:
        unavailable(
            "15_historical_prospective.png", "Historical versus prospective", "Evidence unavailable"
        )
    external = read_csv("evidence/external/benchmark/external_benchmark.csv")
    twins = (
        external[external.dataset == "twins"].dropna(subset=["PEHE"])
        if external is not None
        else None
    )
    if twins is not None and len(twins):
        plt.figure(figsize=(8, 4))
        plt.bar(twins.model, twins.PEHE)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Root PEHE")
        save_figure("16_twins.png", "Twins known-outcome comparison")
    else:
        unavailable("16_twins.png", "Twins comparison", "Evidence unavailable")
    unavailable("17_ihdp.png", "IHDP aggregate", "SKIPPED_RESOURCE_LIMIT; no metric fabricated.")


def copy_tables() -> None:
    sources = {
        "model_benchmark.csv": "evidence/retrospective/model_benchmark/model_benchmark.csv",
        "systems_scale.csv": "evidence/systems/scale/systems_benchmark.csv",
        "systems_local_scaling.csv": "evidence/systems/local_scaling/systems_benchmark.csv",
        "systems_spark_backends.csv": "evidence/systems/spark_backends/systems_benchmark.csv",
        "gpu_benchmark.csv": "evidence/gpu/controlled_benchmark/gpu_benchmark.csv",
        "external_benchmark.csv": "evidence/external/benchmark/external_benchmark.csv",
        "prospective_summary.csv": "evidence/prospective/prospective_summary.csv",
    }
    for target, source in sources.items():
        if Path(source).is_file():
            shutil.copy2(source, TABLES / target)
    system_frames = []
    for source in (
        "evidence/systems/scale/systems_benchmark.csv",
        "evidence/systems/local_scaling/systems_benchmark.csv",
        "evidence/systems/spark_backends/systems_benchmark.csv",
    ):
        frame = read_csv(source)
        if frame is not None:
            system_frames.append(frame)
    if system_frames:
        pd.concat(system_frames, ignore_index=True, sort=False).to_csv(
            TABLES / "systems_benchmark.csv", index=False
        )


def reports() -> None:
    historical = read_json("evidence/historical_reference/historical_metrics.json") or {}
    environment = read_json("evidence/environment/preflight/environment.json") or {}
    dataset = read_json("evidence/data/criteo_v2_1/dataset_manifest.json") or {}
    models = read_json("evidence/models/baselines/run_manifest.json") or {}
    parity = read_json("evidence/spark/historical_parity/metrics.json") or {}
    scale = read_json("evidence/systems/scale/metrics.json") or {}
    local_scaling = read_json("evidence/systems/local_scaling/metrics.json") or {}
    gpu = read_json("evidence/gpu/controlled_benchmark/metrics.json") or {}
    ablation = read_csv("evidence/retrospective/ablation/ablation.csv")
    capacity = read_csv("evidence/retrospective/capacity_frontier/capacity_frontier.csv")
    external = read_json("evidence/external/benchmark/metrics.json") or {}
    prospective = read_json("evidence/prospective/prospective_results.json") or {}
    engineering = read_json("evidence/qualification/engineering/qualification_summary.json") or {}
    contract_path = Path("evidence/prospective/scientific_contract.json")
    contract_sha = sha256_file(contract_path) if contract_path.is_file() else "NOT_SEALED"
    model_status = (
        ", ".join(f"{item['model']}={item['status']}" for item in models.get("models", []))
        or "not available"
    )
    final_status = prospective.get("status", "NOT_RUN")
    gpu_summary = gpu.get("benchmark", [{}])[0] if gpu.get("benchmark") else {}
    primary_summary = (
        {
            key: prospective.get(key)
            for key in (
                "candidate_value",
                "baseline_value",
                "value_difference",
                "candidate_calibration_rmse",
                "baseline_calibration_rmse",
                "calibration_rmse_difference",
                "gates",
            )
        }
        if prospective
        else "NOT_RUN"
    )
    ablation_summary = (
        ablation[["model", "policy_value", "calibration_rmse"]].to_dict(orient="records")
        if ablation is not None
        else "NOT_RUN"
    )
    capacity_summary = capacity.to_dict(orient="records") if capacity is not None else "NOT_RUN"
    technical = f"""# CAUSAL-SCALE Technical Report

## Scope and claim boundary

CAUSAL-SCALE evaluates capacity-constrained causal targeting. Offline policy value is an estimator of incremental visit probability under the declared policy versus never treating; it is not observed online revenue. Historical, development, retrospective, systems, external, and prospective evidence are not pooled into one claim.

## Historical predecessor

The immutable ORBIT-Causal v0.3.9 committed snapshot anchors the historical semantics. Its locked 2M result was E7={historical.get("candidate_value", "NA")} and C3={historical.get("baseline_value", "NA")}, with delta={historical.get("value_difference", "NA")} and verdict={historical.get("verdict", "NA")}.

## Development

New models used only the deterministic development roles. Model status: {model_status}. Response scores were not subjected to causal calibration. Positive-slope affine calibration preserved ranking while zero-boundary eligibility changes were measured.

## Retrospective

The common benchmark is in `reports/tables/model_benchmark.csv`. Every method uses the same policy engine and evaluator within a view. Criteo PEHE is intentionally absent because unit counterfactual truth is unavailable.

## GPU compute

GPU qualification: {environment.get("xgboost_cuda", {}).get("status", "NA")}. Device speed measurements are systems evidence only and do not establish scientific superiority.

## Distributed parity

Spark parity status: {parity.get("status", "NOT_RUN")}. Spark workers are same-machine processes, not physical nodes. Policy selection uses global descending score and numeric source-row-ID ties.

## Systems scale

Feature-only authentic scale status: {scale.get("status", "NOT_RUN")}; full workload: {scale.get("full_13_98m", {}).get("rows", "NA")} rows. These are scoring systems results, not an independent causal holdout.

## External validation

Twins status: {external.get("status", "NOT_RUN")}. IHDP: {external.get("ihdp", "NOT_RUN")}. Historical E7 was not relabeled as an externally retrained method.

## Prospective

Scientific contract SHA-256: `{contract_sha}`. Final prospective status: **{final_status}**. Point estimates and gates are retained in `evidence/prospective/prospective_results.json` and never inferred from this prose.

## Reproducibility

The repository contains a resumable hash-bound DAG, dependency lock, dataset and row-role manifests, frozen model manifests, Spark event logs, and machine-generated tables/figures. The independent evidence verifier checks every present SHA256SUMS ledger.
"""
    (REPORTS / "TECHNICAL_REPORT.md").write_text(technical, encoding="utf-8")
    methods = """# Methods and Estimands

- Primary estimand: AIPW incremental policy value at 20% capacity versus never treating.
- Sensitivity estimands: randomized-propensity IPW/HT and independently nuisance-fitted AIPW.
- Ranking: deterministic global descending scores with numeric source-row-ID ties.
- Eligibility: causal scores must be positive; response rankings are non-causal comparators; positive constant ATE uses deterministic hash allocation.
- Calibration: ten-bin causal calibration RMSE and development-only positive-slope affine calibration.
- Inference: paired exact pretreatment-profile cluster bootstrap for E7/C3; predeclared cluster-robust normal secondary comparisons with Holm correction within evaluator.
- External truth: Twins uses paired mortality potential outcomes; Criteo never reports PEHE.
"""
    (REPORTS / "METHODS_AND_ESTIMANDS.md").write_text(methods, encoding="utf-8")
    limitations = """# Limitations

- Criteo is an offline randomized advertising benchmark; policy value is neither observed online revenue nor production ROI.
- The authentic 13.98M run is a systems workload and is not an independent causal holdout.
- Local Spark workers share one physical Windows machine and are not distributed physical nodes.
- CUDA speed and CPU/GPU prediction agreement are systems properties, not evidence that GPU fitting is causally superior.
- Twins is a benchmark construction around paired twins and does not establish universal external validity.
- DragonNet, full IHDP replications, Hillstrom, ONNX CUDA, and physical multi-node scaling are optional and may be resource-skipped.
"""
    (REPORTS / "LIMITATIONS.md").write_text(limitations, encoding="utf-8")
    experiment_matrix = """# Experiment Matrix

| Evidence class | Dataset | Required result |
|---|---|---|
| Historical predecessor | Criteo locked 2M | Exact ORBIT replay |
| Development | Criteo permitted 500K roles | Model and calibration fitting |
| Retrospective | Criteo historical locked 2M | Common multi-model comparison |
| GPU compute | Controlled 100K fixture | CPU/CUDA time and agreement |
| Distributed parity | Criteo historical locked 2M | Scores, memberships, AIPW parity |
| Systems scale | Criteo features, up to 13,979,592 | Scoring-only throughput |
| External validation | Twins | Root PEHE and ATE error |
| Prospective | Fresh disjoint Criteo 2M | Sealed E7/C3 co-primary tests |
"""
    (REPORTS / "EXPERIMENT_MATRIX.md").write_text(experiment_matrix, encoding="utf-8")
    final = f"""# Final Status

- Workspace: `{Path.cwd()}`
- Historical reference: ORBIT-Causal v0.3.9; verdict `{historical.get("verdict", "NA")}`
- Environment: Python {environment.get("python", "NA")}; Spark {environment.get("packages", {}).get("pyspark", "NA")}; GPU `{environment.get("xgboost_cuda", {}).get("status", "NA")}`
- Dataset: `{dataset.get("status", "NA")}`; rows `{dataset.get("rows", dataset.get("observed_rows", "NA"))}`
- Models: {model_status}
- GPU path used: `{environment.get("xgboost_cuda", {}).get("status", "NA")}`; controlled CUDA train `{gpu_summary.get("train_time", "NA")}` seconds; CPU/GPU speedup `{gpu_summary.get("cpu_gpu_speedup", "NA")}`
- GPU parallelism actually used: 1 heavy GPU job
- Historical E7/C3/delta: `{historical.get("candidate_value", "NA")}` / `{historical.get("baseline_value", "NA")}` / `{historical.get("value_difference", "NA")}`
- Retrospective benchmark: `reports/tables/model_benchmark.csv`
- ORBIT ablation: `{ablation_summary}`
- Spark parity: `{parity.get("status", "NOT_RUN")}`
- Direct 1M: `{scale.get("direct_1m", "NOT_RUN")}`
- 13.98M feature-only: `{scale.get("full_13_98m", "NOT_RUN")}`
- Local scaling: `{local_scaling.get("records", "NOT_RUN")}`
- Optional worker scaling: `NOT_RUN` (no already-working WSL/Linux Spark Standalone environment)
- Capacity frontier: `{capacity_summary}`
- External benchmark: `{external.get("status", "NOT_RUN")}`
- Prospective contract SHA: `{contract_sha}`
- Prospective final status: **{final_status}**
- Prospective primary estimates: `{primary_summary}`
- Engineering qualification: `{engineering.get("status", "NOT_RUN")}`
- Remaining blockers/optional omissions: Causal Forest `SANDBOX_TOOLING_BLOCKER`; IHDP full suite `SKIPPED_RESOURCE_LIMIT`; Hillstrom, DragonNet/JAX, ONNX CUDA, and physical multi-node Spark not run
- Limitations: offline value is not online revenue; local Spark workers are not physical nodes; CUDA speed is not causal superiority; full-source scale is systems-only
- Reports: `reports/TECHNICAL_REPORT.md`, `reports/METHODS_AND_ESTIMANDS.md`, `reports/LIMITATIONS.md`
"""
    (REPORTS / "FINAL_STATUS.md").write_text(final, encoding="utf-8")
    claims = [
        {
            "claim_text": "Historical locked replay values and verdict",
            "evidence_file": "evidence/historical_reference/historical_metrics.json",
            "exact_metric_fields": [
                "candidate_value",
                "baseline_value",
                "value_difference",
                "verdict",
            ],
            "evidence_class": "HISTORICAL_PREDECESSOR",
            "claim_boundary": "frozen ORBIT v0.3.9 only",
        },
        {
            "claim_text": "Authentic 13.98M feature-only scoring throughput",
            "evidence_file": "evidence/systems/scale/metrics.json",
            "exact_metric_fields": ["full_13_98m"],
            "evidence_class": "SYSTEMS_SCALE",
            "claim_boundary": "systems-only, non-causal",
        },
        {
            "claim_text": "Prospective E7/C3 replication classification",
            "evidence_file": "evidence/prospective/prospective_results.json",
            "exact_metric_fields": [
                "status",
                "value_difference",
                "calibration_rmse_difference",
                "gates",
            ],
            "evidence_class": "PROSPECTIVE",
            "claim_boundary": "fresh sealed 2M cohort",
        },
    ]
    for claim in claims:
        claim["status"] = "APPROVED" if Path(str(claim["evidence_file"])).is_file() else "BLOCKED"
    atomic_json(REPORTS / "resume_claim_ledger.json", claims)
    lines = ["# Resume Evidence Ledger", ""]
    for claim in claims:
        lines.extend(
            [
                f"## {claim['claim_text']}",
                "",
                f"- Evidence: `{claim['evidence_file']}`",
                f"- Fields: `{', '.join(claim['exact_metric_fields'])}`",
                f"- Class/boundary: {claim['evidence_class']} — {claim['claim_boundary']}",
                f"- Status: **{claim['status']}**",
                "",
            ]
        )
    (REPORTS / "RESUME_EVIDENCE_LEDGER.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    copy_tables()
    plot_reports()
    reports()
    print(json.dumps({"status": "PASS", "reports": sorted(str(p) for p in REPORTS.rglob("*"))}))


if __name__ == "__main__":
    main()
