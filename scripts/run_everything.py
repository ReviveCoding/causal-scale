from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from causal_scale.evidence.hashing import sha256_file, stable_hash, tree_manifest
from causal_scale.orchestration.state import DurableState


def command_phase(
    name: str,
    command: list[str],
    outputs: list[Path],
    state: DurableState,
    context: dict[str, object],
    resume: bool,
) -> bool:
    input_hash = stable_hash(
        {
            "name": name,
            "command": command,
            "context": context,
            "source": tree_manifest("src", ("*.py",)),
            "configs": tree_manifest("configs"),
        }
    )
    current_outputs = {str(path): sha256_file(path) for path in outputs if path.is_file()}
    if (
        resume
        and len(current_outputs) == len(outputs)
        and state.valid(name, input_hash, current_outputs)
    ):
        print(json.dumps({"phase": name, "status": "CACHED"}))
        return True
    process = subprocess.run(command)
    current_outputs = {str(path): sha256_file(path) for path in outputs if path.is_file()}
    status = (
        "PASS" if process.returncode == 0 and len(current_outputs) == len(outputs) else "BLOCKED"
    )
    state.record(
        name, status, input_hash, current_outputs, exit_code=process.returncode, command=command
    )
    print(json.dumps({"phase": name, "status": status, "exit_code": process.returncode}))
    return status == "PASS"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["smoke", "standard", "full"], default="standard")
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--gpu", choices=["auto", "off", "force"], default="auto")
    parser.add_argument("--gpu-parallel", default="auto")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.8)
    parser.add_argument("--spark", choices=["auto", "off", "force"], default="auto")
    parser.add_argument("--max-cpu-workers", type=int)
    parser.add_argument(
        "--resource-profile", choices=["conservative", "auto", "aggressive"], default="auto"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-before-prospective", action="store_true")
    args = parser.parse_args()
    state = DurableState(".local-runs/orchestrator/state.json")
    python = sys.executable
    context = vars(args)
    phases = [
        (
            "00_environment_preflight",
            [python, "scripts/preflight.py"],
            [Path("evidence/environment/preflight/environment.json")],
        ),
        (
            "01_historical_audit",
            [python, "scripts/audit_historical_reference.py"],
            [Path("evidence/historical_reference/reference_manifest.json")],
        ),
        (
            "03_dataset_acquisition",
            [
                python,
                "scripts/fetch_datasets.py",
                "--required" if args.download_missing else "--verify-only",
            ],
            [Path("evidence/data/criteo_v2_1/dataset_manifest.json")],
        ),
        (
            "04_row_roles",
            [python, "scripts/reconstruct_roles.py"],
            [Path("evidence/data/row_roles/role_manifest.json")],
        ),
        (
            "05_criteo_marts",
            [python, "scripts/build_criteo_mart.py"],
            [Path("evidence/data/criteo_marts/run_manifest.json")],
        ),
        (
            "06_known_truth",
            [
                python,
                "-m",
                "pytest",
                "-q",
                "tests/test_known_truth_estimators.py",
                "--basetemp",
                ".local-runs/pytest-known-truth",
            ],
            [],
        ),
        (
            "07_models",
            [
                python,
                "scripts/fit_baselines.py",
                "--profile",
                args.profile,
                "--gpu",
                args.gpu,
                *(["--include-forest"] if args.profile == "full" else []),
            ],
            [Path("evidence/models/baselines/run_manifest.json")],
        ),
        (
            "08_common_calibration",
            [python, "scripts/run_common_calibration.py"],
            [Path("evidence/models/common_calibration/metrics.json")],
        ),
        (
            "11_retrospective",
            [python, "scripts/run_retrospective.py"],
            [Path("evidence/retrospective/model_benchmark/metrics.json")],
        ),
        (
            "11_retrospective_inference",
            [python, "scripts/add_retrospective_inference.py"],
            [Path("evidence/retrospective/model_benchmark/model_benchmark.csv")],
        ),
        (
            "12_ablation",
            [python, "scripts/run_ablation.py"],
            [Path("evidence/retrospective/ablation/ablation.csv")],
        ),
    ]
    if args.spark != "off":
        phases += [
            (
                "14_spark_parity",
                [python, "scripts/run_spark_parity.py"],
                [Path("evidence/spark/historical_parity/metrics.json")],
            ),
            (
                "15_partition_invariance",
                [python, "scripts/run_partition_invariance.py"],
                [Path("evidence/spark/partition_invariance/metrics.json")],
            ),
            (
                "17_scale",
                [python, "scripts/run_scale_suite.py"],
                [Path("evidence/systems/scale/metrics.json")],
            ),
            (
                "18_local_scaling",
                [python, "scripts/run_local_scaling.py"],
                [Path("evidence/systems/local_scaling/metrics.json")],
            ),
            (
                "13_spark_backends",
                [python, "scripts/run_spark_backends.py"],
                [Path("evidence/systems/spark_backends/metrics.json")],
            ),
        ]
    phases += [
        (
            "16_gpu_benchmark",
            [python, "scripts/run_gpu_suite.py"],
            [Path("evidence/gpu/controlled_benchmark/metrics.json")],
        ),
        (
            "20_capacity",
            [python, "scripts/run_capacity_suite.py"],
            [Path("evidence/retrospective/capacity_frontier/capacity_frontier.csv")],
        ),
        (
            "21_external",
            [python, "scripts/run_external_benchmarks.py"],
            [Path("evidence/external/benchmark/metrics.json")],
        ),
        (
            "22_23_reviews",
            [python, "scripts/run_reviews.py"],
            [
                Path("evidence/reviews/science_review.json"),
                Path("evidence/reviews/systems_review.json"),
            ],
        ),
        (
            "23_evidence_manifests",
            [python, "scripts/finalize_evidence.py"],
            [Path("evidence/historical_reference/SHA256SUMS")],
        ),
        (
            "24_quality",
            [python, "scripts/qualify.py"],
            [Path("evidence/qualification/engineering/qualification_summary.json")],
        ),
    ]
    for name, command, outputs in phases:
        if not command_phase(name, command, outputs, state, context, args.resume):
            print(json.dumps({"orchestrator": "HALTED", "blocked_phase": name}))
            return
    if args.stop_before_prospective:
        print(json.dumps({"orchestrator": "STOPPED_BEFORE_PROSPECTIVE", "status": "PASS"}))
        return
    if not command_phase(
        "26_scientific_contract",
        [python, "scripts/seal_prospective.py"],
        [Path("evidence/prospective/scientific_contract.json")],
        state,
        context,
        args.resume,
    ):
        return
    if not command_phase(
        "27_29_feature_policy_seal",
        [python, "scripts/run_prospective.py", "--stage", "seal-features"],
        [Path("evidence/prospective/feature_policy_seal.json")],
        state,
        context,
        args.resume,
    ):
        return
    if not command_phase(
        "30_32_prospective_evaluation",
        [python, "scripts/run_prospective.py", "--stage", "evaluate"],
        [Path("evidence/prospective/prospective_results.json")],
        state,
        context,
        args.resume,
    ):
        return
    command_phase(
        "33_evidence_audit",
        [python, "scripts/finalize_evidence.py"],
        [Path("evidence/prospective/SHA256SUMS")],
        state,
        context,
        False,
    )
    command_phase(
        "35_39_reports",
        [python, "scripts/generate_reports.py"],
        [Path("reports/FINAL_STATUS.md"), Path("reports/TECHNICAL_REPORT.md")],
        state,
        context,
        args.resume,
    )


if __name__ == "__main__":
    main()
