from __future__ import annotations

import argparse
import json
import shutil
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from causal_scale.evidence.hashing import sha256_file
from causal_scale.evidence.manifest import atomic_json
from causal_scale.models.historical_adapter import HistoricalPortfolioAdapter
from causal_scale.policy.engine import capacity_policy
from causal_scale.spark.aipw import aggregate_aipw, with_aipw_score
from causal_scale.spark.artifacts import distribute_artifact
from causal_scale.spark.eventlog import event_log_for_application, summarize_event_log
from causal_scale.spark.policy import exact_capacity_policy
from causal_scale.spark.scoring import load_distributed_historical, score_map_in_pandas
from causal_scale.spark.session import create_session

EXPECTED = {"HIST_E7": 0.006041015750860274, "HIST_C3": 0.005643564219054695}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=Path(".local-data/marts/historical_locked_2m.parquet")
    )
    parser.add_argument("--master", default="local[4]")
    parser.add_argument("--partitions", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("evidence/spark/historical_parity"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    event_dir = Path(".local-spark/eventlog")
    spark = create_session(args.master, "causal-scale-historical-parity", event_dir)
    application_id = spark.sparkContext.applicationId
    try:
        source_zip = Path(".local-spark/orbit_causal_source")
        source_zip.parent.mkdir(parents=True, exist_ok=True)
        archive = Path(shutil.make_archive(str(source_zip), "zip", ".reference/ORBIT-Causal/src"))
        spark.sparkContext.addPyFile(str(archive.resolve()))
        artifact = Path(".reference/ORBIT-Causal/artifacts/frozen_candidate_portfolio.joblib")
        artifact_name = distribute_artifact(spark, artifact)
        source = spark.read.parquet(str(args.data.resolve())).repartition(args.partitions).cache()
        row_count = source.count()
        if row_count != 2_000_000:
            raise ValueError("historical parity requires exactly 2M rows")
        pandas = pd.read_parquet(args.data)
        adapter = HistoricalPortfolioAdapter()
        mu0, mu1 = adapter.predict_outcomes(pandas)
        results = {}
        for model_id in ("HIST_E7", "HIST_C3"):
            start = time.perf_counter()
            loader = partial(load_distributed_historical, artifact_name, model_id)
            scored = score_map_in_pandas(source, loader).cache()
            scored_count = scored.count()
            spark_policy = (
                exact_capacity_policy(scored, 0.2)
                .select("source_row_id", "score", "policy")
                .cache()
            )
            spark_policy.count()
            reference_score = adapter.predict_effect(model_id, pandas)
            reference_policy = capacity_policy(
                reference_score, 0.2, historical_position=np.arange(len(pandas))
            ).membership
            reference = pd.DataFrame(
                {
                    "source_row_id": pandas.source_row_id,
                    "reference_score": reference_score,
                    "reference_policy": reference_policy,
                    "visit": pandas.visit.astype(float),
                    "treatment": pandas.treatment.astype(int),
                    "mu0": mu0,
                    "mu1": mu1,
                }
            )
            reference_path = Path(f".local-spark/{model_id.lower()}_reference.parquet")
            reference.to_parquet(reference_path, index=False)
            reference_spark = spark.read.parquet(str(reference_path.resolve()))
            from pyspark.sql import functions as f

            joined = spark_policy.join(reference_spark, "source_row_id")
            summary = joined.agg(
                f.count("*").alias("rows"),
                f.max(f.abs(f.col("score") - f.col("reference_score"))).alias("max_score_error"),
                f.sum((f.col("policy") != f.col("reference_policy")).cast("long")).alias(
                    "membership_mismatches"
                ),
            ).first()
            spark_aipw = aggregate_aipw(
                with_aipw_score(
                    joined,
                    adapter.propensity,
                    policy="policy",
                    outcome="visit",
                    treatment="treatment",
                    mu0="mu0",
                    mu1="mu1",
                )
            )
            value = spark_aipw["policy_value"]
            results[model_id] = {
                "rows": int(summary.rows),
                "scored_rows": scored_count,
                "max_score_error": float(summary.max_score_error),
                "membership_mismatches": int(summary.membership_mismatches),
                "policy_agreement": 1 - int(summary.membership_mismatches) / row_count,
                "aipw_value": value,
                "aipw_rows": spark_aipw["rows"],
                "expected_aipw_value": EXPECTED[model_id],
                "aipw_abs_error": abs(value - EXPECTED[model_id]),
                "runtime_seconds": time.perf_counter() - start,
            }
        passed = all(
            item["rows"] == 2_000_000
            and item["max_score_error"] <= 1e-12
            and item["membership_mismatches"] == 0
            and item["aipw_abs_error"] <= 1e-12
            for item in results.values()
        )
        report = {
            "schema": "causal-scale-spark-historical-parity-v1",
            "status": "PASS" if passed else "FAIL",
            "spark_version": spark.version,
            "application_id": spark.sparkContext.applicationId,
            "master": args.master,
            "partitions": args.partitions,
            "artifact_sha256": sha256_file(artifact),
            "models": results,
            "historical_verdict": "RELEASE_ELIGIBLE" if passed else "PARITY_FAILURE",
        }
        atomic_json(args.output / "metrics.json", report)
        atomic_json(
            args.output / "spark_application.json",
            {
                "application_id": spark.sparkContext.applicationId,
                "spark_version": spark.version,
                "master": args.master,
            },
        )
        print(json.dumps(report, indent=2))
        if not passed:
            raise SystemExit(1)
    finally:
        spark.stop()
    event_summary = summarize_event_log(event_log_for_application(event_dir, application_id))
    atomic_json(args.output / "stage_metrics.json", event_summary)
    atomic_json(
        args.output / "executor_metrics.json",
        {
            "application_id": application_id,
            "executors": event_summary["executors"],
            "shuffle_read_bytes": event_summary["shuffle_read_bytes"],
            "shuffle_write_bytes": event_summary["shuffle_write_bytes"],
            "spill_bytes": event_summary["spill_bytes"],
        },
    )


if __name__ == "__main__":
    main()
