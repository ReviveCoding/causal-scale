from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from causal_scale.evidence.manifest import atomic_json
from causal_scale.spark.eventlog import event_log_for_application, summarize_event_log
from causal_scale.spark.policy import exact_capacity_policy
from causal_scale.spark.session import create_session


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores", type=Path, default=Path("evidence/retrospective/model_benchmark/scores.parquet")
    )
    parser.add_argument("--output", type=Path, default=Path("evidence/spark/partition_invariance"))
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args()
    spark = create_session(
        "local[4]", "causal-scale-partition-invariance", Path(".local-spark/eventlog")
    )
    application_id = spark.sparkContext.applicationId
    partitions = [2, 4, 8] if args.fixture else [32, 64, 128, 256, 512]
    try:
        if args.fixture:
            base = spark.createDataFrame(
                pd.DataFrame(
                    {
                        "source_row_id": range(10_000),
                        "score": [float(i % 101) for i in range(10_000)],
                    }
                )
            )
        else:
            base = spark.read.parquet(str(args.scores.resolve())).select(
                "source_row_id", pd_col("HIST_E7").alias("score")
            )
        memberships = []
        for count in partitions:
            current = exact_capacity_policy(base.repartition(count), 0.2).select(
                "source_row_id", "policy"
            )
            memberships.append(current.withColumnRenamed("policy", f"policy_{count}"))
        joined = memberships[0]
        for current in memberships[1:]:
            joined = joined.join(current, "source_row_id")
        from pyspark.sql import functions as f

        columns = [f"policy_{count}" for count in partitions]
        mismatch = joined.filter(
            f.greatest(*[f.col(name) for name in columns])
            != f.least(*[f.col(name) for name in columns])
        ).count()
        report = {
            "status": "PASS" if mismatch == 0 else "FAIL",
            "partitions": partitions,
            "membership_mismatches": mismatch,
            "rows": joined.count(),
        }
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output / "metrics.json", report)
        print(json.dumps(report, indent=2))
        if mismatch:
            raise SystemExit(1)
    finally:
        spark.stop()
    event_dir = Path(".local-spark/eventlog")
    summary = summarize_event_log(event_log_for_application(event_dir, application_id))
    atomic_json(args.output / "spark_application.json", summary)
    atomic_json(args.output / "stage_metrics.json", summary)
    atomic_json(
        args.output / "executor_metrics.json",
        {key: summary[key] for key in ("application_id", "executors", "spill_bytes")},
    )


def pd_col(name: str):
    from pyspark.sql import functions as f

    return f.col(name)


if __name__ == "__main__":
    main()
