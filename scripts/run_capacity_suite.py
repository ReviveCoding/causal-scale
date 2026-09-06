from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("evidence/retrospective/model_benchmark/model_benchmark.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evidence/retrospective/capacity_frontier/capacity_frontier.csv"),
    )
    args = parser.parse_args()
    frame = pd.read_csv(args.benchmark)
    subset = frame[
        (frame.evaluator == "HISTORICAL_AIPW") & frame.model.isin(["HIST_E7", "HIST_C3"])
    ]
    pivot = subset.pivot(index="capacity", columns="model", values="policy_value").reset_index()
    pivot["e7_minus_c3"] = pivot.HIST_E7 - pivot.HIST_C3
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
