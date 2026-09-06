from __future__ import annotations


def calibration_aggregates(
    frame, bins: int = 10, score_column: str = "score", pseudo_outcome_column: str = "gamma"
):
    from pyspark.sql import Window
    from pyspark.sql import functions as f

    binned = frame.withColumn(
        "calibration_bin", f.ntile(bins).over(Window.orderBy(f.col(score_column)))
    )
    return (
        binned.groupBy("calibration_bin")
        .agg(
            f.count("*").alias("rows"),
            f.avg(score_column).alias("predicted_effect"),
            f.avg(pseudo_outcome_column).alias("dr_observed_effect"),
        )
        .orderBy("calibration_bin")
    )
