from __future__ import annotations


def cluster_sufficient_statistics(frame, cluster_column: str, value_columns: list[str]):
    from pyspark.sql import functions as f

    return frame.groupBy(cluster_column).agg(
        f.count("*").alias("cluster_rows"),
        *[f.sum(column).alias(f"{column}_sum") for column in value_columns],
    )
