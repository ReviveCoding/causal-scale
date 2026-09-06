from __future__ import annotations


def exact_capacity_policy(
    frame,
    capacity: float,
    *,
    score_column: str = "score",
    tie_column: str = "source_row_id",
    positive_only: bool = True,
):
    if not 0 < capacity <= 1:
        raise ValueError("capacity must be in (0,1]")
    from pyspark.sql import Window
    from pyspark.sql import functions as functions

    total = frame.count()
    limit = int(capacity * total)
    window = Window.orderBy(functions.desc(score_column), functions.asc(tie_column))
    ranked = frame.withColumn("rank", functions.row_number().over(window))
    eligible = functions.col(score_column) > 0 if positive_only else functions.lit(True)
    return ranked.withColumn("policy", (eligible & (functions.col("rank") <= limit)).cast("byte"))
