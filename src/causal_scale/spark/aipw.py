from __future__ import annotations


def with_aipw_score(
    frame,
    propensity: float,
    *,
    outcome: str = "visit",
    treatment: str = "treatment",
    policy: str = "policy",
    mu0: str = "mu0",
    mu1: str = "mu1",
):
    if not 0 < propensity < 1:
        raise ValueError("propensity must be in (0,1)")
    from pyspark.sql import functions as f

    t, y, pi = f.col(treatment), f.col(outcome), f.col(policy)
    m0, m1 = f.col(mu0), f.col(mu1)
    observed_mu = f.when(t == 1, m1).otherwise(m0)
    observed_probability = f.when(t == 1, f.lit(propensity)).otherwise(f.lit(1 - propensity))
    target_mu = f.when(pi == 1, m1).otherwise(m0)
    target = target_mu + f.when(t == pi, (y - observed_mu) / observed_probability).otherwise(
        f.lit(0.0)
    )
    never = m0 + f.when(t == 0, (y - observed_mu) / observed_probability).otherwise(f.lit(0.0))
    return frame.withColumn("aipw_score", target - never)


def aggregate_aipw(frame) -> dict[str, float]:
    from pyspark.sql import functions as f

    row = frame.agg(f.avg("aipw_score").alias("policy_value"), f.count("*").alias("rows")).first()
    return {"policy_value": float(row["policy_value"]), "rows": int(row["rows"])}
