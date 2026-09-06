# Public Evidence Index

Only compact, aggregate, public-safe artifacts are indexed here. Raw data, row-level outcomes, feature-policy Parquet, model artifacts, Spark event logs, local telemetry dumps, and machine-specific manifests remain excluded. For comparison-aware interpretation, start with [Results Analysis](RESULTS_ANALYSIS.md).

| Public artifact | What it supports | What it does **not** support |
|---|---|---|
| [Prospective summary](tables/prospective_summary.csv) | E7-minus-comparator point estimates, intervals, p-values, Holm decisions, and the primary 20% E7/C3 result on the fresh 2M cohort | A claim that E7 beat every comparator; raw outcomes; reuse of the cohort |
| [Final prospective audit](../evidence/prospective/final_audit.json) | Contract/seal digest matches, row-identity match, attempt history, report agreement, `REPLICATED`, and both primary gates | Disclosure of row-level features/outcomes; proof of online lift |
| [Completion gates](../evidence/final_audit/completion_gates.json) | Final PASS gates, the retained Causal Forest blocker, and optional omissions | Evidence that omitted work ran or that the system is production ready |
| [Historical parity](../evidence/spark/historical_parity/metrics.json) | Exact Spark/reference score, membership, and AIPW parity for E7 and C3 on 2M rows | Physical multi-node execution, faster Spark performance, or a new causal holdout |
| [Systems scale](../evidence/systems/scale/metrics.json) and [table](tables/systems_scale.csv) | Authentic 13,979,592-row feature-only scoring throughput and absence of outcome access | End-to-end ingestion/training/evaluation throughput, speedup at 13.98M, or causal validity |
| [Local scaling](../evidence/systems/local_scaling/metrics.json) and [table](tables/systems_local_scaling.csv) | Identical 1M scoring workload at one, two, and four same-machine workers, including speedup and efficiency | Monotonic scaling, physical nodes, or general cluster efficiency |
| [GPU benchmark](../evidence/gpu/controlled_benchmark/metrics.json) and [table](tables/gpu_benchmark.csv) | Actual RTX/XGBoost CUDA qualification, recorded GPU times, inference ratio, peak VRAM, and agreement | GPU training speedup, statistically stable hardware performance, or improved causal validity |
| [Model benchmark](../evidence/retrospective/model_benchmark/metrics.json) and [table](tables/model_benchmark.csv) | Common retrospective comparison at fixed capacities/evaluators, with primary-capacity uncertainty | Prospective superiority for all models; Criteo PEHE; direct response-versus-E7 significance where not stored |
| [Capacity frontier](../evidence/retrospective/capacity_frontier/capacity_frontier.csv) | Retrospective E7/C3 point estimates across 5%, 10%, 20%, 30%, and 50% capacity | Prospective results outside 20% or uncertainty at every capacity |
| [Historical metrics](../evidence/historical_reference/historical_metrics.json) | Frozen ORBIT-Causal v0.3.9 E7/C3 estimates, intervals, gates, and release-eligible verdict | A fresh CAUSAL-SCALE result or a claim about other comparators |
| [Causal Forest blocker](../evidence/models/causal_forest_retry/metrics.json) | Two unchanged-parameter attempts and the exact sandbox failure classification | Evidence that Causal Forest underperformed scientifically |
| [Final Status](FINAL_STATUS.md) | Human-readable completion snapshot and durable limitations | A substitute for machine-readable evidence or an expanded claim boundary |
| [Technical Report](TECHNICAL_REPORT.md) | Study scope, evidence-class separation, methods narrative, and prospective status | Row-level evidence, online outcomes, or unrecorded experiments |
| [Resume Evidence Ledger](RESUME_EVIDENCE_LEDGER.md) | Approved source-to-claim mappings for the most important public statements | Authorization for claims beyond the cited fields |
| [Results Analysis](RESULTS_ANALYSIS.md) | Baseline-aware interpretation, descriptive ratios, uncertainty, systems overhead, and resume-safe wording | New scientific estimates or a rerun of any experiment |

The selected public machine-readable files are intentionally a subset of the local evidence tree. Ignored originals remain available to the release operator for local verification, but are not publication candidates when they contain machine-specific paths, raw artifacts, or unnecessary environment detail.
