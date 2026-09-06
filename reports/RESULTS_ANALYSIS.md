# Results Analysis

This document interprets the completed CAUSAL-SCALE evidence without rerunning or changing the experiment. Percentages below are descriptive ratios computed from the stored point estimates; all scientific estimates and intervals come from the linked evidence.

## Prospective causal result

The predeclared comparison was frozen **HIST_E7 minus HIST_C3**, where HIST_C3 is the frozen historical doubly robust baseline. Both policies were evaluated at 20% capacity on a fresh, sealed, disjoint 2,000,000-row cohort.

| Quantity | E7 | HIST_C3 baseline | E7 − C3 |
|---|---:|---:|---:|
| Incremental policy value | 0.006312739031 | 0.005908982394 | **+0.000403756637** |
| Descriptive relative difference | — | denominator | **+6.8329%** |
| Paired two-sided 95% interval for difference | — | — | **[0.000121794021, 0.000701070814]** |
| Predeclared one-sided 95% lower bound | — | margin = -0.00025 | **0.000162144489** |

The H1 rule was noninferiority: the one-sided lower bound had to exceed `-0.00025`. It did, so H1 passed. The paired two-sided interval also lies above zero, but the release claim remains the predeclared one: **E7 prospectively replicated relative to HIST_C3**. This is prospective causal evidence under the declared AIPW estimand. It is not evidence of online revenue, universal external validity, or superiority over every causal and non-causal comparator. Sources: [prospective comparison table](tables/prospective_summary.csv) and [final audit](../evidence/prospective/final_audit.json).

## Baseline comparison

### Prospective direct comparisons

Every entry below is `E7 − comparator` on the same fresh cohort. Positive favors E7; negative favors the comparator. Intervals are cluster-robust normal 95% intervals for secondary comparisons, with two-sided p-values evaluated against the shown Holm threshold within evaluator. Relative differences are not reported for these secondary contrasts because the public paired-comparison table does not retain every comparator's absolute value under every evaluator; applying E7's primary AIPW level to other evaluators would be invalid.

**Frozen historical AIPW evaluator**

| Comparator baseline | Absolute difference | 95% interval | p / Holm threshold | Statistically supported? |
|---|---:|---:|---:|---|
| RESPONSE_MARGINAL | -0.0002827131 | [-0.0006360881, 0.0000706620] | 0.1169 / 0.0250 | No; response has the higher point estimate |
| RESPONSE_TREATED | -0.0002934786 | [-0.0006452752, 0.0000583180] | 0.1020 / 0.0167 | No; response has the higher point estimate |
| HIST_C1 | +0.0001091179 | [-0.0000625599, 0.0002807958] | 0.2129 / 0.0500 | No |
| GPU_T | +0.0004865651 | [0.0002037547, 0.0007693755] | 0.000746 / 0.0100 | Yes, favors E7 |
| X_LEARNER | +0.0004207516 | [0.0001123408, 0.0007291625] | 0.007498 / 0.0125 | Yes, favors E7 |
| R_LEARNER | +0.0006896867 | [0.0003766528, 0.0010027205] | 0.0000157 / 0.00833 | Yes, favors E7 |

**Randomized-propensity IPW/HT evaluator**

| Comparator baseline | Absolute difference | 95% interval | p / Holm threshold | Statistically supported? |
|---|---:|---:|---:|---|
| RESPONSE_MARGINAL | +0.0000190580 | [-0.0004274064, 0.0004655224] | 0.9333 / 0.0250 | No |
| RESPONSE_TREATED | +0.0000032047 | [-0.0004419656, 0.0004483751] | 0.9887 / 0.0500 | No |
| HIST_C1 | +0.0002881528 | [0.0000611882, 0.0005151174] | 0.01283 / 0.0167 | Yes, favors E7 |
| GPU_T | +0.0009350238 | [0.0005767353, 0.0012933122] | 0.000000314 / 0.0100 | Yes, favors E7 |
| X_LEARNER | +0.0007628696 | [0.0003772444, 0.0011484948] | 0.0001056 / 0.0125 | Yes, favors E7 |
| R_LEARNER | +0.0011066228 | [0.0007150913, 0.0014981543] | 0.0000000303 / 0.00833 | Yes, favors E7 |

**Independently nuisance-fitted AIPW evaluator**

| Comparator baseline | Absolute difference | 95% interval | p / Holm threshold | Statistically supported? |
|---|---:|---:|---:|---|
| RESPONSE_MARGINAL | -0.0002998246 | [-0.0006545310, 0.0000548819] | 0.09758 / 0.0250 | No; response has the higher point estimate |
| RESPONSE_TREATED | -0.0003096991 | [-0.0006628363, 0.0000434380] | 0.08564 / 0.0167 | No; response has the higher point estimate |
| HIST_C1 | +0.0001246119 | [-0.0000475854, 0.0002968091] | 0.1561 / 0.0500 | No |
| GPU_T | +0.0004357279 | [0.0001515598, 0.0007198959] | 0.002653 / 0.0100 | Yes, favors E7 |
| X_LEARNER | +0.0003697927 | [0.0000604517, 0.0006791336] | 0.01913 / 0.0125 | No after Holm |
| R_LEARNER | +0.0006534437 | [0.0003393761, 0.0009675114] | 0.0000455 / 0.00833 | Yes, favors E7 |

The direct prospective evidence therefore supports E7 over GPU_T and R_LEARNER across all three evaluators, and over X_LEARNER in two of three after Holm correction. It does not support E7 superiority over either response policy. Response point estimates are higher under the two AIPW evaluators, while E7 is marginally higher under IPW/HT; none of those six direct response comparisons is significant.

Causal Forest has no prospective comparison because two unchanged-parameter fitting attempts ended in the recorded sandbox multiprocessing blocker. Constant ATE and deterministic random allocation were retrospective controls, not members of the predeclared prospective secondary family.

### Retrospective major-baseline context

These historical 2M results use the historical AIPW evaluator at 20% capacity. Differences, relative differences, and intervals are all against HIST_C3—not against E7.

| Model | Policy value | Difference vs C3 | Relative vs C3 | 95% interval for difference | Supported vs C3? |
|---|---:|---:|---:|---:|---|
| HIST_C1 | 0.00588035 | +0.00023679 | +4.20% | [-0.00009825, 0.00057182] | No |
| HIST_C3 | 0.00564356 | 0 | 0% | reference | — |
| HIST_E7 | 0.00604102 | +0.00039745 | +7.04% | [0.00011070, 0.00068190] | Yes |
| CONSTANT_ATE | 0.00132405 | -0.00431952 | -76.54% | [-0.00480306, -0.00383597] | Yes, favors C3 |
| GPU_T | 0.00581725 | +0.00017369 | +3.08% | [-0.00014530, 0.00049268] | No |
| X_LEARNER | 0.00568660 | +0.00004304 | +0.76% | [-0.00026587, 0.00035195] | No |
| R_LEARNER | 0.00557473 | -0.00006883 | -1.22% | [-0.00038005, 0.00024239] | No |
| RESPONSE_MARGINAL | 0.00659629 | +0.00095272 | +16.88% | [0.00060135, 0.00130410] | Yes |
| RESPONSE_TREATED | 0.00661577 | +0.00097220 | +17.23% | [0.00062287, 0.00132154] | Yes |
| RANDOM | 0.00132405 | -0.00431952 | -76.54% | [-0.00480306, -0.00383597] | Yes, favors C3 |

The response policies rank above E7 by point estimate in this retrospective view, but the stored intervals compare each to C3, not response to E7. They cannot establish response superiority over E7. The prospective direct response comparisons above also fail to establish a difference. Source: [model benchmark](tables/model_benchmark.csv).

The historical ablation adds C3 common calibration, C5, and the pre-final blend. Common calibration changes C3 calibration RMSE (`0.00947047` to `0.00637642`) without changing its ranking or value; E7 and the pre-final blend share policy value `0.00604102`, while E7 improves their calibration RMSE from `0.00982456` to `0.00508840`. Those are retrospective component results, not fresh comparative claims.

## Calibration analysis

Prospectively, E7 calibration RMSE was `0.0049530331` versus `0.0088863005` for C3. The absolute change was `-0.0039332673`, a **44.2621% reduction** relative to C3. The predeclared one-sided 95% upper bound was `-0.0024691233`, below zero, so H2 calibration superiority passed. The claim is metric- and setting-specific: better ten-bin causal calibration RMSE than frozen C3 on the sealed prospective cohort. It does not show universal probability calibration, better online decisions, or calibration superiority over response models (which were not subjected to causal calibration).

Historically, E7's RMSE was `0.0050883993` versus C3's `0.0094704722`, an absolute reduction of `0.0043820729` (**46.2709%**). The paired two-sided 95% interval for E7 minus C3 was `[-0.0059205252, -0.0028630619]`. This retrospective result agrees in direction with the prospective result but remains a separate evidence class.

## Capacity analysis

The stored capacity frontier is retrospective and compares E7 with C3 under the historical evaluator. Positive differences favor E7.

| Capacity | E7 | C3 baseline | Absolute difference | Relative difference vs C3 | Uncertainty |
|---:|---:|---:|---:|---:|---|
| 5% | 0.00371291 | 0.00362927 | +0.00008364 | +2.30% | Not stored |
| 10% | 0.00500122 | 0.00489660 | +0.00010462 | +2.14% | Not stored |
| 20% | 0.00604102 | 0.00564356 | +0.00039745 | +7.04% | 95% difference interval [0.00011070, 0.00068190] |
| 30% | 0.00631294 | 0.00599850 | +0.00031444 | +5.24% | Not stored |
| 50% | 0.00647952 | 0.00633009 | +0.00014943 | +2.36% | Not stored |

E7 has the higher retrospective point estimate at every stored capacity. Only the 20% primary-capacity difference has retained uncertainty here, and the fresh prospective test was conducted only at 20%. The evidence does not establish prospective superiority across capacities. Source: [capacity frontier](../evidence/retrospective/capacity_frontier/capacity_frontier.csv).

## Spark parity

The baseline for distributed parity is the frozen reference implementation, not another learned model. On the historical 2M rows, Spark reproduced both HIST_C3 and HIST_E7 with maximum score error `0`, policy membership mismatches `0`, and policy agreement `1.0`. AIPW absolute errors were `5.03e-17` for C3 and `5.29e-17` for E7. This is an exact deterministic check, so a sampling interval and relative effect are not meaningful.

The claim supported is scientific implementation parity for scores, global capacity policy selection, and AIPW results on the tested same-machine Spark workload. It is not a new causal estimate, a claim of faster execution, partition-independent performance, or evidence from physical cluster nodes.

The historical-parity event summary recorded one executor, 39 jobs/stages, 1,336 tasks, about 297 MB each of shuffle read and write, and about 860 MB of spill. Those records support attributing substantial execution overhead to shuffle/spill around global policy work; they do not isolate a causal runtime contribution for each Spark stage. Source: [parity metrics](../evidence/spark/historical_parity/metrics.json).

## Scaling analysis

### Identical-workload local scaling

The only valid worker speedups compare the same 1M-row `MAP_IN_PANDAS` scoring-only workload against the one-worker run.

| Same-machine workers | Runtime | Throughput | Speedup vs 1 | Parallel efficiency | Rank |
|---:|---:|---:|---:|---:|---|
| 1 | 36.6363 s | 27,295 rows/s | 1.000x | 100.00% | 3 |
| 2 | **22.2093 s** | **45,026 rows/s** | **1.650x** | **82.48%** | **1 (best)** |
| 4 | 28.0226 s | 35,685 rows/s | 1.307x | 32.68% | 2 |

Scaling is non-monotonic: four workers are slower than two. Event summaries show a single executor for every run, nearly constant shuffle read/write volume around 26.64 MB, zero spill, and task counts rising from 8 to 12 to 22. Together with the shared machine, that is consistent with task scheduling, serialization, coordination, and local resource contention overwhelming added parallelism at four workers. The summaries do not contain stage-level causal decomposition, so this remains an evidence-supported likely explanation, not a proven attribution.

### Authentic-row-count workload

The 13,979,592-row feature-only `SPARK_MAP_IN_PANDAS` workload on `local[4]` took `68.0202 s` at `205,521 rows/s`. Its comparison baseline is simply the same recorded workload definition; no alternative configuration at 13.98M was recorded, so no speedup, efficiency, or uncertainty interval is available. The event summary shows one executor, 14 jobs/stages, 35 tasks, only 1,380 bytes each of shuffle read/write, and zero spill. This supports describing the workload as low-shuffle scoring.

The smaller rows in the same scale sweep have lower throughput (for example, 1M at `51,507 rows/s`), but row counts differ and the feature frame was cached before the sweep. The higher full-size throughput is consistent with fixed Spark startup/scheduling overhead being amortized; it is not a valid parallel speedup. These are **scoring-only** measurements: they do not include model training, complete raw-data ingestion, causal evaluation, or a production end-to-end service path. No end-to-end headline result was recorded. Sources: [scale table](tables/systems_scale.csv) and [local scaling table](tables/systems_local_scaling.csv).

## GPU performance

The comparison baseline is an XGBoost T-learner run on CPU against the corresponding CUDA T-learner on the same controlled 100K fixture. The evidence directly retains GPU training and inference time plus the ratio `CPU inference / GPU inference`.

| Metric | CPU | GPU | Difference / ratio |
|---|---:|---:|---:|
| Inference runtime | 0.0765772 s (derived from stored ratio) | 0.1105055 s | GPU +0.0339283 s; `0.693x` CPU/GPU speedup, so GPU was 44.31% slower |
| Training runtime | Not retained | 0.4352295 s | Training speedup cannot be audited |
| Peak VRAM | — | 212 MiB | Recorded telemetry maximum |
| Prediction agreement | baseline | 0.903081 correlation | Not exact prediction parity |
| Policy agreement | baseline | 91.494% | 8.506 percentage points disagree |

There is no repeated-run uncertainty interval. The empirically supported systems claim is that XGBoost used the NVIDIA GeForce RTX 4090 Laptop GPU and produced broadly aligned—but not identical—predictions and policies. The controlled run does not show an inference speedup, does not retain enough evidence for a training speedup, and says nothing about causal validity. Source: [GPU metrics](../evidence/gpu/controlled_benchmark/metrics.json).

## Failure/limitation analysis

- **Causal Forest:** two attempts failed with a Windows sandbox multiprocessing `PermissionError`. The retry moved joblib temporary storage locally but retained 400 trees, three-fold CV, and four jobs. Scientific parameters were not weakened. This supports a tooling blocker, not a negative scientific result for Causal Forest.
- **Optional work:** DragonNet/JAX was skipped for GPU environment reasons; full IHDP replications were resource-skipped; Hillstrom, ONNX CUDA, and physical multi-node Spark were not run. Absence of these results cannot be interpreted as success or failure.
- **External evidence:** Twins passed as a known-paired-potential-outcomes benchmark, but a single benchmark construction does not establish universal external validity.
- **Outcome boundary:** offline incremental visit-policy value is neither observed online revenue nor production ROI. The authentic 13.98M scale run accessed no outcomes and is not an independent causal holdout.
- **Hardware boundary:** Spark processes shared one machine, and one heavy GPU job ran at a time. Neither physical cluster scaling nor concurrent multi-GPU scaling was demonstrated.
- **Response nuance:** response targeting has higher point estimates in several AIPW views. Direct prospective differences from E7 are not significant, so neither E7-over-response nor response-over-E7 superiority is supported.

## Resume-safe claims

The following wording is supported by the completed evidence:

- “E7 prospectively replicated relative to the frozen HIST_C3 doubly robust baseline at 20% capacity on a sealed, disjoint 2M cohort.”
- “Both predeclared co-primary gates passed: value noninferiority and calibration superiority versus HIST_C3.”
- “Spark achieved exact historical policy parity for E7 and C3 on the tested 2M same-machine workload.”
- “A feature-only, scoring-only Spark workload processed 13,979,592 authentic rows in 68.02 seconds on `local[4]`; this is systems evidence.”
- “Two workers were the best tested same-machine 1M scoring configuration, at 1.65x speedup and 82.48% efficiency versus one worker.”
- “XGBoost CUDA execution was verified on an RTX 4090 Laptop GPU; the controlled inference measurement was slower than CPU.”
- “Engineering qualification and the final evidence gates passed, with the Causal Forest blocker and optional omissions retained.”

Do not shorten those claims into “beat all baselines,” “online lift,” “production ready,” “GPU acceleration improved performance,” “end-to-end 13.98M,” or “multi-node cluster.” Those stronger claims are not supported.
