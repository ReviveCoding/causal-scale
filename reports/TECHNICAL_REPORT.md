# CAUSAL-SCALE Technical Report

## Scope and claim boundary

CAUSAL-SCALE evaluates capacity-constrained causal targeting. Offline policy value is an estimator of incremental visit probability under the declared policy versus never treating; it is not observed online revenue. Historical, development, retrospective, systems, external, and prospective evidence are not pooled into one claim.

## Historical predecessor

The immutable ORBIT-Causal v0.3.9 committed snapshot anchors the historical semantics. Its locked 2M result was E7=0.006041015750860274 and C3=0.005643564219054695, with delta=0.0003974515318055772 and verdict=RELEASE_ELIGIBLE.

## Development

New models used only the deterministic development roles. Model status: CONSTANT_ATE=PASS, GPU_T=PASS, INDEPENDENT_AIPW_NUISANCE=PASS, R_LEARNER=PASS, RESPONSE_MARGINAL=PASS, RESPONSE_TREATED=PASS, X_LEARNER=PASS, CAUSAL_FOREST=SANDBOX_TOOLING_BLOCKER. Response scores were not subjected to causal calibration. Positive-slope affine calibration preserved ranking while zero-boundary eligibility changes were measured.

## Retrospective

The common benchmark is in `reports/tables/model_benchmark.csv`. Every method uses the same policy engine and evaluator within a view. Criteo PEHE is intentionally absent because unit counterfactual truth is unavailable.

## GPU compute

GPU qualification: GPU_XGBOOST_AVAILABLE. Device speed measurements are systems evidence only and do not establish scientific superiority.

## Distributed parity

Spark parity status: PASS. Spark workers are same-machine processes, not physical nodes. Policy selection uses global descending score and numeric source-row-ID ties.

## Systems scale

Feature-only authentic scale status: PASS; full workload: 13979592 rows. These are scoring systems results, not an independent causal holdout.

## External validation

Twins status: PASS. IHDP: SKIPPED_RESOURCE_LIMIT. Historical E7 was not relabeled as an externally retrained method.

## Prospective

Scientific contract SHA-256: `240dfe619f5e251bcc2673ef876dc37911618caae5421f76e8f218f55e2fb5e3`. Final prospective status: **REPLICATED**. Point estimates and gates are retained in `evidence/prospective/prospective_results.json` and never inferred from this prose.

## Reproducibility

The repository contains a resumable hash-bound DAG, dependency lock, dataset and row-role manifests, frozen model manifests, Spark event logs, and machine-generated tables/figures. The independent evidence verifier checks every present SHA256SUMS ledger.
