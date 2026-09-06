# CAUSAL-SCALE v1.0.0 — Prospective Replication Release

CAUSAL-SCALE evaluates capacity-constrained causal targeting while keeping scientific validation separate from compute-scale qualification.

## Release highlights

- **Prospective replication:** on a sealed, disjoint 2M cohort at 20% capacity, E7 policy value was `0.0063127390` versus `0.0059089824` for the frozen HIST_C3 doubly robust baseline. The absolute difference was `+0.0004037566` (`+6.83%` relative to C3), with paired 95% interval `[0.0001217940, 0.0007010708]`. The predeclared noninferiority lower bound was `0.0001621445`, above the `-0.00025` margin; H1 passed. This supports prospective replication over C3, not superiority over every comparator.
- **Calibration:** prospective E7 RMSE was `0.0049530331` versus C3's `0.0088863005`, an absolute reduction of `0.0039332673` (`44.26%`). The one-sided 95% upper bound for E7 minus C3 was `-0.0024691233`; H2 calibration superiority passed. This is metric-specific and does not compare causal calibration to response models.
- **Comparator boundary:** response-targeting point estimates were higher than E7 under both prospective AIPW evaluators, but direct intervals crossed zero and Holm-adjusted decisions were non-significant. E7 was significantly higher than GPU_T and R_LEARNER under all three evaluators, and X_LEARNER under two of three. The release does not claim “beat all baselines.”
- **Distributed parity:** Spark matched frozen E7 and C3 reference scores and policy memberships on 2M historical rows with zero score error, zero membership mismatches, and 100% agreement. AIPW error was approximately `5e-17`. This supports same-workload implementation parity, not a physical multi-node claim or a new causal result.
- **Authentic systems scale:** a feature-only, scoring-only 13,979,592-row workload ran in `68.0202 s` at `205,521 rows/s` on same-machine `local[4]`. No same-size alternative was recorded, so no speedup or uncertainty is claimed. This is not end-to-end processing and accessed no outcomes.
- **Local scaling:** for the identical 1M scoring workload, two workers were best at `22.2093 s`, `1.650x` speedup, and `82.48%` efficiency versus one worker. Four workers regressed to `28.0226 s`, `1.307x`, and `32.68%`. Event summaries show increased task count, nearly fixed shuffle, and no spill, consistent with local scheduling/coordination overhead.
- **GPU qualification:** XGBoost CUDA ran on an RTX 4090 Laptop GPU. GPU inference was `0.1105 s` versus derived CPU inference `0.0766 s`, or `0.693x` CPU/GPU speedup; CUDA was slower in this run. Peak VRAM was `212 MiB`, prediction correlation `0.9031`, and policy agreement `91.494%`. GPU training took `0.4352 s`, but CPU training time was not retained, so no training speedup or causal-validity improvement is claimed.
- **Engineering qualification:** formatting, lint, governed mypy scope, pytest, package build, installed-wheel import, and evidence verification passed in the completed qualification. The public CI reproduces a synthetic, no-data, no-GPU subset plus a small Spark parity fixture and public-release audit.

## Recorded limitations and omissions

- Offline incremental policy value is not online revenue, ROI, or deployment impact.
- The 13.98M run is systems-only and scoring-only; same-machine Spark workers are not physical nodes.
- Causal Forest remained a `SANDBOX_TOOLING_BLOCKER` after two attempts without weakening scientific parameters.
- Full IHDP replications were resource-skipped. DragonNet/JAX, Hillstrom, ONNX CUDA, and physical multi-node Spark were optional omissions.
- Twins external validation passed, but one benchmark does not establish universal external validity.

See [Results Analysis](RESULTS_ANALYSIS.md), the [Public Evidence Index](PUBLIC_EVIDENCE_INDEX.md), and the [completion gates](../evidence/final_audit/completion_gates.json) for exact sources and claim boundaries.
