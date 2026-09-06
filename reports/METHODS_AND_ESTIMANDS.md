# Methods and Estimands

- Primary estimand: AIPW incremental policy value at 20% capacity versus never treating.
- Sensitivity estimands: randomized-propensity IPW/HT and independently nuisance-fitted AIPW.
- Ranking: deterministic global descending scores with numeric source-row-ID ties.
- Eligibility: causal scores must be positive; response rankings are non-causal comparators; positive constant ATE uses deterministic hash allocation.
- Calibration: ten-bin causal calibration RMSE and development-only positive-slope affine calibration.
- Inference: paired exact pretreatment-profile cluster bootstrap for E7/C3; predeclared cluster-robust normal secondary comparisons with Holm correction within evaluator.
- External truth: Twins uses paired mortality potential outcomes; Criteo never reports PEHE.
