---
name: qamr-covariance-risk
description: Estimate, validate, repair, compare, and explain covariance and correlation matrices with qamr. Use when asked to turn validated returns into sample, EWMA, shrinkage, or spectral covariance estimates; choose a PSD policy; inspect volatility/correlation; or report numerical diagnostics before portfolio construction.
---

# QAMR Covariance and Risk

Use a validated `ResearchDataset` or labelled returns from the data-boundary
workflow. Keep every annualisation factor, missing-data rule, PSD policy, and
spectral rank choice explicit.

## Estimator Selection

- Use `SampleCovariance` as the transparent baseline.
- Use `EWMACovariance` when a caller specifies how rapidly older observations
  should decay.
- Use `ShrinkageCovariance` when the sample is noisy relative to the number of
  instruments; state the selected target.
- Use `SpectralDenoisedCovariance` only with an explicit rank or defensible
  effective-observation basis. Explain the denoising choice.

## Workflow

1. Confirm return convention, sample window, missing-data behavior, and the
   caller-supplied annualisation factor.
2. Estimate covariance from labelled returns. Preserve instrument order and
   never add implicit data alignment.
3. Inspect covariance, correlation, volatility, observation count, and
   numerical diagnostics. Apply `PSDPolicy` only when a matrix requires the
   documented policy; report that repair.
4. Compare estimators only under the same input window and assumptions.
5. Hand the chosen `CovarianceEstimate` to portfolio construction; do not
   infer weights or backtest performance in this skill.

## Minimal Recipe

```python
from qamr.risk import SampleCovariance

assert dataset.returns is not None
estimate = SampleCovariance(annualization_factor=252.0).estimate(dataset.returns)
print(estimate.volatility.values)
print(estimate.correlation.values)
```

## Output

Report the estimator, input observations, annualisation factor, estimator
parameters, matrix policy, and diagnostics. Mark all choices as research
assumptions, not predictions.

## Boundaries

- Do not infer annualisation from labels such as `daily` or a market name.
- Do not treat a repaired PSD matrix as equivalent to the original without
  disclosing the policy and diagnostics.
- Do not fetch data, generate signals, execute trades, or claim future returns.
