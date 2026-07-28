---
name: qamr-portfolio-construction
description: Construct transparent long-only portfolios from user-provided return data with the Quant Asset Management Research Toolkit (qamr). Use when asked to estimate covariance, compare equal-weight/inverse-volatility/HRP/HERC allocations, inspect portfolio risk contributions, or produce reproducible portfolio-construction evidence without data fetching, signals, backtesting, or trade execution.
---

# QAMR Portfolio Construction

Use this skill only with returns supplied by the user or already present in the
workspace. Do not fetch market data, infer an annualisation factor, run a
backtest, or produce trade instructions.

## Workflow

1. Locate the `qamr` repository or package. Use its project environment or a
   disposable virtual environment; do not install this Skill into a global
   Codex/Claude skill directory.
2. Confirm the input represents dated returns, the return convention, and the
   caller-supplied annualisation factor if annualised covariance is required.
   Preserve instrument labels exactly. If these facts are missing, state the
   assumption or ask for them before presenting investment conclusions.
3. Adapt tabular returns through `PandasAdapter`, then choose one covariance
   estimator:
   - `SampleCovariance` for a transparent baseline;
   - `EWMACovariance` when recency should matter;
   - `ShrinkageCovariance` when the sample is noisy relative to its dimension;
   - `SpectralDenoisedCovariance` only when rank/effective-observation choices
     can be stated explicitly.
4. Construct the requested allocation. If none is specified, compare all four:
   `equal_weights`, `inverse_volatility_weights`, `hrp_weights`, and
   `herc_weights`. Pass `PortfolioConstraints` explicitly whenever constraints
   are requested. Do not claim that these baseline constructors solve arbitrary
   optimisation constraints.
5. Calculate `portfolio_volatility` and `risk_contributions` for each selected
   portfolio. Treat structured validation, infeasibility, and numerical errors
   as reportable results; do not silently clip, relabel, or replace inputs.
6. Return a compact evidence-first report. Separate facts from interpretation.

## Minimal Recipe

```python
import pandas as pd

from qamr.allocation import herc_weights, risk_contributions
from qamr.contracts import DatasetMetadata, ReturnConvention
from qamr.contracts.pandas_adapter import PandasAdapter, PandasResearchInput
from qamr.risk import ShrinkageCovariance, ShrinkageTarget

returns = pd.read_csv("returns.csv", index_col=0, parse_dates=True)
dataset = PandasAdapter(
    metadata=DatasetMetadata(
        frequency="business-day",
        timezone="UTC",
        return_convention=ReturnConvention.SIMPLE,
    )
).adapt(PandasResearchInput(returns=returns))
assert dataset.returns is not None

estimate = ShrinkageCovariance(
    target=ShrinkageTarget.DIAGONAL,
    annualization_factor=252.0,  # Explicit caller choice.
).estimate(dataset.returns)
weights = herc_weights(estimate, linkage_method="average")
contributions = risk_contributions(estimate, weights)
```

## Report Shape

Use this default structure and adapt it to the request:

```markdown
## Portfolio construction result

Assumptions: return convention, sample dates, annualisation factor, covariance
estimator, allocation method, and constraints.

| Instrument | Weight | Risk contribution |
|---|---:|---:|

Portfolio volatility: …

Notes: data-quality warnings, numerical diagnostics, and limitations. This is
research output, not investment advice or an execution instruction.
```

## Boundaries

- Do not fetch data or use hidden vendor/calendar assumptions.
- Do not add signal generation, performance attribution, backtests, or trading.
- Do not change labels to force matrix alignment.
- Do not annualise unless the caller explicitly supplies the factor.
