---
name: qamr-data-boundary
description: Convert user-provided tabular returns into validated, immutable qamr research contracts. Use when asked to validate labelled return data, adapt a pandas DataFrame, set return conventions or missing-data rules, preserve point-in-time metadata, or diagnose label/alignment errors before covariance estimation or portfolio construction.
---

# QAMR Data Boundary

Accept only data supplied by the user or already in the workspace. Do not fetch
market data, infer exchange calendars, or silently alter dates, labels, missing
values, or return conventions.

## Workflow

1. Identify the input as returns, not prices; ask for or state the return
   convention (`SIMPLE` or `LOG`), frequency, and timezone.
2. Validate that the observation index is ordered, unique, and point-in-time
   appropriate. Preserve instrument column labels exactly.
3. Create `DatasetMetadata` explicitly. Record the intended missing-data policy
   for the later risk estimator or `ResearchConfig`; never fill implicitly.
4. Adapt a pandas DataFrame with `PandasAdapter` and `PandasResearchInput`.
   Keep pandas at this boundary; estimators receive qamr labelled arrays.
5. Report the resolved metadata, observation count, instrument labels, and any
   rejected records or structured validation errors.

## Minimal Recipe

```python
import pandas as pd

from qamr.contracts import DatasetMetadata, ReturnConvention
from qamr.contracts.pandas_adapter import PandasAdapter, PandasResearchInput

returns = pd.read_csv("returns.csv", index_col=0, parse_dates=True)
metadata = DatasetMetadata(
    frequency="business-day",
    timezone="UTC",
    return_convention=ReturnConvention.SIMPLE,
)
dataset = PandasAdapter(
    metadata=metadata,
).adapt(PandasResearchInput(returns=returns))
assert dataset.returns is not None
```

## Output

State the input location, row range, labels, frequency, timezone, return
convention, missing-data policy, and every assumption. Do not present a
portfolio or performance result; hand the validated dataset to the risk or
portfolio-construction Skill.

## Boundaries

- Never align assets by position when labels differ.
- Never infer annualisation, calendar, currency, or vendor semantics.
- Never convert prices to returns unless the user explicitly requests and
  defines that transformation.
