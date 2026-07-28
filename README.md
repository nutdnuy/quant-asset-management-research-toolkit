# Quant Asset Management Research Toolkit

`qamr` is a generic, reproducible Python library for covariance estimation and
portfolio construction, independent of asset class, market, vendor, currency,
and observation frequency.

This project is a clean-room implementation. It excludes prior employer code,
data, credentials, schemas, identifiers, and business rules.

No open-source licence has been selected. All rights are reserved until a
licence is selected.

The package covers immutable labelled research-data contracts, an optional
Pandas adapter, deterministic covariance estimators, portfolio risk
contributions, equal weight, inverse volatility, HRP, and HERC. Data fetching,
signals, backtesting, live trading, and vendor integrations are out of scope.

## Foundation API

Calculations consume canonical labelled arrays. Pandas is an optional adapter,
not part of estimator signatures:

```python
import pandas as pd

from qamr.contracts import DatasetMetadata, ReturnConvention
from qamr.contracts.pandas_adapter import PandasAdapter, PandasResearchInput
from qamr.risk import SampleCovariance

returns = pd.DataFrame(
    [[0.01, 0.02], [0.03, 0.01], [0.02, 0.04]],
    index=pd.date_range("2026-01-01", periods=3, tz="UTC"),
    columns=["asset-a", "asset-b"],
)
dataset = PandasAdapter(
    metadata=DatasetMetadata(
        frequency="business-day",
        timezone="UTC",
        return_convention=ReturnConvention.SIMPLE,
    )
).adapt(PandasResearchInput(returns=returns))

assert dataset.returns is not None
estimate = SampleCovariance(annualization_factor=252.0).estimate(dataset.returns)
print(estimate.covariance.values)
```

The number `252.0` above is supplied by the caller; the library never infers an
annualisation factor from a market or frequency name.

## Portfolio Construction

Portfolio constructors consume a labelled covariance estimate and preserve its
instrument labels:

```python
from qamr.allocation import (
    equal_weights,
    herc_weights,
    hrp_weights,
    inverse_volatility_weights,
    risk_contributions,
)

equal = equal_weights(estimate)
inverse_volatility = inverse_volatility_weights(estimate)
hrp = hrp_weights(estimate, linkage_method="average")
herc = herc_weights(estimate, linkage_method="average")
component_risk = risk_contributions(estimate, inverse_volatility)
```

Allocators accept an optional `PortfolioConstraints` contract. Invalid,
infeasible, singular, or numerically unrepresentable inputs fail explicitly
instead of being silently repaired.
