"""Transparent baseline allocation rules."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from qamr.allocation.risk import (
    _bounded_type_name,
    _finite_real_values,
    _require_estimate_structure,
)
from qamr.contracts.arrays import LabeledVector
from qamr.contracts.config import PortfolioConstraints
from qamr.errors import (
    DataValidationError,
    InfeasiblePortfolioError,
    LabelAlignmentError,
    NumericalStabilityError,
)
from qamr.risk.estimates import CovarianceEstimate

_CONSTRAINT_TOLERANCE = 1e-10


def _resolved_constraints(value: object) -> PortfolioConstraints:
    if value is None:
        return PortfolioConstraints()
    if type(value) is not PortfolioConstraints:
        raise DataValidationError(
            "constraints must be None or an exact PortfolioConstraints",
            context={
                "field": "constraints",
                "dtype": _bounded_type_name(value),
            },
        )
    return value


def _violates_lower(value: float, lower: float) -> bool:
    return value < lower and not math.isclose(
        value,
        lower,
        rel_tol=0.0,
        abs_tol=_CONSTRAINT_TOLERANCE,
    )


def _violates_upper(value: float, upper: float) -> bool:
    return value > upper and not math.isclose(
        value,
        upper,
        rel_tol=0.0,
        abs_tol=_CONSTRAINT_TOLERANCE,
    )


def _constraint_error(message: str, name: str) -> InfeasiblePortfolioError:
    return InfeasiblePortfolioError(
        message,
        context={"constraint": name, "reason": "violated"},
    )


def _constrained_weights(
    values: NDArray[np.float64],
    estimate: CovarianceEstimate,
    constraints: PortfolioConstraints,
) -> LabeledVector:
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    gross = math.fsum(abs(float(value)) for value in values)
    net = math.fsum(float(value) for value in values)
    if constraints.long_only and _violates_lower(minimum, 0.0):
        raise _constraint_error(
            "allocator violates long-only constraints",
            "long_only",
        )
    if constraints.min_weight is not None and _violates_lower(
        minimum,
        constraints.min_weight,
    ):
        raise _constraint_error(
            "allocator violates minimum weight",
            "min_weight",
        )
    if constraints.max_weight is not None and _violates_upper(
        maximum,
        constraints.max_weight,
    ):
        raise _constraint_error(
            "allocator violates maximum weight",
            "max_weight",
        )
    if constraints.gross_leverage is not None and _violates_upper(
        gross,
        constraints.gross_leverage,
    ):
        raise _constraint_error(
            "allocator violates gross leverage",
            "gross_leverage",
        )
    if constraints.net_exposure is not None and not math.isclose(
        net,
        constraints.net_exposure,
        rel_tol=0.0,
        abs_tol=_CONSTRAINT_TOLERANCE,
    ):
        raise _constraint_error(
            "allocator violates net exposure",
            "net_exposure",
        )
    return LabeledVector(
        values,
        estimate.labels,
        estimate.covariance.column_name,
    )


def equal_weights(
    estimate: CovarianceEstimate,
    constraints: PortfolioConstraints | None = None,
) -> LabeledVector:
    """Return a fully invested equal-weight baseline."""
    validated_estimate, _, _ = _require_estimate_structure(estimate)
    resolved_constraints = _resolved_constraints(constraints)
    count = len(validated_estimate.labels)
    values = np.full(count, 1.0 / count, dtype=np.float64)
    return _constrained_weights(
        values,
        validated_estimate,
        resolved_constraints,
    )


def inverse_volatility_weights(
    estimate: CovarianceEstimate,
    constraints: PortfolioConstraints | None = None,
) -> LabeledVector:
    """Return a stable fully invested inverse-volatility baseline."""
    validated_estimate, covariance, _ = _require_estimate_structure(estimate)
    resolved_constraints = _resolved_constraints(constraints)
    volatility = getattr(validated_estimate, "volatility", None)
    if type(volatility) is not LabeledVector:
        raise DataValidationError(
            "estimate volatility must be an exact LabeledVector",
            context={
                "field": "volatility",
                "dtype": _bounded_type_name(volatility),
            },
        )
    if volatility.labels != validated_estimate.labels:
        raise LabelAlignmentError(
            "volatility labels must match risk estimate labels",
            context={"reason": "labels"},
        )
    if volatility.axis_name != covariance.column_name:
        raise LabelAlignmentError(
            "volatility axis must match risk estimate axis",
            context={"reason": "axis_name"},
        )
    values = _finite_real_values(volatility, field="volatility")
    if np.any(values <= 0.0):
        raise NumericalStabilityError(
            "volatility values must be strictly positive",
            context={"field": "volatility", "reason": "not_positive"},
        )
    with np.errstate(divide="ignore", invalid="ignore", under="ignore"):
        inverse_logs = -np.log(values)
        relative_inverse = np.exp(inverse_logs - float(np.max(inverse_logs)))
    if np.any(relative_inverse <= 0.0):
        raise NumericalStabilityError(
            "strictly positive inverse-volatility weights must be representable",
            context={
                "field": "weights",
                "reason": "positive_weight_not_float64_representable",
            },
        )
    normalizer = math.fsum(float(value) for value in relative_inverse)
    normalized = relative_inverse / normalizer
    if np.any(normalized <= 0.0):
        raise NumericalStabilityError(
            "strictly positive inverse-volatility weights must be representable",
            context={
                "field": "weights",
                "reason": "positive_weight_not_float64_representable",
            },
        )
    return _constrained_weights(
        normalized,
        validated_estimate,
        resolved_constraints,
    )
