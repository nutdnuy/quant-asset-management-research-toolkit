"""Portfolio risk measures for labelled allocations."""

from __future__ import annotations

import math
from collections.abc import Hashable

import numpy as np
from numpy.typing import NDArray

from qamr.contracts.arrays import LabeledMatrix, LabeledVector
from qamr.errors import (
    DataValidationError,
    LabelAlignmentError,
    NumericalStabilityError,
)
from qamr.risk.estimates import CovarianceEstimate

_MIN_NORMAL_FLOAT64 = float(np.finfo(np.float64).tiny)


def _bounded_type_name(value: object) -> str:
    return type(value).__name__[:64]


def _require_estimate(value: object) -> CovarianceEstimate:
    if type(value) is not CovarianceEstimate:
        raise DataValidationError(
            "estimate must be an exact CovarianceEstimate",
            context={
                "field": "estimate",
                "dtype": _bounded_type_name(value),
            },
        )
    return value


def _require_estimate_structure(
    value: object,
) -> tuple[CovarianceEstimate, LabeledMatrix, tuple[Hashable, ...]]:
    estimate = _require_estimate(value)
    covariance = getattr(estimate, "covariance", None)
    if type(covariance) is not LabeledMatrix:
        raise DataValidationError(
            "estimate covariance must be an exact LabeledMatrix",
            context={
                "field": "covariance",
                "dtype": _bounded_type_name(covariance),
            },
        )
    labels = covariance.column_labels
    if covariance.row_labels != labels or covariance.shape != (
        len(labels),
        len(labels),
    ):
        raise LabelAlignmentError(
            "covariance labels and shape must define a square risk universe",
            context={"reason": "labels_or_shape"},
        )
    if covariance.row_name != covariance.column_name:
        raise LabelAlignmentError(
            "covariance axis names must align",
            context={"reason": "axis_names"},
        )
    if not labels:
        raise NumericalStabilityError(
            "at least one instrument is required",
            context={"reason": "empty_asset_universe"},
        )
    return estimate, covariance, labels


def _finite_real_values(
    value: LabeledMatrix | LabeledVector,
    *,
    field: str,
) -> NDArray[np.float64]:
    source = value.values
    if source.dtype.kind not in {"i", "u", "f"}:
        raise DataValidationError(
            f"{field} values must be finite real numeric values",
            context={
                "field": field,
                "dtype": str(source.dtype)[:64],
                "reason": "not_real_numeric",
            },
        )
    if not np.isfinite(source).all():
        raise NumericalStabilityError(
            f"{field} numeric values must be finite",
            context={
                "field": field,
                "reason": "not_finite",
            },
        )
    with np.errstate(over="ignore", invalid="ignore"):
        converted = source.astype(np.float64, copy=True)
    if not np.isfinite(converted).all():
        raise DataValidationError(
            f"{field} values must be safely representable as finite float64 values",
            context={
                "field": field,
                "dtype": str(source.dtype)[:64],
                "reason": "not_float64_representable",
            },
        )
    if source.dtype.kind in {"i", "u"}:
        with np.errstate(over="ignore", invalid="ignore"):
            round_trip = converted.astype(source.dtype)
        if not np.array_equal(round_trip, source):
            raise DataValidationError(
                f"{field} integer values must be exactly representable as float64 values",
                context={
                    "field": field,
                    "dtype": str(source.dtype)[:64],
                    "reason": "not_exactly_representable",
                },
            )
    return converted


def _validated_inputs(
    estimate: object,
    weights: object,
) -> tuple[
    CovarianceEstimate,
    NDArray[np.float64],
    NDArray[np.float64],
    tuple[Hashable, ...],
]:
    validated_estimate, covariance, labels = _require_estimate_structure(
        estimate,
    )
    if type(weights) is not LabeledVector:
        raise DataValidationError(
            "weights must be an exact LabeledVector",
            context={
                "field": "weights",
                "dtype": _bounded_type_name(weights),
            },
        )
    if weights.labels != labels:
        raise LabelAlignmentError(
            "weight labels must match risk estimate labels exactly and in order",
            context={"reason": "labels"},
        )
    risk_axis = covariance.column_name
    if weights.axis_name != risk_axis:
        raise LabelAlignmentError(
            "weight axis must match the risk estimate axis",
            context={"reason": "axis_name"},
        )
    weight_values = _finite_real_values(weights, field="weights")
    covariance_values = _finite_real_values(covariance, field="covariance")
    if covariance_values.shape != (len(labels), len(labels)):
        raise LabelAlignmentError(
            "covariance shape must match risk estimate labels",
            context={"reason": "shape"},
        )
    return validated_estimate, weight_values, covariance_values, labels


def _volatility_from_log_variance(log_variance: float) -> float:
    try:
        volatility = math.exp(0.5 * log_variance)
    except OverflowError as error:
        raise NumericalStabilityError(
            "portfolio volatility is outside the supported float64 range",
            context={"reason": "not_float64_representable"},
        ) from error
    if not math.isfinite(volatility) or volatility <= 0.0:
        raise NumericalStabilityError(
            "portfolio volatility is outside the supported float64 range",
            context={"reason": "not_float64_representable"},
        )
    return volatility


def _log_scaled_portfolio_components(
    weights: NDArray[np.float64],
    covariance: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64]]:
    absolute_weights = np.abs(weights)
    absolute_covariance = np.abs(covariance)
    active_terms = (
        (absolute_weights[:, np.newaxis] > 0.0)
        & (absolute_covariance > 0.0)
        & (absolute_weights[np.newaxis, :] > 0.0)
    )
    if not np.any(active_terms):
        raise NumericalStabilityError(
            "portfolio variance must be finite and strictly positive",
            context={"reason": "not_positive"},
        )
    weight_logs = np.full(weights.shape, -np.inf, dtype=np.float64)
    covariance_logs = np.full(covariance.shape, -np.inf, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.log(absolute_weights, out=weight_logs, where=absolute_weights > 0.0)
        np.log(
            absolute_covariance,
            out=covariance_logs,
            where=absolute_covariance > 0.0,
        )
        term_logs = weight_logs[:, np.newaxis] + covariance_logs + weight_logs[np.newaxis, :]
    maximum_log = float(np.max(term_logs[active_terms]))
    scaled_terms = np.zeros_like(covariance)
    signs = np.sign(weights)[:, np.newaxis] * np.sign(covariance) * np.sign(weights)[np.newaxis, :]
    with np.errstate(under="ignore", invalid="ignore"):
        scaled_terms[active_terms] = signs[active_terms] * np.exp(
            term_logs[active_terms] - maximum_log
        )
    row_contributions = np.sum(scaled_terms, axis=1)
    scaled_variance = math.fsum(float(value) for value in row_contributions)
    if not math.isfinite(scaled_variance):
        raise NumericalStabilityError(
            "portfolio variance could not be computed stably",
            context={"reason": "not_finite"},
        )
    if scaled_variance <= 0.0:
        raise NumericalStabilityError(
            "portfolio variance must be finite and strictly positive",
            context={"reason": "not_positive"},
        )
    volatility = _volatility_from_log_variance(maximum_log + math.log(scaled_variance))
    return volatility, row_contributions / scaled_variance


def _scaled_portfolio_components(
    weights: NDArray[np.float64],
    covariance: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64]]:
    weight_scale = float(np.max(np.abs(weights)))
    covariance_scale = float(np.max(np.abs(covariance)))
    if weight_scale == 0.0 or covariance_scale == 0.0:
        return _log_scaled_portfolio_components(weights, covariance)
    scaled_weights = weights / weight_scale
    scaled_covariance = covariance / covariance_scale
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        covariance_times_weights = scaled_covariance @ scaled_weights
        scaled_variance = float(scaled_weights @ covariance_times_weights)
    if math.isfinite(scaled_variance) and scaled_variance >= _MIN_NORMAL_FLOAT64:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            contribution_shares = scaled_weights * covariance_times_weights / scaled_variance
        if np.isfinite(contribution_shares).all():
            volatility = _volatility_from_log_variance(
                2.0 * math.log(weight_scale)
                + math.log(covariance_scale)
                + math.log(scaled_variance)
            )
            return volatility, contribution_shares
    return _log_scaled_portfolio_components(
        weights,
        covariance,
    )


def portfolio_volatility(
    estimate: CovarianceEstimate,
    weights: LabeledVector,
) -> float:
    """Return positive portfolio volatility under a labelled covariance estimate."""
    _, weight_values, covariance_values, _ = _validated_inputs(estimate, weights)
    volatility, _ = _scaled_portfolio_components(
        weight_values,
        covariance_values,
    )
    return volatility


def risk_contributions(
    estimate: CovarianceEstimate,
    weights: LabeledVector,
) -> LabeledVector:
    """Return Euler volatility contributions in risk-estimate label order."""
    _, weight_values, covariance_values, labels = _validated_inputs(estimate, weights)
    volatility, contribution_shares = _scaled_portfolio_components(
        weight_values,
        covariance_values,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        contributions = volatility * contribution_shares
    if not np.isfinite(contributions).all():
        raise NumericalStabilityError(
            "risk contributions are outside the supported float64 range",
            context={"reason": "not_float64_representable"},
        )
    return LabeledVector(
        contributions,
        labels,
        estimate.covariance.column_name,
    )
