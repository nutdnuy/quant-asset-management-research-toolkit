"""Labelled risk-estimate result objects."""

import math
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from qamr._types import JsonValue
from qamr.contracts.arrays import LabeledMatrix, LabeledVector
from qamr.contracts.results import NumericalDiagnostic
from qamr.errors import (
    DataValidationError,
    InsufficientHistoryError,
    LabelAlignmentError,
    NumericalStabilityError,
)
from qamr.risk.matrices import PSDPolicy, apply_psd_policy, covariance_to_correlation

# Validation starts at float64 roundoff tolerance and rises to four machine
# epsilons of the least-precise floating source. The one-percent cap prevents a
# low-precision dtype from making semantic contradictions acceptable. Integer
# sources contribute no additional tolerance. Positive semidefiniteness remains
# deliberately deferred to the later matrix policy.
_BASE_RELATIVE_TOLERANCE = 1e-10
_BASE_ABSOLUTE_TOLERANCE = 1e-12
_DTYPE_EPSILON_MULTIPLIER = 4.0
_MAX_RELATIVE_TOLERANCE = 1e-2
_MAX_CONTEXT_POSITIONS = 32
_DEFAULT_MAX_COVARIANCE_DIMENSION = 2048


def _bounded_position_context(
    positions: NDArray[np.intp],
) -> dict[str, JsonValue]:
    listed: list[JsonValue] = [int(position) for position in positions[:_MAX_CONTEXT_POSITIONS]]
    context: dict[str, JsonValue] = {"positions": listed}
    if positions.size > _MAX_CONTEXT_POSITIONS:
        context.update(
            {
                "position_count": int(positions.size),
                "positions_truncated": True,
            }
        )
    return context


def _bounded_type_name(value: object) -> str:
    return type(value).__name__[:64]


def _require_result_type(name: str, value: object, expected_type: type[object]) -> None:
    if type(value) is not expected_type:
        raise DataValidationError(
            f"{name} must be a {expected_type.__name__}",
            context={"field": name, "dtype": _bounded_type_name(value)},
        )


def _require_finite_real_numeric(
    name: str,
    data: LabeledMatrix | LabeledVector,
) -> tuple[NDArray[np.float64], tuple[float, float]]:
    values = data.values
    if values.dtype.kind not in {"i", "u", "f"}:
        raise DataValidationError(
            f"{name} values must be finite real numeric values",
            context={
                "field": name,
                "dtype": str(values.dtype),
                "reason": "not_real_numeric",
            },
        )
    if not np.isfinite(values).all():
        raise DataValidationError(
            f"{name} values must be finite real numeric values",
            context={
                "field": name,
                "dtype": str(values.dtype),
                "reason": "not_finite",
            },
        )
    with np.errstate(over="ignore", invalid="ignore"):
        working_values = values.astype(np.float64, copy=True)
    if not np.isfinite(working_values).all():
        raise DataValidationError(
            f"{name} values must be safely representable as finite float64 values",
            context={
                "field": name,
                "dtype": str(values.dtype)[:64],
                "reason": "not_float64_representable",
            },
        )
    source_precision = (
        (
            float(np.finfo(values.dtype).eps),
            float(np.finfo(values.dtype).smallest_subnormal),
        )
        if values.dtype.kind == "f"
        else (0.0, 0.0)
    )
    return working_values, source_precision


@dataclass(frozen=True, slots=True)
class _ValidationTolerance:
    relative: float
    zero_absolute: float

    def absolute_for(self, *values: NDArray[np.float64]) -> float:
        scale = max(
            (float(np.max(np.abs(value))) for value in values if value.size),
            default=0.0,
        )
        return max(_BASE_ABSOLUTE_TOLERANCE, self.relative * scale)


def _validation_tolerance(
    *source_precisions: tuple[float, float],
) -> _ValidationTolerance:
    dtype_relative = _DTYPE_EPSILON_MULTIPLIER * max(
        (precision[0] for precision in source_precisions),
        default=0.0,
    )
    dtype_zero_absolute = _DTYPE_EPSILON_MULTIPLIER * max(
        (precision[1] for precision in source_precisions),
        default=0.0,
    )
    return _ValidationTolerance(
        relative=min(
            _MAX_RELATIVE_TOLERANCE,
            max(_BASE_RELATIVE_TOLERANCE, dtype_relative),
        ),
        zero_absolute=max(_BASE_ABSOLUTE_TOLERANCE, dtype_zero_absolute),
    )


def _require_symmetric(
    name: str,
    values: NDArray[np.float64],
    tolerance: _ValidationTolerance,
) -> None:
    if not np.allclose(
        values,
        values.T,
        rtol=tolerance.relative,
        atol=tolerance.absolute_for(values),
    ):
        raise DataValidationError(
            f"{name} must be symmetric within numerical tolerance",
            context={"field": name, "reason": "not_symmetric"},
        )


def _validate_diagnostics(value: object) -> None:
    if type(value) is not tuple:
        raise DataValidationError(
            "diagnostics must be a tuple of NumericalDiagnostic values",
            context={"field": "diagnostics", "dtype": _bounded_type_name(value)},
        )
    for position, diagnostic in enumerate(value):
        if type(diagnostic) is not NumericalDiagnostic:
            raise DataValidationError(
                "diagnostics must be a tuple of NumericalDiagnostic values",
                context={
                    "field": "diagnostics",
                    "position": position,
                    "dtype": _bounded_type_name(diagnostic),
                },
            )


def _require_covariance_correlation_consistency(
    covariance: NDArray[np.float64],
    correlation: NDArray[np.float64],
    volatility: NDArray[np.float64],
    volatility_tolerance: _ValidationTolerance,
    covariance_volatility_tolerance: _ValidationTolerance,
    correlation_tolerance: _ValidationTolerance,
    reconciliation_tolerance: _ValidationTolerance,
) -> None:
    zero_volatility = np.isclose(
        volatility,
        0.0,
        rtol=0.0,
        atol=volatility_tolerance.zero_absolute,
    )
    zero_pairs = np.logical_or.outer(zero_volatility, zero_volatility)
    # Zero status belongs to volatility alone. Once an asset is classified as
    # zero-volatility, its covariance entries are products involving that
    # measured volatility, so their local absolute allowance combines exactly
    # covariance and volatility source precision (without matrix-scale growth).
    if np.any(np.abs(covariance[zero_pairs]) > covariance_volatility_tolerance.zero_absolute):
        raise DataValidationError(
            "covariance involving a zero-volatility asset must be numerically zero",
            context={
                "field": "covariance",
                "reason": "zero_volatility_covariance",
            },
        )

    off_diagonal = ~np.eye(len(volatility), dtype=bool)
    zero_correlation_pairs = zero_pairs & off_diagonal
    if np.any(np.abs(correlation[zero_correlation_pairs]) > correlation_tolerance.zero_absolute):
        raise DataValidationError(
            "off-diagonal correlation for a zero-volatility asset must be numerically zero",
            context={
                "field": "correlation",
                "reason": "zero_volatility_correlation",
            },
        )

    nonzero_pairs = ~zero_pairs
    denominator = np.multiply.outer(volatility, volatility)
    implied_correlation = np.zeros_like(covariance)
    np.divide(
        covariance,
        denominator,
        out=implied_correlation,
        where=nonzero_pairs,
    )
    if not np.allclose(
        correlation[nonzero_pairs],
        implied_correlation[nonzero_pairs],
        rtol=reconciliation_tolerance.relative,
        atol=reconciliation_tolerance.zero_absolute,
    ):
        raise DataValidationError(
            "correlation must be consistent with covariance and volatility",
            context={
                "field": "correlation",
                "reason": "covariance_mismatch",
            },
        )


@dataclass(frozen=True, slots=True, eq=False)
class CovarianceEstimate:
    """A complete, semantically validated labelled covariance result."""

    covariance: LabeledMatrix
    correlation: LabeledMatrix
    volatility: LabeledVector
    observation_count: int
    diagnostics: tuple[NumericalDiagnostic, ...] = field(default_factory=tuple)

    __hash__: ClassVar[None] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        _require_result_type("covariance", self.covariance, LabeledMatrix)
        _require_result_type("correlation", self.correlation, LabeledMatrix)
        _require_result_type("volatility", self.volatility, LabeledVector)

        labels = self.covariance.column_labels
        labels_are_aligned = (
            self.covariance.row_labels == labels
            and self.correlation.row_labels == labels
            and self.correlation.column_labels == labels
            and self.volatility.labels == labels
        )
        shapes_are_aligned = (
            self.covariance.shape == (len(labels), len(labels))
            and self.correlation.shape == (len(labels), len(labels))
            and self.volatility.shape == (len(labels),)
        )
        if not labels_are_aligned or not shapes_are_aligned:
            raise LabelAlignmentError(
                "risk estimate labels and shapes must align",
                context={"reason": "labels_or_shape"},
            )
        if not labels:
            raise LabelAlignmentError(
                "risk estimate requires a non-empty asset universe",
                context={"reason": "empty_asset_universe"},
            )

        risk_axis_name = self.covariance.column_name
        axes_are_aligned = (
            self.covariance.row_name == risk_axis_name
            and self.correlation.row_name == risk_axis_name
            and self.correlation.column_name == risk_axis_name
            and self.volatility.axis_name == risk_axis_name
        )
        if not axes_are_aligned:
            raise LabelAlignmentError(
                "risk estimate axis names must align",
                context={"reason": "axis_names"},
            )

        if type(self.observation_count) is not int:
            raise DataValidationError(
                "observation count must be an actual integer",
                context={
                    "field": "observation_count",
                    "dtype": _bounded_type_name(self.observation_count),
                },
            )
        if self.observation_count <= 0:
            raise DataValidationError(
                "observation count must be positive",
                context={"field": "observation_count", "reason": "not_positive"},
            )
        _validate_diagnostics(self.diagnostics)

        covariance_values, covariance_precision = _require_finite_real_numeric(
            "covariance",
            self.covariance,
        )
        correlation_values, correlation_precision = _require_finite_real_numeric(
            "correlation",
            self.correlation,
        )
        volatility_values, volatility_precision = _require_finite_real_numeric(
            "volatility",
            self.volatility,
        )
        covariance_tolerance = _validation_tolerance(covariance_precision)
        correlation_tolerance = _validation_tolerance(correlation_precision)
        volatility_tolerance = _validation_tolerance(volatility_precision)
        covariance_volatility_tolerance = _validation_tolerance(
            covariance_precision,
            volatility_precision,
        )
        reconciliation_tolerance = _validation_tolerance(
            covariance_precision,
            correlation_precision,
            volatility_precision,
        )
        _require_symmetric("covariance", covariance_values, covariance_tolerance)
        _require_symmetric("correlation", correlation_values, correlation_tolerance)

        covariance_diagonal = np.diag(covariance_values)
        if np.any(covariance_diagonal < -covariance_tolerance.zero_absolute):
            raise DataValidationError(
                "covariance diagonal must be nonnegative within numerical tolerance",
                context={"field": "covariance", "reason": "negative_diagonal"},
            )
        correlation_diagonal = np.diag(correlation_values)
        if not np.allclose(
            correlation_diagonal,
            1.0,
            rtol=correlation_tolerance.relative,
            atol=_BASE_ABSOLUTE_TOLERANCE,
        ):
            raise DataValidationError(
                "correlation diagonal must be approximately 1",
                context={"field": "correlation", "reason": "diagonal_not_one"},
            )
        correlation_bound = 1.0 + correlation_tolerance.relative + _BASE_ABSOLUTE_TOLERANCE
        if np.any(np.abs(correlation_values) > correlation_bound):
            raise DataValidationError(
                "correlation entries must lie within [-1, 1] within numerical tolerance",
                context={"field": "correlation", "reason": "outside_unit_interval"},
            )
        if np.any(volatility_values < -volatility_tolerance.zero_absolute):
            raise DataValidationError(
                "volatility must be nonnegative within numerical tolerance",
                context={"field": "volatility", "reason": "negative"},
            )
        expected_volatility = np.sqrt(np.maximum(covariance_diagonal, 0.0))
        volatility_matches = np.isclose(
            volatility_values,
            expected_volatility,
            rtol=covariance_volatility_tolerance.relative,
            atol=covariance_volatility_tolerance.zero_absolute,
        )
        if not np.all(volatility_matches):
            raise DataValidationError(
                "volatility must match the square root of the covariance diagonal",
                context={"field": "volatility", "reason": "covariance_diagonal_mismatch"},
            )
        _require_covariance_correlation_consistency(
            covariance_values,
            correlation_values,
            volatility_values,
            volatility_tolerance,
            covariance_volatility_tolerance,
            correlation_tolerance,
            reconciliation_tolerance,
        )

    @property
    def labels(self) -> tuple[Hashable, ...]:
        """Return the covariance column labels."""
        return self.covariance.column_labels


def build_covariance_estimate(
    covariance_values: NDArray[np.float64],
    labels: tuple[Hashable, ...],
    *,
    observation_count: int,
    diagnostics: tuple[NumericalDiagnostic, ...],
    psd_policy: PSDPolicy,
    tolerance: float,
    max_dimension: int = _DEFAULT_MAX_COVARIANCE_DIMENSION,
) -> CovarianceEstimate:
    """Build a complete risk estimate under an explicit PSD decision."""
    if type(covariance_values) is not np.ndarray:
        raise DataValidationError(
            "covariance_values must be an exact ndarray",
            context={
                "field": "covariance_values",
                "dtype": _bounded_type_name(covariance_values),
            },
        )
    if type(max_dimension) is not int:
        raise NumericalStabilityError(
            "max_dimension must be an exact built-in integer",
            context={
                "field": "max_dimension",
                "dtype": _bounded_type_name(max_dimension),
            },
        )
    if max_dimension <= 0:
        raise NumericalStabilityError(
            "max_dimension must be strictly positive",
            context={"field": "max_dimension", "reason": "not_positive"},
        )
    if covariance_values.ndim != 2 or covariance_values.shape[0] != covariance_values.shape[1]:
        raise NumericalStabilityError(
            "covariance_values must be a two-dimensional square array",
            context={
                "field": "covariance_values",
                "ndim": int(covariance_values.ndim),
                "shape": [int(size) for size in covariance_values.shape[:8]],
            },
        )
    dimension = int(covariance_values.shape[0])
    if dimension > max_dimension:
        raise NumericalStabilityError(
            "covariance exceeds the configured maximum dimension",
            context={"dimension": dimension, "maximum": max_dimension},
        )
    if type(labels) is not tuple:
        raise DataValidationError(
            "labels must be an exact tuple",
            context={"field": "labels", "dtype": _bounded_type_name(labels)},
        )
    if type(observation_count) is not int:
        raise DataValidationError(
            "observation_count must be an exact integer",
            context={
                "field": "observation_count",
                "dtype": _bounded_type_name(observation_count),
            },
        )
    if observation_count <= 0:
        raise DataValidationError(
            "observation_count must be positive",
            context={"field": "observation_count", "reason": "not_positive"},
        )
    _validate_diagnostics(diagnostics)
    if type(psd_policy) is not PSDPolicy:
        raise NumericalStabilityError(
            "psd_policy must be an exact PSDPolicy",
            context={"field": "psd_policy", "dtype": _bounded_type_name(psd_policy)},
        )
    if type(tolerance) not in {int, float}:
        raise NumericalStabilityError(
            "tolerance must be an exact built-in int or float",
            context={"field": "tolerance", "dtype": _bounded_type_name(tolerance)},
        )
    try:
        validated_tolerance = float(tolerance)
    except (OverflowError, ValueError) as error:
        raise NumericalStabilityError(
            "tolerance must be finite, positive, and at most 1e-2",
            context={"field": "tolerance", "reason": "not_finite"},
        ) from error
    if (
        not math.isfinite(validated_tolerance)
        or validated_tolerance <= 0.0
        or validated_tolerance > _MAX_RELATIVE_TOLERANCE
    ):
        raise NumericalStabilityError(
            "tolerance must be finite, positive, and at most 1e-2",
            context={"field": "tolerance", "reason": "outside_valid_range"},
        )

    covariance = LabeledMatrix(
        covariance_values,
        labels,
        labels,
        "instrument",
        "instrument",
    )
    covariance = apply_psd_policy(
        covariance,
        psd_policy,
        tolerance=validated_tolerance,
        max_dimension=max_dimension,
    )
    working_values = covariance.values
    variances = np.diag(working_values)
    bad_positions = np.flatnonzero(variances <= validated_tolerance)
    if bad_positions.size:
        raise InsufficientHistoryError(
            "risk estimate requires strictly positive variance for every instrument",
            context=_bounded_position_context(bad_positions),
        )
    unsupported_positions = np.flatnonzero(np.sqrt(variances) <= _BASE_ABSOLUTE_TOLERANCE)
    if unsupported_positions.size:
        context = _bounded_position_context(unsupported_positions)
        context["reason"] = "volatility_below_supported_resolution"
        raise NumericalStabilityError(
            "risk estimate volatility is below the supported result resolution",
            context=context,
        )
    correlation, volatility = covariance_to_correlation(
        covariance,
        tolerance=validated_tolerance,
    )
    return CovarianceEstimate(
        covariance=covariance,
        correlation=correlation,
        volatility=volatility,
        observation_count=observation_count,
        diagnostics=diagnostics,
    )
