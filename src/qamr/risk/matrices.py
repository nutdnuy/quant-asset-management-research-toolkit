"""Pure labelled covariance, correlation, and PSD matrix operations.

User tolerances are absolute semantic tolerances.  Comparisons also admit up
to four source-dtype epsilons at the two cells being compared, so values
produced in float16/32 are not judged as though they originated in float64.
The local comparison prevents an unrelated large matrix entry from relaxing a
material violation elsewhere. Public semantic tolerances are capped at
``1e-2``; dtype-derived local machine allowance is calculated separately.
"""

import math
from collections.abc import Hashable
from enum import Enum
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from qamr._types import JsonValue
from qamr.contracts.arrays import LabeledMatrix, LabeledVector
from qamr.errors import LabelAlignmentError, NumericalStabilityError

_DTYPE_EPSILON_MULTIPLIER = 4.0
_MAX_CONTEXT_POSITIONS = 32
_MAX_SEMANTIC_TOLERANCE = 1e-2
_MAX_PSD_CORRECTION_STEPS = 8
_MAX_SHAPE_DIMENSIONS_IN_CONTEXT = 8
_DEFAULT_MAX_PSD_DIMENSION = 2048


class PSDPolicy(str, Enum):
    """Policy for a symmetric matrix with negative eigenvalues."""

    RAISE = "raise"
    CLIP = "clip"


def _bounded_type_name(value: object) -> str:
    return type(value).__name__[:64]


def _validate_tolerance(tolerance: object) -> float:
    if type(tolerance) not in {int, float}:
        raise NumericalStabilityError(
            "tolerance must be an exact built-in int or float",
            context={"field": "tolerance", "dtype": _bounded_type_name(tolerance)},
        )
    try:
        validated = float(cast("int | float", tolerance))
    except (OverflowError, ValueError) as error:
        raise NumericalStabilityError(
            "tolerance must be finite and nonnegative",
            context={"field": "tolerance", "reason": "not_finite"},
        ) from error
    if not math.isfinite(validated) or validated < 0.0:
        raise NumericalStabilityError(
            "tolerance must be finite and nonnegative",
            context={"field": "tolerance", "reason": "not_finite_or_negative"},
        )
    if validated > _MAX_SEMANTIC_TOLERANCE:
        raise NumericalStabilityError(
            "tolerance must not exceed the maximum semantic tolerance",
            context={
                "field": "tolerance",
                "maximum": _MAX_SEMANTIC_TOLERANCE,
            },
        )
    return validated


def _validate_max_dimension(max_dimension: object) -> int:
    if type(max_dimension) is not int:
        raise NumericalStabilityError(
            "max_dimension must be an exact built-in int",
            context={
                "field": "max_dimension",
                "dtype": _bounded_type_name(max_dimension),
            },
        )
    validated = max_dimension
    if validated <= 0:
        raise NumericalStabilityError(
            "max_dimension must be strictly positive",
            context={"field": "max_dimension", "reason": "not_positive"},
        )
    return validated


def _matrix_dimension_without_value_copy(matrix: LabeledMatrix) -> int:
    dimension = max(len(matrix.row_labels), len(matrix.column_labels))
    try:
        reported_shape = matrix.shape
    except (IndexError, TypeError, ValueError):
        # Malformed rank is diagnosed by the bounded shape validator after the
        # guard; a normal LabeledMatrix reports two dimensions without copying.
        return dimension
    return max(dimension, *(int(size) for size in reported_shape))


def _require_exact_type(name: str, value: object, expected: type[object]) -> None:
    if type(value) is not expected:
        raise NumericalStabilityError(
            f"{name} must be an exact {expected.__name__}",
            context={"field": name, "dtype": _bounded_type_name(value)},
        )


def _position_context(positions: NDArray[np.intp]) -> dict[str, JsonValue]:
    listed: list[JsonValue] = [int(position) for position in positions[:_MAX_CONTEXT_POSITIONS]]
    if positions.size <= _MAX_CONTEXT_POSITIONS:
        return {"positions": listed}
    return {
        "positions": listed,
        "position_count": int(positions.size),
        "positions_truncated": True,
    }


def _source_epsilon(values: NDArray[np.generic]) -> float:
    if values.dtype.kind != "f":
        return 0.0
    floating_type = cast("type[np.floating[Any]]", values.dtype.type)
    return float(np.finfo(floating_type).eps)


def _finite_real_float64(
    name: str,
    data: LabeledMatrix | LabeledVector,
) -> tuple[NDArray[np.float64], float]:
    values = data.values
    if values.dtype.kind not in {"i", "u", "f"}:
        raise NumericalStabilityError(
            f"{name} values must be finite real numeric values",
            context={
                "field": name,
                "dtype": str(values.dtype)[:64],
                "reason": "not_real_numeric",
            },
        )
    if not np.isfinite(values).all():
        raise NumericalStabilityError(
            f"{name} values must be finite real numeric values",
            context={
                "field": name,
                "dtype": str(values.dtype)[:64],
                "reason": "not_finite",
            },
        )
    with np.errstate(over="ignore", invalid="ignore"):
        working = values.astype(np.float64, copy=True)
    if not np.isfinite(working).all():
        raise NumericalStabilityError(
            f"{name} values must be safely representable as finite float64 values",
            context={
                "field": name,
                "dtype": str(values.dtype)[:64],
                "reason": "not_float64_representable",
            },
        )
    return working, _source_epsilon(values)


def _bounded_shape(values: NDArray[np.generic]) -> list[JsonValue]:
    return [int(size) for size in values.shape[:_MAX_SHAPE_DIMENSIONS_IN_CONTEXT]]


def _validate_matrix_value_shape(
    name: str,
    values: NDArray[np.float64],
    matrix: LabeledMatrix,
) -> None:
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise NumericalStabilityError(
            f"{name} values must be a two-dimensional square array",
            context={
                "field": name,
                "ndim": int(values.ndim),
                "shape": _bounded_shape(values),
                "shape_truncated": values.ndim > _MAX_SHAPE_DIMENSIONS_IN_CONTEXT,
            },
        )
    expected_shape = (len(matrix.row_labels), len(matrix.column_labels))
    if values.shape != expected_shape:
        raise LabelAlignmentError(
            f"{name} value dimensions must match row and column label counts",
            context={
                "field": name,
                "shape": _bounded_shape(values),
                "expected": [int(size) for size in expected_shape],
            },
        )


def _validate_vector_value_shape(
    name: str,
    values: NDArray[np.float64],
    vector: LabeledVector,
) -> None:
    if values.ndim != 1:
        raise NumericalStabilityError(
            f"{name} values must be a one-dimensional array",
            context={
                "field": name,
                "ndim": int(values.ndim),
                "shape": _bounded_shape(values),
                "shape_truncated": values.ndim > _MAX_SHAPE_DIMENSIONS_IN_CONTEXT,
            },
        )
    if values.shape[0] != len(vector.labels):
        raise LabelAlignmentError(
            f"{name} value length must match its label count",
            context={
                "field": name,
                "length": int(values.shape[0]),
                "expected": len(vector.labels),
            },
        )


def _require_square_alignment(
    name: str,
    matrix: LabeledMatrix,
) -> tuple[tuple[Hashable, ...], str]:
    labels = matrix.row_labels
    if matrix.column_labels != labels:
        raise LabelAlignmentError(
            f"{name} row and column labels must match exactly",
            context={"field": name, "reason": "labels"},
        )
    if not labels:
        raise LabelAlignmentError(
            f"{name} requires a non-empty asset universe",
            context={"field": name, "reason": "empty_asset_universe"},
        )
    if matrix.row_name != matrix.column_name:
        raise LabelAlignmentError(
            f"{name} row and column axis names must match exactly",
            context={"field": name, "reason": "axis_names"},
        )
    return labels, matrix.column_name


def _local_allowance(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    tolerance: float,
    source_epsilon: float,
) -> NDArray[np.float64]:
    local_scale = np.maximum(np.abs(left), np.abs(right))
    dtype_allowance = local_scale * (_DTYPE_EPSILON_MULTIPLIER * source_epsilon)
    return np.asarray(np.maximum(tolerance, dtype_allowance), dtype=np.float64)


def _validated_symmetric_values(
    name: str,
    matrix: LabeledMatrix,
    *,
    tolerance: float,
) -> tuple[NDArray[np.float64], float]:
    _require_square_alignment(name, matrix)
    values, source_epsilon = _finite_real_float64(name, matrix)
    _validate_matrix_value_shape(name, values, matrix)
    transposed = values.T
    exact = values == transposed
    with np.errstate(over="ignore", invalid="ignore"):
        difference = np.abs(values - transposed)
    allowance = _local_allowance(values, transposed, tolerance, source_epsilon)
    if not np.all(exact | (difference <= allowance)):
        raise NumericalStabilityError(
            f"{name} must be symmetric within numerical tolerance",
            context={"field": name, "reason": "not_symmetric"},
        )
    # Preserve exactly equal cells (including extreme finite diagonals) and
    # average only tolerated off-diagonal discrepancies without overflow.
    symmetric = np.asarray(
        np.where(exact, values, values / 2.0 + transposed / 2.0),
        dtype=np.float64,
    )
    if not np.isfinite(symmetric).all():
        raise NumericalStabilityError(
            f"{name} could not be symmetrized safely",
            context={"field": name, "reason": "symmetrization_failed"},
        )
    return symmetric, source_epsilon


def _matrix_result(
    values: NDArray[np.float64],
    source: LabeledMatrix,
) -> LabeledMatrix:
    return LabeledMatrix(
        values,
        source.row_labels,
        source.column_labels,
        source.row_name,
        source.column_name,
    )


def _eigh(
    values: NDArray[np.float64],
    *,
    operation: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            result = np.linalg.eigh(values)
    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as error:
        raise NumericalStabilityError(
            "symmetric eigendecomposition failed safely",
            context={"operation": operation, "reason": type(error).__name__[:64]},
        ) from error
    try:
        eigenvalues, eigenvectors = result
    except (TypeError, ValueError) as error:
        raise NumericalStabilityError(
            "eigensolver output must contain eigenvalues and eigenvectors",
            context={"operation": operation, "reason": "invalid_result_count"},
        ) from error
    dimension = int(values.shape[0])
    validated_eigenvalues = _validated_eigensolver_output(
        "eigenvalues",
        eigenvalues,
        expected_shape=(dimension,),
        operation=operation,
    )
    validated_eigenvectors = _validated_eigensolver_output(
        "eigenvectors",
        eigenvectors,
        expected_shape=(dimension, dimension),
        operation=operation,
    )
    return validated_eigenvalues, validated_eigenvectors


def _validated_eigensolver_output(
    name: str,
    output: object,
    *,
    expected_shape: tuple[int, ...],
    operation: str,
) -> NDArray[np.float64]:
    try:
        values = np.asarray(output)
    except (TypeError, ValueError) as error:
        raise NumericalStabilityError(
            "eigensolver output could not be represented as an array",
            context={
                "operation": operation,
                "field": name,
                "reason": type(error).__name__[:64],
            },
        ) from error
    if values.ndim != len(expected_shape) or values.shape != expected_shape:
        raise NumericalStabilityError(
            "eigensolver output has an invalid shape",
            context={
                "operation": operation,
                "field": name,
                "shape": _bounded_shape(values),
                "expected": [int(size) for size in expected_shape],
            },
        )
    if values.dtype.kind not in {"i", "u", "f"} or not np.isfinite(values).all():
        raise NumericalStabilityError(
            "eigensolver output must contain finite real numeric values",
            context={
                "operation": operation,
                "field": name,
                "dtype": str(values.dtype)[:64],
            },
        )
    with np.errstate(over="ignore", invalid="ignore"):
        validated = values.astype(np.float64, copy=True)
    if not np.isfinite(validated).all():
        raise NumericalStabilityError(
            "eigensolver output must be finite in float64",
            context={
                "operation": operation,
                "field": name,
                "reason": "not_float64_representable",
            },
        )
    return validated


def _eigvalsh(
    values: NDArray[np.float64],
    *,
    operation: str,
) -> NDArray[np.float64]:
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            eigenvalues = np.linalg.eigvalsh(values)
    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as error:
        raise NumericalStabilityError(
            "symmetric eigenvalue verification failed safely",
            context={"operation": operation, "reason": type(error).__name__[:64]},
        ) from error
    dimension = int(values.shape[0])
    return _validated_eigensolver_output(
        "eigenvalues",
        eigenvalues,
        expected_shape=(dimension,),
        operation=operation,
    )


def _correct_psd_roundoff(
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    corrected = values.copy()
    minimum = 0.0
    for _ in range(_MAX_PSD_CORRECTION_STEPS):
        minimum = float(_eigvalsh(corrected, operation="verify_psd_clip")[0])
        if minimum >= 0.0:
            return corrected
        required_shift = np.nextafter(-minimum, np.inf)
        diagonal = np.diag(corrected).copy()
        try:
            with np.errstate(over="raise", invalid="raise"):
                shifted_diagonal = np.nextafter(
                    diagonal + required_shift,
                    np.inf,
                )
        except FloatingPointError as error:
            raise NumericalStabilityError(
                "PSD clipping diagonal correction failed safely",
                context={
                    "operation": "verify_psd_clip",
                    "reason": type(error).__name__[:64],
                },
            ) from error
        if not np.isfinite(shifted_diagonal).all() or np.any(shifted_diagonal <= diagonal):
            raise NumericalStabilityError(
                "PSD clipping diagonal correction was not representable",
                context={
                    "operation": "verify_psd_clip",
                    "reason": "correction_not_representable",
                },
            )
        np.fill_diagonal(corrected, shifted_diagonal)
    raise NumericalStabilityError(
        "PSD clipping did not produce a nonnegative eigenvalue floor",
        context={
            "minimum_eigenvalue": minimum,
            "correction_steps": _MAX_PSD_CORRECTION_STEPS,
        },
    )


def apply_psd_policy(
    matrix: LabeledMatrix,
    policy: PSDPolicy,
    *,
    tolerance: float = 1e-10,
    max_dimension: int = _DEFAULT_MAX_PSD_DIMENSION,
) -> LabeledMatrix:
    """Apply an explicit PSD policy without mutating the labelled matrix.

    ``RAISE`` accepts eigenvalues in ``[-tolerance, 0)`` unchanged; this is a
    numerical acceptance band and does not claim the returned matrix is exactly
    PSD. ``CLIP`` projects negative eigenvalues to zero. ``max_dimension``
    defaults to 2048 and protects untrusted ingress from unbounded
    eigendecomposition work; trusted callers may explicitly choose a higher
    positive built-in integer. ``tolerance`` must not exceed ``1e-2``.
    """

    _require_exact_type("matrix", matrix, LabeledMatrix)
    if type(policy) is not PSDPolicy:
        raise NumericalStabilityError(
            "policy must be an exact PSDPolicy",
            context={"field": "policy", "dtype": _bounded_type_name(policy)},
        )
    validated_tolerance = _validate_tolerance(tolerance)
    validated_max_dimension = _validate_max_dimension(max_dimension)
    matrix_dimension = _matrix_dimension_without_value_copy(matrix)
    if matrix_dimension > validated_max_dimension:
        raise NumericalStabilityError(
            "matrix exceeds the configured maximum dimension",
            context={
                "dimension": matrix_dimension,
                "maximum": validated_max_dimension,
            },
        )
    values, _ = _validated_symmetric_values(
        "matrix",
        matrix,
        tolerance=validated_tolerance,
    )
    eigenvalues, eigenvectors = _eigh(values, operation="apply_psd_policy")
    minimum = float(eigenvalues[0])
    if minimum < -validated_tolerance and policy is PSDPolicy.RAISE:
        raise NumericalStabilityError(
            "matrix is not positive semidefinite",
            context={
                "minimum_eigenvalue": minimum,
                "tolerance": validated_tolerance,
            },
        )
    if policy is PSDPolicy.RAISE:
        return _matrix_result(values, matrix)
    if minimum >= 0.0:
        verified = _correct_psd_roundoff(values)
        return _matrix_result(verified, matrix)

    clipped = np.maximum(eigenvalues, 0.0)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            repaired = (eigenvectors * clipped) @ eigenvectors.T
            repaired = repaired / 2.0 + repaired.T / 2.0
    except (FloatingPointError, ValueError) as error:
        raise NumericalStabilityError(
            "PSD clipping could not be reconstructed safely",
            context={"operation": "apply_psd_policy", "reason": type(error).__name__[:64]},
        ) from error
    if not np.isfinite(repaired).all():
        raise NumericalStabilityError(
            "PSD clipping returned non-finite values",
            context={"operation": "apply_psd_policy", "reason": "non_finite_result"},
        )
    repaired = _correct_psd_roundoff(repaired)
    return _matrix_result(repaired, matrix)


def covariance_to_correlation(
    covariance: LabeledMatrix,
    *,
    tolerance: float = 1e-12,
) -> tuple[LabeledMatrix, LabeledVector]:
    """Convert covariance to correlation and volatility without a PSD policy.

    This validates local covariance/correlation semantics but intentionally
    neither repairs nor certifies global positive semidefiniteness. Call
    :func:`apply_psd_policy` explicitly when a PSD decision is required.
    ``tolerance`` must not exceed ``1e-2``.
    """

    _require_exact_type("covariance", covariance, LabeledMatrix)
    validated_tolerance = _validate_tolerance(tolerance)
    values, source_epsilon = _validated_symmetric_values(
        "covariance",
        covariance,
        tolerance=validated_tolerance,
    )
    variances = np.diag(values).copy()
    bad_positions = np.flatnonzero(variances <= validated_tolerance)
    if bad_positions.size:
        raise NumericalStabilityError(
            "covariance must have strictly positive variance above tolerance",
            context=_position_context(bad_positions),
        )
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            volatility_values = np.sqrt(variances)
            denominator = np.multiply.outer(volatility_values, volatility_values)
            correlation_values = values / denominator
    except FloatingPointError as error:
        raise NumericalStabilityError(
            "covariance could not be scaled safely",
            context={"operation": "covariance_to_correlation", "reason": type(error).__name__},
        ) from error
    if (
        not np.isfinite(volatility_values).all()
        or not np.isfinite(denominator).all()
        or np.any(denominator <= 0.0)
        or not np.isfinite(correlation_values).all()
    ):
        raise NumericalStabilityError(
            "covariance could not be scaled safely",
            context={"operation": "covariance_to_correlation", "reason": "non_finite_scale"},
        )

    boundary_tolerance = max(
        validated_tolerance,
        _DTYPE_EPSILON_MULTIPLIER * source_epsilon,
    )
    if np.any(np.abs(correlation_values) > 1.0 + boundary_tolerance):
        raise NumericalStabilityError(
            "implied correlation entries must lie within [-1, 1]",
            context={
                "operation": "covariance_to_correlation",
                "reason": "outside_unit_interval",
            },
        )
    correlation_values = np.clip(correlation_values, -1.0, 1.0)
    correlation_values = correlation_values / 2.0 + correlation_values.T / 2.0
    np.fill_diagonal(correlation_values, 1.0)
    labels = covariance.column_labels
    axis_name = covariance.column_name
    return (
        LabeledMatrix(
            correlation_values,
            labels,
            labels,
            covariance.row_name,
            covariance.column_name,
        ),
        LabeledVector(volatility_values, labels, axis_name),
    )


def correlation_to_covariance(
    correlation: LabeledMatrix,
    volatility: LabeledVector,
    *,
    tolerance: float = 1e-10,
) -> LabeledMatrix:
    """Scale correlation by volatility without applying a global PSD policy.

    This validates local correlation/covariance semantics but intentionally
    neither repairs nor certifies global positive semidefiniteness. Call
    :func:`apply_psd_policy` explicitly when a PSD decision is required.
    ``tolerance`` must not exceed ``1e-2``.
    """

    _require_exact_type("correlation", correlation, LabeledMatrix)
    _require_exact_type("volatility", volatility, LabeledVector)
    validated_tolerance = _validate_tolerance(tolerance)
    values, source_epsilon = _validated_symmetric_values(
        "correlation",
        correlation,
        tolerance=validated_tolerance,
    )
    if correlation.column_labels != volatility.labels:
        raise LabelAlignmentError(
            "volatility labels must match correlation labels exactly",
            context={"reason": "volatility_labels"},
        )
    if correlation.column_name != volatility.axis_name:
        raise LabelAlignmentError(
            "volatility axis name must match the correlation axis name",
            context={"reason": "axis_names"},
        )
    volatility_values, _ = _finite_real_float64("volatility", volatility)
    _validate_vector_value_shape("volatility", volatility_values, volatility)
    if np.any(volatility_values <= 0.0):
        raise NumericalStabilityError(
            "volatility must be finite real and strictly positive",
            context={
                "field": "volatility",
                "reason": "not_strictly_positive",
            },
        )

    semantic_tolerance = max(
        validated_tolerance,
        _DTYPE_EPSILON_MULTIPLIER * source_epsilon,
    )
    diagonal = np.diag(values)
    if np.any(np.abs(diagonal - 1.0) > semantic_tolerance):
        raise NumericalStabilityError(
            "correlation diagonal must equal one within tolerance",
            context={"field": "correlation", "reason": "diagonal_not_one"},
        )
    if np.any(np.abs(values) > 1.0 + semantic_tolerance):
        raise NumericalStabilityError(
            "correlation entries must lie within [-1, 1]",
            context={"field": "correlation", "reason": "outside_unit_interval"},
        )
    values = np.clip(values, -1.0, 1.0)
    np.fill_diagonal(values, 1.0)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            scale = np.multiply.outer(volatility_values, volatility_values)
            covariance_values = values * scale
    except FloatingPointError as error:
        raise NumericalStabilityError(
            "correlation and volatility could not be scaled safely",
            context={"operation": "correlation_to_covariance", "reason": type(error).__name__},
        ) from error
    if not np.isfinite(scale).all() or not np.isfinite(covariance_values).all():
        raise NumericalStabilityError(
            "correlation and volatility could not be scaled safely",
            context={"operation": "correlation_to_covariance", "reason": "non_finite_scale"},
        )
    scale_diagonal = np.diag(scale)
    covariance_diagonal = np.diag(covariance_values)
    if (
        np.any(scale_diagonal <= 0.0)
        or np.any(covariance_diagonal <= 0.0)
        or not np.array_equal(covariance_diagonal, scale_diagonal)
    ):
        raise NumericalStabilityError(
            "volatility outer-product scaling underflowed",
            context={
                "operation": "correlation_to_covariance",
                "reason": "positive_variance_underflow",
            },
        )
    return _matrix_result(covariance_values, correlation)
