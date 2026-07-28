"""Shared explicit preparation of return observations."""

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from qamr._types import JsonValue
from qamr.contracts.arrays import LabeledMatrix
from qamr.contracts.dataset import MissingDataPolicy
from qamr.contracts.results import DiagnosticSeverity, NumericalDiagnostic
from qamr.errors import DataValidationError, NumericalStabilityError, QAMRError

_MAX_CONTEXT_DIMENSIONS = 8
_ANNUALIZATION_ROUND_TRIP_EPSILONS = 64.0


def _bounded_type_name(value: object) -> str:
    return type(value).__name__[:64]


def _bounded_axis_name(value: object) -> str:
    return value[:64] if type(value) is str else _bounded_type_name(value)


def _bounded_shape(values: NDArray[Any]) -> list[JsonValue]:
    return [int(size) for size in values.shape[:_MAX_CONTEXT_DIMENSIONS]]


def _public_snapshot(values: NDArray[np.float64]) -> NDArray[np.float64]:
    snapshot = np.array(values, dtype=np.float64, copy=True)
    snapshot.flags.writeable = False
    return snapshot


def _annualize_covariance(
    covariance: NDArray[np.float64],
    factor: int | float,
) -> NDArray[np.float64]:
    """Scale covariance while rejecting float64 precision loss."""
    try:
        validated_factor = float(factor)
        with np.errstate(
            over="ignore",
            invalid="ignore",
            divide="ignore",
            under="ignore",
        ):
            scaled = covariance * validated_factor
            nonzero_base = covariance != 0.0
            if not np.isfinite(scaled).all():
                raise NumericalStabilityError(
                    "covariance annualization produced non-finite values",
                    context={
                        "operation": "annualization",
                        "reason": "annualization_precision_loss",
                    },
                )
            if np.any(nonzero_base & (scaled == 0.0)):
                raise NumericalStabilityError(
                    "covariance annualization lost nonzero values",
                    context={
                        "operation": "annualization",
                        "reason": "annualization_precision_loss",
                    },
                )
            recovered = scaled[nonzero_base] / validated_factor
        base_values = covariance[nonzero_base]
        allowance = (
            _ANNUALIZATION_ROUND_TRIP_EPSILONS * np.finfo(np.float64).eps * np.abs(base_values)
        )
        if not np.isfinite(recovered).all() or np.any(np.abs(recovered - base_values) > allowance):
            raise NumericalStabilityError(
                "covariance annualization lost numerical precision",
                context={
                    "operation": "annualization",
                    "reason": "annualization_precision_loss",
                },
            )
        return np.asarray(scaled, dtype=np.float64)
    except MemoryError:
        raise
    except QAMRError:
        raise
    except Exception as error:
        raise NumericalStabilityError(
            "covariance annualization failed safely",
            context={
                "operation": "annualization",
                "reason": type(error).__name__[:64],
            },
        ) from error


@dataclass(frozen=True, slots=True, init=False, eq=False)
class PreparedReturns:
    """Internal immutable snapshot of prepared time-by-instrument returns."""

    _values: NDArray[np.float64] = field(repr=False)
    observation_count: int
    diagnostics: tuple[NumericalDiagnostic, ...]

    __hash__: ClassVar[None] = None  # type: ignore[assignment]

    def __init__(
        self,
        values: NDArray[np.float64],
        observation_count: int,
        diagnostics: tuple[NumericalDiagnostic, ...],
    ) -> None:
        self._validate_components(values, observation_count, diagnostics)
        copied = _public_snapshot(values)
        object.__setattr__(self, "_values", copied)
        object.__setattr__(self, "observation_count", observation_count)
        object.__setattr__(self, "diagnostics", diagnostics)

    @staticmethod
    def _validate_components(
        values: object,
        observation_count: object,
        diagnostics: object,
    ) -> None:
        if type(values) is not np.ndarray:
            raise DataValidationError(
                "prepared values must be an exact ndarray",
                context={"field": "values", "dtype": _bounded_type_name(values)},
            )
        if values.ndim != 2:
            raise DataValidationError(
                "prepared values must be two-dimensional",
                context={"field": "values", "ndim": int(values.ndim)},
            )
        if values.dtype != np.dtype(np.float64):
            raise DataValidationError(
                "prepared values must have float64 dtype",
                context={"field": "values", "dtype": str(values.dtype)[:64]},
            )
        if not np.isfinite(values).all():
            raise DataValidationError(
                "prepared values must be finite",
                context={"field": "values", "reason": "not_finite"},
            )
        if type(observation_count) is not int or observation_count != values.shape[0]:
            raise DataValidationError(
                "observation_count must exactly match prepared rows",
                context={
                    "field": "observation_count",
                    "dtype": _bounded_type_name(observation_count),
                },
            )
        if type(diagnostics) is not tuple or any(
            type(item) is not NumericalDiagnostic for item in diagnostics
        ):
            raise DataValidationError(
                "diagnostics must be a tuple of NumericalDiagnostic values",
                context={
                    "field": "diagnostics",
                    "dtype": _bounded_type_name(diagnostics),
                },
            )

    @classmethod
    def _from_owned(
        cls,
        values: NDArray[np.float64],
        diagnostics: tuple[NumericalDiagnostic, ...],
    ) -> "PreparedReturns":
        observation_count = int(values.shape[0])
        cls._validate_components(values, observation_count, diagnostics)
        owned = values if values.flags.owndata else np.array(values, copy=True)
        owned.flags.writeable = False
        prepared = object.__new__(cls)
        object.__setattr__(prepared, "_values", owned)
        object.__setattr__(prepared, "observation_count", observation_count)
        object.__setattr__(prepared, "diagnostics", diagnostics)
        return prepared

    @property
    def values(self) -> NDArray[np.float64]:
        """Return an isolated read-only snapshot of the prepared values."""
        return _public_snapshot(self._values)

    def _computation_values(self) -> NDArray[np.float64]:
        """Borrow the internal read-only array for trusted risk kernels."""
        borrowed = self._values.view()
        borrowed.flags.writeable = False
        return borrowed


def _require_exact_runtime_contract(
    returns: object,
    policy: object,
) -> tuple[LabeledMatrix, MissingDataPolicy]:
    if type(returns) is not LabeledMatrix:
        raise DataValidationError(
            "returns must be an exact LabeledMatrix",
            context={"field": "returns", "dtype": _bounded_type_name(returns)},
        )
    if type(policy) is not MissingDataPolicy:
        raise DataValidationError(
            "policy must be an exact MissingDataPolicy",
            context={"field": "policy", "dtype": _bounded_type_name(policy)},
        )
    return returns, policy


def _finite_real_float64(values: NDArray[Any]) -> NDArray[np.float64]:
    if values.dtype.kind not in {"i", "u", "f"}:
        raise DataValidationError(
            "returns values must be real numeric values",
            context={
                "field": "returns",
                "dtype": str(values.dtype)[:64],
                "reason": "not_real_numeric",
            },
        )
    if np.isinf(values).any():
        raise DataValidationError(
            "returns must not contain infinity",
            context={"field": "returns", "reason": "infinite"},
        )
    if values.dtype.kind == "f" and values.dtype.itemsize > np.dtype(np.float64).itemsize:
        raise DataValidationError(
            "returns floating values must be exactly representable as float64",
            context={
                "field": "returns",
                "dtype": str(values.dtype)[:64],
                "reason": "not_exactly_representable",
            },
        )
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            converted = values.astype(np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise DataValidationError(
            "returns could not be converted safely to float64",
            context={
                "field": "returns",
                "dtype": str(values.dtype)[:64],
                "reason": type(error).__name__[:64],
            },
        ) from error
    if np.isinf(converted).any():
        raise DataValidationError(
            "returns must be safely representable as finite float64 values",
            context={
                "field": "returns",
                "dtype": str(values.dtype)[:64],
                "reason": "not_float64_representable",
            },
        )
    if values.dtype.kind in {"i", "u"} and any(
        int(converted_value) != int(source_value)
        for source_value, converted_value in zip(
            values.flat,
            converted.flat,
            strict=True,
        )
    ):
        raise DataValidationError(
            "returns integers must be exactly representable as float64",
            context={
                "field": "returns",
                "dtype": str(values.dtype)[:64],
                "reason": "not_exactly_representable",
            },
        )
    return converted


def prepare_returns(
    returns: LabeledMatrix,
    policy: MissingDataPolicy,
) -> PreparedReturns:
    """Validate and copy time-by-instrument returns under an explicit NaN policy."""
    validated_returns, validated_policy = _require_exact_runtime_contract(returns, policy)
    row_name = validated_returns.row_name
    column_name = validated_returns.column_name
    if type(row_name) is not str or type(column_name) is not str:
        raise DataValidationError(
            "returns must use time-by-instrument axes",
            context={
                "row_name": _bounded_axis_name(row_name),
                "column_name": _bounded_axis_name(column_name),
            },
        )
    if row_name != "time" or column_name != "instrument":
        raise DataValidationError(
            "returns must use time-by-instrument axes",
            context={
                "row_name": row_name[:64],
                "column_name": column_name[:64],
            },
        )
    values = validated_returns.values
    if values.ndim != 2:
        raise DataValidationError(
            "returns values must be a two-dimensional array",
            context={
                "field": "returns",
                "ndim": int(values.ndim),
                "shape": _bounded_shape(values),
                "shape_truncated": values.ndim > _MAX_CONTEXT_DIMENSIONS,
            },
        )
    if values.shape != (
        len(validated_returns.row_labels),
        len(validated_returns.column_labels),
    ):
        expected_shape: list[JsonValue] = [
            len(validated_returns.row_labels),
            len(validated_returns.column_labels),
        ]
        raise DataValidationError(
            "returns value dimensions must match labels",
            context={
                "field": "returns",
                "shape": _bounded_shape(values),
                "expected": expected_shape,
            },
        )
    if values.shape[1] == 0:
        raise DataValidationError(
            "returns require a non-empty instrument universe",
            context={"field": "returns", "reason": "empty_instrument_universe"},
        )
    converted = _finite_real_float64(values)
    missing_count = int(np.isnan(converted).sum())
    if missing_count and validated_policy is MissingDataPolicy.RAISE:
        raise DataValidationError(
            "returns contain missing returns",
            context={
                "missing_count": missing_count,
                "policy": validated_policy.value,
            },
        )

    diagnostics: tuple[NumericalDiagnostic, ...] = ()
    if missing_count:
        complete_rows = ~np.isnan(converted).any(axis=1)
        dropped_count = int((~complete_rows).sum())
        converted = converted[complete_rows]
        diagnostics = (
            NumericalDiagnostic(
                code="dropped_missing_observations",
                severity=DiagnosticSeverity.WARNING,
                message="rows with one or more missing returns were dropped",
                context={"dropped_count": dropped_count},
            ),
        )

    return PreparedReturns._from_owned(converted, diagnostics)
