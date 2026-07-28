"""Label-preserving sample covariance estimator."""

import math
import sys
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from qamr._types import JsonValue
from qamr.contracts.arrays import LabeledMatrix
from qamr.contracts.dataset import MissingDataPolicy
from qamr.contracts.results import DiagnosticSeverity, NumericalDiagnostic
from qamr.errors import (
    DataValidationError,
    InsufficientHistoryError,
    NumericalStabilityError,
    QAMRError,
)
from qamr.risk._preparation import _annualize_covariance, prepare_returns
from qamr.risk.estimates import CovarianceEstimate, build_covariance_estimate
from qamr.risk.matrices import PSDPolicy

_MAX_TOLERANCE = 1e-2
_DEFAULT_MAX_DIMENSION = 2048
_MAX_CONTEXT_POSITIONS = 32


def _bounded_type_name(value: object) -> str:
    return type(value).__name__[:64]


def _configuration_error(
    field: str,
    message: str,
    value: object,
    *,
    reason: str,
) -> DataValidationError:
    return DataValidationError(
        message,
        context={
            "field": field,
            "dtype": _bounded_type_name(value),
            "reason": reason,
        },
    )


@dataclass(frozen=True, slots=True)
class SampleCovariance:
    """Ordinary sample covariance with explicit preparation and scaling policies."""

    ddof: int = 1
    missing_data_policy: MissingDataPolicy = MissingDataPolicy.RAISE
    psd_policy: PSDPolicy = PSDPolicy.RAISE
    annualization_factor: int | float | None = None
    tolerance: int | float = 1e-10
    max_dimension: int = _DEFAULT_MAX_DIMENSION

    def __post_init__(self) -> None:
        if type(self.ddof) is not int:
            raise _configuration_error(
                "ddof",
                "ddof must be an exact built-in integer",
                self.ddof,
                reason="invalid_type",
            )
        if self.ddof < 0:
            raise _configuration_error(
                "ddof",
                "ddof must be non-negative",
                self.ddof,
                reason="negative",
            )
        if self.ddof > sys.maxsize:
            raise _configuration_error(
                "ddof",
                "ddof must not exceed sys.maxsize",
                self.ddof,
                reason="too_large",
            )
        if type(self.missing_data_policy) is not MissingDataPolicy:
            raise _configuration_error(
                "missing_data_policy",
                "missing_data_policy must be an exact MissingDataPolicy",
                self.missing_data_policy,
                reason="invalid_type",
            )
        if type(self.psd_policy) is not PSDPolicy:
            raise _configuration_error(
                "psd_policy",
                "psd_policy must be an exact PSDPolicy",
                self.psd_policy,
                reason="invalid_type",
            )
        annualization = self.annualization_factor
        if annualization is not None:
            if type(annualization) not in {int, float}:
                raise _configuration_error(
                    "annualization_factor",
                    "annualization factor must be an exact built-in int or float",
                    annualization,
                    reason="invalid_type",
                )
            try:
                validated_annualization = float(annualization)
            except (OverflowError, ValueError) as error:
                raise _configuration_error(
                    "annualization_factor",
                    "annualization factor must be finite and positive",
                    annualization,
                    reason="not_finite_or_positive",
                ) from error
            if not math.isfinite(validated_annualization) or annualization <= 0:
                raise _configuration_error(
                    "annualization_factor",
                    "annualization factor must be finite and positive",
                    annualization,
                    reason="not_finite_or_positive",
                )
        if type(self.tolerance) not in {int, float}:
            raise _configuration_error(
                "tolerance",
                "tolerance must be an exact built-in int or float",
                self.tolerance,
                reason="invalid_type",
            )
        try:
            validated_tolerance = float(self.tolerance)
        except (OverflowError, ValueError) as error:
            raise _configuration_error(
                "tolerance",
                "tolerance must be finite, positive, and at most 1e-2",
                self.tolerance,
                reason="outside_valid_range",
            ) from error
        if (
            not math.isfinite(validated_tolerance)
            or validated_tolerance <= 0.0
            or validated_tolerance > _MAX_TOLERANCE
        ):
            raise _configuration_error(
                "tolerance",
                "tolerance must be finite, positive, and at most 1e-2",
                self.tolerance,
                reason="outside_valid_range",
            )
        if type(self.max_dimension) is not int:
            raise _configuration_error(
                "max_dimension",
                "max_dimension must be an exact built-in integer",
                self.max_dimension,
                reason="invalid_type",
            )
        if self.max_dimension <= 0:
            raise _configuration_error(
                "max_dimension",
                "max_dimension must be strictly positive",
                self.max_dimension,
                reason="not_positive",
            )

    def estimate(self, returns: LabeledMatrix) -> CovarianceEstimate:
        """Estimate covariance from complete time-by-instrument observations."""
        prepared = prepare_returns(returns, self.missing_data_policy)
        if prepared.observation_count <= self.ddof:
            raise InsufficientHistoryError(
                "sample covariance requires more observations than ddof",
                context={
                    "observation_count": prepared.observation_count,
                    "ddof": self.ddof,
                },
            )

        instrument_count = len(returns.column_labels)
        if instrument_count > self.max_dimension:
            raise NumericalStabilityError(
                "sample covariance exceeds the configured maximum dimension",
                context={
                    "dimension": instrument_count,
                    "maximum": self.max_dimension,
                },
            )
        try:
            raw_covariance = _sample_covariance(
                prepared._computation_values(),
                self.ddof,
            )
        except MemoryError:
            raise
        except QAMRError:
            raise
        except Exception as error:
            raise NumericalStabilityError(
                "sample covariance calculation failed safely",
                context={
                    "operation": "sample_covariance",
                    "reason": type(error).__name__[:64],
                },
            ) from error
        covariance = _validated_covariance_result(raw_covariance, instrument_count)

        if self.annualization_factor is not None:
            covariance = _annualize_covariance(
                covariance,
                self.annualization_factor,
            )

        diagnostic = NumericalDiagnostic(
            code="sample_covariance",
            severity=DiagnosticSeverity.INFO,
            message="sample covariance estimated from prepared returns",
            context={
                "ddof": self.ddof,
                "annualization_factor": self.annualization_factor,
            },
        )
        return build_covariance_estimate(
            covariance,
            returns.column_labels,
            observation_count=prepared.observation_count,
            diagnostics=(*prepared.diagnostics, diagnostic),
            psd_policy=self.psd_policy,
            tolerance=float(self.tolerance),
            max_dimension=self.max_dimension,
        )


def _sample_covariance(
    values: NDArray[np.float64],
    ddof: int,
) -> NDArray[np.float64]:
    try:
        observation_count, instrument_count = values.shape
        with np.errstate(
            over="raise",
            invalid="raise",
            divide="raise",
            under="ignore",
        ):
            anchor = values[0]
            deltas = values - anchor
            mean_deltas = np.sum(deltas, axis=0, dtype=np.float64) / observation_count
            centers = anchor + mean_deltas
            deltas -= mean_deltas
            if not np.isfinite(centers).all() or not np.isfinite(deltas).all():
                raise FloatingPointError("non-finite centered returns")
            covariance = deltas.T @ deltas
            covariance = covariance / (observation_count - ddof)
            covariance = covariance / 2.0 + covariance.T / 2.0
        varying_columns = np.any(deltas != 0.0, axis=0)
        underflow_positions = np.flatnonzero(varying_columns & (np.diag(covariance) == 0.0))
        if underflow_positions.size:
            context = _bounded_position_context(underflow_positions)
            context.update(
                {
                    "operation": "sample_covariance",
                    "reason": "covariance_underflow",
                }
            )
            raise NumericalStabilityError(
                "sample covariance underflowed nonzero centered observations",
                context=context,
            )
    except MemoryError:
        raise
    except QAMRError:
        raise
    except Exception as error:
        raise NumericalStabilityError(
            "sample covariance kernel failed safely",
            context={
                "operation": "sample_covariance",
                "reason": type(error).__name__[:64],
            },
        ) from error
    return _validated_covariance_result(covariance, instrument_count)


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


def _validated_covariance_result(
    result: object,
    instrument_count: int,
) -> NDArray[np.float64]:
    try:
        values = np.asarray(result)
    except (TypeError, ValueError, OverflowError) as error:
        raise NumericalStabilityError(
            "sample covariance result could not be represented as an array",
            context={
                "operation": "sample_covariance",
                "reason": type(error).__name__[:64],
            },
        ) from error
    if instrument_count == 1 and values.ndim == 0:
        values = values.reshape(1, 1)
    expected_shape = (instrument_count, instrument_count)
    if values.ndim != 2 or values.shape != expected_shape:
        raise NumericalStabilityError(
            "sample covariance result has an invalid shape",
            context={
                "operation": "sample_covariance",
                "shape": [int(size) for size in values.shape[:8]],
                "expected": [instrument_count, instrument_count],
            },
        )
    if values.dtype.kind not in {"i", "u", "f"}:
        raise NumericalStabilityError(
            "sample covariance result must contain real numeric values",
            context={
                "operation": "sample_covariance",
                "dtype": str(values.dtype)[:64],
            },
        )
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            converted = values.astype(np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise NumericalStabilityError(
            "sample covariance result could not be converted to float64",
            context={
                "operation": "sample_covariance",
                "reason": type(error).__name__[:64],
            },
        ) from error
    if not np.isfinite(converted).all():
        raise NumericalStabilityError(
            "sample covariance result must contain finite values",
            context={
                "operation": "sample_covariance",
                "reason": "non_finite_result",
            },
        )
    return converted
