"""Exponentially weighted covariance with explicit decay semantics."""

import math
from collections.abc import Hashable
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
# ESS within sixteen float64 epsilons of one is not usable demeaning history.
_EFFECTIVE_HISTORY_EPSILONS = 16.0


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
class EWMACovariance:
    """Normalized population weighted moment with recent observations weighted most.

    The estimator applies no Bessel or effective-sample correction. With
    ``decay=1``, it therefore matches population sample covariance with
    ``ddof=0``.
    """

    decay: int | float = 0.94
    demean: bool = True
    missing_data_policy: MissingDataPolicy = MissingDataPolicy.RAISE
    psd_policy: PSDPolicy = PSDPolicy.CLIP
    annualization_factor: int | float | None = None
    tolerance: int | float = 1e-10
    max_dimension: int = _DEFAULT_MAX_DIMENSION

    def __post_init__(self) -> None:
        if type(self.decay) not in {int, float}:
            raise _configuration_error(
                "decay",
                "decay must be an exact built-in int or float",
                self.decay,
                reason="invalid_type",
            )
        try:
            validated_decay = float(self.decay)
        except (OverflowError, ValueError) as error:
            raise _configuration_error(
                "decay",
                "decay must be finite, greater than zero, and at most one",
                self.decay,
                reason="outside_valid_range",
            ) from error
        if not math.isfinite(validated_decay) or validated_decay <= 0.0 or validated_decay > 1.0:
            raise _configuration_error(
                "decay",
                "decay must be finite, greater than zero, and at most one",
                self.decay,
                reason="outside_valid_range",
            )
        if type(self.demean) is not bool:
            raise _configuration_error(
                "demean",
                "demean must be an exact bool",
                self.demean,
                reason="invalid_type",
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
            if not math.isfinite(validated_annualization) or validated_annualization <= 0.0:
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
        labels, instrument_count = _preflight_dimension(returns, self.max_dimension)
        prepared = prepare_returns(returns, self.missing_data_policy)
        if prepared.observation_count < 2:
            raise InsufficientHistoryError(
                "EWMA covariance requires at least two complete observations",
                context={"observation_count": prepared.observation_count},
            )

        weights = _normalized_weights(prepared.observation_count, float(self.decay))
        effective_sample_size = _effective_sample_size(weights)
        positive_weight_count = int(np.count_nonzero(weights))
        effective_history_floor = 1.0 + (_EFFECTIVE_HISTORY_EPSILONS * np.finfo(np.float64).eps)
        if self.demean and (
            positive_weight_count < 2 or effective_sample_size <= effective_history_floor
        ):
            raise InsufficientHistoryError(
                "EWMA covariance requires effective history beyond one observation",
                context={
                    "observation_count": prepared.observation_count,
                    "effective_sample_size": effective_sample_size,
                },
            )

        try:
            raw_covariance = _ewma_covariance(
                prepared._computation_values(),
                weights,
                self.demean,
            )
            covariance = _validated_covariance_result(raw_covariance, instrument_count)
        except MemoryError:
            raise
        except QAMRError:
            raise
        except Exception as error:
            raise NumericalStabilityError(
                "EWMA covariance calculation failed safely",
                context={
                    "operation": "ewma_covariance",
                    "reason": type(error).__name__[:64],
                },
            ) from error

        if self.annualization_factor is not None:
            covariance = _annualize_covariance(covariance, self.annualization_factor)

        diagnostic = NumericalDiagnostic(
            code="ewma_covariance",
            severity=DiagnosticSeverity.INFO,
            message="exponentially weighted covariance estimated from prepared returns",
            context={
                "decay": self.decay,
                "demean": self.demean,
                "annualization_factor": self.annualization_factor,
                "effective_sample_size": effective_sample_size,
            },
        )
        return build_covariance_estimate(
            covariance,
            labels,
            observation_count=prepared.observation_count,
            diagnostics=(*prepared.diagnostics, diagnostic),
            psd_policy=self.psd_policy,
            tolerance=float(self.tolerance),
            max_dimension=self.max_dimension,
        )


def _preflight_dimension(
    returns: object,
    max_dimension: int,
) -> tuple[tuple[Hashable, ...], int]:
    if type(returns) is not LabeledMatrix:
        raise DataValidationError(
            "returns must be an exact LabeledMatrix",
            context={
                "field": "returns",
                "dtype": _bounded_type_name(returns),
            },
        )
    labels = returns.column_labels
    if type(labels) is not tuple:
        raise DataValidationError(
            "returns column_labels must be an exact tuple",
            context={
                "field": "column_labels",
                "dtype": _bounded_type_name(labels),
            },
        )
    instrument_count = len(labels)
    if instrument_count > max_dimension:
        raise NumericalStabilityError(
            "EWMA covariance exceeds the configured maximum dimension",
            context={
                "dimension": instrument_count,
                "maximum": max_dimension,
            },
        )
    return labels, instrument_count


def _normalized_weights(
    observation_count: int,
    decay: float,
) -> NDArray[np.float64]:
    try:
        powers = np.arange(observation_count - 1, -1, -1, dtype=np.float64)
        if decay == 1.0:
            weights = np.ones(observation_count, dtype=np.float64)
        else:
            log_decay = math.log(decay)
            with np.errstate(over="raise", invalid="raise", under="ignore"):
                weights = np.exp(powers * log_decay)
        weight_sum = float(np.sum(weights, dtype=np.float64))
        if not math.isfinite(weight_sum) or weight_sum <= 0.0:
            raise NumericalStabilityError(
                "EWMA weights could not be normalized safely",
                context={
                    "operation": "ewma_weights",
                    "reason": "invalid_weight_sum",
                },
            )
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            weights /= weight_sum
        if not np.isfinite(weights).all() or weights[-1] <= 0.0:
            raise NumericalStabilityError(
                "EWMA weights could not be normalized safely",
                context={
                    "operation": "ewma_weights",
                    "reason": "invalid_normalized_weights",
                },
            )
        return weights
    except MemoryError:
        raise
    except QAMRError:
        raise
    except Exception as error:
        raise NumericalStabilityError(
            "EWMA weight construction failed safely",
            context={
                "operation": "ewma_weights",
                "reason": type(error).__name__[:64],
            },
        ) from error


def _effective_sample_size(weights: NDArray[np.float64]) -> float:
    try:
        with np.errstate(
            over="raise",
            invalid="raise",
            divide="raise",
            under="ignore",
        ):
            squared_sum = float(weights @ weights)
            effective_sample_size = 1.0 / squared_sum
        if (
            not math.isfinite(squared_sum)
            or squared_sum <= 0.0
            or not math.isfinite(effective_sample_size)
            or effective_sample_size < 1.0
        ):
            raise NumericalStabilityError(
                "EWMA effective history could not be calculated safely",
                context={
                    "operation": "ewma_weights",
                    "reason": "invalid_effective_sample_size",
                },
            )
        return effective_sample_size
    except MemoryError:
        raise
    except QAMRError:
        raise
    except Exception as error:
        raise NumericalStabilityError(
            "EWMA effective history calculation failed safely",
            context={
                "operation": "ewma_weights",
                "reason": type(error).__name__[:64],
            },
        ) from error


def _ewma_covariance(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
    demean: bool,
) -> NDArray[np.float64]:
    """Calculate a normalized EWMA covariance through a square-root-weight Gram matrix."""
    try:
        _, instrument_count = values.shape
        with np.errstate(
            over="raise",
            invalid="raise",
            divide="raise",
            under="ignore",
        ):
            if demean:
                deltas = values - values[-1]
                center_delta = np.sum(
                    deltas * weights[:, None],
                    axis=0,
                    dtype=np.float64,
                )
                centered = deltas - center_delta
            else:
                centered = np.array(values, dtype=np.float64, copy=True)
            if not np.isfinite(centered).all():
                raise FloatingPointError("non-finite centered returns")
            square_root_weights = np.sqrt(weights)
            weighted_centered = centered * square_root_weights[:, None]
            covariance = weighted_centered.T @ weighted_centered
        if not np.isfinite(covariance).all():
            raise NumericalStabilityError(
                "EWMA covariance produced non-finite values",
                context={
                    "operation": "ewma_covariance",
                    "reason": "non_finite_result",
                },
            )
        scale = float(np.max(np.abs(covariance), initial=0.0))
        symmetry_allowance = 64.0 * np.finfo(np.float64).eps * scale
        if np.any(np.abs(covariance - covariance.T) > symmetry_allowance):
            raise NumericalStabilityError(
                "EWMA covariance Gram matrix lost numerical symmetry",
                context={
                    "operation": "ewma_covariance",
                    "reason": "asymmetric_gram_result",
                },
            )
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            covariance = covariance / 2.0 + covariance.T / 2.0
        varying_columns = np.any(centered != 0.0, axis=0)
        underflow_positions = np.flatnonzero(varying_columns & (np.diag(covariance) == 0.0))
        if underflow_positions.size:
            context = _bounded_position_context(underflow_positions)
            context.update(
                {
                    "operation": "ewma_covariance",
                    "reason": "covariance_underflow",
                }
            )
            raise NumericalStabilityError(
                "EWMA covariance underflowed nonzero observations",
                context=context,
            )
    except MemoryError:
        raise
    except QAMRError:
        raise
    except Exception as error:
        raise NumericalStabilityError(
            "EWMA covariance kernel failed safely",
            context={
                "operation": "ewma_covariance",
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
            "EWMA covariance result could not be represented as an array",
            context={
                "operation": "ewma_covariance",
                "reason": type(error).__name__[:64],
            },
        ) from error
    if instrument_count == 1 and values.ndim == 0:
        values = values.reshape(1, 1)
    expected_shape = (instrument_count, instrument_count)
    if values.ndim != 2 or values.shape != expected_shape:
        raise NumericalStabilityError(
            "EWMA covariance result has an invalid shape",
            context={
                "operation": "ewma_covariance",
                "shape": [int(size) for size in values.shape[:8]],
                "expected": [instrument_count, instrument_count],
            },
        )
    if values.dtype.kind not in {"i", "u", "f"}:
        raise NumericalStabilityError(
            "EWMA covariance result must contain real numeric values",
            context={
                "operation": "ewma_covariance",
                "dtype": str(values.dtype)[:64],
            },
        )
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            converted = values.astype(np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise NumericalStabilityError(
            "EWMA covariance result could not be converted to float64",
            context={
                "operation": "ewma_covariance",
                "reason": type(error).__name__[:64],
            },
        ) from error
    if not np.isfinite(converted).all():
        raise NumericalStabilityError(
            "EWMA covariance result must contain finite values",
            context={
                "operation": "ewma_covariance",
                "reason": "non_finite_result",
            },
        )
    return converted
