"""Spectral covariance denoising with explicit rank-selection assumptions."""

from __future__ import annotations

import math
import sys
from collections.abc import Hashable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from qamr.contracts.arrays import LabeledMatrix
from qamr.contracts.interfaces import RiskEstimator
from qamr.contracts.results import DiagnosticSeverity, NumericalDiagnostic
from qamr.errors import (
    DataValidationError,
    InsufficientHistoryError,
    LabelAlignmentError,
    NumericalStabilityError,
    QAMRError,
)
from qamr.risk.estimates import CovarianceEstimate, build_covariance_estimate
from qamr.risk.matrices import (
    PSDPolicy,
    apply_psd_policy,
    correlation_to_covariance,
    covariance_to_correlation,
)
from qamr.risk.sample import SampleCovariance

_MAX_TOLERANCE = 1e-2
_DEFAULT_MAX_DIMENSION = 2048
_MAX_SHAPE_DIMENSIONS_IN_CONTEXT = 8
_MP_EDGE_EPSILON_MULTIPLIER = 8.0


def _bounded_type_name(value: object) -> str:
    return type(value).__name__[:64]


def _configuration_error(
    field_name: str,
    message: str,
    value: object,
    *,
    reason: str,
) -> DataValidationError:
    return DataValidationError(
        message,
        context={
            "field": field_name,
            "dtype": _bounded_type_name(value),
            "reason": reason,
        },
    )


@dataclass(frozen=True, slots=True)
class SpectralDenoisedCovariance:
    """Replace the selected noise eigenvalues by one stable common mean.

    ``base_estimator`` is accepted structurally. Its result is checked against
    the full exact :class:`CovarianceEstimate` contract at call time.
    """

    signal_rank: int | None = None
    base_estimator: RiskEstimator = field(default_factory=SampleCovariance)
    psd_policy: PSDPolicy = PSDPolicy.CLIP
    tolerance: int | float = 1e-10
    max_dimension: int = _DEFAULT_MAX_DIMENSION
    mp_effective_observations: int | float | None = None

    def __post_init__(self) -> None:
        if self.signal_rank is not None and type(self.signal_rank) is not int:
            raise _configuration_error(
                "signal_rank",
                "signal_rank must be None or an exact built-in integer",
                self.signal_rank,
                reason="invalid_type",
            )
        effective = self.mp_effective_observations
        if effective is not None:
            if type(effective) not in {int, float}:
                raise _configuration_error(
                    "mp_effective_observations",
                    "mp_effective_observations must be an exact built-in int or float",
                    effective,
                    reason="invalid_type",
                )
            try:
                validated_effective = float(effective)
            except (OverflowError, ValueError) as error:
                raise _configuration_error(
                    "mp_effective_observations",
                    "mp_effective_observations must be finite and strictly positive",
                    effective,
                    reason="not_finite_or_positive",
                ) from error
            if not math.isfinite(validated_effective) or validated_effective <= 0.0:
                raise _configuration_error(
                    "mp_effective_observations",
                    "mp_effective_observations must be finite and strictly positive",
                    effective,
                    reason="not_finite_or_positive",
                )
        if type(self.psd_policy) is not PSDPolicy:
            raise _configuration_error(
                "psd_policy",
                "psd_policy must be an exact PSDPolicy",
                self.psd_policy,
                reason="invalid_type",
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
        try:
            estimate = self.base_estimator.estimate
        except MemoryError:
            raise
        except Exception as error:
            raise _configuration_error(
                "base_estimator",
                "base_estimator must expose a callable estimate attribute",
                self.base_estimator,
                reason=_bounded_type_name(error),
            ) from error
        if not callable(estimate):
            raise _configuration_error(
                "base_estimator",
                "base_estimator must expose a callable estimate attribute",
                self.base_estimator,
                reason="estimate_not_callable",
            )

    def estimate(self, returns: LabeledMatrix) -> CovarianceEstimate:
        """Denoise an injected correlation spectrum without mutating inputs."""
        labels, dimension = _preflight_returns(returns, self.max_dimension)
        _validate_explicit_rank(self.signal_rank, dimension)
        try:
            base = self.base_estimator.estimate(returns)
        except MemoryError:
            raise
        except QAMRError:
            raise
        except Exception as error:
            raise NumericalStabilityError(
                "base estimator failed safely during spectral denoising",
                context={
                    "operation": "base_covariance_estimation",
                    "reason": _bounded_type_name(error),
                },
            ) from error
        if type(base) is not CovarianceEstimate:
            raise DataValidationError(
                "base_estimator must return an exact CovarianceEstimate",
                context={
                    "field": "base_estimate",
                    "dtype": _bounded_type_name(base),
                },
            )
        try:
            base.__post_init__()
            base_labels = base.labels
            if base_labels != labels:
                raise LabelAlignmentError(
                    "base covariance labels must match returns labels exactly and in order",
                    context={"reason": "base_labels_mismatch"},
                )
            if len(base_labels) != dimension:
                raise LabelAlignmentError(
                    "base covariance dimension must match the returns universe",
                    context={"reason": "base_dimension_mismatch"},
                )
            if dimension > self.max_dimension:
                raise NumericalStabilityError(
                    "base covariance exceeds the configured maximum dimension",
                    context={"dimension": dimension, "maximum": self.max_dimension},
                )
        except MemoryError:
            raise
        except QAMRError:
            raise
        except Exception as error:
            raise NumericalStabilityError(
                "base covariance result validation failed safely",
                context={
                    "operation": "spectral_denoising",
                    "reason": _bounded_type_name(error),
                },
            ) from error

        effective_observations: int | float | None = None
        effective_source: str | None = None
        if self.signal_rank is None:
            effective_observations, effective_source = _resolve_mp_effective_observations(
                base,
                self.mp_effective_observations,
            )
            if effective_observations < dimension:
                raise InsufficientHistoryError(
                    "Marchenko-Pastur selection requires effective observations "
                    "at least equal to instruments",
                    context={
                        "effective_observations": effective_observations,
                        "instrument_count": dimension,
                    },
                )
            signal_rank = 0
            selection = "marchenko_pastur"
        else:
            signal_rank = self.signal_rank
            selection = "explicit"
        upper_edge: float | None = None
        diagnostic = _diagnostic(
            signal_rank,
            selection,
            upper_edge,
            dimension=dimension,
            effective_observations=effective_observations,
            effective_source=effective_source,
        )
        if signal_rank == dimension:
            return _full_rank_result_safely(
                base,
                diagnostic,
                psd_policy=self.psd_policy,
                tolerance=float(self.tolerance),
                max_dimension=self.max_dimension,
            )

        try:
            correlation = _correlation_snapshot(base.correlation, dimension)
            eigenvalues, eigenvectors = _safe_eigh(correlation, dimension)
            order = np.argsort(eigenvalues, kind="stable")[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]
            if self.signal_rank is None:
                if effective_observations is None:
                    raise NumericalStabilityError(
                        "Marchenko-Pastur effective observations were not resolved",
                        context={
                            "operation": "spectral_denoising",
                            "reason": "missing_effective_observations",
                        },
                    )
                signal_rank, selection, upper_edge = _select_marchenko_pastur_rank(
                    eigenvalues,
                    effective_observations,
                    tolerance=float(self.tolerance),
                )
                diagnostic = _diagnostic(
                    signal_rank,
                    selection,
                    upper_edge,
                    dimension=dimension,
                    effective_observations=effective_observations,
                    effective_source=effective_source,
                )
            else:
                _reject_split_eigenspace(
                    eigenvalues,
                    signal_rank,
                    tolerance=float(self.tolerance),
                )
            denoised_eigenvalues = eigenvalues.copy()
            if signal_rank < dimension:
                denoised_eigenvalues[signal_rank:] = _stable_mean(
                    eigenvalues[signal_rank:],
                )
            with np.errstate(
                over="raise",
                invalid="raise",
                divide="raise",
                under="ignore",
            ):
                reconstructed = (eigenvectors * denoised_eigenvalues) @ eigenvectors.T
                reconstructed = reconstructed / 2.0 + reconstructed.T / 2.0
            if not np.isfinite(reconstructed).all():
                raise NumericalStabilityError(
                    "spectral reconstruction returned non-finite values",
                    context={
                        "operation": "spectral_reconstruction",
                        "reason": "non_finite_result",
                    },
                )
            provisional = LabeledMatrix(
                reconstructed,
                labels,
                labels,
                "instrument",
                "instrument",
            )
            provisional = apply_psd_policy(
                provisional,
                self.psd_policy,
                tolerance=float(self.tolerance),
                max_dimension=self.max_dimension,
            )
            normalized_correlation, _ = covariance_to_correlation(
                provisional,
                tolerance=float(self.tolerance),
            )
            denoised_covariance = correlation_to_covariance(
                normalized_correlation,
                base.volatility,
                tolerance=float(self.tolerance),
            )
        except MemoryError:
            raise
        except QAMRError:
            raise
        except Exception as error:
            raise NumericalStabilityError(
                "spectral denoising failed safely",
                context={
                    "operation": "spectral_denoising",
                    "reason": _bounded_type_name(error),
                },
            ) from error

        return _build_result_safely(
            denoised_covariance,
            labels,
            observation_count=base.observation_count,
            diagnostics=(*base.diagnostics, diagnostic),
            psd_policy=self.psd_policy,
            tolerance=float(self.tolerance),
            max_dimension=self.max_dimension,
        )


def _preflight_returns(
    returns: object,
    max_dimension: int,
) -> tuple[tuple[Hashable, ...], int]:
    if type(returns) is not LabeledMatrix:
        raise DataValidationError(
            "returns must be an exact LabeledMatrix",
            context={"field": "returns", "dtype": _bounded_type_name(returns)},
        )
    labels = returns.column_labels
    dimension = len(labels)
    if dimension > max_dimension:
        raise NumericalStabilityError(
            "spectral denoising exceeds the configured maximum dimension",
            context={"dimension": dimension, "maximum": max_dimension},
        )
    return labels, dimension


def _validate_explicit_rank(signal_rank: int | None, dimension: int) -> None:
    if signal_rank is not None and not 0 <= signal_rank <= dimension:
        raise DataValidationError(
            "signal rank must be between zero and instrument count",
            context={
                "signal_rank": signal_rank,
                "instrument_count": dimension,
            },
        )


def _resolve_mp_effective_observations(
    base: CovarianceEstimate,
    configured: int | float | None,
) -> tuple[int | float, str]:
    if configured is not None:
        return configured, "explicit"
    if not base.diagnostics or base.diagnostics[-1].code != "sample_covariance":
        latest_code = base.diagnostics[-1].code[:64] if base.diagnostics else None
        raise DataValidationError(
            "automatic Marchenko-Pastur rank selection requires a directly "
            "compatible sample covariance spectrum or explicit effective observations",
            context={
                "reason": "incompatible_base_spectrum",
                "latest_diagnostic_code": latest_code,
            },
        )
    latest = base.diagnostics[-1]
    try:
        ddof = latest.context["ddof"]
    except KeyError as error:
        raise DataValidationError(
            "automatic Marchenko-Pastur sample diagnostic must declare an exact ddof",
            context={"reason": "invalid_sample_diagnostic"},
        ) from error
    if type(ddof) is not int or ddof < 0 or ddof >= base.observation_count:
        raise DataValidationError(
            "automatic Marchenko-Pastur sample diagnostic must declare an exact ddof",
            context={"reason": "invalid_sample_diagnostic"},
        )
    return base.observation_count - 1, "sample_covariance"


def _full_rank_result_safely(
    base: CovarianceEstimate,
    diagnostic: NumericalDiagnostic,
    *,
    psd_policy: PSDPolicy,
    tolerance: float,
    max_dimension: int,
) -> CovarianceEstimate:
    try:
        dimension = len(base.labels)
        correlation_values = _correlation_snapshot(base.correlation, dimension)
        provisional = LabeledMatrix(
            correlation_values,
            base.labels,
            base.labels,
            "instrument",
            "instrument",
        )
        verified = apply_psd_policy(
            provisional,
            psd_policy,
            tolerance=tolerance,
            max_dimension=max_dimension,
        )
        verified_values = verified.values
        base_values = base.correlation.values
        diagnostics = (*base.diagnostics, diagnostic)
        if np.array_equal(verified_values, base_values):
            return CovarianceEstimate(
                covariance=base.covariance,
                correlation=base.correlation,
                volatility=base.volatility,
                observation_count=base.observation_count,
                diagnostics=diagnostics,
            )
        normalized_correlation, _ = covariance_to_correlation(
            verified,
            tolerance=tolerance,
        )
        repaired_covariance = correlation_to_covariance(
            normalized_correlation,
            base.volatility,
            tolerance=tolerance,
        )
        return CovarianceEstimate(
            covariance=repaired_covariance,
            correlation=normalized_correlation,
            volatility=base.volatility,
            observation_count=base.observation_count,
            diagnostics=diagnostics,
        )
    except MemoryError:
        raise
    except QAMRError:
        raise
    except Exception as error:
        raise NumericalStabilityError(
            "full-rank spectral result construction failed safely",
            context={
                "operation": "spectral_result_construction",
                "component": "full_rank_result",
                "reason": _bounded_type_name(error),
            },
        ) from error
    except BaseException:
        raise


def _build_result_safely(
    covariance: LabeledMatrix,
    labels: tuple[Hashable, ...],
    *,
    observation_count: int,
    diagnostics: tuple[NumericalDiagnostic, ...],
    psd_policy: PSDPolicy,
    tolerance: float,
    max_dimension: int,
) -> CovarianceEstimate:
    try:
        covariance_values = np.asarray(covariance.values, dtype=np.float64)
        return build_covariance_estimate(
            covariance_values,
            labels,
            observation_count=observation_count,
            diagnostics=diagnostics,
            psd_policy=psd_policy,
            tolerance=tolerance,
            max_dimension=max_dimension,
        )
    except MemoryError:
        raise
    except QAMRError:
        raise
    except Exception as error:
        raise NumericalStabilityError(
            "final covariance estimate construction failed safely",
            context={
                "operation": "spectral_result_construction",
                "component": "covariance_estimate_builder",
                "reason": _bounded_type_name(error),
            },
        ) from error
    except BaseException:
        raise


def _correlation_snapshot(
    correlation: LabeledMatrix,
    dimension: int,
) -> NDArray[np.float64]:
    raw = np.asarray(correlation.values)
    expected_shape = (dimension, dimension)
    if raw.ndim != 2 or raw.shape != expected_shape:
        raise NumericalStabilityError(
            "base correlation has an invalid shape",
            context={
                "operation": "spectral_denoising",
                "shape": [int(size) for size in raw.shape[:_MAX_SHAPE_DIMENSIONS_IN_CONTEXT]],
                "expected": [dimension, dimension],
            },
        )
    if raw.dtype.kind not in {"i", "u", "f"} or not np.isfinite(raw).all():
        raise DataValidationError(
            "base correlation must contain finite real numeric values",
            context={
                "field": "base_correlation",
                "dtype": str(raw.dtype)[:64],
                "reason": "not_finite_real_numeric",
            },
        )
    with np.errstate(over="ignore", invalid="ignore"):
        values = raw.astype(np.float64, copy=True)
    if not np.isfinite(values).all():
        raise DataValidationError(
            "base correlation must be safely representable as finite float64 values",
            context={
                "field": "base_correlation",
                "dtype": str(raw.dtype)[:64],
                "reason": "not_float64_representable",
            },
        )
    if not np.allclose(values, values.T, rtol=1e-10, atol=1e-12):
        raise DataValidationError(
            "base correlation must be symmetric within numerical tolerance",
            context={"field": "base_correlation", "reason": "not_symmetric"},
        )
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        return values / 2.0 + values.T / 2.0


def _safe_eigh(
    values: NDArray[np.float64],
    dimension: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            result = np.linalg.eigh(values)
    except MemoryError:
        raise
    except QAMRError:
        raise
    except Exception as error:
        raise NumericalStabilityError(
            "spectral eigendecomposition failed safely",
            context={
                "operation": "spectral_eigendecomposition",
                "reason": _bounded_type_name(error),
            },
        ) from error
    try:
        eigenvalues, eigenvectors = result
    except (TypeError, ValueError) as error:
        raise NumericalStabilityError(
            "spectral eigendecomposition returned an invalid result",
            context={
                "operation": "spectral_eigendecomposition",
                "reason": "invalid_result_count",
            },
        ) from error
    return (
        _validated_solver_output(
            "eigenvalues",
            eigenvalues,
            expected_shape=(dimension,),
        ),
        _validated_solver_output(
            "eigenvectors",
            eigenvectors,
            expected_shape=(dimension, dimension),
        ),
    )


def _validated_solver_output(
    field_name: str,
    output: object,
    *,
    expected_shape: tuple[int, ...],
) -> NDArray[np.float64]:
    try:
        values = np.asarray(output)
    except (TypeError, ValueError, OverflowError) as error:
        raise NumericalStabilityError(
            "spectral eigendecomposition output could not be represented",
            context={
                "operation": "spectral_eigendecomposition",
                "field": field_name,
                "reason": _bounded_type_name(error),
            },
        ) from error
    if values.ndim != len(expected_shape) or values.shape != expected_shape:
        raise NumericalStabilityError(
            "spectral eigendecomposition output has an invalid shape",
            context={
                "operation": "spectral_eigendecomposition",
                "field": field_name,
                "shape": [int(size) for size in values.shape[:_MAX_SHAPE_DIMENSIONS_IN_CONTEXT]],
                "expected": [int(size) for size in expected_shape],
            },
        )
    if values.dtype.kind not in {"i", "u", "f"} or not np.isfinite(values).all():
        raise NumericalStabilityError(
            "spectral eigendecomposition output must contain finite real values",
            context={
                "operation": "spectral_eigendecomposition",
                "field": field_name,
                "dtype": str(values.dtype)[:64],
            },
        )
    with np.errstate(over="ignore", invalid="ignore"):
        validated = values.astype(np.float64, copy=True)
    if not np.isfinite(validated).all():
        raise NumericalStabilityError(
            "spectral eigendecomposition output must be finite in float64",
            context={
                "operation": "spectral_eigendecomposition",
                "field": field_name,
                "reason": "not_float64_representable",
            },
        )
    return validated


def _select_marchenko_pastur_rank(
    eigenvalues_descending: NDArray[np.float64],
    effective_observations: int | float,
    *,
    tolerance: float,
) -> tuple[int, str, float]:
    dimension = int(eigenvalues_descending.size)
    if effective_observations < dimension:
        raise InsufficientHistoryError(
            "Marchenko-Pastur selection requires effective observations at least "
            "equal to instruments",
            context={
                "effective_observations": effective_observations,
                "instrument_count": dimension,
            },
        )
    concentration = (
        0.0
        if effective_observations > sys.float_info.max
        else float(dimension) / float(effective_observations)
    )
    upper_edge = float((1.0 + math.sqrt(concentration)) ** 2)
    rank = 0
    for start, stop in _eigenvalue_groups(
        eigenvalues_descending,
        tolerance=tolerance,
    ):
        representative = _stable_mean(eigenvalues_descending[start:stop])
        if not _strictly_above_mp_edge(representative, upper_edge):
            break
        rank = stop
    return rank, "marchenko_pastur", upper_edge


def _strictly_above_mp_edge(value: float, upper_edge: float) -> bool:
    scale = max(1.0, abs(value), abs(upper_edge))
    allowance = scale * _MP_EDGE_EPSILON_MULTIPLIER * np.finfo(np.float64).eps
    return bool(value > upper_edge + allowance)


def _reject_split_eigenspace(
    eigenvalues_descending: NDArray[np.float64],
    signal_rank: int,
    *,
    tolerance: float,
) -> None:
    dimension = int(eigenvalues_descending.size)
    if signal_rank == 0 or signal_rank == dimension:
        return
    upper = float(eigenvalues_descending[signal_rank - 1])
    lower = float(eigenvalues_descending[signal_rank])
    if _same_eigenvalue_group(upper, lower, tolerance=tolerance):
        raise DataValidationError(
            "signal rank must not split an eigenvalue multiplicity group",
            context={
                "signal_rank": signal_rank,
                "reason": "split_eigenvalue_multiplicity",
            },
        )


def _eigenvalue_groups(
    eigenvalues_descending: NDArray[np.float64],
    *,
    tolerance: float,
) -> tuple[tuple[int, int], ...]:
    groups: list[tuple[int, int]] = []
    start = 0
    dimension = int(eigenvalues_descending.size)
    while start < dimension:
        stop = start + 1
        while stop < dimension and _same_eigenvalue_group(
            float(eigenvalues_descending[stop - 1]),
            float(eigenvalues_descending[stop]),
            tolerance=tolerance,
        ):
            stop += 1
        groups.append((start, stop))
        start = stop
    return tuple(groups)


def _same_eigenvalue_group(
    left: float,
    right: float,
    *,
    tolerance: float,
) -> bool:
    scale = max(1.0, abs(left), abs(right))
    allowance = scale * (tolerance + 8.0 * np.finfo(np.float64).eps)
    return bool(abs(left - right) <= allowance)


def _stable_mean(values: NDArray[np.float64]) -> float:
    if values.size == 0:
        raise NumericalStabilityError(
            "noise spectrum must be non-empty",
            context={
                "operation": "spectral_denoising",
                "reason": "empty_noise_spectrum",
            },
        )
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return 0.0
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        mean = scale * float(np.sum(values / scale, dtype=np.float64) / values.size)
    if not math.isfinite(mean):
        raise NumericalStabilityError(
            "noise eigenvalue mean was not finite",
            context={
                "operation": "spectral_denoising",
                "reason": "non_finite_noise_mean",
            },
        )
    return mean


def _diagnostic(
    signal_rank: int,
    rank_selection: str,
    upper_edge: float | None,
    *,
    dimension: int,
    effective_observations: int | float | None,
    effective_source: str | None,
) -> NumericalDiagnostic:
    denoising_applied = signal_rank < dimension
    noise_count = dimension - signal_rank
    return NumericalDiagnostic(
        code="spectral_denoising",
        severity=DiagnosticSeverity.INFO,
        message=(
            "noise eigenvalues replaced by their common mean"
            if denoising_applied
            else "no spectral denoising was applied"
        ),
        context={
            "signal_rank": signal_rank,
            "rank_selection": rank_selection,
            "marchenko_pastur_upper_edge": upper_edge,
            "effective_observations": effective_observations,
            "effective_observations_source": effective_source,
            "denoising_applied": denoising_applied,
            "noise_eigenvalue_count": noise_count,
        },
    )
