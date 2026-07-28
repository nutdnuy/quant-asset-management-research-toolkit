"""Explicit convex shrinkage of an injected covariance estimate."""

from __future__ import annotations

import math
from collections.abc import Hashable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from qamr.contracts.arrays import LabeledMatrix
from qamr.contracts.interfaces import RiskEstimator
from qamr.contracts.results import DiagnosticSeverity, NumericalDiagnostic
from qamr.errors import (
    DataValidationError,
    LabelAlignmentError,
    NumericalStabilityError,
    QAMRError,
)
from qamr.risk.estimates import CovarianceEstimate, build_covariance_estimate
from qamr.risk.matrices import PSDPolicy
from qamr.risk.sample import SampleCovariance

_MAX_TOLERANCE = 1e-2
_DEFAULT_MAX_DIMENSION = 2048


class ShrinkageTarget(str, Enum):
    """Supported explicit covariance shrinkage targets."""

    DIAGONAL = "diagonal"
    SCALED_IDENTITY = "scaled_identity"


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
class ShrinkageCovariance:
    """Shrink an injected covariance toward a caller-selected explicit target.

    ``base_estimator`` is accepted structurally. Runtime validation can prove
    only that it exposes a callable ``estimate`` attribute; its signature and
    return contract are checked when :meth:`estimate` is called.
    """

    shrinkage: int | float
    target: ShrinkageTarget = ShrinkageTarget.DIAGONAL
    base_estimator: RiskEstimator = field(default_factory=SampleCovariance)
    psd_policy: PSDPolicy = PSDPolicy.CLIP
    tolerance: int | float = 1e-10
    max_dimension: int = _DEFAULT_MAX_DIMENSION

    def __post_init__(self) -> None:
        if type(self.shrinkage) not in {int, float}:
            raise _configuration_error(
                "shrinkage",
                "shrinkage must be an exact built-in int or float",
                self.shrinkage,
                reason="invalid_type",
            )
        try:
            validated_shrinkage = float(self.shrinkage)
        except (OverflowError, ValueError) as error:
            raise _configuration_error(
                "shrinkage",
                "shrinkage must be finite and between zero and one",
                self.shrinkage,
                reason="outside_valid_range",
            ) from error
        if (
            not math.isfinite(validated_shrinkage)
            or validated_shrinkage < 0.0
            or validated_shrinkage > 1.0
        ):
            raise _configuration_error(
                "shrinkage",
                "shrinkage must be finite and between zero and one",
                self.shrinkage,
                reason="outside_valid_range",
            )
        if type(self.target) is not ShrinkageTarget:
            raise _configuration_error(
                "target",
                "target must be an exact ShrinkageTarget",
                self.target,
                reason="invalid_type",
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
        """Estimate and explicitly shrink covariance without mutating inputs."""
        labels, dimension = _preflight_returns(returns, self.max_dimension)
        try:
            base = self.base_estimator.estimate(returns)
        except MemoryError:
            raise
        except QAMRError:
            raise
        except Exception as error:
            raise NumericalStabilityError(
                "base estimator failed safely during covariance shrinkage",
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
        # Enforce the complete public result contract even if a custom backend
        # bypassed normal construction.
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
        except MemoryError:
            raise
        except QAMRError:
            raise
        except Exception as error:
            raise NumericalStabilityError(
                "base covariance result validation failed safely",
                context={
                    "operation": "covariance_shrinkage",
                    "reason": _bounded_type_name(error),
                },
            ) from error

        try:
            shrunk = _shrink_covariance(
                base.covariance.values,
                float(self.shrinkage),
                self.target,
                dimension,
            )
        except MemoryError:
            raise
        except QAMRError:
            raise
        except Exception as error:
            raise NumericalStabilityError(
                "covariance shrinkage failed safely",
                context={
                    "operation": "covariance_shrinkage",
                    "reason": _bounded_type_name(error),
                },
            ) from error

        diagnostic = NumericalDiagnostic(
            code="covariance_shrinkage",
            severity=DiagnosticSeverity.INFO,
            message="covariance shrunk toward an explicit target",
            context={
                "shrinkage": self.shrinkage,
                "target": self.target.value,
            },
        )
        return build_covariance_estimate(
            shrunk,
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
            context={
                "field": "returns",
                "dtype": _bounded_type_name(returns),
            },
        )
    labels = returns.column_labels
    dimension = len(labels)
    if dimension > max_dimension:
        raise NumericalStabilityError(
            "covariance shrinkage exceeds the configured maximum dimension",
            context={
                "dimension": dimension,
                "maximum": max_dimension,
            },
        )
    return labels, dimension


def _shrink_covariance(
    raw_covariance: object,
    shrinkage: float,
    target_kind: ShrinkageTarget,
    dimension: int,
) -> NDArray[np.float64]:
    try:
        raw_values = np.asarray(raw_covariance)
    except (TypeError, ValueError, OverflowError) as error:
        raise NumericalStabilityError(
            "base covariance could not be represented as an array",
            context={
                "operation": "covariance_shrinkage",
                "reason": _bounded_type_name(error),
            },
        ) from error
    expected_shape = (dimension, dimension)
    if raw_values.ndim != 2 or raw_values.shape != expected_shape:
        raise NumericalStabilityError(
            "base covariance has an invalid shape",
            context={
                "operation": "covariance_shrinkage",
                "shape": [int(size) for size in raw_values.shape[:8]],
                "expected": [dimension, dimension],
            },
        )
    if raw_values.dtype.kind not in {"i", "u", "f"}:
        raise DataValidationError(
            "base covariance must contain finite real numeric values",
            context={
                "field": "base_covariance",
                "dtype": str(raw_values.dtype)[:64],
                "reason": "not_real_numeric",
            },
        )
    if not np.isfinite(raw_values).all():
        raise DataValidationError(
            "base covariance must contain finite real numeric values",
            context={
                "field": "base_covariance",
                "dtype": str(raw_values.dtype)[:64],
                "reason": "not_finite",
            },
        )
    with np.errstate(over="ignore", invalid="ignore"):
        covariance = raw_values.astype(np.float64, copy=True)
    if not np.isfinite(covariance).all():
        raise DataValidationError(
            "base covariance must be safely representable as finite float64 values",
            context={
                "field": "base_covariance",
                "dtype": str(raw_values.dtype)[:64],
                "reason": "not_float64_representable",
            },
        )
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        symmetric = np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12)
    if not symmetric:
        raise DataValidationError(
            "base covariance must be symmetric within numerical tolerance",
            context={"field": "base_covariance", "reason": "not_symmetric"},
        )
    exact_symmetry = covariance == covariance.T
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        covariance = np.asarray(
            np.where(exact_symmetry, covariance, covariance / 2.0 + covariance.T / 2.0),
            dtype=np.float64,
        )
    diagonal = np.diag(covariance).copy()
    if np.any(diagonal < 0.0):
        raise DataValidationError(
            "base covariance diagonal must be nonnegative",
            context={"field": "base_covariance", "reason": "negative_diagonal"},
        )
    if shrinkage == 0.0:
        return covariance

    if target_kind is ShrinkageTarget.DIAGONAL:
        if shrinkage == 1.0:
            target = np.zeros_like(covariance)
            np.fill_diagonal(target, diagonal)
            return target
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                shrunk = covariance.copy()
                shrunk *= 1.0 - shrinkage
        except FloatingPointError as error:
            raise NumericalStabilityError(
                "convex covariance shrinkage was not representable",
                context={
                    "operation": "covariance_shrinkage",
                    "reason": _bounded_type_name(error),
                },
            ) from error
        np.fill_diagonal(shrunk, diagonal)
        return shrunk

    target = np.zeros_like(covariance)
    mean_variance = _stable_nonnegative_mean(diagonal)
    np.fill_diagonal(target, mean_variance)
    if shrinkage == 1.0:
        return target

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            shrunk = (1.0 - shrinkage) * covariance + shrinkage * target
    except FloatingPointError as error:
        raise NumericalStabilityError(
            "convex covariance shrinkage was not representable",
            context={
                "operation": "covariance_shrinkage",
                "reason": _bounded_type_name(error),
            },
        ) from error
    if not np.isfinite(shrunk).all():
        raise NumericalStabilityError(
            "convex covariance shrinkage returned non-finite values",
            context={
                "operation": "covariance_shrinkage",
                "reason": "non_finite_result",
            },
        )
    return shrunk


def _stable_nonnegative_mean(values: NDArray[np.float64]) -> float:
    if values.size == 0:
        raise DataValidationError(
            "scaled-identity shrinkage requires a non-empty covariance",
            context={"field": "base_covariance", "reason": "empty_asset_universe"},
        )
    anchor = float(np.max(values))
    if anchor == 0.0:
        return 0.0
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        scaled_values = values / anchor
    scaled_total = math.fsum(float(value) for value in scaled_values)
    scaled_mean = min(1.0, scaled_total / int(values.size))
    mean = anchor * scaled_mean
    if not math.isfinite(mean):
        raise NumericalStabilityError(
            "scaled-identity mean variance was not representable",
            context={
                "operation": "covariance_shrinkage",
                "reason": "mean_variance_not_finite",
            },
        )
    return mean
