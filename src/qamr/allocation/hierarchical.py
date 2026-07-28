"""Hierarchical risk-parity allocation over condensed correlation distance."""

from __future__ import annotations

import math
from collections.abc import Hashable

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from qamr.allocation.risk import (
    _bounded_type_name,
    _finite_real_values,
    _require_estimate_structure,
    _scaled_portfolio_components,
)
from qamr.allocation.weights import _constrained_weights, _resolved_constraints
from qamr.contracts.arrays import LabeledMatrix, LabeledVector
from qamr.contracts.config import PortfolioConstraints
from qamr.errors import (
    DataValidationError,
    LabelAlignmentError,
    NumericalStabilityError,
)
from qamr.risk.estimates import CovarianceEstimate

_CORRELATION_TOLERANCE = 1e-12
_SUPPORTED_LINKAGE_METHODS = frozenset({"single", "complete", "average"})


def _validated_risk_inputs(
    value: object,
) -> tuple[
    CovarianceEstimate,
    NDArray[np.float64],
    NDArray[np.float64],
    tuple[Hashable, ...],
]:
    estimate, covariance, labels = _require_estimate_structure(value)
    correlation = getattr(estimate, "correlation", None)
    if type(correlation) is not LabeledMatrix:
        raise DataValidationError(
            "estimate correlation must be an exact LabeledMatrix",
            context={
                "field": "correlation",
                "dtype": _bounded_type_name(correlation),
            },
        )
    if (
        correlation.row_labels != labels
        or correlation.column_labels != labels
        or correlation.shape != (len(labels), len(labels))
    ):
        raise LabelAlignmentError(
            "correlation labels and shape must match covariance exactly",
            context={"reason": "labels"},
        )
    if (
        correlation.row_name != covariance.row_name
        or correlation.column_name != covariance.column_name
    ):
        raise LabelAlignmentError(
            "correlation axis names must match covariance exactly",
            context={"reason": "axis_names"},
        )
    covariance_values = _finite_real_values(
        covariance,
        field="covariance",
    )
    correlation_values = _finite_real_values(
        correlation,
        field="correlation",
    )
    return estimate, covariance_values, correlation_values, labels


def _validated_correlation(
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    if not np.allclose(
        values,
        values.T,
        rtol=0.0,
        atol=_CORRELATION_TOLERANCE,
    ):
        raise DataValidationError(
            "correlation must be symmetric within numerical tolerance",
            context={"field": "correlation", "reason": "not_symmetric"},
        )
    if not np.allclose(
        np.diag(values),
        1.0,
        rtol=0.0,
        atol=_CORRELATION_TOLERANCE,
    ):
        raise DataValidationError(
            "correlation diagonal must equal one within numerical tolerance",
            context={"field": "correlation", "reason": "diagonal_not_one"},
        )
    if np.any(values < -1.0 - _CORRELATION_TOLERANCE) or np.any(
        values > 1.0 + _CORRELATION_TOLERANCE
    ):
        raise DataValidationError(
            "correlation values must be within [-1, 1]",
            context={"field": "correlation", "reason": "outside_bounds"},
        )
    symmetric = 0.5 * (values + values.T)
    return np.clip(symmetric, -1.0, 1.0)


def _condensed_distance(
    correlation: NDArray[np.float64],
) -> NDArray[np.float64]:
    validated = _validated_correlation(correlation)
    distance = np.sqrt(np.clip((1.0 - validated) / 2.0, 0.0, 1.0))
    np.fill_diagonal(distance, 0.0)
    try:
        condensed = squareform(distance, checks=False)
    except (TypeError, ValueError, FloatingPointError) as error:
        raise DataValidationError(
            "correlation distance could not be condensed",
            context={"field": "correlation", "reason": "invalid_distance"},
        ) from error
    return np.asarray(condensed, dtype=np.float64)


def condensed_correlation_distance(
    estimate: CovarianceEstimate,
) -> NDArray[np.float64]:
    """Return SciPy's condensed distance vector derived from correlation."""
    _, _, correlation, _ = _validated_risk_inputs(estimate)
    return _condensed_distance(correlation)


def _validated_linkage_method(value: object) -> str:
    if type(value) is not str:
        raise DataValidationError(
            "linkage_method must be an exact string",
            context={
                "field": "linkage_method",
                "dtype": _bounded_type_name(value),
                "reason": "wrong_type",
            },
        )
    if value not in _SUPPORTED_LINKAGE_METHODS:
        raise DataValidationError(
            "unsupported linkage method",
            context={
                "field": "linkage_method",
                "reason": "unsupported",
                "value": value[:64],
            },
        )
    return value


def _tree_structure(
    correlation: NDArray[np.float64],
    linkage_method: str,
) -> tuple[
    tuple[tuple[int, int], ...],
    dict[int, list[int]],
]:
    condensed = _condensed_distance(correlation)
    try:
        tree = linkage(condensed, method=linkage_method)
    except (TypeError, ValueError, FloatingPointError) as error:
        raise DataValidationError(
            "hierarchical clustering could not be constructed",
            context={"field": "correlation", "reason": "clustering_failed"},
        ) from error
    leaf_count = correlation.shape[0]
    if tree.shape != (leaf_count - 1, 4) or not np.isfinite(tree).all():
        raise NumericalStabilityError(
            "hierarchical clustering returned an invalid tree",
            context={"reason": "invalid_tree"},
        )
    node_members: dict[int, list[int]] = {leaf: [leaf] for leaf in range(leaf_count)}
    children: list[tuple[int, int]] = []
    for row_index, row in enumerate(tree):
        node = leaf_count + row_index
        raw_left = float(row[0])
        raw_right = float(row[1])
        left = int(raw_left)
        right = int(raw_right)
        if (
            not raw_left.is_integer()
            or not raw_right.is_integer()
            or left == right
            or left not in node_members
            or right not in node_members
        ):
            raise NumericalStabilityError(
                "hierarchical clustering returned an invalid tree",
                context={"reason": "invalid_tree"},
            )
        left_members = node_members[left]
        right_members = node_members[right]
        if set(left_members).intersection(right_members):
            raise NumericalStabilityError(
                "hierarchical clustering returned an invalid tree",
                context={"reason": "invalid_tree"},
            )
        merged = left_members + right_members
        if int(row[3]) != len(merged):
            raise NumericalStabilityError(
                "hierarchical clustering returned an invalid tree",
                context={"reason": "invalid_tree"},
            )
        children.append((left, right))
        node_members[node] = merged
    root = leaf_count + len(children) - 1
    if sorted(node_members[root]) != list(range(leaf_count)):
        raise NumericalStabilityError(
            "hierarchical clustering returned an invalid tree",
            context={"reason": "invalid_tree"},
        )
    return tuple(children), node_members


def _positive_normalized_from_logs(
    log_values: NDArray[np.float64],
) -> NDArray[np.float64]:
    maximum = float(np.max(log_values))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        relative = np.exp(log_values - maximum)
    if not np.isfinite(relative).all() or np.any(relative <= 0.0):
        raise NumericalStabilityError(
            "strictly positive weights must be representable as float64",
            context={
                "field": "weights",
                "reason": "positive_weight_not_float64_representable",
            },
        )
    normalizer = math.fsum(float(value) for value in relative)
    normalized = relative / normalizer
    if not np.isfinite(normalized).all() or np.any(normalized <= 0.0):
        raise NumericalStabilityError(
            "strictly positive weights must be representable as float64",
            context={
                "field": "weights",
                "reason": "positive_weight_not_float64_representable",
            },
        )
    return normalized


def _cluster_log_variance(
    covariance: NDArray[np.float64],
    members: list[int],
) -> float:
    submatrix = covariance[np.ix_(members, members)]
    diagonal = np.diag(submatrix)
    if np.any(diagonal <= 0.0):
        raise NumericalStabilityError(
            "cluster variances must be strictly positive",
            context={"reason": "not_positive"},
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse_variance_logs = -np.log(diagonal)
    cluster_weights = _positive_normalized_from_logs(
        inverse_variance_logs,
    )
    volatility, _ = _scaled_portfolio_components(
        cluster_weights,
        submatrix,
    )
    return 2.0 * math.log(volatility)


def _normalized_hierarchical_weights(
    covariance: NDArray[np.float64],
    children: tuple[tuple[int, int], ...],
    node_members: dict[int, list[int]],
    *,
    variance_exponent: float,
) -> NDArray[np.float64]:
    leaf_count = covariance.shape[0]
    log_weights = np.zeros(leaf_count, dtype=np.float64)
    pending_nodes = [leaf_count + len(children) - 1]
    while pending_nodes:
        node = pending_nodes.pop()
        if node < leaf_count:
            continue
        left, right = children[node - leaf_count]
        left_members = node_members[left]
        right_members = node_members[right]
        left_log_risk = variance_exponent * _cluster_log_variance(
            covariance,
            left_members,
        )
        right_log_risk = variance_exponent * _cluster_log_variance(
            covariance,
            right_members,
        )
        log_denominator = float(np.logaddexp(left_log_risk, right_log_risk))
        log_weights[left_members] += right_log_risk - log_denominator
        log_weights[right_members] += left_log_risk - log_denominator
        pending_nodes.extend((left, right))
    return _positive_normalized_from_logs(log_weights)


def hrp_weights(
    estimate: CovarianceEstimate,
    constraints: PortfolioConstraints | None = None,
    *,
    linkage_method: str = "single",
) -> LabeledVector:
    """Return HRP weights.

    Results are deterministic for a fixed input order. Distance ties may make
    otherwise equivalent input permutations choose different valid trees.
    """
    validated_estimate, covariance, correlation, labels = _validated_risk_inputs(estimate)
    resolved_constraints = _resolved_constraints(constraints)
    method = _validated_linkage_method(linkage_method)
    if len(labels) == 1:
        weights = np.ones(1, dtype=np.float64)
    else:
        children, node_members = _tree_structure(correlation, method)
        weights = _normalized_hierarchical_weights(
            covariance,
            children,
            node_members,
            variance_exponent=1.0,
        )
    _scaled_portfolio_components(weights, covariance)
    return _constrained_weights(
        weights,
        validated_estimate,
        resolved_constraints,
    )


def herc_weights(
    estimate: CovarianceEstimate,
    constraints: PortfolioConstraints | None = None,
    *,
    linkage_method: str = "single",
) -> LabeledVector:
    """Return HERC weights using cluster standard deviation as split risk.

    Results are deterministic for a fixed input order. Distance ties may make
    otherwise equivalent input permutations choose different valid trees.
    """
    validated_estimate, covariance, correlation, labels = _validated_risk_inputs(estimate)
    resolved_constraints = _resolved_constraints(constraints)
    method = _validated_linkage_method(linkage_method)
    if len(labels) == 1:
        weights = np.ones(1, dtype=np.float64)
    else:
        children, node_members = _tree_structure(correlation, method)
        weights = _normalized_hierarchical_weights(
            covariance,
            children,
            node_members,
            variance_exponent=0.5,
        )
    _scaled_portfolio_components(weights, covariance)
    return _constrained_weights(
        weights,
        validated_estimate,
        resolved_constraints,
    )
