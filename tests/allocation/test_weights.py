from __future__ import annotations

import json

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import qamr.allocation as allocation
from qamr.allocation.weights import equal_weights, inverse_volatility_weights
from qamr.contracts import LabeledMatrix, LabeledVector, PortfolioConstraints
from qamr.errors import (
    DataValidationError,
    InfeasiblePortfolioError,
    NumericalStabilityError,
)
from qamr.risk import CovarianceEstimate
from tests.allocation.helpers import estimate


def test_equal_weights_are_fully_invested() -> None:
    result = equal_weights(estimate())
    np.testing.assert_allclose(result.values, [0.5, 0.5])
    assert result.labels == ("a", "b")
    assert result.axis_name == "instrument"


def test_public_allocation_api_is_explicit_and_introspection_safe() -> None:
    expected = {
        "condensed_correlation_distance",
        "equal_weights",
        "herc_weights",
        "hrp_weights",
        "inverse_volatility_weights",
        "portfolio_volatility",
        "risk_contributions",
    }
    assert set(allocation.__all__) == expected
    assert len(allocation.__all__) == len(expected)
    assert expected.issubset(dir(allocation))
    assert dir(allocation) == sorted(set(dir(allocation)))


@pytest.mark.parametrize("allocator", [equal_weights, inverse_volatility_weights])
def test_baseline_weights_preserve_nonstandard_risk_axis(allocator: object) -> None:
    labels = ("a", "b")
    risk = CovarianceEstimate(
        covariance=LabeledMatrix(
            np.diag([0.04, 0.09]),
            labels,
            labels,
            "asset",
            "asset",
        ),
        correlation=LabeledMatrix(
            np.eye(2),
            labels,
            labels,
            "asset",
            "asset",
        ),
        volatility=LabeledVector(np.array([0.2, 0.3]), labels, "asset"),
        observation_count=24,
    )
    assert allocator(risk).axis_name == "asset"  # type: ignore[operator]


def test_inverse_volatility_weights_match_hand_calculation() -> None:
    result = inverse_volatility_weights(estimate())
    np.testing.assert_allclose(result.values, [0.6, 0.4])
    assert np.isclose(result.values.sum(), 1.0)


@pytest.mark.parametrize("allocator", [equal_weights, inverse_volatility_weights])
def test_one_asset_baselines_return_unit_weight(allocator: object) -> None:
    one_asset = estimate(labels=("solo",), covariance=np.array([[0.09]]))
    result = allocator(one_asset)  # type: ignore[operator]
    np.testing.assert_array_equal(result.values, [1.0])
    assert result.labels == ("solo",)


def test_inverse_volatility_is_stable_across_extreme_supported_scales() -> None:
    extreme = estimate(covariance=np.diag([1e-24, 1e308]))
    result = inverse_volatility_weights(extreme)
    assert np.isfinite(result.values).all()
    assert result.values.sum() == pytest.approx(1.0)
    assert result.values[0] == pytest.approx(1.0)
    assert result.values[1] > 0.0


def test_inverse_volatility_preserves_representable_tiny_positive_weight() -> None:
    risk = estimate()
    object.__setattr__(
        risk,
        "volatility",
        LabeledVector(np.array([1e-154, 1e154]), ("a", "b"), "instrument"),
    )
    result = inverse_volatility_weights(risk)
    assert result.values[1] > 0.0
    assert result.values[1] == pytest.approx(1e-308, rel=1e-12, abs=0.0)


def test_inverse_volatility_rejects_unrepresentable_positive_weight() -> None:
    risk = estimate()
    object.__setattr__(
        risk,
        "volatility",
        LabeledVector(
            np.array([np.nextafter(0.0, 1.0), 1e154]),
            ("a", "b"),
            "instrument",
        ),
    )
    with pytest.raises(NumericalStabilityError, match="representable") as captured:
        inverse_volatility_weights(risk)
    assert captured.value.context == {
        "field": "weights",
        "reason": "positive_weight_not_float64_representable",
    }


@pytest.mark.parametrize("allocator", [equal_weights, inverse_volatility_weights])
def test_baseline_allocator_rejects_incompatible_net_exposure(allocator: object) -> None:
    with pytest.raises(InfeasiblePortfolioError, match="net exposure") as captured:
        allocator(  # type: ignore[operator]
            estimate(),
            PortfolioConstraints(
                long_only=False,
                min_weight=-1.0,
                max_weight=1.0,
                gross_leverage=1.0,
                net_exposure=0.0,
            ),
        )
    assert captured.value.context["constraint"] == "net_exposure"


@pytest.mark.parametrize(
    ("constraints", "message", "constraint"),
    [
        (
            PortfolioConstraints(min_weight=0.500000001),
            "minimum weight",
            "min_weight",
        ),
        (
            PortfolioConstraints(max_weight=0.499999999),
            "maximum weight",
            "max_weight",
        ),
        (
            PortfolioConstraints(gross_leverage=0.999999999, net_exposure=None),
            "gross leverage",
            "gross_leverage",
        ),
        (
            PortfolioConstraints(
                long_only=False,
                min_weight=-1.0,
                gross_leverage=1.0,
                net_exposure=-1.0,
            ),
            "net exposure",
            "net_exposure",
        ),
    ],
)
def test_all_constraint_failures_are_structured_and_bounded(
    constraints: PortfolioConstraints,
    message: str,
    constraint: str,
) -> None:
    with pytest.raises(InfeasiblePortfolioError, match=message) as captured:
        equal_weights(estimate(), constraints)
    assert captured.value.context == {"constraint": constraint, "reason": "violated"}
    assert len(json.dumps(captured.value.as_dict())) < 512


@pytest.mark.parametrize(
    "constraints",
    [
        PortfolioConstraints(min_weight=0.5),
        PortfolioConstraints(max_weight=0.5),
        PortfolioConstraints(gross_leverage=1.0),
        PortfolioConstraints(net_exposure=1.0),
        PortfolioConstraints(min_weight=0.5 + 0.5e-10),
        PortfolioConstraints(max_weight=0.5 - 0.5e-10),
        PortfolioConstraints(gross_leverage=1.0 - 0.5e-10, net_exposure=None),
        PortfolioConstraints(gross_leverage=None, net_exposure=1.0 + 0.5e-10),
    ],
)
def test_constraints_accept_exact_and_documented_tolerance_boundaries(
    constraints: PortfolioConstraints,
) -> None:
    np.testing.assert_allclose(equal_weights(estimate(), constraints).values, [0.5, 0.5])


def test_long_only_constraint_accepts_zero_at_tolerance_boundary() -> None:
    constrained = PortfolioConstraints(
        long_only=True,
        min_weight=None,
        max_weight=None,
        gross_leverage=None,
        net_exposure=None,
    )
    assert np.all(equal_weights(estimate(), constrained).values >= 0.0)


@pytest.mark.parametrize("candidate", [object(), False, 1, "constraints"])
def test_constraints_argument_requires_exact_portfolio_constraints(candidate: object) -> None:
    with pytest.raises(DataValidationError, match="PortfolioConstraints") as captured:
        equal_weights(estimate(), candidate)  # type: ignore[arg-type]
    assert captured.value.context["field"] == "constraints"


@pytest.mark.parametrize("candidate", [None, object(), "estimate"])
def test_allocators_require_exact_covariance_estimate(candidate: object) -> None:
    with pytest.raises(DataValidationError, match="exact CovarianceEstimate"):
        equal_weights(candidate)  # type: ignore[arg-type]


def test_inverse_volatility_revalidates_public_numeric_boundary() -> None:
    labels = ("a", "b")
    valid = estimate()
    object.__setattr__(
        valid,
        "volatility",
        LabeledVector(np.array([True, False]), labels, "instrument"),
    )
    with pytest.raises(DataValidationError, match="finite real numeric"):
        inverse_volatility_weights(valid)


def test_inverse_volatility_rejects_nonpositive_values() -> None:
    valid = estimate()
    object.__setattr__(
        valid,
        "volatility",
        LabeledVector(np.array([0.0, 0.3]), ("a", "b"), "instrument"),
    )
    with pytest.raises(NumericalStabilityError, match="strictly positive") as captured:
        inverse_volatility_weights(valid)
    assert captured.value.context["reason"] == "not_positive"


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_inverse_volatility_nonfinite_values_are_numerical_stability_errors(
    bad_value: float,
) -> None:
    risk = estimate()
    object.__setattr__(
        risk,
        "volatility",
        LabeledVector(np.array([bad_value, 0.3]), ("a", "b"), "instrument"),
    )
    with pytest.raises(NumericalStabilityError, match="finite") as captured:
        inverse_volatility_weights(risk)
    assert captured.value.context == {
        "field": "volatility",
        "reason": "not_finite",
    }


@pytest.mark.parametrize("allocator", [equal_weights, inverse_volatility_weights])
def test_empty_defensive_estimate_is_rejected_before_allocation(
    allocator: object,
) -> None:
    empty = object.__new__(CovarianceEstimate)
    object.__setattr__(
        empty,
        "covariance",
        LabeledMatrix(
            np.empty((0, 0)),
            (),
            (),
            "instrument",
            "instrument",
        ),
    )
    with pytest.raises(
        NumericalStabilityError,
        match="at least one instrument",
    ) as captured:
        allocator(empty)  # type: ignore[operator]
    assert captured.value.context == {"reason": "empty_asset_universe"}


@pytest.mark.parametrize("allocator", [equal_weights, inverse_volatility_weights])
@pytest.mark.parametrize("bad_covariance", [object(), None, "covariance"])
def test_corrupted_exact_estimate_covariance_is_structured_for_allocators(
    allocator: object,
    bad_covariance: object,
) -> None:
    risk = object.__new__(CovarianceEstimate)
    object.__setattr__(risk, "covariance", bad_covariance)
    with pytest.raises(DataValidationError, match="exact LabeledMatrix") as captured:
        allocator(risk)  # type: ignore[operator]
    assert captured.value.context["field"] == "covariance"


def test_estimate_covariance_labels_are_used_without_aliasing() -> None:
    labels = ["a", "b"]
    covariance = np.diag([0.04, 0.09])
    risk = CovarianceEstimate(
        covariance=LabeledMatrix(
            covariance,
            labels,
            labels,
            "instrument",
            "instrument",
        ),
        correlation=LabeledMatrix(
            np.eye(2),
            labels,
            labels,
            "instrument",
            "instrument",
        ),
        volatility=LabeledVector(np.array([0.2, 0.3]), labels, "instrument"),
        observation_count=10,
    )
    result = inverse_volatility_weights(risk)
    labels[:] = ["changed", "changed-again"]
    covariance[:] = 0.0
    assert result.labels == ("a", "b")
    np.testing.assert_allclose(result.values, [0.6, 0.4])


@given(
    first=st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False),
    second=st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False),
)
def test_inverse_volatility_property_is_finite_positive_and_fully_invested(
    first: float,
    second: float,
) -> None:
    risk = estimate(covariance=np.diag([first * first, second * second]))
    result = inverse_volatility_weights(risk)
    assert np.isfinite(result.values).all()
    assert np.all(result.values > 0.0)
    assert result.values.sum() == pytest.approx(1.0)
