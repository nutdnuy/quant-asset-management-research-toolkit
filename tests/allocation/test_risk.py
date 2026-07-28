from __future__ import annotations

import json

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from qamr.allocation.risk import portfolio_volatility, risk_contributions
from qamr.contracts import LabeledMatrix, LabeledVector
from qamr.errors import (
    DataValidationError,
    LabelAlignmentError,
    NumericalStabilityError,
)
from qamr.risk import CovarianceEstimate
from tests.allocation.helpers import estimate


def weights(values: object, *, labels: tuple[str, ...] = ("a", "b")) -> LabeledVector:
    return LabeledVector(values, labels, "instrument")


def test_risk_contributions_sum_to_portfolio_volatility() -> None:
    allocation = weights(np.array([0.6, 0.4]))
    contributions = risk_contributions(estimate(), allocation)
    assert np.isclose(
        contributions.values.sum(),
        portfolio_volatility(estimate(), allocation),
    )


def test_risk_contributions_preserve_instrument_labels_and_axis() -> None:
    result = risk_contributions(estimate(), weights(np.array([0.5, 0.5])))
    assert result.labels == ("a", "b")
    assert result.axis_name == "instrument"


def test_risk_contributions_preserve_a_matching_nonstandard_axis_name() -> None:
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
    allocation = LabeledVector(np.array([0.5, 0.5]), labels, "asset")
    assert risk_contributions(risk, allocation).axis_name == "asset"


def test_risk_contributions_match_hand_calculation() -> None:
    result = risk_contributions(estimate(), weights(np.array([0.6, 0.4])))
    np.testing.assert_allclose(result.values, [0.0848528137, 0.0848528137])


def test_one_asset_portfolio_has_all_volatility_as_its_contribution() -> None:
    one_asset = estimate(labels=("solo",), covariance=np.array([[0.09]]))
    allocation = LabeledVector(np.array([1.0]), ("solo",), "instrument")
    assert portfolio_volatility(one_asset, allocation) == pytest.approx(0.3)
    np.testing.assert_allclose(risk_contributions(one_asset, allocation).values, [0.3])


def test_long_short_contributions_may_be_negative_and_still_reconcile() -> None:
    correlated = estimate(covariance=np.array([[0.04, 0.054], [0.054, 0.09]]))
    allocation = weights(np.array([-1.0, 2.0]))
    result = risk_contributions(correlated, allocation)
    assert result.values[0] < 0.0
    assert result.values[1] > 0.0
    assert np.sum(result.values) == pytest.approx(
        portfolio_volatility(correlated, allocation),
    )


def test_weight_labels_must_match_exactly_and_in_order() -> None:
    with pytest.raises(LabelAlignmentError, match="labels") as captured:
        portfolio_volatility(
            estimate(),
            weights(np.array([0.5, 0.5]), labels=("b", "a")),
        )
    assert captured.value.context == {"reason": "labels"}


def test_weight_axis_must_match_risk_axis() -> None:
    allocation = LabeledVector(np.array([0.5, 0.5]), ("a", "b"), "asset")
    with pytest.raises(LabelAlignmentError, match="axis") as captured:
        risk_contributions(estimate(), allocation)
    assert captured.value.context == {"reason": "axis_name"}


@pytest.mark.parametrize(
    "allocation",
    [
        np.array([True, False]),
        np.array([0.5 + 0.0j, 0.5 + 0.0j]),
        np.array(["0.5", "0.5"]),
        np.array([10**400, 1], dtype=object),
    ],
)
def test_weights_require_finite_safely_float64_representable_real_numeric_values(
    allocation: np.ndarray,
) -> None:
    # LabeledVector is deliberately generic, so the allocation boundary owns
    # numeric validation, including the canonical NaN case in the plan.
    with pytest.raises(DataValidationError, match="finite real numeric") as captured:
        risk_contributions(estimate(), weights(allocation))
    assert len(json.dumps(captured.value.as_dict())) < 512
    assert set(captured.value.context) <= {"dtype", "field", "reason"}


@pytest.mark.parametrize(
    "allocation",
    [
        np.array([np.nan, 0.5]),
        np.array([np.inf, 0.5]),
        np.array([-np.inf, 0.5]),
    ],
)
def test_nonfinite_numeric_weights_are_a_numerical_stability_error(
    allocation: np.ndarray,
) -> None:
    with pytest.raises(NumericalStabilityError, match="finite") as captured:
        risk_contributions(estimate(), weights(allocation))
    assert captured.value.context == {"field": "weights", "reason": "not_finite"}


def test_nonfinite_numeric_covariance_is_a_numerical_stability_error() -> None:
    risk = estimate()
    object.__setattr__(
        risk,
        "covariance",
        LabeledMatrix(
            np.array([[np.nan, 0.0], [0.0, 0.09]]),
            ("a", "b"),
            ("a", "b"),
            "instrument",
            "instrument",
        ),
    )
    with pytest.raises(NumericalStabilityError, match="finite") as captured:
        portfolio_volatility(risk, weights(np.array([0.5, 0.5])))
    assert captured.value.context == {
        "field": "covariance",
        "reason": "not_finite",
    }


def test_integer_value_that_is_not_exact_in_float64_is_rejected() -> None:
    allocation = np.array([2**63 + 1, 1], dtype=np.uint64)
    with pytest.raises(DataValidationError, match="exactly representable") as captured:
        portfolio_volatility(estimate(), weights(allocation))
    assert captured.value.context["reason"] == "not_exactly_representable"


@pytest.mark.parametrize("candidate", [None, object(), "weights"])
def test_weight_argument_requires_exact_labeled_vector(candidate: object) -> None:
    with pytest.raises(DataValidationError, match="exact LabeledVector"):
        portfolio_volatility(estimate(), candidate)  # type: ignore[arg-type]


@pytest.mark.parametrize("candidate", [None, object(), "estimate"])
def test_estimate_argument_requires_exact_covariance_estimate(candidate: object) -> None:
    with pytest.raises(DataValidationError, match="exact CovarianceEstimate"):
        portfolio_volatility(candidate, weights(np.array([0.5, 0.5])))  # type: ignore[arg-type]


def test_zero_portfolio_variance_is_rejected() -> None:
    singular = estimate(covariance=np.array([[1.0, -1.0], [-1.0, 1.0]]))
    with pytest.raises(NumericalStabilityError, match="strictly positive") as captured:
        portfolio_volatility(singular, weights(np.array([0.5, 0.5])))
    assert captured.value.context["reason"] == "not_positive"


def test_negative_portfolio_variance_is_rejected() -> None:
    covariance = np.array(
        [
            [1.0, -0.9, -0.9],
            [-0.9, 1.0, -0.9],
            [-0.9, -0.9, 1.0],
        ]
    )
    indefinite = estimate(labels=("a", "b", "c"), covariance=covariance)
    allocation = LabeledVector(np.ones(3) / 3.0, ("a", "b", "c"), "instrument")
    with pytest.raises(NumericalStabilityError, match="strictly positive"):
        risk_contributions(indefinite, allocation)


def test_extreme_scale_volatility_avoids_intermediate_variance_overflow() -> None:
    huge = estimate(covariance=np.array([[1e308]]), labels=("a",))
    allocation = LabeledVector(np.array([1e-154]), ("a",), "instrument")
    assert portfolio_volatility(huge, allocation) == pytest.approx(1.0)
    np.testing.assert_allclose(risk_contributions(huge, allocation).values, [1.0])


def test_extreme_covariance_dynamic_range_preserves_selected_small_variance() -> None:
    extreme = estimate(covariance=np.diag([1e-24, 1e308]))
    allocation = weights(np.array([1.0, 0.0]))
    assert portfolio_volatility(extreme, allocation) == pytest.approx(1e-12)
    np.testing.assert_allclose(
        risk_contributions(extreme, allocation).values,
        [1e-12, 0.0],
    )


@pytest.mark.parametrize("middle_weight", [1.5e-62, 2e-62, 2.5e-62])
def test_subnormal_scaled_quadratic_uses_log_resolution_fallback(
    middle_weight: float,
) -> None:
    risk = estimate(
        labels=("a", "b", "c"),
        covariance=np.diag([1e200, 1.0, 1e-300]),
    )
    allocation = LabeledVector(
        np.array([1e-200, middle_weight, 1.0]),
        ("a", "b", "c"),
        "instrument",
    )
    assert portfolio_volatility(risk, allocation) == pytest.approx(
        middle_weight,
        rel=1e-12,
        abs=0.0,
    )
    assert np.sum(risk_contributions(risk, allocation).values) == pytest.approx(
        middle_weight,
        rel=1e-12,
        abs=0.0,
    )


@given(
    exponent=st.integers(min_value=-69, max_value=-55),
    coefficient=st.integers(min_value=1, max_value=9),
)
def test_low_resolution_scaled_quadratic_matches_dominant_exact_term(
    exponent: int,
    coefficient: int,
) -> None:
    middle_weight = coefficient * 10.0**exponent
    risk = estimate(
        labels=("a", "b", "c"),
        covariance=np.diag([1e200, 1.0, 1e-300]),
    )
    allocation = LabeledVector(
        np.array([1e-200, middle_weight, 1.0]),
        ("a", "b", "c"),
        "instrument",
    )
    assert portfolio_volatility(risk, allocation) == pytest.approx(
        middle_weight,
        rel=1e-12,
        abs=0.0,
    )


@pytest.mark.parametrize("function", [portfolio_volatility, risk_contributions])
def test_empty_defensive_estimate_is_structured_for_risk_functions(
    function: object,
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
    empty_weights = LabeledVector(np.empty(0), (), "instrument")
    with pytest.raises(
        NumericalStabilityError,
        match="at least one instrument",
    ) as captured:
        function(empty, empty_weights)  # type: ignore[operator]
    assert captured.value.context == {"reason": "empty_asset_universe"}


@pytest.mark.parametrize(
    "bad_covariance",
    [object(), None, "covariance"],
)
def test_corrupted_exact_estimate_covariance_is_structured(
    bad_covariance: object,
) -> None:
    risk = object.__new__(CovarianceEstimate)
    object.__setattr__(risk, "covariance", bad_covariance)
    with pytest.raises(DataValidationError, match="exact LabeledMatrix") as captured:
        portfolio_volatility(risk, weights(np.array([0.5, 0.5])))
    assert captured.value.context["field"] == "covariance"


def test_output_does_not_alias_input_weights() -> None:
    source = np.array([0.6, 0.4])
    allocation = weights(source)
    result = risk_contributions(estimate(), allocation)
    source[:] = 0.0
    assert result.values.sum() > 0.0
    snapshot = result.values
    assert not snapshot.flags.writeable
    with pytest.raises(ValueError):
        snapshot[0] = 0.0


@given(
    first=st.integers(min_value=-1000, max_value=1000).map(lambda value: value / 100.0),
    second=st.integers(min_value=-1000, max_value=1000).map(lambda value: value / 100.0),
)
def test_diagonal_risk_contributions_reconcile_for_nonzero_weights(
    first: float,
    second: float,
) -> None:
    if first == 0.0 and second == 0.0:
        return
    allocation = weights(np.array([first, second]))
    result = risk_contributions(estimate(), allocation)
    assert np.sum(result.values) == pytest.approx(
        portfolio_volatility(estimate(), allocation),
        rel=1e-12,
        abs=1e-12,
    )
