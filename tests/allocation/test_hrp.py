from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.spatial.distance import squareform

from qamr.allocation.hierarchical import (
    condensed_correlation_distance,
    hrp_weights,
)
from qamr.contracts import LabeledMatrix, LabeledVector, PortfolioConstraints
from qamr.errors import (
    DataValidationError,
    InfeasiblePortfolioError,
    LabelAlignmentError,
    NumericalStabilityError,
)
from qamr.risk import CovarianceEstimate
from tests.allocation.helpers import estimate, four_asset_estimate


def _replace_correlation(
    risk: CovarianceEstimate,
    values: Any,
    *,
    row_labels: tuple[str, ...] | None = None,
    column_labels: tuple[str, ...] | None = None,
    row_name: str | None = None,
    column_name: str | None = None,
) -> CovarianceEstimate:
    labels = tuple(str(label) for label in risk.labels)
    object.__setattr__(
        risk,
        "correlation",
        LabeledMatrix(
            values,
            labels if row_labels is None else row_labels,
            labels if column_labels is None else column_labels,
            risk.covariance.row_name if row_name is None else row_name,
            risk.covariance.column_name if column_name is None else column_name,
        ),
    )
    return risk


def test_correlation_distance_is_condensed_not_square() -> None:
    result = condensed_correlation_distance(estimate())

    assert type(result) is np.ndarray
    assert result.dtype == np.dtype(np.float64)
    assert result.shape == (1,)
    assert np.isclose(result[0], np.sqrt(0.5))


def test_condensed_distance_matches_squareform_order() -> None:
    correlation = np.array(
        [
            [1.0, 0.5, 0.0],
            [0.5, 1.0, 0.25],
            [0.0, 0.25, 1.0],
        ]
    )
    risk = estimate(labels=("a", "b", "c"), covariance=correlation)

    condensed = condensed_correlation_distance(risk)

    expected_square = np.sqrt((1.0 - correlation) / 2.0)
    np.fill_diagonal(expected_square, 0.0)
    np.testing.assert_allclose(squareform(condensed), expected_square)


def test_two_asset_hrp_matches_inverse_cluster_variance() -> None:
    result = hrp_weights(estimate())

    assert type(result) is LabeledVector
    np.testing.assert_allclose(result.values, [0.6923076923, 0.3076923077])
    assert result.labels == ("a", "b")
    assert result.axis_name == "instrument"
    assert np.isclose(result.values.sum(), 1.0)


def test_one_asset_distance_and_hrp_are_well_defined() -> None:
    risk = estimate(labels=("solo",), covariance=np.array([[0.09]]))

    distance = condensed_correlation_distance(risk)
    result = hrp_weights(risk)

    assert distance.shape == (0,)
    np.testing.assert_array_equal(result.values, [1.0])
    assert result.labels == ("solo",)
    assert result.axis_name == "instrument"


def test_hrp_preserves_nonstandard_axis_and_exact_label_order() -> None:
    labels = ("asset-a", "asset-b")
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

    result = hrp_weights(risk)

    assert result.labels == labels
    assert result.axis_name == "asset"


@pytest.mark.parametrize(
    "function",
    [condensed_correlation_distance, hrp_weights],
)
@pytest.mark.parametrize("candidate", [None, object(), "estimate"])
def test_hrp_boundaries_require_exact_covariance_estimate(
    function: Any,
    candidate: object,
) -> None:
    with pytest.raises(DataValidationError, match="exact CovarianceEstimate"):
        function(candidate)


def test_correlation_requires_exact_labeled_matrix() -> None:
    risk = estimate()
    object.__setattr__(risk, "correlation", object())

    with pytest.raises(DataValidationError, match="exact LabeledMatrix") as captured:
        condensed_correlation_distance(risk)

    assert captured.value.context["field"] == "correlation"


@pytest.mark.parametrize(
    ("replacement", "reason"),
    [
        (
            {
                "row_labels": ("b", "a"),
                "column_labels": ("b", "a"),
            },
            "labels",
        ),
        (
            {
                "row_name": "asset",
                "column_name": "asset",
            },
            "axis_names",
        ),
    ],
)
def test_correlation_labels_and_axes_must_match_covariance(
    replacement: dict[str, Any],
    reason: str,
) -> None:
    risk = _replace_correlation(estimate(), np.eye(2), **replacement)

    with pytest.raises(LabelAlignmentError) as captured:
        condensed_correlation_distance(risk)

    assert captured.value.context == {"reason": reason}


@pytest.mark.parametrize(
    "values",
    [
        np.array([[True, False], [False, True]]),
        np.array([["1", "0"], ["0", "1"]]),
    ],
)
def test_correlation_requires_real_numeric_values(values: np.ndarray) -> None:
    risk = _replace_correlation(estimate(), values)

    with pytest.raises(DataValidationError, match="finite real numeric") as captured:
        condensed_correlation_distance(risk)

    assert captured.value.context["field"] == "correlation"


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_correlation_is_structured(bad_value: float) -> None:
    values = np.eye(2)
    values[0, 1] = values[1, 0] = bad_value
    risk = _replace_correlation(estimate(), values)

    with pytest.raises(NumericalStabilityError, match="finite") as captured:
        condensed_correlation_distance(risk)

    assert captured.value.context == {
        "field": "correlation",
        "reason": "not_finite",
    }


@pytest.mark.parametrize("roundoff", [1.0 + 5e-13, -1.0 - 5e-13])
def test_correlation_roundoff_is_clipped_at_bounds(roundoff: float) -> None:
    values = np.array([[1.0, roundoff], [roundoff, 1.0]])
    result = condensed_correlation_distance(
        _replace_correlation(estimate(), values),
    )

    expected = 0.0 if roundoff > 0.0 else 1.0
    np.testing.assert_allclose(result, [expected], rtol=0.0, atol=0.0)


@pytest.mark.parametrize("outside", [1.0 + 1e-6, -1.0 - 1e-6])
def test_correlation_outside_roundoff_bounds_is_rejected(outside: float) -> None:
    values = np.array([[1.0, outside], [outside, 1.0]])

    with pytest.raises(DataValidationError, match="within") as captured:
        condensed_correlation_distance(
            _replace_correlation(estimate(), values),
        )

    assert captured.value.context == {
        "field": "correlation",
        "reason": "outside_bounds",
    }


def test_asymmetric_correlation_is_rejected_before_squareform() -> None:
    values = np.array([[1.0, 0.2], [0.1, 1.0]])

    with pytest.raises(DataValidationError, match="symmetric") as captured:
        condensed_correlation_distance(
            _replace_correlation(estimate(), values),
        )

    assert captured.value.context["reason"] == "not_symmetric"


def test_correlation_diagonal_must_be_unit() -> None:
    values = np.array([[0.9, 0.0], [0.0, 1.0]])

    with pytest.raises(DataValidationError, match="diagonal") as captured:
        condensed_correlation_distance(
            _replace_correlation(estimate(), values),
        )

    assert captured.value.context["reason"] == "diagonal_not_one"


class _LinkageName(str):
    pass


@pytest.mark.parametrize("candidate", [None, 1, np.str_("single"), _LinkageName("single")])
def test_linkage_method_requires_exact_string(candidate: object) -> None:
    with pytest.raises(DataValidationError, match="exact string") as captured:
        hrp_weights(estimate(), linkage_method=candidate)  # type: ignore[arg-type]

    assert captured.value.context["field"] == "linkage_method"
    assert captured.value.context["reason"] == "wrong_type"


@pytest.mark.parametrize("candidate", ["ward", "centroid", "", "SINGLE"])
def test_linkage_method_rejects_unsupported_values(candidate: str) -> None:
    with pytest.raises(DataValidationError, match="unsupported") as captured:
        hrp_weights(estimate(), linkage_method=candidate)

    assert captured.value.context == {
        "field": "linkage_method",
        "reason": "unsupported",
        "value": candidate,
    }


def test_hrp_respects_constraints() -> None:
    with pytest.raises(InfeasiblePortfolioError, match="maximum weight"):
        hrp_weights(
            estimate(),
            PortfolioConstraints(max_weight=0.6),
        )


@pytest.mark.parametrize("candidate", [object(), False, 1, "constraints"])
def test_hrp_constraints_require_exact_portfolio_constraints(candidate: object) -> None:
    with pytest.raises(DataValidationError, match="PortfolioConstraints"):
        hrp_weights(estimate(), candidate)  # type: ignore[arg-type]


def test_hrp_does_not_mutate_risk_estimate_arrays() -> None:
    risk = four_asset_estimate(np.arange(4))
    covariance = risk.covariance.values
    correlation = risk.correlation.values
    volatility = risk.volatility.values

    hrp_weights(risk, linkage_method="average")

    np.testing.assert_array_equal(risk.covariance.values, covariance)
    np.testing.assert_array_equal(risk.correlation.values, correlation)
    np.testing.assert_array_equal(risk.volatility.values, volatility)


def test_positive_semidefinite_singular_covariance_remains_supported() -> None:
    risk = estimate(covariance=np.ones((2, 2)))

    result = hrp_weights(risk)

    np.testing.assert_allclose(result.values, [0.5, 0.5])


def test_zero_variance_hrp_portfolio_is_rejected() -> None:
    singular = estimate(
        covariance=np.array([[1.0, -1.0], [-1.0, 1.0]]),
    )

    with pytest.raises(NumericalStabilityError, match="strictly positive") as captured:
        hrp_weights(singular)

    assert captured.value.context["reason"] == "not_positive"


@pytest.mark.parametrize(
    ("diagonal", "expected"),
    [
        (
            np.array(
                [
                    np.nextafter(0.0, 1.0),
                    2.0 * np.nextafter(0.0, 1.0),
                ]
            ),
            np.array([2.0 / 3.0, 1.0 / 3.0]),
        ),
        (np.array([1e-300, 4e-300]), np.array([0.8, 0.2])),
        (np.array([1e300, 4e300]), np.array([0.8, 0.2])),
    ],
)
def test_hrp_handles_supported_extreme_covariance_scales(
    diagonal: np.ndarray,
    expected: np.ndarray,
) -> None:
    risk = estimate(covariance=np.diag(diagonal))

    result = hrp_weights(risk)

    np.testing.assert_allclose(result.values, expected, rtol=5e-13, atol=0.0)


def test_unrepresentable_positive_weight_fails_explicitly() -> None:
    risk = estimate(
        covariance=np.diag([np.nextafter(0.0, 1.0), 1e308]),
    )

    with pytest.raises(NumericalStabilityError, match="representable") as captured:
        hrp_weights(risk)

    assert captured.value.context == {
        "field": "weights",
        "reason": "positive_weight_not_float64_representable",
    }


@given(st.integers(min_value=2, max_value=8))
@settings(max_examples=20, deadline=None)
def test_hrp_is_finite_and_fully_invested(size: int) -> None:
    rng = np.random.default_rng(size)
    samples = rng.normal(size=(size + 10, size))
    covariance = np.cov(samples, rowvar=False)
    labels = tuple(f"a{index}" for index in range(size))

    result = hrp_weights(
        estimate(labels=labels, covariance=covariance),
    )

    assert np.isfinite(result.values).all()
    assert (result.values > 0.0).all()
    assert np.isclose(result.values.sum(), 1.0)


def test_hrp_is_permutation_equivariant_without_distance_ties() -> None:
    base = hrp_weights(
        four_asset_estimate(np.arange(4)),
        linkage_method="average",
    )
    permutation = np.array([2, 0, 3, 1])
    permuted = hrp_weights(
        four_asset_estimate(permutation),
        linkage_method="average",
    )
    restored = dict(zip(permuted.labels, permuted.values, strict=True))

    np.testing.assert_allclose(
        base.values,
        [restored[label] for label in base.labels],
    )


def _restored_permuted_weights(
    covariance: np.ndarray,
    labels: tuple[str, ...],
    permutation: np.ndarray,
    linkage_method: str,
) -> tuple[np.ndarray, np.ndarray]:
    base = hrp_weights(
        estimate(labels=labels, covariance=covariance),
        linkage_method=linkage_method,
    )
    permuted_labels = tuple(labels[index] for index in permutation)
    permuted = hrp_weights(
        estimate(
            labels=permuted_labels,
            covariance=covariance[np.ix_(permutation, permutation)],
        ),
        linkage_method=linkage_method,
    )
    restored = dict(zip(permuted.labels, permuted.values, strict=True))
    return base.values, np.array([restored[label] for label in labels])


def test_five_asset_unique_distance_permutation_regression() -> None:
    rng = np.random.default_rng(5001)
    covariance = np.cov(rng.normal(size=(15, 5)), rowvar=False)
    labels = tuple(f"a{index}" for index in range(5))
    permutation = np.array([0, 1, 3, 4, 2])
    distance = condensed_correlation_distance(
        estimate(labels=labels, covariance=covariance),
    )
    assert np.unique(distance).size == distance.size

    base, restored = _restored_permuted_weights(
        covariance,
        labels,
        permutation,
        "average",
    )

    np.testing.assert_allclose(base, restored, rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize("linkage_method", ["single", "complete", "average"])
@pytest.mark.parametrize("size", [3, 5, 7])
def test_odd_universes_are_equivariant_across_multiple_permutations(
    size: int,
    linkage_method: str,
) -> None:
    rng = np.random.default_rng(7000 + size)
    covariance = np.cov(rng.normal(size=(size + 20, size)), rowvar=False)
    labels = tuple(f"a{index}" for index in range(size))
    risk = estimate(labels=labels, covariance=covariance)
    distance = condensed_correlation_distance(risk)
    assert np.unique(distance).size == distance.size
    permutations = (
        np.arange(size)[::-1],
        np.roll(np.arange(size), 2),
        rng.permutation(size),
    )

    for permutation in permutations:
        base, restored = _restored_permuted_weights(
            covariance,
            labels,
            permutation,
            linkage_method,
        )
        np.testing.assert_allclose(
            base,
            restored,
            rtol=1e-12,
            atol=1e-15,
        )


def test_distance_ties_are_deterministic_for_fixed_input_order() -> None:
    tied = estimate(labels=("a", "b", "c"), covariance=np.eye(3))

    first = hrp_weights(tied, linkage_method="single")
    second = hrp_weights(tied, linkage_method="single")

    np.testing.assert_array_equal(first.values, second.values)
    assert "ties" in (hrp_weights.__doc__ or "")


def test_public_allocation_exports_are_lazy_and_introspection_safe() -> None:
    code = textwrap.dedent(
        """
        import sys
        import qamr.allocation as allocation

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
        assert expected.issubset(dir(allocation))
        assert dir(allocation) == sorted(set(dir(allocation)))
        assert "qamr.allocation.hierarchical" not in sys.modules
        assert callable(allocation.hrp_weights)
        assert "qamr.allocation.hierarchical" in sys.modules
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_structured_failures_are_bounded() -> None:
    values = np.array([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(DataValidationError) as captured:
        condensed_correlation_distance(
            _replace_correlation(estimate(), values),
        )

    assert len(json.dumps(captured.value.as_dict())) < 512
