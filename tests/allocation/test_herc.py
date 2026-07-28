from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from qamr.allocation.hierarchical import (
    condensed_correlation_distance,
    herc_weights,
)
from qamr.allocation.risk import risk_contributions
from qamr.contracts import LabeledVector, PortfolioConstraints
from qamr.errors import (
    DataValidationError,
    InfeasiblePortfolioError,
    NumericalStabilityError,
)
from tests.allocation.helpers import estimate, four_asset_estimate


def _restored_permuted_weights(
    covariance: np.ndarray,
    labels: tuple[str, ...],
    permutation: np.ndarray,
    linkage_method: str,
) -> tuple[np.ndarray, np.ndarray]:
    base = herc_weights(
        estimate(labels=labels, covariance=covariance),
        linkage_method=linkage_method,
    )
    permuted_labels = tuple(labels[index] for index in permutation)
    permuted = herc_weights(
        estimate(
            labels=permuted_labels,
            covariance=covariance[np.ix_(permutation, permutation)],
        ),
        linkage_method=linkage_method,
    )
    restored = dict(zip(permuted.labels, permuted.values, strict=True))
    return base.values, np.array([restored[label] for label in labels])


def test_two_asset_herc_equalises_component_risk() -> None:
    result = herc_weights(estimate())

    assert type(result) is LabeledVector
    np.testing.assert_allclose(result.values, [0.6, 0.4])
    contributions = risk_contributions(estimate(), result)
    np.testing.assert_allclose(
        contributions.values,
        np.repeat(contributions.values.mean(), 2),
    )
    assert result.labels == ("a", "b")
    assert result.axis_name == "instrument"


def test_one_asset_herc_is_fully_invested() -> None:
    result = herc_weights(
        estimate(labels=("solo",), covariance=np.array([[0.09]])),
    )

    np.testing.assert_array_equal(result.values, [1.0])
    assert result.labels == ("solo",)


@given(st.integers(min_value=2, max_value=8))
@settings(max_examples=20, deadline=None)
def test_herc_is_finite_and_fully_invested(size: int) -> None:
    rng = np.random.default_rng(9000 + size)
    covariance = np.cov(rng.normal(size=(size + 12, size)), rowvar=False)
    labels = tuple(f"a{index}" for index in range(size))

    result = herc_weights(
        estimate(labels=labels, covariance=covariance),
    )

    assert np.isfinite(result.values).all()
    assert (result.values > 0.0).all()
    assert np.isclose(result.values.sum(), 1.0)


def test_herc_is_permutation_equivariant_without_distance_ties() -> None:
    base = herc_weights(
        four_asset_estimate(np.arange(4)),
        linkage_method="average",
    )
    permutation = np.array([2, 0, 3, 1])
    permuted = herc_weights(
        four_asset_estimate(permutation),
        linkage_method="average",
    )
    restored = dict(zip(permuted.labels, permuted.values, strict=True))

    np.testing.assert_allclose(
        base.values,
        [restored[label] for label in base.labels],
    )


@pytest.mark.parametrize("linkage_method", ["single", "complete", "average"])
def test_five_asset_herc_is_equivariant_for_multiple_permutations(
    linkage_method: str,
) -> None:
    rng = np.random.default_rng(5001)
    covariance = np.cov(rng.normal(size=(15, 5)), rowvar=False)
    labels = tuple(f"a{index}" for index in range(5))
    distance = condensed_correlation_distance(
        estimate(labels=labels, covariance=covariance),
    )
    assert np.unique(distance).size == distance.size
    permutations = (
        np.array([0, 1, 3, 4, 2]),
        np.arange(5)[::-1],
        np.array([2, 4, 1, 0, 3]),
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


def test_herc_respects_constraints() -> None:
    with pytest.raises(InfeasiblePortfolioError, match="maximum weight"):
        herc_weights(
            estimate(),
            PortfolioConstraints(max_weight=0.55),
        )


@pytest.mark.parametrize("candidate", [object(), False, 1, "constraints"])
def test_herc_constraints_require_exact_portfolio_constraints(candidate: object) -> None:
    with pytest.raises(DataValidationError, match="PortfolioConstraints"):
        herc_weights(estimate(), candidate)  # type: ignore[arg-type]


@pytest.mark.parametrize("candidate", [None, 1, "ward"])
def test_herc_rejects_invalid_linkage_methods(candidate: object) -> None:
    with pytest.raises(DataValidationError, match="linkage"):
        herc_weights(estimate(), linkage_method=candidate)  # type: ignore[arg-type]


def test_positive_semidefinite_singular_covariance_remains_supported() -> None:
    result = herc_weights(estimate(covariance=np.ones((2, 2))))

    np.testing.assert_allclose(result.values, [0.5, 0.5])


def test_zero_variance_herc_portfolio_is_rejected() -> None:
    singular = estimate(
        covariance=np.array([[1.0, -1.0], [-1.0, 1.0]]),
    )

    with pytest.raises(NumericalStabilityError, match="strictly positive"):
        herc_weights(singular)


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
            np.array(
                [
                    np.sqrt(2.0) / (1.0 + np.sqrt(2.0)),
                    1.0 / (1.0 + np.sqrt(2.0)),
                ]
            ),
        ),
        (np.array([1e-300, 4e-300]), np.array([2.0 / 3.0, 1.0 / 3.0])),
        (np.array([1e300, 4e300]), np.array([2.0 / 3.0, 1.0 / 3.0])),
    ],
)
def test_herc_handles_supported_extreme_covariance_scales(
    diagonal: np.ndarray,
    expected: np.ndarray,
) -> None:
    result = herc_weights(estimate(covariance=np.diag(diagonal)))

    np.testing.assert_allclose(result.values, expected, rtol=5e-13, atol=0.0)


def test_distance_ties_are_deterministic_for_fixed_input_order() -> None:
    tied = estimate(labels=("a", "b", "c"), covariance=np.eye(3))

    first = herc_weights(tied)
    second = herc_weights(tied)

    np.testing.assert_array_equal(first.values, second.values)
    assert "ties" in (herc_weights.__doc__ or "")


def test_public_herc_export_is_lazy_and_introspection_safe() -> None:
    code = textwrap.dedent(
        """
        import sys
        import qamr.allocation as allocation

        assert "herc_weights" in allocation.__all__
        assert "herc_weights" in dir(allocation)
        assert "qamr.allocation.hierarchical" not in sys.modules
        assert callable(allocation.herc_weights)
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
