from __future__ import annotations

from enum import Enum
from typing import Any, cast

import numpy as np
import pytest

from qamr.contracts.arrays import LabeledMatrix, LabeledVector
from qamr.errors import LabelAlignmentError, NumericalStabilityError
from qamr.risk.matrices import (
    PSDPolicy,
    apply_psd_policy,
    correlation_to_covariance,
    covariance_to_correlation,
)


def square(
    values: Any,
    labels: tuple[str, ...] = ("a", "b"),
    *,
    row_name: str = "asset",
    column_name: str = "asset",
) -> LabeledMatrix:
    return LabeledMatrix(values, labels, labels, row_name, column_name)


def test_covariance_correlation_round_trip_preserves_metadata() -> None:
    covariance = square([[0.04, 0.03], [0.03, 0.09]])

    correlation, volatility = covariance_to_correlation(covariance)
    rebuilt = correlation_to_covariance(correlation, volatility)

    np.testing.assert_allclose(correlation.values, [[1.0, 0.5], [0.5, 1.0]])
    np.testing.assert_allclose(volatility.values, [0.2, 0.3])
    np.testing.assert_allclose(rebuilt.values, covariance.values)
    assert correlation.row_labels == covariance.row_labels
    assert correlation.column_labels == covariance.column_labels
    assert (correlation.row_name, correlation.column_name) == ("asset", "asset")
    assert volatility.labels == covariance.column_labels
    assert volatility.axis_name == "asset"
    assert (rebuilt.row_name, rebuilt.column_name) == ("asset", "asset")


def test_conversions_do_not_mutate_inputs() -> None:
    covariance = square([[0.04, -0.03], [-0.03, 0.09]])
    covariance_before = covariance.values
    correlation, volatility = covariance_to_correlation(covariance)
    correlation_before = correlation.values
    volatility_before = volatility.values

    correlation_to_covariance(correlation, volatility)

    np.testing.assert_array_equal(covariance.values, covariance_before)
    np.testing.assert_array_equal(correlation.values, correlation_before)
    np.testing.assert_array_equal(volatility.values, volatility_before)


def test_zero_variance_is_a_structured_numerical_error() -> None:
    with pytest.raises(
        NumericalStabilityError,
        match="strictly positive variance",
    ) as captured:
        covariance_to_correlation(square([[0.0, 0.0], [0.0, 1.0]]))

    assert captured.value.context == {"positions": (0,)}


def test_variance_at_tolerance_is_rejected_at_exact_position() -> None:
    covariance = square([[1e-12, 0.0], [0.0, 1.0]])

    with pytest.raises(NumericalStabilityError) as captured:
        covariance_to_correlation(covariance, tolerance=1e-12)

    assert captured.value.context == {"positions": (0,)}


def test_correlation_and_volatility_labels_must_align() -> None:
    with pytest.raises(LabelAlignmentError, match="volatility labels"):
        correlation_to_covariance(
            square([[1.0, 0.0], [0.0, 1.0]]),
            LabeledVector(np.array([1.0, 1.0]), ("b", "a"), "asset"),
        )


def test_correlation_and_volatility_axis_names_must_align() -> None:
    with pytest.raises(LabelAlignmentError, match="axis name"):
        correlation_to_covariance(
            square([[1.0, 0.0], [0.0, 1.0]]),
            LabeledVector(np.array([1.0, 1.0]), ("a", "b"), "instrument"),
        )


def test_raise_psd_policy_reports_negative_eigenvalue_with_bounded_context() -> None:
    indefinite = square([[1.0, 2.0], [2.0, 1.0]])

    with pytest.raises(
        NumericalStabilityError,
        match="not positive semidefinite",
    ) as captured:
        apply_psd_policy(indefinite, PSDPolicy.RAISE)

    assert set(captured.value.context) == {"minimum_eigenvalue", "tolerance"}
    assert float(captured.value.context["minimum_eigenvalue"]) == pytest.approx(-1.0)


def test_raise_psd_policy_accepts_negative_eigenvalue_within_tolerance() -> None:
    matrix = square([[1.0, 1.0 + 5e-11], [1.0 + 5e-11, 1.0]])

    accepted = apply_psd_policy(matrix, PSDPolicy.RAISE, tolerance=1e-10)

    np.testing.assert_array_equal(accepted.values, matrix.values)


def test_clip_psd_policy_returns_symmetric_psd_matrix_and_preserves_metadata() -> None:
    repaired = apply_psd_policy(square([[1.0, 2.0], [2.0, 1.0]]), PSDPolicy.CLIP)

    np.testing.assert_allclose(repaired.values, repaired.values.T, atol=1e-12)
    assert np.linalg.eigvalsh(repaired.values).min() >= 0.0
    assert repaired.row_labels == ("a", "b")
    assert repaired.column_labels == ("a", "b")
    assert (repaired.row_name, repaired.column_name) == ("asset", "asset")


def test_asymmetric_matrix_is_rejected_before_psd_handling() -> None:
    with pytest.raises(NumericalStabilityError, match="must be symmetric"):
        apply_psd_policy(square([[1.0, 0.2], [0.1, 1.0]]), PSDPolicy.CLIP)


def test_local_symmetry_check_is_not_relaxed_by_unrelated_large_value() -> None:
    matrix = square(
        [[1e24, 0.0, 0.0], [0.0, 1.0, 1e-4], [0.0, 0.0, 1.0]],
        ("a", "b", "c"),
    )

    with pytest.raises(NumericalStabilityError, match="must be symmetric"):
        apply_psd_policy(matrix, PSDPolicy.CLIP, tolerance=1e-10)


@pytest.mark.parametrize(
    "bad_values",
    [
        np.array([[True, False], [False, True]]),
        np.array([[1 + 0j, 0j], [0j, 1 + 0j]]),
        np.array([["1", "0"], ["0", "1"]]),
        np.array([[1, 0], [0, 1]], dtype=object),
        np.array(
            [["2026-01-01", "2026-01-01"], ["2026-01-01", "2026-01-01"]],
            dtype="datetime64[D]",
        ),
        np.array([[1, 0], [0, 1]], dtype="timedelta64[D]"),
        np.array([[1.0, np.nan], [np.nan, 1.0]]),
        np.array([[1.0, np.inf], [np.inf, 1.0]]),
    ],
)
def test_non_real_finite_numeric_matrix_dtypes_are_rejected(bad_values: np.ndarray) -> None:
    with pytest.raises(NumericalStabilityError, match="finite real numeric"):
        apply_psd_policy(square(bad_values), PSDPolicy.RAISE)


def test_signed_zero_and_int64_extrema_are_valid_numeric_inputs() -> None:
    signed_zero = square(np.array([[1.0, -0.0], [0.0, 1.0]]))
    integer = square(np.diag(np.array([np.iinfo(np.int64).max, 1], dtype=np.int64)))

    correlation, volatility = covariance_to_correlation(integer, tolerance=0)
    repaired = apply_psd_policy(signed_zero, PSDPolicy.CLIP, tolerance=0)

    np.testing.assert_array_equal(correlation.values, np.eye(2))
    assert np.isfinite(volatility.values).all()
    np.testing.assert_array_equal(repaired.values, np.eye(2))


def test_materially_invalid_implied_correlation_is_rejected() -> None:
    covariance = square([[1.0, 1.01], [1.01, 1.0]])

    with pytest.raises(NumericalStabilityError, match=r"\[-1, 1\]"):
        covariance_to_correlation(covariance, tolerance=1e-12)


def test_tolerance_sized_implied_correlation_excess_is_clipped() -> None:
    covariance = square([[1.0, 1.0 + 5e-11], [1.0 + 5e-11, 1.0]])

    correlation, _ = covariance_to_correlation(covariance, tolerance=1e-10)

    np.testing.assert_array_equal(correlation.values, np.ones((2, 2)))


def test_correlation_input_semantics_are_validated() -> None:
    volatility = LabeledVector([1.0, 2.0], ("a", "b"), "asset")

    with pytest.raises(NumericalStabilityError, match="diagonal"):
        correlation_to_covariance(square([[0.9, 0.0], [0.0, 1.0]]), volatility)
    with pytest.raises(NumericalStabilityError, match=r"\[-1, 1\]"):
        correlation_to_covariance(square([[1.0, 1.01], [1.01, 1.0]]), volatility)


@pytest.mark.parametrize(
    "bad_values",
    [
        [0.0, 1.0],
        [-1.0, 1.0],
        [np.nan, 1.0],
        [np.inf, 1.0],
        [True, False],
        [1 + 0j, 1 + 0j],
        ["1", "1"],
    ],
)
def test_volatility_must_be_finite_real_and_strictly_positive(
    bad_values: list[object],
) -> None:
    volatility = LabeledVector(bad_values, ("a", "b"), "asset")

    with pytest.raises(NumericalStabilityError, match="volatility"):
        correlation_to_covariance(square([[1.0, 0.0], [0.0, 1.0]]), volatility)


def test_correlation_scaling_overflow_is_structured() -> None:
    correlation = square([[1.0, 0.0], [0.0, 1.0]])
    volatility = LabeledVector([np.finfo(np.float64).max, 1.0], ("a", "b"), "asset")

    with pytest.raises(NumericalStabilityError, match="safely"):
        correlation_to_covariance(correlation, volatility)


@pytest.mark.parametrize("bad_tolerance", [True, np.float64(1e-10), -1.0, np.inf, np.nan])
def test_tolerance_requires_nonnegative_finite_exact_builtin_number(
    bad_tolerance: object,
) -> None:
    with pytest.raises(NumericalStabilityError, match="tolerance"):
        apply_psd_policy(
            square([[1.0, 0.0], [0.0, 1.0]]),
            PSDPolicy.RAISE,
            tolerance=cast(Any, bad_tolerance),
        )


def test_huge_builtin_integer_tolerance_is_rejected_structurally() -> None:
    with pytest.raises(NumericalStabilityError, match="tolerance"):
        apply_psd_policy(
            square([[1.0, 0.0], [0.0, 1.0]]),
            PSDPolicy.RAISE,
            tolerance=10**1000,
        )


class _OtherPolicy(str, Enum):
    RAISE = "raise"


@pytest.mark.parametrize("policy", ["raise", _OtherPolicy.RAISE])
def test_psd_policy_requires_exact_enum_instance(policy: object) -> None:
    with pytest.raises(NumericalStabilityError, match="PSDPolicy"):
        apply_psd_policy(
            square([[1.0, 0.0], [0.0, 1.0]]),
            cast(Any, policy),
        )


class _Hostile:
    def __getattribute__(self, name: str) -> Any:
        raise AssertionError(f"property accessed: {name}")


def test_exact_public_types_are_checked_before_property_access() -> None:
    hostile = _Hostile()

    with pytest.raises(NumericalStabilityError, match="LabeledMatrix"):
        covariance_to_correlation(cast(Any, hostile))
    with pytest.raises(NumericalStabilityError, match="LabeledMatrix"):
        correlation_to_covariance(
            cast(Any, hostile),
            LabeledVector([1.0], ("a",), "asset"),
        )
    with pytest.raises(NumericalStabilityError, match="LabeledVector"):
        correlation_to_covariance(
            square([[1.0]], ("a",)),
            cast(Any, hostile),
        )


def test_empty_universe_is_rejected() -> None:
    empty = LabeledMatrix(np.empty((0, 0)), (), (), "asset", "asset")

    with pytest.raises(LabelAlignmentError, match="non-empty"):
        covariance_to_correlation(empty)


def test_axis_names_and_labels_must_be_square_aligned() -> None:
    mismatched_labels = LabeledMatrix(
        np.eye(2),
        ("a", "b"),
        ("b", "a"),
        "asset",
        "asset",
    )
    mismatched_names = LabeledMatrix(
        np.eye(2),
        ("a", "b"),
        ("a", "b"),
        "row asset",
        "column asset",
    )

    with pytest.raises(LabelAlignmentError, match="labels"):
        covariance_to_correlation(mismatched_labels)
    with pytest.raises(LabelAlignmentError, match="axis names"):
        covariance_to_correlation(mismatched_names)


@pytest.mark.parametrize(
    "corrupt_values",
    [
        np.ones((2, 3)),
        np.ones((2, 2, 1)),
        np.ones(2),
    ],
)
def test_corrupt_matrix_shape_is_rejected_before_numpy_arithmetic(
    corrupt_values: np.ndarray,
) -> None:
    matrix = square(np.eye(2))
    object.__setattr__(matrix, "_values", corrupt_values)

    with pytest.raises(NumericalStabilityError, match="two-dimensional square"):
        apply_psd_policy(matrix, PSDPolicy.CLIP)


def test_corrupt_matrix_shape_must_match_label_counts() -> None:
    matrix = square(np.eye(2))
    object.__setattr__(matrix, "_values", np.eye(3))

    with pytest.raises(LabelAlignmentError, match="label counts"):
        covariance_to_correlation(matrix)


def test_corrupt_vector_shape_is_rejected_before_scaling() -> None:
    volatility = LabeledVector([1.0, 1.0], ("a", "b"), "asset")
    object.__setattr__(volatility, "_values", np.ones((2, 1)))

    with pytest.raises(NumericalStabilityError, match="one-dimensional"):
        correlation_to_covariance(square(np.eye(2)), volatility)


def test_corrupt_vector_length_must_match_label_count() -> None:
    volatility = LabeledVector([1.0, 1.0], ("a", "b"), "asset")
    object.__setattr__(volatility, "_values", np.ones(3))

    with pytest.raises(LabelAlignmentError, match="label count"):
        correlation_to_covariance(square(np.eye(2)), volatility)


def test_volatility_outer_product_underflow_is_structured() -> None:
    volatility = LabeledVector([1e-200, 1.0], ("a", "b"), "asset")

    with pytest.raises(NumericalStabilityError, match="underflow"):
        correlation_to_covariance(square(np.eye(2)), volatility)


@pytest.mark.parametrize(
    "dtype, small_volatility",
    [
        (
            np.float64,
            float(
                np.nextafter(
                    np.sqrt(np.nextafter(0.0, 1.0)),
                    np.inf,
                )
            ),
        ),
        (np.float32, 1e-30),
        (np.float16, float(np.nextafter(np.float16(0), np.float16(1)))),
    ],
)
def test_just_safe_positive_volatility_scaling_stays_positive(
    dtype: type[np.floating[Any]],
    small_volatility: float,
) -> None:
    volatility = LabeledVector(
        np.asarray([small_volatility, 1.0], dtype=dtype),
        ("a", "b"),
        "asset",
    )

    covariance = correlation_to_covariance(square(np.eye(2)), volatility)

    assert covariance.values[0, 0] > 0.0
    assert np.isfinite(covariance.values).all()


def test_conversions_leave_global_psd_policy_explicit() -> None:
    labels = ("a", "b", "c")
    indefinite_values = np.asarray([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]])
    covariance = square(indefinite_values, labels)
    volatility = LabeledVector(np.ones(3), labels, "asset")

    correlation, converted_volatility = covariance_to_correlation(covariance)
    rebuilt_covariance = correlation_to_covariance(correlation, volatility)

    np.testing.assert_array_equal(correlation.values, indefinite_values)
    np.testing.assert_array_equal(converted_volatility.values, np.ones(3))
    np.testing.assert_array_equal(rebuilt_covariance.values, indefinite_values)
    assert np.linalg.eigvalsh(correlation.values).min() < 0.0
    assert np.linalg.eigvalsh(rebuilt_covariance.values).min() < 0.0
    with pytest.raises(NumericalStabilityError, match="not positive semidefinite"):
        apply_psd_policy(rebuilt_covariance, PSDPolicy.RAISE)
    repaired = apply_psd_policy(rebuilt_covariance, PSDPolicy.CLIP)
    assert np.linalg.eigvalsh(repaired.values).min() >= 0.0


def test_maximum_semantic_tolerance_is_accepted_by_all_apis() -> None:
    identity = square(np.eye(2))
    volatility = LabeledVector(np.ones(2), ("a", "b"), "asset")

    apply_psd_policy(identity, PSDPolicy.RAISE, tolerance=1e-2)
    covariance_to_correlation(identity, tolerance=1e-2)
    correlation_to_covariance(identity, volatility, tolerance=1e-2)


@pytest.mark.parametrize(
    "tolerance",
    [
        float(np.nextafter(1e-2, np.inf)),
        float(np.finfo(np.float64).max),
    ],
)
def test_tolerance_above_semantic_maximum_is_rejected_by_all_apis(
    tolerance: float,
) -> None:
    identity = square(np.eye(2))
    volatility = LabeledVector(np.ones(2), ("a", "b"), "asset")

    with pytest.raises(NumericalStabilityError, match="maximum"):
        apply_psd_policy(
            identity,
            PSDPolicy.RAISE,
            tolerance=tolerance,
        )
    with pytest.raises(NumericalStabilityError, match="maximum"):
        covariance_to_correlation(identity, tolerance=tolerance)
    with pytest.raises(NumericalStabilityError, match="maximum"):
        correlation_to_covariance(identity, volatility, tolerance=tolerance)


def test_huge_tolerance_cannot_relax_material_matrix_semantics() -> None:
    huge = float(np.finfo(np.float64).max)
    volatility = LabeledVector(np.ones(2), ("a", "b"), "asset")

    with pytest.raises(NumericalStabilityError, match="maximum"):
        apply_psd_policy(
            square([[1.0, 1e300], [0.0, 1.0]]),
            PSDPolicy.RAISE,
            tolerance=huge,
        )
    with pytest.raises(NumericalStabilityError, match="maximum"):
        correlation_to_covariance(
            square([[-1.0, 0.0], [0.0, 1.0]]),
            volatility,
            tolerance=huge,
        )


@pytest.mark.parametrize("max_dimension", [True, np.int64(3), 0, -1])
def test_psd_max_dimension_requires_exact_positive_builtin_int(
    max_dimension: object,
) -> None:
    with pytest.raises(NumericalStabilityError, match="max_dimension"):
        apply_psd_policy(
            square(np.eye(2)),
            PSDPolicy.CLIP,
            max_dimension=cast(Any, max_dimension),
        )


def test_psd_dimension_guard_runs_before_any_eigensolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = square(np.eye(2))
    object.__setattr__(matrix, "_values", np.eye(3))
    calls = {"eigh": 0, "eigvalsh": 0}

    def forbidden_eigh(_: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        calls["eigh"] += 1
        raise AssertionError("dimension guard must run first")

    def forbidden_eigvalsh(_: np.ndarray) -> np.ndarray:
        calls["eigvalsh"] += 1
        raise AssertionError("dimension guard must run first")

    monkeypatch.setattr(np.linalg, "eigh", forbidden_eigh)
    monkeypatch.setattr(np.linalg, "eigvalsh", forbidden_eigvalsh)

    with pytest.raises(NumericalStabilityError, match="maximum dimension"):
        apply_psd_policy(
            matrix,
            PSDPolicy.CLIP,
            max_dimension=2,
        )

    assert calls == {"eigh": 0, "eigvalsh": 0}


def test_psd_custom_max_dimension_accepts_its_exact_boundary() -> None:
    labels = ("a", "b", "c")

    accepted = apply_psd_policy(
        square(np.eye(3), labels),
        PSDPolicy.RAISE,
        max_dimension=3,
    )

    np.testing.assert_array_equal(accepted.values, np.eye(3))


@pytest.mark.parametrize(
    "eigenvalues,eigenvectors",
    [
        (np.asarray([]), np.eye(2)),
        (np.zeros((2, 1)), np.eye(2)),
        (np.zeros(3), np.eye(2)),
        (np.asarray([-1.0, 1.0]), np.ones((2, 1))),
        (np.asarray([np.nan, 1.0]), np.eye(2)),
        (np.asarray([-1.0, 1.0]), np.asarray([[np.nan, 0.0], [0.0, 1.0]])),
    ],
)
def test_malformed_eigh_outputs_are_structured(
    monkeypatch: pytest.MonkeyPatch,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
) -> None:
    def malformed_eigh(_: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return eigenvalues, eigenvectors

    monkeypatch.setattr(np.linalg, "eigh", malformed_eigh)

    with pytest.raises(NumericalStabilityError, match="eigensolver output"):
        apply_psd_policy(square(np.eye(2)), PSDPolicy.CLIP)


@pytest.mark.parametrize(
    "eigenvalues",
    [
        np.asarray([]),
        np.zeros((2, 1)),
        np.zeros(3),
        np.asarray([np.nan, 1.0]),
    ],
)
def test_malformed_eigvalsh_outputs_are_structured(
    monkeypatch: pytest.MonkeyPatch,
    eigenvalues: np.ndarray,
) -> None:
    monkeypatch.setattr(
        np.linalg,
        "eigh",
        lambda _: (np.ones(2), np.eye(2)),
    )
    monkeypatch.setattr(np.linalg, "eigvalsh", lambda _: eigenvalues)

    with pytest.raises(NumericalStabilityError, match="eigensolver output"):
        apply_psd_policy(square(np.eye(2)), PSDPolicy.CLIP)
