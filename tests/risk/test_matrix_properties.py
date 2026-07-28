from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from qamr.contracts.arrays import LabeledMatrix
from qamr.risk.matrices import (
    PSDPolicy,
    apply_psd_policy,
    correlation_to_covariance,
    covariance_to_correlation,
)


@given(
    st.lists(
        st.floats(
            min_value=-3.0,
            max_value=3.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=9,
        max_size=9,
    )
)
@settings(max_examples=50, deadline=None, derandomize=True)
def test_positive_definite_covariance_round_trip(values: list[float]) -> None:
    loading = np.asarray(values).reshape(3, 3)
    covariance_values = loading.T @ loading + np.eye(3) * 0.25
    labels = ("a", "b", "c")
    covariance = LabeledMatrix(
        covariance_values,
        labels,
        labels,
        "instrument",
        "instrument",
    )

    correlation, volatility = covariance_to_correlation(covariance)
    rebuilt = correlation_to_covariance(correlation, volatility)

    np.testing.assert_allclose(
        rebuilt.values,
        covariance.values,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_array_equal(np.diag(correlation.values), np.ones(3))
    assert np.linalg.eigvalsh(correlation.values).min() >= -1e-10
    assert rebuilt.row_labels == labels
    assert rebuilt.column_labels == labels
    assert (rebuilt.row_name, rebuilt.column_name) == ("instrument", "instrument")


@given(
    st.lists(
        st.floats(
            min_value=-10.0,
            max_value=10.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=16,
        max_size=16,
    )
)
@settings(max_examples=50, deadline=None, derandomize=True)
def test_psd_clip_is_symmetric_psd_idempotent_and_does_not_mutate(
    values: list[float],
) -> None:
    raw = np.asarray(values).reshape(4, 4)
    symmetric = (raw + raw.T) / 2.0
    labels = ("a", "b", "c", "d")
    matrix = LabeledMatrix(symmetric, labels, labels, "asset", "asset")
    before = matrix.values

    clipped = apply_psd_policy(matrix, PSDPolicy.CLIP)
    clipped_again = apply_psd_policy(clipped, PSDPolicy.CLIP)

    np.testing.assert_array_equal(matrix.values, before)
    np.testing.assert_allclose(clipped.values, clipped.values.T, rtol=0, atol=1e-12)
    assert np.linalg.eigvalsh(clipped.values).min() >= 0.0
    np.testing.assert_allclose(
        clipped_again.values,
        clipped.values,
        rtol=1e-10,
        atol=1e-10,
    )
    assert clipped.row_labels == labels
    assert clipped.column_labels == labels
    assert (clipped.row_name, clipped.column_name) == ("asset", "asset")


def test_seeded_adversarial_psd_clips_are_exactly_nonnegative_and_stable() -> None:
    generator = np.random.default_rng(20260728)
    labels = tuple(f"asset-{position}" for position in range(6))

    for _ in range(50):
        raw = generator.normal(size=(6, 6))
        symmetric = raw / 2.0 + raw.T / 2.0
        matrix = LabeledMatrix(symmetric, labels, labels, "asset", "asset")
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        spectral_projection = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
        spectral_projection = spectral_projection / 2.0 + spectral_projection.T / 2.0

        clipped = apply_psd_policy(matrix, PSDPolicy.CLIP)
        clipped_again = apply_psd_policy(clipped, PSDPolicy.CLIP)

        assert np.linalg.eigvalsh(clipped.values).min() >= 0.0
        np.testing.assert_allclose(
            clipped_again.values,
            clipped.values,
            rtol=1e-12,
            atol=1e-12,
        )
        assert np.linalg.norm(
            clipped.values - spectral_projection,
            ord="fro",
        ) <= (
            np.finfo(np.float64).eps
            * 100.0
            * max(1.0, np.linalg.norm(spectral_projection, ord="fro"))
        )


def test_clip_verifies_forced_eigh_eigvalsh_sign_disagreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = ("a", "b", "c")
    values = np.eye(3)
    matrix = LabeledMatrix(values, labels, labels, "asset", "asset")
    real_eigvalsh = np.linalg.eigvalsh
    eigvalsh_call_count = 0

    def forced_eigh(
        _: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return np.ones(3), np.eye(3)

    def forced_eigvalsh(candidate: np.ndarray) -> np.ndarray:
        nonlocal eigvalsh_call_count
        eigvalsh_call_count += 1
        if eigvalsh_call_count == 1:
            return np.asarray([-np.finfo(np.float64).eps, 1.0, 1.0])
        return real_eigvalsh(candidate)

    monkeypatch.setattr(np.linalg, "eigh", forced_eigh)
    monkeypatch.setattr(np.linalg, "eigvalsh", forced_eigvalsh)

    clipped = apply_psd_policy(matrix, PSDPolicy.CLIP)

    assert eigvalsh_call_count >= 2
    assert real_eigvalsh(clipped.values).min() >= 0.0
    np.testing.assert_allclose(
        clipped.values,
        values,
        rtol=0.0,
        atol=np.finfo(np.float64).eps * 2.0,
    )


def test_clip_keeps_strictly_positive_definite_input_exactly_unchanged() -> None:
    labels = ("a", "b", "c")
    values = np.asarray([[2.0, 0.25, -0.1], [0.25, 1.5, 0.2], [-0.1, 0.2, 1.0]])
    matrix = LabeledMatrix(values, labels, labels, "asset", "asset")

    clipped = apply_psd_policy(matrix, PSDPolicy.CLIP)

    np.testing.assert_array_equal(clipped.values, values)


def test_raise_does_not_invoke_clip_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = np.eye(3)
    labels = ("a", "b", "c")
    matrix = LabeledMatrix(values, labels, labels, "asset", "asset")

    def forced_eigh(
        _: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return np.ones(3), np.eye(3)

    def forbidden_eigvalsh(_: np.ndarray) -> np.ndarray:
        raise AssertionError("RAISE must not invoke the CLIP verifier")

    monkeypatch.setattr(np.linalg, "eigh", forced_eigh)
    monkeypatch.setattr(np.linalg, "eigvalsh", forbidden_eigvalsh)

    accepted = apply_psd_policy(matrix, PSDPolicy.RAISE)

    np.testing.assert_array_equal(accepted.values, values)
