import copy
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from typing import Any, cast

import numpy as np
import pytest

from qamr.contracts.arrays import LabeledMatrix, LabeledVector
from qamr.contracts.interfaces import RiskEstimator
from qamr.contracts.results import DiagnosticSeverity, NumericalDiagnostic
from qamr.errors import (
    DataValidationError,
    InsufficientHistoryError,
    LabelAlignmentError,
    NumericalStabilityError,
)
from qamr.risk.estimates import CovarianceEstimate
from qamr.risk.ewma import EWMACovariance
from qamr.risk.matrices import PSDPolicy
from qamr.risk.sample import SampleCovariance
from qamr.risk.shrinkage import ShrinkageCovariance
from qamr.risk.spectral import SpectralDenoisedCovariance


class IntSubclass(int):
    pass


class FloatSubclass(float):
    pass


class LabeledMatrixSubclass(LabeledMatrix):
    pass


class CovarianceEstimateSubclass(CovarianceEstimate):
    pass


class StringSubclass(str):
    pass


class CustomBackendError(Exception):
    pass


class CustomBaseError(BaseException):
    pass


def fixed_estimate(
    observation_count: int = 100,
    *,
    labels: tuple[str, ...] = ("a", "b", "c"),
    correlation: np.ndarray[Any, Any] | None = None,
    volatility: np.ndarray[Any, Any] | None = None,
    diagnostics: tuple[NumericalDiagnostic, ...] = (),
) -> CovarianceEstimate:
    correlation_values = np.asarray(
        correlation
        if correlation is not None
        else [[1.0, 0.80, 0.10], [0.80, 1.0, 0.10], [0.10, 0.10, 1.0]],
        dtype=np.float64,
    )
    volatility_values = np.asarray(
        volatility if volatility is not None else [0.2, 0.3, 0.4],
        dtype=np.float64,
    )
    covariance_values = correlation_values * np.outer(volatility_values, volatility_values)
    return CovarianceEstimate(
        covariance=LabeledMatrix(
            covariance_values,
            labels,
            labels,
            "instrument",
            "instrument",
        ),
        correlation=LabeledMatrix(
            correlation_values,
            labels,
            labels,
            "instrument",
            "instrument",
        ),
        volatility=LabeledVector(volatility_values, labels, "instrument"),
        observation_count=observation_count,
        diagnostics=diagnostics,
    )


class FixedEstimator:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[LabeledMatrix] = []

    def estimate(self, matrix: LabeledMatrix) -> object:
        self.calls.append(matrix)
        return self.result


def dummy_returns(
    *,
    labels: tuple[str, ...] = ("a", "b", "c"),
) -> LabeledMatrix:
    return LabeledMatrix(
        np.zeros((4, len(labels))),
        ("t0", "t1", "t2", "t3"),
        labels,
        "time",
        "instrument",
    )


def variable_returns() -> LabeledMatrix:
    return LabeledMatrix(
        np.array(
            [
                [0.01, 0.04, 0.02],
                [0.03, 0.01, 0.01],
                [0.02, 0.05, 0.04],
                [0.04, 0.02, 0.03],
            ]
        ),
        ("t0", "t1", "t2", "t3"),
        ("a", "b", "c"),
        "time",
        "instrument",
    )


def degenerate_spectrum_fixture(
    angle: float,
    *,
    split: float = 0.0,
) -> tuple[CovarianceEstimate, np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    eigenvalues = np.array([0.0, 1.0 - split, 1.0 + split, 2.0])
    eigenvectors = (
        np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, -1.0, 1.0, -1.0],
                [1.0, 1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0, 1.0],
            ]
        )
        / 2.0
    )
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )
    eigenvectors[:, 1:3] = eigenvectors[:, 1:3] @ rotation
    correlation = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    np.fill_diagonal(correlation, 1.0)
    base = fixed_estimate(
        labels=("a", "b", "c", "d"),
        correlation=correlation,
        volatility=np.ones(4),
    )
    return base, eigenvalues, eigenvectors


def test_explicit_rank_preserves_labels_volatility_and_psd() -> None:
    base = fixed_estimate()
    denoised = SpectralDenoisedCovariance(
        signal_rank=1,
        base_estimator=cast("Any", FixedEstimator(base)),
    ).estimate(dummy_returns())

    assert denoised.labels == base.labels
    np.testing.assert_allclose(denoised.volatility.values, base.volatility.values)
    np.testing.assert_allclose(np.diag(denoised.correlation.values), np.ones(3))
    assert np.linalg.eigvalsh(denoised.correlation.values).min() >= -1e-10
    assert not np.allclose(denoised.correlation.values, base.correlation.values)
    assert denoised.diagnostics[-1].context["rank_selection"] == "explicit"


def test_marchenko_pastur_selection_is_deterministic_rank_one() -> None:
    estimator = SpectralDenoisedCovariance(
        signal_rank=None,
        base_estimator=cast("Any", FixedEstimator(fixed_estimate(observation_count=100))),
        mp_effective_observations=100,
    )

    first = estimator.estimate(dummy_returns())
    second = estimator.estimate(dummy_returns())

    np.testing.assert_array_equal(first.covariance.values, second.covariance.values)
    assert first.diagnostics[-1].context["rank_selection"] == "marchenko_pastur"
    assert first.diagnostics[-1].context["signal_rank"] == 1


def test_marchenko_pastur_requires_more_observations_than_instruments() -> None:
    estimator = SpectralDenoisedCovariance(
        signal_rank=None,
        base_estimator=cast("Any", FixedEstimator(fixed_estimate(observation_count=3))),
        mp_effective_observations=2,
    )

    with pytest.raises(InsufficientHistoryError, match="effective observations"):
        estimator.estimate(dummy_returns())


def test_inferred_sample_effective_observations_use_ddof_at_equality_boundary() -> None:
    matrix = LabeledMatrix(
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [-1.0, -1.0, -1.0, -1.0],
            ]
        ),
        ("t0", "t1", "t2", "t3", "t4"),
        ("a", "b", "c", "d"),
        "time",
        "instrument",
    )

    estimate = SpectralDenoisedCovariance(
        base_estimator=SampleCovariance(ddof=1),
    ).estimate(matrix)
    diagnostic = estimate.diagnostics[-1]

    assert diagnostic.context["effective_observations"] == 4.0
    assert diagnostic.context["effective_observations_source"] == "sample_covariance"
    assert diagnostic.context["marchenko_pastur_upper_edge"] == 4.0
    assert diagnostic.context["signal_rank"] == 0


def test_inferred_sample_ddof_zero_still_uses_demeaned_t_minus_one_history() -> None:
    matrix = LabeledMatrix(
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [-1.0, -1.0, -1.0, -1.0],
            ]
        ),
        ("t0", "t1", "t2", "t3", "t4"),
        ("a", "b", "c", "d"),
        "time",
        "instrument",
    )

    estimate = SpectralDenoisedCovariance(
        base_estimator=SampleCovariance(ddof=0),
    ).estimate(matrix)
    diagnostic = estimate.diagnostics[-1]

    assert diagnostic.context["effective_observations"] == 4.0
    assert diagnostic.context["effective_observations_source"] == "sample_covariance"
    assert diagnostic.context["marchenko_pastur_upper_edge"] == 4.0
    assert diagnostic.context["signal_rank"] == 0


def test_inferred_sample_t_one_has_no_effective_demeaned_history() -> None:
    sample_diagnostic = NumericalDiagnostic(
        code="sample_covariance",
        severity=DiagnosticSeverity.INFO,
        message="sample fixture",
        context={"ddof": 0, "annualization_factor": None},
    )
    base = fixed_estimate(
        observation_count=1,
        labels=("solo",),
        correlation=np.ones((1, 1)),
        volatility=np.ones(1),
        diagnostics=(sample_diagnostic,),
    )

    with pytest.raises(InsufficientHistoryError, match="effective observations") as captured:
        SpectralDenoisedCovariance(
            base_estimator=cast("Any", FixedEstimator(base)),
        ).estimate(dummy_returns(labels=("solo",)))

    assert captured.value.context == {
        "effective_observations": 0,
        "instrument_count": 1,
    }


@pytest.mark.parametrize(
    "base",
    [
        EWMACovariance(),
        ShrinkageCovariance(shrinkage=0.25),
        FixedEstimator(fixed_estimate()),
    ],
    ids=["ewma", "shrinkage", "fixed"],
)
def test_automatic_mp_rejects_non_sample_spectrum_without_explicit_effective_history(
    base: object,
) -> None:
    with pytest.raises(DataValidationError, match="automatic Marchenko-Pastur") as captured:
        SpectralDenoisedCovariance(
            base_estimator=cast("Any", base),
        ).estimate(variable_returns())

    assert captured.value.context["reason"] == "incompatible_base_spectrum"
    assert len(json.dumps(captured.value.as_dict())) < 512


@pytest.mark.parametrize(
    "base",
    [
        EWMACovariance(),
        ShrinkageCovariance(shrinkage=0.25),
        FixedEstimator(fixed_estimate()),
    ],
    ids=["ewma", "shrinkage", "fixed"],
)
def test_explicit_mp_effective_observations_override_supports_composed_bases(
    base: object,
) -> None:
    estimate = SpectralDenoisedCovariance(
        base_estimator=cast("Any", base),
        mp_effective_observations=100,
    ).estimate(variable_returns())

    diagnostic = estimate.diagnostics[-1]
    assert diagnostic.context["effective_observations"] == 100.0
    assert diagnostic.context["effective_observations_source"] == "explicit"


@pytest.mark.parametrize("rank", [-1, 4])
def test_explicit_rank_must_fit_matrix(rank: int) -> None:
    base = FixedEstimator(fixed_estimate())
    estimator = SpectralDenoisedCovariance(
        signal_rank=rank,
        base_estimator=cast("Any", base),
    )

    with pytest.raises(DataValidationError, match="signal rank"):
        estimator.estimate(dummy_returns())
    assert base.calls == []


@pytest.mark.parametrize(
    "rank",
    [True, False, 1.0, np.int64(1), IntSubclass(1), "1"],
)
def test_signal_rank_requires_none_or_exact_builtin_int(rank: object) -> None:
    with pytest.raises(DataValidationError, match="signal_rank") as captured:
        SpectralDenoisedCovariance(signal_rank=cast("Any", rank))

    assert captured.value.context["field"] == "signal_rank"


@pytest.mark.parametrize(
    "effective",
    [
        True,
        False,
        np.int64(100),
        np.float64(100.0),
        IntSubclass(100),
        FloatSubclass(100.0),
        0,
        -1,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_mp_effective_observations_requires_exact_positive_finite_builtin_number(
    effective: object,
) -> None:
    with pytest.raises(DataValidationError, match="mp_effective_observations") as captured:
        SpectralDenoisedCovariance(mp_effective_observations=cast("Any", effective))

    assert captured.value.context["field"] == "mp_effective_observations"


@pytest.mark.parametrize("effective", [1, 3.5, 10**300])
def test_valid_mp_effective_observations_is_preserved(
    effective: int | float,
) -> None:
    estimator = SpectralDenoisedCovariance(mp_effective_observations=effective)

    assert estimator.mp_effective_observations is effective


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("psd_policy", "clip"),
        ("psd_policy", StringSubclass("clip")),
    ],
)
def test_enum_configuration_requires_exact_member(field: str, value: object) -> None:
    with pytest.raises(DataValidationError) as captured:
        SpectralDenoisedCovariance(**{field: value})

    assert captured.value.context["field"] == field


@pytest.mark.parametrize(
    "tolerance",
    [
        True,
        np.float64(1e-10),
        IntSubclass(1),
        FloatSubclass(1e-10),
        0,
        -0.0,
        -1e-10,
        1.000001e-2,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_tolerance_requires_exact_finite_bounded_builtin_number(tolerance: object) -> None:
    with pytest.raises(DataValidationError, match="tolerance") as captured:
        SpectralDenoisedCovariance(tolerance=cast("Any", tolerance))

    assert captured.value.context["field"] == "tolerance"


@pytest.mark.parametrize("maximum", [True, 1.0, np.int64(2), IntSubclass(2), 0, -1])
def test_max_dimension_requires_exact_positive_int(maximum: object) -> None:
    with pytest.raises(DataValidationError, match="max_dimension") as captured:
        SpectralDenoisedCovariance(max_dimension=cast("Any", maximum))

    assert captured.value.context["field"] == "max_dimension"


@pytest.mark.parametrize("base", [None, object()])
def test_base_estimator_must_expose_callable_estimate(base: object) -> None:
    with pytest.raises(DataValidationError, match="base_estimator"):
        SpectralDenoisedCovariance(base_estimator=cast("Any", base))


def test_noncallable_base_estimate_is_rejected() -> None:
    class NonCallable:
        estimate = 42

    with pytest.raises(DataValidationError, match="callable"):
        SpectralDenoisedCovariance(base_estimator=cast("Any", NonCallable()))


def test_base_descriptor_error_policy_and_single_access() -> None:
    class Once:
        accesses = 0

        @property
        def estimate(self) -> object:
            self.accesses += 1
            return lambda matrix: matrix

    base = Once()
    SpectralDenoisedCovariance(base_estimator=cast("Any", base))
    assert base.accesses == 1

    memory = MemoryError("descriptor")

    class MemoryDescriptor:
        @property
        def estimate(self) -> object:
            raise memory

    with pytest.raises(MemoryError) as captured:
        SpectralDenoisedCovariance(base_estimator=cast("Any", MemoryDescriptor()))
    assert captured.value is memory

    secret = "DO-NOT-LEAK-DESCRIPTOR"

    class FailingDescriptor:
        @property
        def estimate(self) -> object:
            raise RuntimeError(secret)

    with pytest.raises(DataValidationError) as captured:
        SpectralDenoisedCovariance(base_estimator=cast("Any", FailingDescriptor()))
    assert captured.value.context == {
        "field": "base_estimator",
        "dtype": "FailingDescriptor",
        "reason": "RuntimeError",
    }
    assert secret not in str(captured.value)


def test_default_configuration_protocol_frozen_slots_and_pickle() -> None:
    estimator = SpectralDenoisedCovariance()

    assert isinstance(estimator, RiskEstimator)
    assert estimator.signal_rank is None
    assert type(estimator.base_estimator) is SampleCovariance
    assert estimator.psd_policy is PSDPolicy.CLIP
    assert estimator.tolerance == 1e-10
    assert estimator.max_dimension == 2048
    assert estimator.mp_effective_observations is None
    assert not hasattr(estimator, "__dict__")
    with pytest.raises((FrozenInstanceError, AttributeError)):
        estimator.signal_rank = 1  # type: ignore[misc]
    assert pickle.loads(pickle.dumps(estimator)) == estimator


def test_returns_exact_type_and_dimension_preflight_before_base() -> None:
    base = FixedEstimator(fixed_estimate())
    subclass = LabeledMatrixSubclass(
        np.zeros((4, 3)),
        ("t0", "t1", "t2", "t3"),
        ("a", "b", "c"),
        "time",
        "instrument",
    )
    with pytest.raises(DataValidationError, match="exact LabeledMatrix"):
        SpectralDenoisedCovariance(
            base_estimator=cast("Any", base),
        ).estimate(subclass)
    assert base.calls == []

    with pytest.raises(NumericalStabilityError, match="maximum dimension"):
        SpectralDenoisedCovariance(
            base_estimator=cast("Any", base),
            max_dimension=2,
        ).estimate(dummy_returns())
    assert base.calls == []


def test_base_call_error_policy() -> None:
    for error in (
        MemoryError("memory"),
        DataValidationError("qamr"),
        CustomBaseError("base"),
    ):

        class Failing:
            def __init__(self, failure: BaseException) -> None:
                self.failure = failure

            def estimate(self, matrix: LabeledMatrix) -> CovarianceEstimate:
                del matrix
                raise self.failure

        with pytest.raises(type(error)) as captured:
            SpectralDenoisedCovariance(
                base_estimator=cast("Any", Failing(error)),
            ).estimate(dummy_returns())
        assert captured.value is error

    secret = "DO-NOT-LEAK-BASE"

    class OrdinaryFailing:
        def estimate(self, matrix: LabeledMatrix) -> CovarianceEstimate:
            del matrix
            raise CustomBackendError(secret)

    with pytest.raises(NumericalStabilityError) as captured:
        SpectralDenoisedCovariance(
            base_estimator=cast("Any", OrdinaryFailing()),
        ).estimate(dummy_returns())
    assert captured.value.context == {
        "operation": "base_covariance_estimation",
        "reason": "CustomBackendError",
    }
    assert secret not in str(captured.value)


@pytest.mark.parametrize("result", [None, object()])
def test_base_must_return_exact_covariance_estimate(result: object) -> None:
    with pytest.raises(DataValidationError, match="CovarianceEstimate"):
        SpectralDenoisedCovariance(
            base_estimator=cast("Any", FixedEstimator(result)),
        ).estimate(dummy_returns())


def test_covariance_estimate_subclass_and_label_mismatch_are_rejected() -> None:
    base = fixed_estimate()
    subclass = CovarianceEstimateSubclass(
        covariance=base.covariance,
        correlation=base.correlation,
        volatility=base.volatility,
        observation_count=base.observation_count,
        diagnostics=base.diagnostics,
    )
    with pytest.raises(DataValidationError, match="CovarianceEstimate"):
        SpectralDenoisedCovariance(
            base_estimator=cast("Any", FixedEstimator(subclass)),
        ).estimate(dummy_returns())

    mismatched = fixed_estimate(labels=("b", "a", "c"))
    with pytest.raises(LabelAlignmentError, match="labels"):
        SpectralDenoisedCovariance(
            base_estimator=cast("Any", FixedEstimator(mismatched)),
        ).estimate(dummy_returns())


def test_full_rank_fast_path_preserves_base_risk_tightly() -> None:
    base = fixed_estimate()
    estimate = SpectralDenoisedCovariance(
        signal_rank=3,
        base_estimator=cast("Any", FixedEstimator(base)),
        psd_policy=PSDPolicy.RAISE,
    ).estimate(dummy_returns())

    np.testing.assert_allclose(estimate.covariance.values, base.covariance.values, atol=1e-15)
    np.testing.assert_allclose(estimate.correlation.values, base.correlation.values, atol=1e-15)
    np.testing.assert_allclose(estimate.volatility.values, base.volatility.values, atol=1e-15)
    assert estimate.covariance is base.covariance
    assert estimate.correlation is base.correlation
    assert estimate.volatility is base.volatility
    assert estimate.diagnostics[-1].message == "no spectral denoising was applied"
    assert estimate.diagnostics[-1].context["denoising_applied"] is False
    assert estimate.diagnostics[-1].context["noise_eigenvalue_count"] == 0


def test_full_rank_repairs_correlation_before_rescaling_exact_base_volatility() -> None:
    correlation = np.array(
        [
            [1.0, -0.9, -0.9],
            [-0.9, 1.0, -0.9],
            [-0.9, -0.9, 1.0],
        ]
    )
    volatility = np.array([0.2, 0.3, 0.4])
    base = fixed_estimate(correlation=correlation, volatility=volatility)

    with pytest.raises(NumericalStabilityError, match="positive semidefinite"):
        SpectralDenoisedCovariance(
            signal_rank=3,
            base_estimator=cast("Any", FixedEstimator(base)),
            psd_policy=PSDPolicy.RAISE,
        ).estimate(dummy_returns())

    clipped = SpectralDenoisedCovariance(
        signal_rank=3,
        base_estimator=cast("Any", FixedEstimator(base)),
        psd_policy=PSDPolicy.CLIP,
    ).estimate(dummy_returns())

    np.testing.assert_array_equal(clipped.volatility.values, volatility)
    np.testing.assert_allclose(np.diag(clipped.correlation.values), np.ones(3))
    assert np.linalg.eigvalsh(clipped.correlation.values).min() >= -1e-10


def test_full_rank_true_noop_supports_extreme_valid_base_volatility_resolution() -> None:
    labels = ("large", "small")
    volatility = np.array([1e154, 1e-154])
    covariance = np.diag(volatility * volatility)
    base = CovarianceEstimate(
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
        volatility=LabeledVector(volatility, labels, "instrument"),
        observation_count=10,
    )

    estimate = SpectralDenoisedCovariance(
        signal_rank=2,
        base_estimator=cast("Any", FixedEstimator(base)),
    ).estimate(dummy_returns(labels=labels))

    assert estimate.covariance is base.covariance
    assert estimate.correlation is base.correlation
    assert estimate.volatility is base.volatility
    np.testing.assert_array_equal(estimate.volatility.values, volatility)


def test_rank_zero_produces_identity_correlation() -> None:
    estimate = SpectralDenoisedCovariance(
        signal_rank=0,
        base_estimator=cast("Any", FixedEstimator(fixed_estimate())),
    ).estimate(dummy_returns())

    np.testing.assert_allclose(estimate.correlation.values, np.eye(3), atol=1e-14)


def test_intermediate_rank_matches_hand_eigendecomposition_at_matrix_level() -> None:
    correlation = np.array(
        [[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    base = fixed_estimate(correlation=correlation)
    estimate = SpectralDenoisedCovariance(
        signal_rank=1,
        base_estimator=cast("Any", FixedEstimator(base)),
    ).estimate(dummy_returns())

    expected = np.array(
        [[1.0, 1.0 / 3.0, 0.0], [1.0 / 3.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    np.testing.assert_allclose(estimate.correlation.values, expected, atol=1e-14)


def test_explicit_rank_cannot_split_degenerate_eigenspace_across_rotated_bases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qamr.risk import spectral as spectral_module

    errors: list[dict[str, Any]] = []
    for angle in (0.0, 0.713):
        base, eigenvalues, eigenvectors = degenerate_spectrum_fixture(angle)
        monkeypatch.setattr(
            spectral_module,
            "_safe_eigh",
            lambda values, dimension, _ev=eigenvalues, _q=eigenvectors: (_ev, _q),
        )
        with pytest.raises(DataValidationError, match="eigenvalue multiplicity") as captured:
            SpectralDenoisedCovariance(
                signal_rank=2,
                base_estimator=cast("Any", FixedEstimator(base)),
            ).estimate(dummy_returns(labels=("a", "b", "c", "d")))
        errors.append(captured.value.as_dict())

    assert errors[0] == errors[1]


@pytest.mark.parametrize("rank", [1, 3])
def test_group_aligned_explicit_ranks_are_rotation_invariant(
    monkeypatch: pytest.MonkeyPatch,
    rank: int,
) -> None:
    from qamr.risk import spectral as spectral_module

    results: list[np.ndarray[Any, Any]] = []
    for angle in (0.0, 0.713):
        base, eigenvalues, eigenvectors = degenerate_spectrum_fixture(angle)
        monkeypatch.setattr(
            spectral_module,
            "_safe_eigh",
            lambda values, dimension, _ev=eigenvalues, _q=eigenvectors: (_ev, _q),
        )
        estimate = SpectralDenoisedCovariance(
            signal_rank=rank,
            base_estimator=cast("Any", FixedEstimator(base)),
        ).estimate(dummy_returns(labels=("a", "b", "c", "d")))
        results.append(estimate.correlation.values)

    np.testing.assert_allclose(results[0], results[1], atol=1e-14)


def test_mp_groups_numerically_equal_boundary_eigenvalues_across_rotated_bases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qamr.risk import spectral as spectral_module

    ranks: list[object] = []
    for angle in (0.0, 0.713):
        base, eigenvalues, eigenvectors = degenerate_spectrum_fixture(
            angle,
            split=1e-12,
        )
        monkeypatch.setattr(
            spectral_module,
            "_safe_eigh",
            lambda values, dimension, _ev=eigenvalues, _q=eigenvectors: (_ev, _q),
        )
        estimate = SpectralDenoisedCovariance(
            base_estimator=cast("Any", FixedEstimator(base)),
            mp_effective_observations=1e300,
        ).estimate(dummy_returns(labels=("a", "b", "c", "d")))
        ranks.append(estimate.diagnostics[-1].context["signal_rank"])

    assert ranks == [1, 1]


def test_one_asset_is_supported_for_explicit_and_automatic_rank() -> None:
    base = fixed_estimate(
        labels=("solo",),
        correlation=np.ones((1, 1)),
        volatility=np.array([0.3]),
        observation_count=2,
    )
    for rank in (0, 1, None):
        estimate = SpectralDenoisedCovariance(
            signal_rank=rank,
            base_estimator=cast("Any", FixedEstimator(base)),
            mp_effective_observations=2 if rank is None else None,
        ).estimate(dummy_returns(labels=("solo",)))
        np.testing.assert_allclose(estimate.covariance.values, [[0.09]])


def test_marchenko_pastur_boundary_is_strict_and_huge_count_is_finite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qamr.risk import spectral as spectral_module

    edge = (1.0 + np.sqrt(3.0 / 100.0)) ** 2

    def boundary_eigh(
        matrix: np.ndarray[Any, Any],
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        del matrix
        return np.array([0.5, 1.0, edge]), np.eye(3)

    monkeypatch.setattr(spectral_module.np.linalg, "eigh", boundary_eigh)
    estimate = SpectralDenoisedCovariance(
        base_estimator=cast("Any", FixedEstimator(fixed_estimate(observation_count=100))),
        mp_effective_observations=100,
    ).estimate(dummy_returns())
    assert estimate.diagnostics[-1].context["signal_rank"] == 0

    huge = fixed_estimate(observation_count=10**300)
    estimate = SpectralDenoisedCovariance(
        base_estimator=cast("Any", FixedEstimator(huge)),
        mp_effective_observations=10**300,
    ).estimate(dummy_returns())
    upper = estimate.diagnostics[-1].context["marchenko_pastur_upper_edge"]
    assert isinstance(upper, float)
    assert np.isfinite(upper)


def test_q_one_all_ones_correlation_keeps_theoretical_edge_as_noise() -> None:
    base = fixed_estimate(
        observation_count=4,
        labels=("a", "b", "c", "d"),
        correlation=np.ones((4, 4)),
        volatility=np.ones(4),
    )

    estimate = SpectralDenoisedCovariance(
        base_estimator=cast("Any", FixedEstimator(base)),
        mp_effective_observations=4,
    ).estimate(dummy_returns(labels=("a", "b", "c", "d")))

    assert estimate.diagnostics[-1].context["marchenko_pastur_upper_edge"] == 4.0
    assert estimate.diagnostics[-1].context["signal_rank"] == 0


@pytest.mark.parametrize(
    ("largest", "expected_rank"),
    [
        (np.nextafter(4.0, np.inf), 0),
        (4.0 + 1e-8, 1),
    ],
    ids=["one_ulp_above_edge", "genuinely_above_edge"],
)
def test_mp_edge_comparison_absorbs_only_roundoff_scale(
    monkeypatch: pytest.MonkeyPatch,
    largest: float,
    expected_rank: int,
) -> None:
    from qamr.risk import spectral as spectral_module

    base, _, eigenvectors = degenerate_spectrum_fixture(0.0)
    eigenvalues = np.array([0.0, 0.0, 0.0, largest])
    monkeypatch.setattr(
        spectral_module,
        "_safe_eigh",
        lambda values, dimension: (eigenvalues, eigenvectors),
    )

    estimate = SpectralDenoisedCovariance(
        base_estimator=cast("Any", FixedEstimator(base)),
        mp_effective_observations=4,
    ).estimate(dummy_returns(labels=("a", "b", "c", "d")))

    assert estimate.diagnostics[-1].context["signal_rank"] == expected_rank


@pytest.mark.parametrize(
    ("eigenvalues", "eigenvectors"),
    [
        (np.array([]), np.empty((0, 0))),
        (np.ones(2), np.eye(3)),
        (np.ones(3), np.ones((2, 3))),
        (np.array([1.0, np.nan, 2.0]), np.eye(3)),
        (np.ones(3), np.full((3, 3), np.inf)),
        (np.ones(3, dtype=np.complex128), np.eye(3)),
        (np.ones(3), np.eye(3, dtype=np.complex128)),
    ],
)
def test_malformed_eigensolver_outputs_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
    eigenvalues: np.ndarray[Any, Any],
    eigenvectors: np.ndarray[Any, Any],
) -> None:
    from qamr.risk import spectral as spectral_module

    monkeypatch.setattr(
        spectral_module.np.linalg,
        "eigh",
        lambda matrix: (eigenvalues, eigenvectors),
    )
    with pytest.raises(NumericalStabilityError, match="eigendecomposition"):
        SpectralDenoisedCovariance(
            signal_rank=1,
            base_estimator=cast("Any", FixedEstimator(fixed_estimate())),
        ).estimate(dummy_returns())


def test_eigensolver_ordinary_error_is_bounded_and_memory_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qamr.risk import spectral as spectral_module

    secret = "DO-NOT-LEAK-EIGEN"

    def ordinary(matrix: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        del matrix
        raise CustomBackendError(secret)

    monkeypatch.setattr(spectral_module.np.linalg, "eigh", ordinary)
    with pytest.raises(NumericalStabilityError) as captured:
        SpectralDenoisedCovariance(
            signal_rank=1,
            base_estimator=cast("Any", FixedEstimator(fixed_estimate())),
        ).estimate(dummy_returns())
    assert captured.value.context == {
        "operation": "spectral_eigendecomposition",
        "reason": "CustomBackendError",
    }
    assert secret not in str(captured.value)

    memory = MemoryError("eigh")

    def exhausted(
        matrix: np.ndarray[Any, Any],
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        del matrix
        raise memory

    monkeypatch.setattr(spectral_module.np.linalg, "eigh", exhausted)
    with pytest.raises(MemoryError) as memory_captured:
        SpectralDenoisedCovariance(
            signal_rank=1,
            base_estimator=cast("Any", FixedEstimator(fixed_estimate())),
        ).estimate(dummy_returns())
    assert memory_captured.value is memory


def test_max_dimension_guard_occurs_before_eigensolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qamr.risk import spectral as spectral_module

    called = False

    def forbidden(
        matrix: np.ndarray[Any, Any],
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        nonlocal called
        called = True
        raise AssertionError("must not run")

    monkeypatch.setattr(spectral_module.np.linalg, "eigh", forbidden)
    with pytest.raises(NumericalStabilityError, match="maximum dimension"):
        SpectralDenoisedCovariance(max_dimension=2).estimate(dummy_returns())
    assert not called


def test_negative_spectrum_is_deferred_to_explicit_psd_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qamr.risk import spectral as spectral_module

    correlation = np.array(
        [[1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]],
    )
    indefinite_result = np.linalg.eigh(correlation)
    monkeypatch.setattr(
        spectral_module,
        "_safe_eigh",
        lambda values, dimension: indefinite_result,
    )
    with pytest.raises(NumericalStabilityError, match="positive semidefinite"):
        SpectralDenoisedCovariance(
            signal_rank=2,
            base_estimator=cast("Any", FixedEstimator(fixed_estimate())),
            psd_policy=PSDPolicy.RAISE,
        ).estimate(dummy_returns())

    clipped = SpectralDenoisedCovariance(
        signal_rank=2,
        base_estimator=cast("Any", FixedEstimator(fixed_estimate())),
        psd_policy=PSDPolicy.CLIP,
    ).estimate(dummy_returns())
    assert np.linalg.eigvalsh(clipped.covariance.values).min() >= -1e-10


def test_diagnostics_are_bounded_json_safe_and_preserve_history() -> None:
    prior = NumericalDiagnostic(
        code="fixed_fixture",
        severity=DiagnosticSeverity.INFO,
        message="fixture",
    )
    base = fixed_estimate(diagnostics=(prior,))
    estimate = SpectralDenoisedCovariance(
        signal_rank=1,
        base_estimator=cast("Any", FixedEstimator(base)),
    ).estimate(dummy_returns())

    assert estimate.diagnostics[:-1] == (prior,)
    diagnostic = estimate.diagnostics[-1]
    assert diagnostic.code == "spectral_denoising"
    assert diagnostic.severity is DiagnosticSeverity.INFO
    assert dict(diagnostic.context) == {
        "signal_rank": 1,
        "rank_selection": "explicit",
        "marchenko_pastur_upper_edge": None,
        "effective_observations": None,
        "effective_observations_source": None,
        "denoising_applied": True,
        "noise_eigenvalue_count": 2,
    }
    json.dumps(dict(diagnostic.context))


def test_no_mutation_and_serial_concurrent_determinism() -> None:
    matrix = dummy_returns()
    base = fixed_estimate()
    matrix_before = matrix.values
    base_before = copy.deepcopy(base)
    estimator = SpectralDenoisedCovariance(
        signal_rank=1,
        base_estimator=cast("Any", FixedEstimator(base)),
    )

    serial = [estimator.estimate(matrix).covariance.values for _ in range(3)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        concurrent = list(
            executor.map(lambda _: estimator.estimate(matrix).covariance.values, range(8))
        )

    for result in serial[1:] + concurrent:
        np.testing.assert_array_equal(result, serial[0])
    np.testing.assert_array_equal(matrix.values, matrix_before)
    np.testing.assert_array_equal(base.covariance.values, base_before.covariance.values)


@pytest.mark.parametrize(
    ("rank", "target", "component"),
    [
        (3, "apply_psd_policy", "full_rank_result"),
        (1, "build_covariance_estimate", "covariance_estimate_builder"),
    ],
    ids=["full_rank", "normal"],
)
@pytest.mark.parametrize("error_type", [RuntimeError, CustomBackendError])
def test_final_result_ordinary_error_is_bounded_on_both_paths(
    monkeypatch: pytest.MonkeyPatch,
    rank: int,
    target: str,
    component: str,
    error_type: type[Exception],
) -> None:
    from qamr.risk import spectral as spectral_module

    secret = f"DO-NOT-LEAK-FINAL-BUILDER-{error_type.__name__}"

    def fail(*args: object, **kwargs: object) -> CovarianceEstimate:
        del args, kwargs
        raise error_type(secret)

    monkeypatch.setattr(spectral_module, target, fail)
    with pytest.raises(NumericalStabilityError) as captured:
        SpectralDenoisedCovariance(
            signal_rank=rank,
            base_estimator=cast("Any", FixedEstimator(fixed_estimate())),
        ).estimate(dummy_returns())

    assert captured.value.context == {
        "operation": "spectral_result_construction",
        "component": component,
        "reason": error_type.__name__,
    }
    serialized = json.dumps(captured.value.as_dict())
    assert secret not in str(captured.value)
    assert secret not in serialized
    assert len(serialized) < 512


@pytest.mark.parametrize(
    ("rank", "target"),
    [(3, "apply_psd_policy"), (1, "build_covariance_estimate")],
    ids=["full_rank", "normal"],
)
@pytest.mark.parametrize(
    "error",
    [
        DataValidationError("builder qamr", context={"safe": True}),
        MemoryError("builder memory"),
        KeyboardInterrupt("builder interrupt"),
    ],
)
def test_final_builder_qamr_memory_and_base_errors_propagate_by_identity(
    monkeypatch: pytest.MonkeyPatch,
    rank: int,
    target: str,
    error: BaseException,
) -> None:
    from qamr.risk import spectral as spectral_module

    def fail(*args: object, **kwargs: object) -> CovarianceEstimate:
        del args, kwargs
        raise error

    monkeypatch.setattr(spectral_module, target, fail)
    with pytest.raises(type(error)) as captured:
        SpectralDenoisedCovariance(
            signal_rank=rank,
            base_estimator=cast("Any", FixedEstimator(fixed_estimate())),
        ).estimate(dummy_returns())

    assert captured.value is error
