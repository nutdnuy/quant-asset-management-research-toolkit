import copy
import pickle
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from threading import Barrier
from typing import Any, cast

import numpy as np
import pytest

from qamr.contracts.arrays import LabeledMatrix
from qamr.contracts.dataset import MissingDataPolicy
from qamr.contracts.interfaces import RiskEstimator
from qamr.errors import (
    DataValidationError,
    InsufficientHistoryError,
    NumericalStabilityError,
    QAMRError,
)
from qamr.risk.ewma import EWMACovariance
from qamr.risk.matrices import PSDPolicy
from qamr.risk.sample import SampleCovariance


class IntSubclass(int):
    pass


class FloatSubclass(float):
    pass


class CustomBackendError(Exception):
    pass


class CustomBackendBaseError(BaseException):
    pass


class MalformedCovarianceOutput:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    def __array__(
        self,
        dtype: np.dtype[Any] | None = None,
        copy: bool | None = None,
    ) -> np.ndarray[Any, Any]:
        del dtype, copy
        raise self.error


def returns(
    values: object,
    *,
    labels: tuple[str, ...] = ("asset-a", "asset-b"),
) -> LabeledMatrix:
    array = np.asarray(values)
    return LabeledMatrix(
        array,
        tuple(f"t{i}" for i in range(array.shape[0])),
        labels,
        "time",
        "instrument",
    )


def test_recent_observations_receive_larger_normalized_weights() -> None:
    estimate = EWMACovariance(decay=0.5, demean=True).estimate(
        returns([[-1.0, -2.0], [0.0, 0.0], [1.0, 2.0]])
    )

    variance = 182.0 / 343.0
    np.testing.assert_allclose(
        estimate.covariance.values,
        [[variance, 2 * variance], [2 * variance, 4 * variance]],
        rtol=1e-12,
        atol=1e-12,
    )
    assert estimate.diagnostics[-1].context["decay"] == 0.5
    assert estimate.diagnostics[-1].context["demean"] is True
    assert estimate.diagnostics[-1].context["effective_sample_size"] == pytest.approx(7.0 / 3.0)


def test_no_demean_follows_riskmetrics_zero_mean_definition() -> None:
    estimate = EWMACovariance(decay=0.5, demean=False).estimate(
        returns([[-1.0, -2.0], [0.0, 0.0], [1.0, 2.0]])
    )

    second_moment = 5.0 / 7.0
    np.testing.assert_allclose(
        estimate.covariance.values,
        [[second_moment, 2 * second_moment], [2 * second_moment, 4 * second_moment]],
    )


def test_ewma_uses_common_explicit_missing_policy() -> None:
    estimate = EWMACovariance(
        decay=0.8,
        missing_data_policy=MissingDataPolicy.DROP_OBSERVATION,
    ).estimate(returns([[0.01, 0.02], [np.nan, 0.03], [0.04, 0.05]]))

    assert estimate.observation_count == 2
    assert estimate.diagnostics[0].code == "dropped_missing_observations"


def test_raise_missing_policy_rejects_nan() -> None:
    with pytest.raises(DataValidationError, match="missing returns"):
        EWMACovariance().estimate(returns([[0.01, np.nan], [0.02, 0.03]]))


def test_drop_policy_requires_two_remaining_observations() -> None:
    with pytest.raises(InsufficientHistoryError) as captured:
        EWMACovariance(
            missing_data_policy=MissingDataPolicy.DROP_OBSERVATION,
        ).estimate(returns([[np.nan, 0.01], [0.02, 0.03]]))

    assert captured.value.context == {"observation_count": 1}


def test_drop_policy_rejects_all_missing_history() -> None:
    with pytest.raises(InsufficientHistoryError) as captured:
        EWMACovariance(
            missing_data_policy=MissingDataPolicy.DROP_OBSERVATION,
        ).estimate(returns([[np.nan, 0.01], [0.02, np.nan]]))

    assert captured.value.context == {"observation_count": 0}


@pytest.mark.parametrize("decay", [0.0, -0.1, 1.01, np.nan, np.inf, -np.inf])
def test_decay_must_be_finite_in_open_closed_unit_interval(decay: float) -> None:
    with pytest.raises(DataValidationError, match="decay"):
        EWMACovariance(decay=decay)


@pytest.mark.parametrize(
    ("changes", "field"),
    [
        ({"decay": True}, "decay"),
        ({"decay": np.float64(0.5)}, "decay"),
        ({"decay": FloatSubclass(0.5)}, "decay"),
        ({"demean": 1}, "demean"),
        ({"missing_data_policy": "raise"}, "missing_data_policy"),
        ({"psd_policy": "clip"}, "psd_policy"),
        ({"annualization_factor": True}, "annualization"),
        ({"annualization_factor": np.float64(2.0)}, "annualization"),
        ({"annualization_factor": FloatSubclass(2.0)}, "annualization"),
        ({"annualization_factor": 0}, "annualization"),
        ({"annualization_factor": -1.0}, "annualization"),
        ({"annualization_factor": np.nan}, "annualization"),
        ({"annualization_factor": np.inf}, "annualization"),
        ({"annualization_factor": 10**1000}, "annualization"),
        ({"tolerance": True}, "tolerance"),
        ({"tolerance": np.float64(1e-10)}, "tolerance"),
        ({"tolerance": FloatSubclass(1e-10)}, "tolerance"),
        ({"tolerance": 0.0}, "tolerance"),
        ({"tolerance": -1.0}, "tolerance"),
        ({"tolerance": np.nan}, "tolerance"),
        ({"tolerance": np.inf}, "tolerance"),
        ({"tolerance": 1.01e-2}, "tolerance"),
        ({"max_dimension": True}, "max_dimension"),
        ({"max_dimension": np.int64(2)}, "max_dimension"),
        ({"max_dimension": IntSubclass(2)}, "max_dimension"),
        ({"max_dimension": 0}, "max_dimension"),
        ({"max_dimension": -1}, "max_dimension"),
    ],
)
def test_configuration_requires_exact_safe_runtime_types(
    changes: dict[str, object],
    field: str,
) -> None:
    with pytest.raises(DataValidationError, match=field):
        EWMACovariance(**cast(Any, changes))


def test_builtin_integer_parameters_are_accepted() -> None:
    estimator = EWMACovariance(decay=1, annualization_factor=12, tolerance=1e-3)

    assert estimator.decay == 1
    assert estimator.annualization_factor == 12
    assert estimator.tolerance == 1e-3


def test_defaults_and_structural_protocol_are_explicit() -> None:
    estimator = EWMACovariance()

    assert isinstance(estimator, RiskEstimator)
    assert estimator == EWMACovariance(
        decay=0.94,
        demean=True,
        missing_data_policy=MissingDataPolicy.RAISE,
        psd_policy=PSDPolicy.CLIP,
        annualization_factor=None,
        tolerance=1e-10,
        max_dimension=2048,
    )


def test_docstring_defines_population_moment_without_sample_correction() -> None:
    docstring = (EWMACovariance.__doc__ or "").lower()

    assert "normalized population weighted moment" in docstring
    assert "no bessel" in docstring
    assert "ddof=0" in docstring


def test_decay_one_matches_population_sample_covariance() -> None:
    matrix = returns([[1.0, 2.0], [2.0, 5.0], [4.0, 8.0], [7.0, 9.0]])

    actual = EWMACovariance(decay=1.0).estimate(matrix)
    expected = SampleCovariance(ddof=0, psd_policy=PSDPolicy.CLIP).estimate(matrix)

    np.testing.assert_allclose(actual.covariance.values, expected.covariance.values)


def test_large_integer_offset_is_centered_stably() -> None:
    actual = EWMACovariance(decay=1.0).estimate(
        returns([[2**53], [2**53 - 1], [2**53 - 2]], labels=("large",))
    )

    np.testing.assert_allclose(actual.covariance.values, [[2.0 / 3.0]], rtol=0.0, atol=0.0)


def test_large_float_offset_is_centered_stably() -> None:
    offset = 1e12
    actual = EWMACovariance(decay=0.5).estimate(
        returns([[offset - 2.0, offset - 1.0], [offset, offset + 1.0], [offset + 2.0, offset]])
    )
    centered = np.array(
        [[-20.0 / 7.0, -8.0 / 7.0], [-6.0 / 7.0, 6.0 / 7.0], [8.0 / 7.0, -1.0 / 7.0]]
    )
    weights = np.array([1.0, 2.0, 4.0]) / 7.0
    expected = centered.T @ (centered * weights[:, None])

    np.testing.assert_allclose(actual.covariance.values, expected, rtol=1e-15, atol=1e-15)


def test_nextafter_zero_decay_fails_safely_when_demeaning_collapses() -> None:
    with pytest.raises(InsufficientHistoryError) as captured:
        EWMACovariance(decay=float(np.nextafter(0.0, 1.0))).estimate(
            returns([[0.0], [1.0]], labels=("solo",))
        )

    assert captured.value.context["observation_count"] == 2
    assert float(captured.value.context["effective_sample_size"]) <= (
        1.0 + 16.0 * np.finfo(np.float64).eps
    )


def test_extreme_decay_remains_valid_without_demeaning() -> None:
    estimate = EWMACovariance(
        decay=float(np.nextafter(0.0, 1.0)),
        demean=False,
    ).estimate(returns([[0.0], [2.0]], labels=("solo",)))

    np.testing.assert_allclose(estimate.covariance.values, [[4.0]])
    assert estimate.diagnostics[-1].context["effective_sample_size"] == pytest.approx(1.0)


@pytest.mark.parametrize("scale", [1.0, 1e100])
def test_demeaned_effective_history_rejects_scale_independently(scale: float) -> None:
    with pytest.raises(InsufficientHistoryError) as captured:
        EWMACovariance(decay=1e-16).estimate(returns([[-scale], [scale]], labels=("solo",)))

    assert captured.value.context["observation_count"] == 2
    assert "effective_sample_size" in captured.value.context


def test_effective_history_requires_strictly_more_than_margin_boundary() -> None:
    epsilon = np.finfo(np.float64).eps

    with pytest.raises(InsufficientHistoryError):
        EWMACovariance(decay=float(8.0 * epsilon)).estimate(
            returns([[-1e8], [1e8]], labels=("solo",))
        )


def test_effective_history_accepts_value_above_margin_boundary() -> None:
    epsilon = np.finfo(np.float64).eps

    estimate = EWMACovariance(decay=float(9.0 * epsilon)).estimate(
        returns([[-1e8], [1e8]], labels=("solo",))
    )

    assert float(estimate.diagnostics[-1].context["effective_sample_size"]) > (1.0 + 16.0 * epsilon)


def test_decay_one_reports_observation_count_as_effective_history() -> None:
    estimate = EWMACovariance(decay=1.0).estimate(
        returns([[1.0], [2.0], [4.0], [8.0]], labels=("solo",))
    )

    assert estimate.diagnostics[-1].context["effective_sample_size"] == pytest.approx(4.0)


def test_sqrt_weighted_gram_preserves_extreme_cross_covariance_orientation() -> None:
    estimate = EWMACovariance(
        decay=1e-200,
        demean=False,
        tolerance=1e-300,
    ).estimate(
        returns(
            [[1e200, 1e-124], [0.0, 1.0]],
            labels=("large", "small"),
        )
    )

    covariance = estimate.covariance.values
    assert covariance[0, 0] == pytest.approx(1e200)
    assert covariance[1, 1] == pytest.approx(1.0)
    assert covariance[0, 1] == pytest.approx(1e-124, rel=1e-12, abs=0.0)
    assert covariance[1, 0] == pytest.approx(1e-124, rel=1e-12, abs=0.0)


def test_variable_column_covariance_underflow_is_distinct() -> None:
    with pytest.raises(NumericalStabilityError) as captured:
        EWMACovariance().estimate(returns([[-1e-200], [0.0], [1e-200]], labels=("tiny",)))

    assert captured.value.context["reason"] == "covariance_underflow"
    assert captured.value.context["positions"] == (0,)


def test_constant_column_remains_insufficient_history() -> None:
    with pytest.raises(InsufficientHistoryError):
        EWMACovariance().estimate(returns([[1.0, 1.0], [1.0, 2.0], [1.0, 4.0]]))


def test_weighted_cross_product_overflow_is_structured() -> None:
    with pytest.raises(NumericalStabilityError) as captured:
        EWMACovariance().estimate(returns([[-1e308, -1.0], [0.0, 0.0], [1e308, 1.0]]))

    assert captured.value.context["operation"] == "ewma_covariance"


def test_one_instrument_covariance_stays_two_dimensional() -> None:
    estimate = EWMACovariance(decay=1.0).estimate(returns([[1.0], [2.0], [4.0]], labels=("solo",)))

    assert estimate.covariance.shape == (1, 1)
    assert estimate.labels == ("solo",)


def test_annualization_is_explicit_and_diagnostic_is_complete() -> None:
    matrix = returns([[1.0, 2.0], [2.0, 4.0], [4.0, 9.0]])
    base = EWMACovariance(decay=0.8).estimate(matrix)
    annualized = EWMACovariance(decay=0.8, annualization_factor=12).estimate(matrix)

    np.testing.assert_allclose(annualized.covariance.values, base.covariance.values * 12)
    assert annualized.diagnostics[-1].code == "ewma_covariance"
    assert dict(annualized.diagnostics[-1].context) == {
        "decay": 0.8,
        "demean": True,
        "annualization_factor": 12,
        "effective_sample_size": pytest.approx(2.9047619047619047),
    }


def test_annualization_overflow_is_structured() -> None:
    with pytest.raises(NumericalStabilityError) as captured:
        EWMACovariance(annualization_factor=1e308).estimate(
            returns([[-1e100], [0.0], [1e100]], labels=("solo",))
        )

    assert captured.value.context == {
        "operation": "annualization",
        "reason": "annualization_precision_loss",
    }


def test_annualization_underflow_is_structured() -> None:
    with pytest.raises(NumericalStabilityError) as captured:
        EWMACovariance(
            decay=1.0,
            annualization_factor=float(np.nextafter(0.0, 1.0)),
        ).estimate(returns([[-1.0], [0.0], [1.0]], labels=("solo",)))

    assert captured.value.context == {
        "operation": "annualization",
        "reason": "annualization_precision_loss",
    }


def test_shared_annualization_helper_has_estimator_neutral_errors() -> None:
    from qamr.risk._preparation import _annualize_covariance

    with pytest.raises(NumericalStabilityError, match="covariance annualization") as captured:
        _annualize_covariance(np.array([[0.5]]), float(np.nextafter(0.0, 1.0)))

    assert "sample" not in captured.value.message.lower()
    assert "ewma" not in captured.value.message.lower()


def test_max_dimension_is_checked_before_covariance_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qamr.risk import ewma as ewma_module

    def forbidden(*_args: object, **_kwargs: object) -> np.ndarray[Any, Any]:
        raise AssertionError("quadratic kernel must not run")

    monkeypatch.setattr(ewma_module, "_ewma_covariance", forbidden)
    with pytest.raises(NumericalStabilityError, match="maximum dimension") as captured:
        EWMACovariance(max_dimension=1).estimate(returns([[1.0, 2.0], [2.0, 4.0]]))

    assert captured.value.context == {"dimension": 2, "maximum": 1}


def test_max_dimension_is_checked_before_preparation_or_value_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qamr.risk import ewma as ewma_module

    matrix = returns([[1.0, 2.0], [2.0, 4.0]])

    def forbidden_prepare(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("preparation must not run")

    def forbidden_values(_self: LabeledMatrix) -> np.ndarray[Any, Any]:
        raise AssertionError("values must not be accessed")

    monkeypatch.setattr(ewma_module, "prepare_returns", forbidden_prepare)
    monkeypatch.setattr(LabeledMatrix, "values", property(forbidden_values))

    with pytest.raises(NumericalStabilityError, match="maximum dimension") as captured:
        EWMACovariance(max_dimension=1).estimate(matrix)

    assert captured.value.context == {"dimension": 2, "maximum": 1}


def test_preflight_requires_exact_tuple_column_labels() -> None:
    matrix = returns([[1.0, 2.0], [2.0, 4.0]])
    object.__setattr__(matrix, "column_labels", ["asset-a", "asset-b"])

    with pytest.raises(DataValidationError, match="column_labels") as captured:
        EWMACovariance().estimate(matrix)

    assert captured.value.context["field"] == "column_labels"


def test_labels_diagnostics_and_order_are_preserved() -> None:
    labels = ("zeta", "alpha", "middle")
    estimate = EWMACovariance(decay=1.0).estimate(
        returns(
            [[1.0, 2.0, 4.0], [2.0, 5.0, 2.0], [4.0, 8.0, 1.0]],
            labels=labels,
        )
    )

    assert estimate.labels == labels
    assert estimate.covariance.row_labels == labels
    assert estimate.covariance.column_labels == labels
    assert [item.code for item in estimate.diagnostics] == ["ewma_covariance"]


def test_estimator_and_results_are_immutable_and_pickleable() -> None:
    estimator = EWMACovariance()
    estimate = estimator.estimate(returns([[1.0, 2.0], [2.0, 4.0], [4.0, 9.0]]))

    with pytest.raises(FrozenInstanceError):
        estimator.decay = 0.5  # type: ignore[misc]
    with pytest.raises(ValueError):
        estimate.covariance.values[0, 0] = 0.0
    assert copy.deepcopy(estimator) == estimator
    assert pickle.loads(pickle.dumps(estimator)) == estimator


def test_ordinary_kernel_errors_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qamr.risk import ewma as ewma_module

    def fail(*_args: object, **_kwargs: object) -> np.ndarray[Any, Any]:
        raise CustomBackendError("sensitive detail")

    monkeypatch.setattr(ewma_module, "_ewma_covariance", fail)
    with pytest.raises(NumericalStabilityError) as captured:
        EWMACovariance().estimate(returns([[1.0, 2.0], [2.0, 4.0]]))

    assert captured.value.context == {
        "operation": "ewma_covariance",
        "reason": "CustomBackendError",
    }
    assert "sensitive detail" not in str(captured.value)


def test_backend_memory_error_is_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    from qamr.risk import ewma as ewma_module

    def fail(*_args: object, **_kwargs: object) -> np.ndarray[Any, Any]:
        raise MemoryError

    monkeypatch.setattr(ewma_module, "_ewma_covariance", fail)
    with pytest.raises(MemoryError):
        EWMACovariance().estimate(returns([[1.0, 2.0], [2.0, 4.0]]))


def test_backend_base_exception_is_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    from qamr.risk import ewma as ewma_module

    def fail(*_args: object, **_kwargs: object) -> np.ndarray[Any, Any]:
        raise CustomBackendBaseError

    monkeypatch.setattr(ewma_module, "_ewma_covariance", fail)
    with pytest.raises(CustomBackendBaseError):
        EWMACovariance().estimate(returns([[1.0, 2.0], [2.0, 4.0]]))


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("sensitive runtime detail"),
        CustomBackendError("sensitive custom detail"),
    ],
)
def test_malformed_output_ordinary_errors_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    from qamr.risk import ewma as ewma_module

    monkeypatch.setattr(
        ewma_module,
        "_ewma_covariance",
        lambda *_args, **_kwargs: MalformedCovarianceOutput(error),
    )

    with pytest.raises(NumericalStabilityError) as captured:
        EWMACovariance().estimate(returns([[1.0, 2.0], [2.0, 4.0]]))

    assert captured.value.context == {
        "operation": "ewma_covariance",
        "reason": type(error).__name__,
    }
    assert "sensitive" not in captured.value.message


def test_malformed_output_qamr_error_identity_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qamr.risk import ewma as ewma_module

    error = QAMRError("structured")
    monkeypatch.setattr(
        ewma_module,
        "_ewma_covariance",
        lambda *_args, **_kwargs: MalformedCovarianceOutput(error),
    )

    with pytest.raises(QAMRError) as captured:
        EWMACovariance().estimate(returns([[1.0, 2.0], [2.0, 4.0]]))

    assert captured.value is error


@pytest.mark.parametrize("error", [MemoryError("memory"), KeyboardInterrupt()])
def test_malformed_output_fatal_errors_propagate(
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
) -> None:
    from qamr.risk import ewma as ewma_module

    monkeypatch.setattr(
        ewma_module,
        "_ewma_covariance",
        lambda *_args, **_kwargs: MalformedCovarianceOutput(error),
    )

    with pytest.raises(type(error)) as captured:
        EWMACovariance().estimate(returns([[1.0, 2.0], [2.0, 4.0]]))

    assert captured.value is error


def test_parallel_estimators_do_not_mutate_warning_or_numpy_error_state() -> None:
    warning_filters_before = list(warnings.filters)
    numpy_error_state_before = np.geterr()
    barrier = Barrier(4)

    def calculate(decay: float) -> tuple[float, dict[str, str], dict[str, str]]:
        worker_before = np.geterr()
        barrier.wait()
        estimate = EWMACovariance(decay=decay).estimate(
            returns([[1.0, 2.0], [2.0, 5.0], [4.0, 8.0]])
        )
        return float(estimate.covariance.values[0, 0]), worker_before, np.geterr()

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(calculate, [0.5, 0.7, 0.9, 1.0]))

    assert all(result[0] > 0.0 for result in results)
    assert all(worker_before == worker_after for _, worker_before, worker_after in results)
    assert warnings.filters == warning_filters_before
    assert np.geterr() == numpy_error_state_before
