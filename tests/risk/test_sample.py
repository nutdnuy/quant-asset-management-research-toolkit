import copy
import json
import pickle
import sys
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
from qamr.contracts.results import DiagnosticSeverity, NumericalDiagnostic
from qamr.errors import (
    DataValidationError,
    InsufficientHistoryError,
    NumericalStabilityError,
)
from qamr.risk._preparation import PreparedReturns, prepare_returns
from qamr.risk.estimates import build_covariance_estimate
from qamr.risk.matrices import PSDPolicy
from qamr.risk.sample import SampleCovariance


class IntSubclass(int):
    pass


class FloatSubclass(float):
    pass


class LabeledMatrixSubclass(LabeledMatrix):
    pass


class HostileAxis:
    def __eq__(self, other: object) -> bool:
        del other
        raise AssertionError("axis equality must not be invoked")

    def __ne__(self, other: object) -> bool:
        del other
        raise AssertionError("axis inequality must not be invoked")

    def __str__(self) -> str:
        raise AssertionError("axis string conversion must not be invoked")

    def __repr__(self) -> str:
        raise AssertionError("axis representation must not be invoked")


class HostileStringAxis(str):
    def __eq__(self, other: object) -> bool:
        del other
        raise AssertionError("axis equality must not be invoked")

    def __ne__(self, other: object) -> bool:
        del other
        raise AssertionError("axis inequality must not be invoked")

    def strip(self, chars: str | None = None) -> str:
        del chars
        raise AssertionError("axis strip must not be invoked")


class CustomBackendError(Exception):
    pass


class CustomBackendBaseError(BaseException):
    pass


def returns(
    values: object,
    *,
    labels: tuple[str, ...] = ("asset-a", "asset-b"),
    row_name: str = "time",
    column_name: str = "instrument",
) -> LabeledMatrix:
    array = np.asarray(values)
    return LabeledMatrix(
        array,
        tuple(f"t{i}" for i in range(array.shape[0])),
        labels,
        row_name,
        column_name,
    )


def diagnostic(code: str = "prepared") -> NumericalDiagnostic:
    return NumericalDiagnostic(
        code=code,
        severity=DiagnosticSeverity.INFO,
        message="prepared returns",
    )


def test_sample_covariance_matches_two_asset_hand_calculation() -> None:
    estimator = SampleCovariance(ddof=1, annualization_factor=None)

    estimate = estimator.estimate(returns([[0.01, 0.02], [0.03, 0.01], [0.02, 0.03]]))

    expected = np.array([[0.0001, -0.00005], [-0.00005, 0.0001]])
    np.testing.assert_allclose(estimate.covariance.values, expected)
    np.testing.assert_allclose(
        estimate.correlation.values,
        np.array([[1.0, -0.5], [-0.5, 1.0]]),
    )
    np.testing.assert_allclose(estimate.volatility.values, np.array([0.01, 0.01]))
    assert isinstance(estimator, RiskEstimator)
    assert estimate.labels == ("asset-a", "asset-b")
    assert estimate.observation_count == 3
    assert estimate.diagnostics[-1].code == "sample_covariance"
    assert estimate.covariance.row_name == "instrument"
    assert estimate.covariance.column_name == "instrument"


def test_sample_covariance_ddof_zero_matches_population_hand_calculation() -> None:
    estimate = SampleCovariance(ddof=0).estimate(returns([[1.0, 2.0], [3.0, 4.0], [5.0, 8.0]]))

    expected = np.array(
        [
            [8.0 / 3.0, 4.0],
            [4.0, 56.0 / 9.0],
        ]
    )
    np.testing.assert_allclose(estimate.covariance.values, expected)


def test_sample_covariance_preserves_unit_variance_beyond_float64_contiguous_range() -> None:
    estimate = SampleCovariance().estimate(
        returns(
            [[2**53], [2**53 - 1], [2**53 - 2]],
            labels=("large",),
        )
    )

    np.testing.assert_allclose(estimate.covariance.values, np.array([[1.0]]))


def test_sample_covariance_large_offset_matches_centered_hand_calculation() -> None:
    offset = 1e12
    matrix = returns(
        [
            [offset - 2.0, offset - 1.0],
            [offset, offset + 1.0],
            [offset + 2.0, offset],
        ]
    )

    estimate = SampleCovariance().estimate(matrix)

    np.testing.assert_allclose(
        estimate.covariance.values,
        np.array([[4.0, 1.0], [1.0, 1.0]]),
        rtol=0.0,
        atol=0.0,
    )


def test_private_sample_covariance_matches_canonical_numpy_fixture() -> None:
    from qamr.risk.sample import _sample_covariance

    values = np.array(
        [[0.01, 0.02], [0.03, 0.01], [0.02, 0.03]],
        dtype=np.float64,
    )

    actual = _sample_covariance(values, ddof=1)

    np.testing.assert_allclose(actual, np.cov(values, rowvar=False, ddof=1))


def test_private_sample_covariance_keeps_one_instrument_two_dimensional() -> None:
    from qamr.risk.sample import _sample_covariance

    actual = _sample_covariance(
        np.array([[1.0], [2.0], [4.0]], dtype=np.float64),
        ddof=1,
    )

    assert actual.shape == (1, 1)
    np.testing.assert_allclose(actual, np.array([[7.0 / 3.0]]))


def test_sample_covariance_reports_centered_cross_product_underflow() -> None:
    with pytest.raises(NumericalStabilityError) as captured:
        SampleCovariance().estimate(returns([[-1e-200], [0.0], [1e-200]], labels=("tiny",)))

    assert captured.value.context == {
        "operation": "sample_covariance",
        "reason": "covariance_underflow",
        "positions": (0,),
    }


def test_sample_covariance_reports_underflow_in_one_of_multiple_assets() -> None:
    with pytest.raises(NumericalStabilityError) as captured:
        SampleCovariance().estimate(
            returns(
                [[-1e-200, -1.0], [0.0, 0.0], [1e-200, 1.0]],
                labels=("tiny", "normal"),
            )
        )

    assert captured.value.context["reason"] == "covariance_underflow"
    assert captured.value.context["positions"] == (0,)


def test_constant_column_remains_insufficient_history_not_underflow() -> None:
    with pytest.raises(InsufficientHistoryError):
        SampleCovariance(psd_policy=PSDPolicy.CLIP).estimate(
            returns([[1.0, 1.0], [1.0, 2.0], [1.0, 4.0]])
        )


def test_sample_covariance_kernel_uses_vectorized_reduction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qamr.risk import sample as sample_module

    def forbidden(*_args: object, **_kwargs: object) -> float:
        raise AssertionError("per-column Python compensated loops are forbidden")

    monkeypatch.setattr(sample_module.math, "fsum", forbidden)
    offsets = np.arange(64, dtype=np.float64)
    values = np.vstack((1e12 + offsets, 1e12 + offsets + 1.0, 1e12 + offsets + 2.0))

    covariance = sample_module._sample_covariance(values, ddof=1)

    np.testing.assert_allclose(np.diag(covariance), np.ones(64))


def test_sample_covariance_centers_anchor_deltas_in_place() -> None:
    from qamr.risk.sample import _sample_covariance

    operations: list[str] = []

    class TrackedArray(np.ndarray[Any, np.dtype[np.float64]]):
        def __sub__(self, other: object) -> Any:
            operations.append("subtract")
            return super().__sub__(other)

        def __isub__(self, other: object) -> Any:
            operations.append("in_place_subtract")
            return super().__isub__(other)

    values = np.array(
        [[1e12, 1e12 + 1.0], [1e12 + 1.0, 1e12 + 3.0], [1e12 + 2.0, 1e12 + 5.0]],
        dtype=np.float64,
    ).view(TrackedArray)

    covariance = _sample_covariance(values, ddof=1)

    assert operations == ["subtract", "in_place_subtract"]
    np.testing.assert_allclose(covariance, np.array([[1.0, 2.0], [2.0, 4.0]]))


def test_annualization_is_explicit_not_frequency_derived() -> None:
    matrix = returns([[0.01, 0.02], [0.03, 0.01], [0.02, 0.03]])
    base = SampleCovariance().estimate(matrix)
    annualized = SampleCovariance(annualization_factor=12.0).estimate(matrix)

    np.testing.assert_allclose(
        annualized.covariance.values,
        base.covariance.values * 12.0,
    )
    np.testing.assert_allclose(
        annualized.correlation.values,
        base.correlation.values,
    )
    np.testing.assert_allclose(
        annualized.volatility.values,
        base.volatility.values * np.sqrt(12.0),
    )


def test_raise_missing_policy_does_not_impute_nan() -> None:
    with pytest.raises(DataValidationError, match="missing returns") as captured:
        SampleCovariance(
            missing_data_policy=MissingDataPolicy.RAISE,
        ).estimate(returns([[0.01, np.nan], [0.02, 0.03]]))

    assert captured.value.context == {"missing_count": 1, "policy": "raise"}


def test_drop_observation_policy_reports_dropped_rows_once_per_row() -> None:
    estimate = SampleCovariance(
        missing_data_policy=MissingDataPolicy.DROP_OBSERVATION,
    ).estimate(
        returns(
            [
                [0.01, 0.01],
                [np.nan, np.nan],
                [0.03, 0.03],
                [0.02, 0.02],
            ]
        )
    )

    assert estimate.observation_count == 3
    assert estimate.diagnostics[0].code == "dropped_missing_observations"
    assert estimate.diagnostics[0].context["dropped_count"] == 1


def test_ddof_requires_more_complete_observations() -> None:
    with pytest.raises(InsufficientHistoryError, match="sample covariance") as captured:
        SampleCovariance(ddof=1).estimate(returns([[0.01, 0.02]]))

    assert captured.value.context == {"observation_count": 1, "ddof": 1}


def test_constant_instrument_returns_raise_structured_error() -> None:
    with pytest.raises(
        InsufficientHistoryError,
        match="positive variance",
    ) as captured:
        SampleCovariance(psd_policy=PSDPolicy.CLIP).estimate(
            returns([[0.01, 0.02], [0.01, 0.03], [0.01, 0.04]])
        )

    assert captured.value.context == {"positions": (0,)}


def test_one_instrument_returns_a_two_dimensional_covariance() -> None:
    estimate = SampleCovariance().estimate(returns([[1.0], [2.0], [4.0]], labels=("solo",)))

    assert estimate.covariance.shape == (1, 1)
    np.testing.assert_allclose(estimate.covariance.values, np.array([[7.0 / 3.0]]))
    assert estimate.labels == ("solo",)


def test_labels_and_instrument_order_are_preserved() -> None:
    labels = ("zeta", "alpha", "middle")
    estimate = SampleCovariance().estimate(
        returns(
            [[1.0, 1.0, 4.0], [2.0, 3.0, 2.0], [4.0, 8.0, 1.0]],
            labels=labels,
        )
    )

    assert estimate.labels == labels
    assert estimate.covariance.row_labels == labels
    assert estimate.covariance.column_labels == labels
    np.testing.assert_allclose(
        estimate.covariance.values,
        np.cov(np.array([[1.0, 1.0, 4.0], [2.0, 3.0, 2.0], [4.0, 8.0, 1.0]]), rowvar=False),
    )


def test_prepare_returns_drops_rows_in_original_order() -> None:
    prepared = prepare_returns(
        returns([[3.0, 30.0], [np.nan, 20.0], [1.0, 10.0], [2.0, 20.0]]),
        MissingDataPolicy.DROP_OBSERVATION,
    )

    np.testing.assert_array_equal(
        prepared.values,
        np.array([[3.0, 30.0], [1.0, 10.0], [2.0, 20.0]]),
    )


def test_drop_all_rows_becomes_insufficient_history() -> None:
    with pytest.raises(InsufficientHistoryError, match="sample covariance") as captured:
        SampleCovariance(
            ddof=0,
            missing_data_policy=MissingDataPolicy.DROP_OBSERVATION,
        ).estimate(returns([[np.nan, 1.0], [2.0, np.nan]]))

    assert captured.value.context == {"observation_count": 0, "ddof": 0}


@pytest.mark.parametrize(
    ("values", "match"),
    [
        (np.array([[True, False], [False, True]]), "real numeric"),
        (np.array([[1 + 2j, 3 + 0j], [2 + 0j, 4 + 0j]]), "real numeric"),
        (np.array([["1", "2"], ["3", "4"]]), "real numeric"),
        (np.array([[object(), object()], [object(), object()]], dtype=object), "real numeric"),
        (np.array([[np.datetime64("2026-01-01")]], dtype="datetime64[D]"), "real numeric"),
        (np.array([[np.timedelta64(1, "D")]], dtype="timedelta64[D]"), "real numeric"),
        (np.array([[1.0, np.inf], [2.0, 3.0]]), "infinity"),
        (np.array([[1.0, -np.inf], [2.0, 3.0]]), "infinity"),
    ],
)
def test_prepare_returns_rejects_non_real_or_infinite_values(
    values: np.ndarray[Any],
    match: str,
) -> None:
    labels = tuple(f"asset-{position}" for position in range(values.shape[1]))

    with pytest.raises(DataValidationError, match=match):
        prepare_returns(returns(values, labels=labels), MissingDataPolicy.RAISE)


def test_prepare_returns_preserves_signed_zero_and_nan_until_policy() -> None:
    prepared = prepare_returns(
        returns([[-0.0, 1.0], [2.0, np.nan]]),
        MissingDataPolicy.DROP_OBSERVATION,
    )

    assert np.signbit(prepared.values[0, 0])
    assert prepared.observation_count == 1


def test_prepare_returns_rejects_integer_values_not_exact_in_float64() -> None:
    values = np.array([[2**53 + 1], [2**53 + 3]], dtype=np.int64)

    with pytest.raises(DataValidationError, match="exactly representable") as captured:
        prepare_returns(returns(values, labels=("large",)), MissingDataPolicy.RAISE)

    assert captured.value.context["reason"] == "not_exactly_representable"


def test_prepare_returns_accepts_exact_large_integer_values() -> None:
    values = np.array([[2**53], [-(2**53)]], dtype=np.int64)

    prepared = prepare_returns(
        returns(values, labels=("large",)),
        MissingDataPolicy.RAISE,
    )

    np.testing.assert_array_equal(
        prepared.values,
        np.array([[float(2**53)], [-float(2**53)]], dtype=np.float64),
    )


@pytest.mark.parametrize(
    "values",
    [
        np.array([[2**53 + 2], [-(2**63)]], dtype=np.int64),
        np.array([[2**63], [2**53 + 2]], dtype=np.uint64),
    ],
)
def test_prepare_returns_accepts_all_exactly_representable_integer_values(
    values: np.ndarray[Any],
) -> None:
    prepared = prepare_returns(
        returns(values, labels=("large",)),
        MissingDataPolicy.RAISE,
    )

    for source, converted in zip(values.flat, prepared.values.flat, strict=True):
        assert int(converted) == int(source)


def test_prepare_returns_rejects_nonrepresentable_uint64_max() -> None:
    values = np.array([[np.iinfo(np.uint64).max]], dtype=np.uint64)

    with pytest.raises(DataValidationError, match="exactly representable"):
        prepare_returns(
            returns(values, labels=("large",)),
            MissingDataPolicy.RAISE,
        )


def test_prepare_returns_is_deeply_isolated_from_input_and_export_mutation() -> None:
    source = np.array([[1.0, 2.0], [3.0, 5.0]])
    matrix = returns(source)
    prepared = prepare_returns(matrix, MissingDataPolicy.RAISE)
    source[0, 0] = 99.0
    first = prepared.values
    assert not first.flags.writeable
    with pytest.raises(ValueError):
        first[0, 0] = 77.0
    first.setflags(write=True)
    first[0, 0] = 88.0

    np.testing.assert_array_equal(prepared.values, np.array([[1.0, 2.0], [3.0, 5.0]]))


def test_prepared_returns_internal_computation_reuses_owned_read_only_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qamr.risk import sample as sample_module

    prepared = prepare_returns(
        returns([[1.0, 2.0], [3.0, 5.0], [4.0, 8.0]]),
        MissingDataPolicy.RAISE,
    )
    internal = prepared._computation_values()
    public = prepared.values
    captured: list[np.ndarray[Any]] = []

    def kernel(values: np.ndarray[Any], ddof: int) -> np.ndarray[Any]:
        del ddof
        captured.append(values)
        return np.array([[7.0 / 3.0, 4.5], [4.5, 9.0]])

    monkeypatch.setattr(sample_module, "_sample_covariance", kernel)

    SampleCovariance().estimate(returns([[1.0, 2.0], [3.0, 5.0], [4.0, 8.0]]))

    assert not internal.flags.writeable
    assert not internal.flags.owndata
    assert not public.flags.writeable
    assert not np.shares_memory(public, internal)
    assert not captured[0].flags.writeable
    assert np.shares_memory(internal, prepared._computation_values())
    with pytest.raises(ValueError):
        internal.setflags(write=True)


def test_prepared_returns_is_frozen_slotted_copyable_and_pickleable() -> None:
    prepared = prepare_returns(
        returns([[1.0, 2.0], [3.0, 5.0]]),
        MissingDataPolicy.RAISE,
    )

    assert not hasattr(prepared, "__dict__")
    with pytest.raises(FrozenInstanceError):
        prepared.observation_count = 99  # type: ignore[misc]
    for restored in (copy.deepcopy(prepared), pickle.loads(pickle.dumps(prepared))):
        assert isinstance(restored, PreparedReturns)
        assert restored.observation_count == 2
        assert restored.diagnostics == ()
        np.testing.assert_array_equal(restored.values, prepared.values)
        assert not restored.values.flags.writeable


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"values": np.array([1.0, 2.0])}, "two-dimensional"),
        ({"values": np.array([[1, 2]], dtype=np.int64)}, "float64"),
        ({"values": np.array([[1.0, np.nan]])}, "finite"),
        ({"observation_count": True}, "observation_count"),
        ({"observation_count": 1}, "observation_count"),
        ({"diagnostics": []}, "diagnostics"),
        ({"diagnostics": (object(),)}, "diagnostics"),
    ],
)
def test_prepared_returns_constructor_enforces_value_invariants(
    changes: dict[str, object],
    match: str,
) -> None:
    arguments: dict[str, object] = {
        "values": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        "observation_count": 2,
        "diagnostics": (),
    }
    arguments.update(changes)

    with pytest.raises(DataValidationError, match=match):
        PreparedReturns(**arguments)  # type: ignore[arg-type]


def test_prepare_returns_rejects_empty_instrument_universe() -> None:
    matrix = LabeledMatrix(
        np.empty((3, 0)),
        ("t0", "t1", "t2"),
        (),
        "time",
        "instrument",
    )

    with pytest.raises(DataValidationError, match="non-empty instrument"):
        prepare_returns(matrix, MissingDataPolicy.RAISE)


@pytest.mark.parametrize(
    ("returns_value", "policy"),
    [
        (np.array([[1.0, 2.0]]), MissingDataPolicy.RAISE.value),
        (
            LabeledMatrixSubclass(
                np.array([[1.0, 2.0]]),
                ("t0",),
                ("a", "b"),
                "time",
                "instrument",
            ),
            MissingDataPolicy.RAISE,
        ),
    ],
)
def test_prepare_returns_requires_exact_runtime_types(
    returns_value: object,
    policy: object,
) -> None:
    with pytest.raises(DataValidationError, match="exact"):
        prepare_returns(returns_value, policy)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("row_name", "column_name"),
    [("date", "instrument"), ("time", "asset")],
)
def test_prepare_returns_requires_time_by_instrument_axes(
    row_name: str,
    column_name: str,
) -> None:
    with pytest.raises(DataValidationError, match="time-by-instrument"):
        prepare_returns(
            returns(
                [[1.0, 2.0], [2.0, 3.0]],
                row_name=row_name,
                column_name=column_name,
            ),
            MissingDataPolicy.RAISE,
        )


def test_prepare_returns_rejects_nonstring_axis_names_safely() -> None:
    matrix = returns([[1.0, 2.0], [2.0, 3.0]])
    object.__setattr__(matrix, "row_name", 42)

    with pytest.raises(DataValidationError, match="time-by-instrument"):
        prepare_returns(matrix, MissingDataPolicy.RAISE)


@pytest.mark.parametrize("field", ["row_name", "column_name"])
@pytest.mark.parametrize("axis", [HostileAxis(), HostileStringAxis("time")])
def test_prepare_returns_rejects_hostile_axis_metadata_without_invoking_it(
    field: str,
    axis: object,
) -> None:
    matrix = returns([[1.0, 2.0], [2.0, 3.0]])
    object.__setattr__(matrix, field, axis)

    with pytest.raises(DataValidationError, match="time-by-instrument") as captured:
        prepare_returns(matrix, MissingDataPolicy.RAISE)

    assert captured.value.context[field] == type(axis).__name__


def test_prepare_returns_rejects_compromised_value_shape_safely() -> None:
    matrix = returns([[1.0, 2.0], [2.0, 3.0]])
    object.__setattr__(matrix, "_values", np.array([1.0, 2.0]))

    with pytest.raises(DataValidationError, match="two-dimensional"):
        prepare_returns(matrix, MissingDataPolicy.RAISE)


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"ddof": True}, "ddof"),
        ({"ddof": np.int64(1)}, "ddof"),
        ({"ddof": IntSubclass(1)}, "ddof"),
        ({"ddof": -1}, "ddof"),
        ({"missing_data_policy": "raise"}, "missing_data_policy"),
        ({"psd_policy": "raise"}, "psd_policy"),
        ({"annualization_factor": True}, "annualization"),
        ({"annualization_factor": np.float64(2)}, "annualization"),
        ({"annualization_factor": FloatSubclass(2)}, "annualization"),
        ({"annualization_factor": 0}, "annualization"),
        ({"annualization_factor": -1.0}, "annualization"),
        ({"annualization_factor": np.inf}, "annualization"),
        ({"annualization_factor": np.nan}, "annualization"),
        ({"annualization_factor": 10**1000}, "annualization"),
        ({"tolerance": True}, "tolerance"),
        ({"tolerance": np.float64(1e-10)}, "tolerance"),
        ({"tolerance": FloatSubclass(1e-10)}, "tolerance"),
        ({"tolerance": 0.0}, "tolerance"),
        ({"tolerance": -1e-10}, "tolerance"),
        ({"tolerance": np.inf}, "tolerance"),
        ({"tolerance": 10**1000}, "tolerance"),
        ({"tolerance": 1.00001e-2}, "tolerance"),
        ({"max_dimension": True}, "max_dimension"),
        ({"max_dimension": np.int64(2)}, "max_dimension"),
        ({"max_dimension": IntSubclass(2)}, "max_dimension"),
        ({"max_dimension": 0}, "max_dimension"),
    ],
)
def test_sample_covariance_rejects_invalid_configuration(
    changes: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(DataValidationError, match=match):
        SampleCovariance(**changes)  # type: ignore[arg-type]


@pytest.mark.parametrize("tolerance", [1e-300, 1e-10, 1e-2])
def test_sample_covariance_accepts_positive_tolerance_boundaries(
    tolerance: float,
) -> None:
    assert SampleCovariance(tolerance=tolerance).tolerance == tolerance


def test_sample_covariance_accepts_exact_builtin_integer_annualization() -> None:
    estimator = SampleCovariance(annualization_factor=12)

    assert estimator.annualization_factor == 12


def test_sample_covariance_accepts_sys_maxsize_ddof() -> None:
    assert SampleCovariance(ddof=sys.maxsize).ddof == sys.maxsize


def test_sample_covariance_rejects_ddof_above_sys_maxsize_with_json_safe_context() -> None:
    for invalid in (sys.maxsize + 1, 10**5000):
        with pytest.raises(DataValidationError, match="ddof") as captured:
            SampleCovariance(ddof=invalid)

        assert captured.value.context == {
            "field": "ddof",
            "dtype": "int",
            "reason": "too_large",
        }
        json.dumps(captured.value.as_dict(), allow_nan=False)


def test_sample_covariance_rejects_dimension_before_covariance_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qamr.risk import sample as sample_module

    def forbidden(*_args: object, **_kwargs: object) -> np.ndarray[Any]:
        raise AssertionError("covariance kernel must not run")

    monkeypatch.setattr(sample_module, "_sample_covariance", forbidden)

    with pytest.raises(NumericalStabilityError, match="maximum dimension") as captured:
        SampleCovariance(max_dimension=1).estimate(returns([[1.0, 2.0], [2.0, 4.0], [4.0, 9.0]]))

    assert captured.value.context == {"dimension": 2, "maximum": 1}


def test_sample_covariance_is_frozen_slotted_and_has_semantic_value_behavior() -> None:
    estimator = SampleCovariance()
    equal = SampleCovariance()

    assert not hasattr(estimator, "__dict__")
    assert estimator == equal
    assert hash(estimator) == hash(equal)
    assert pickle.loads(pickle.dumps(estimator)) == estimator
    with pytest.raises(FrozenInstanceError):
        estimator.ddof = 0  # type: ignore[misc]


def test_estimate_requires_exact_labeled_matrix() -> None:
    subclass = LabeledMatrixSubclass(
        np.array([[1.0, 2.0], [2.0, 4.0]]),
        ("t0", "t1"),
        ("a", "b"),
        "time",
        "instrument",
    )

    with pytest.raises(DataValidationError, match="exact LabeledMatrix"):
        SampleCovariance().estimate(subclass)


def test_annualization_overflow_is_structured() -> None:
    with pytest.raises(NumericalStabilityError, match="annualization") as captured:
        SampleCovariance(annualization_factor=1e308).estimate(
            returns([[-1.0, -1.0], [0.0, 0.0], [1.0, 2.0]])
        )

    assert captured.value.context == {
        "operation": "annualization",
        "reason": "annualization_precision_loss",
    }


def test_annualization_underflow_of_positive_variance_is_structured() -> None:
    smallest_positive = float(np.nextafter(0.0, 1.0))

    with pytest.raises(NumericalStabilityError, match="annualization") as captured:
        SampleCovariance(annualization_factor=smallest_positive).estimate(
            returns([[-1e-150], [0.0], [1e-150]], labels=("tiny",))
        )

    assert captured.value.context == {
        "operation": "annualization",
        "reason": "annualization_precision_loss",
    }


@pytest.mark.parametrize("correlation", [1.0 / 3.0, 1e-4])
def test_annualization_detects_subnormal_correlation_precision_loss(
    correlation: float,
) -> None:
    from qamr.risk.sample import _annualize_covariance

    covariance = np.array(
        [[1.0, correlation], [correlation, 1.0]],
        dtype=np.float64,
    )

    with pytest.raises(NumericalStabilityError) as captured:
        _annualize_covariance(covariance, 1e-320)

    assert captured.value.context == {
        "operation": "annualization",
        "reason": "annualization_precision_loss",
    }


def test_annualization_accepts_exact_subnormal_round_trip() -> None:
    from qamr.risk.sample import _annualize_covariance

    covariance = np.array(
        [[1.0, 0.5], [0.5, 1.0]],
        dtype=np.float64,
    )

    scaled = _annualize_covariance(covariance, 1e-320)

    assert np.all(scaled != 0.0)
    np.testing.assert_array_equal(scaled / 1e-320, covariance)


def test_exact_subnormal_annualization_below_result_resolution_is_structured() -> None:
    with pytest.raises(NumericalStabilityError) as captured:
        SampleCovariance(
            annualization_factor=1e-320,
            tolerance=5e-324,
        ).estimate(
            returns(
                [[-1.0, 0.0], [0.0, -1.0], [1.0, 1.0]],
                labels=("a", "b"),
            )
        )

    assert captured.value.context == {
        "reason": "volatility_below_supported_resolution",
        "positions": (0, 1),
    }


def test_natural_tiny_covariance_below_result_resolution_is_structured() -> None:
    with pytest.raises(NumericalStabilityError) as captured:
        SampleCovariance(tolerance=5e-324).estimate(
            returns([[-1e-14], [0.0], [1e-14]], labels=("tiny",))
        )

    assert captured.value.context == {
        "reason": "volatility_below_supported_resolution",
        "positions": (0,),
    }


def test_variance_at_user_tolerance_remains_insufficient_history() -> None:
    with pytest.raises(InsufficientHistoryError) as captured:
        build_covariance_estimate(
            np.array([[1e-28]]),
            ("tiny",),
            observation_count=3,
            diagnostics=(),
            psd_policy=PSDPolicy.RAISE,
            tolerance=1e-28,
        )

    assert captured.value.context == {"positions": (0,)}


def test_normal_scale_covariance_remains_supported_at_tiny_user_tolerance() -> None:
    estimate = SampleCovariance(tolerance=5e-324).estimate(
        returns([[-1.0], [0.0], [1.0]], labels=("normal",))
    )

    np.testing.assert_array_equal(estimate.volatility.values, np.array([1.0]))


def test_annualization_preserves_exact_zero_and_representable_subnormal_cells() -> None:
    from qamr.risk.sample import _annualize_covariance

    scaled = _annualize_covariance(np.eye(2), 1e-320)

    assert scaled[0, 1] == 0.0
    assert scaled[1, 0] == 0.0
    assert 0.0 < scaled[0, 0] < np.finfo(np.float64).tiny
    assert 0.0 < scaled[1, 1] < np.finfo(np.float64).tiny


def test_annualization_normal_factor_round_trips_with_tight_precision() -> None:
    from qamr.risk.sample import _annualize_covariance

    covariance = np.array([[4.0, -0.125], [-0.125, 9.0]])

    scaled = _annualize_covariance(covariance, 12.0)

    np.testing.assert_allclose(scaled, covariance * 12.0, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "malformed",
    [
        np.array([1.0, 2.0]),
        np.ones((2, 2, 1)),
        np.array([[1.0, np.nan], [np.nan, 1.0]]),
        np.array([["x", "y"], ["z", "w"]]),
    ],
)
def test_malformed_or_nonfinite_covariance_backend_result_is_structured(
    monkeypatch: pytest.MonkeyPatch,
    malformed: np.ndarray[Any],
) -> None:
    from qamr.risk import sample as sample_module

    monkeypatch.setattr(
        sample_module,
        "_sample_covariance",
        lambda *_args, **_kwargs: malformed,
    )

    with pytest.raises(NumericalStabilityError, match="sample covariance"):
        SampleCovariance().estimate(returns([[1.0, 2.0], [2.0, 4.0], [4.0, 9.0]]))


def test_covariance_backend_error_is_structured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qamr.risk import sample as sample_module

    def fail(*_args: object, **_kwargs: object) -> np.ndarray[Any]:
        raise ValueError("unbounded backend detail")

    monkeypatch.setattr(sample_module, "_sample_covariance", fail)

    with pytest.raises(NumericalStabilityError, match="sample covariance") as captured:
        SampleCovariance().estimate(returns([[1.0, 2.0], [2.0, 4.0], [4.0, 9.0]]))

    assert captured.value.context == {
        "operation": "sample_covariance",
        "reason": "ValueError",
    }
    assert "unbounded backend detail" not in str(captured.value)


def test_covariance_kernel_structured_error_identity_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qamr.risk import sample as sample_module

    error = NumericalStabilityError(
        "kernel precision failure",
        context={
            "operation": "sample_covariance",
            "reason": "covariance_underflow",
        },
    )

    def fail(*_args: object, **_kwargs: object) -> np.ndarray[Any]:
        raise error

    monkeypatch.setattr(sample_module, "_sample_covariance", fail)

    with pytest.raises(NumericalStabilityError) as captured:
        SampleCovariance().estimate(returns([[1.0, 2.0], [2.0, 4.0], [4.0, 9.0]]))

    assert captured.value is error


@pytest.mark.parametrize("error", [RuntimeError("private"), CustomBackendError("private")])
def test_covariance_backend_ordinary_exceptions_are_structured(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    from qamr.risk import sample as sample_module

    def fail(*_args: object, **_kwargs: object) -> np.ndarray[Any]:
        raise error

    monkeypatch.setattr(sample_module, "_sample_covariance", fail)

    with pytest.raises(NumericalStabilityError) as captured:
        SampleCovariance().estimate(returns([[1.0, 2.0], [2.0, 4.0], [4.0, 9.0]]))

    assert captured.value.context == {
        "operation": "sample_covariance",
        "reason": type(error).__name__,
    }
    assert "private" not in str(captured.value)


@pytest.mark.parametrize("error", [MemoryError("memory"), CustomBackendBaseError("base")])
def test_covariance_backend_does_not_mask_fatal_exceptions(
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
) -> None:
    from qamr.risk import sample as sample_module

    def fail(*_args: object, **_kwargs: object) -> np.ndarray[Any]:
        raise error

    monkeypatch.setattr(sample_module, "_sample_covariance", fail)

    with pytest.raises(type(error)) as captured:
        SampleCovariance().estimate(returns([[1.0, 2.0], [2.0, 4.0], [4.0, 9.0]]))

    assert captured.value is error


def test_parallel_estimates_do_not_mutate_warning_filters_or_diverge() -> None:
    matrix = returns([[1.0, 2.0], [2.0, 5.0], [4.0, 8.0], [8.0, 16.0]])
    estimator = SampleCovariance()
    before_filters = list(warnings.filters)
    barrier = Barrier(2)

    def run() -> list[np.ndarray[Any]]:
        barrier.wait()
        return [estimator.estimate(matrix).covariance.values for _ in range(40)]

    with ThreadPoolExecutor(max_workers=2) as executor:
        left_future = executor.submit(run)
        right_future = executor.submit(run)
        left = left_future.result()
        right = right_future.result()

    assert warnings.filters == before_filters
    for left_values, right_values in zip(left, right, strict=True):
        np.testing.assert_array_equal(left_values, right_values)


def test_estimate_does_not_mutate_input() -> None:
    source = np.array([[1.0, 2.0], [2.0, 5.0], [4.0, 8.0]])
    matrix = returns(source)
    before = matrix.values

    SampleCovariance().estimate(matrix)

    np.testing.assert_array_equal(source, before)
    np.testing.assert_array_equal(matrix.values, before)


def test_builder_constructs_complete_labelled_estimate_without_mutation() -> None:
    values = np.array([[4.0, 1.0], [1.0, 9.0]])
    before = values.copy()
    prepared_diagnostic = diagnostic()

    estimate = build_covariance_estimate(
        values,
        ("b", "a"),
        observation_count=5,
        diagnostics=(prepared_diagnostic,),
        psd_policy=PSDPolicy.RAISE,
        tolerance=1e-10,
    )

    np.testing.assert_array_equal(values, before)
    assert estimate.labels == ("b", "a")
    assert estimate.observation_count == 5
    assert estimate.diagnostics == (prepared_diagnostic,)
    assert estimate.covariance.row_name == "instrument"
    np.testing.assert_allclose(estimate.volatility.values, np.array([2.0, 3.0]))
    np.testing.assert_allclose(
        estimate.correlation.values,
        np.array([[1.0, 1.0 / 6.0], [1.0 / 6.0, 1.0]]),
    )


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"covariance_values": [[1.0]]}, DataValidationError),
        ({"labels": ["a"]}, DataValidationError),
        ({"observation_count": True}, DataValidationError),
        ({"observation_count": 0}, DataValidationError),
        ({"diagnostics": []}, DataValidationError),
        ({"diagnostics": (object(),)}, DataValidationError),
        ({"psd_policy": "raise"}, NumericalStabilityError),
        ({"tolerance": True}, NumericalStabilityError),
        ({"tolerance": 0.0}, NumericalStabilityError),
        ({"tolerance": 1.1e-2}, NumericalStabilityError),
        ({"tolerance": 10**1000}, NumericalStabilityError),
        ({"max_dimension": True}, NumericalStabilityError),
        ({"max_dimension": 0}, NumericalStabilityError),
    ],
)
def test_builder_rejects_invalid_runtime_contracts(
    changes: dict[str, object],
    error: type[Exception],
) -> None:
    arguments: dict[str, object] = {
        "covariance_values": np.array([[1.0]]),
        "labels": ("a",),
        "observation_count": 3,
        "diagnostics": (),
        "psd_policy": PSDPolicy.RAISE,
        "tolerance": 1e-10,
        "max_dimension": 2048,
    }
    arguments.update(changes)

    with pytest.raises(error):
        build_covariance_estimate(**arguments)  # type: ignore[arg-type]


def test_builder_propagates_unrelated_correlation_numerical_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qamr.risk import estimates as estimates_module

    error = NumericalStabilityError(
        "covariance could not be scaled safely",
        context={"operation": "covariance_to_correlation", "reason": "forced"},
    )

    def fail(*_args: object, **_kwargs: object) -> Any:
        raise error

    monkeypatch.setattr(estimates_module, "covariance_to_correlation", fail)

    with pytest.raises(NumericalStabilityError, match="scaled safely") as captured:
        build_covariance_estimate(
            np.eye(2),
            ("a", "b"),
            observation_count=3,
            diagnostics=(),
            psd_policy=PSDPolicy.RAISE,
            tolerance=1e-10,
        )

    assert captured.value is error


def test_builder_rejects_dimension_before_labelled_copy_or_psd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qamr.risk import estimates as estimates_module

    def forbidden(*_args: object, **_kwargs: object) -> Any:
        raise AssertionError("expensive covariance construction must not run")

    monkeypatch.setattr(estimates_module, "LabeledMatrix", forbidden)
    monkeypatch.setattr(estimates_module, "apply_psd_policy", forbidden)

    with pytest.raises(NumericalStabilityError, match="maximum dimension") as captured:
        build_covariance_estimate(
            np.eye(3),
            ("a", "b", "c"),
            observation_count=3,
            diagnostics=(),
            psd_policy=PSDPolicy.RAISE,
            tolerance=1e-10,
            max_dimension=2,
        )

    assert captured.value.context == {"dimension": 3, "maximum": 2}


def test_builder_positive_variance_context_is_bounded() -> None:
    labels = tuple(f"asset-{position}" for position in range(40))

    with pytest.raises(InsufficientHistoryError) as captured:
        build_covariance_estimate(
            np.zeros((40, 40)),
            labels,
            observation_count=3,
            diagnostics=(),
            psd_policy=PSDPolicy.CLIP,
            tolerance=1e-10,
        )

    assert len(cast(tuple[int, ...], captured.value.context["positions"])) == 32
    assert captured.value.context["position_count"] == 40
    assert captured.value.context["positions_truncated"] is True
