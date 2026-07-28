import copy
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
    LabelAlignmentError,
    NumericalStabilityError,
)
from qamr.risk.estimates import CovarianceEstimate, build_covariance_estimate
from qamr.risk.matrices import PSDPolicy
from qamr.risk.sample import SampleCovariance
from qamr.risk.shrinkage import ShrinkageCovariance, ShrinkageTarget


class IntSubclass(int):
    pass


class FloatSubclass(float):
    pass


class LabeledMatrixSubclass(LabeledMatrix):
    pass


class CovarianceEstimateSubclass(CovarianceEstimate):
    pass


class PSDPolicySubclass(str):
    pass


class CustomBackendError(Exception):
    pass


class CustomBackendBaseError(BaseException):
    pass


def returns(
    values: object | None = None,
    *,
    labels: tuple[str, ...] = ("a", "b", "c"),
) -> LabeledMatrix:
    array = np.asarray(
        values
        if values is not None
        else [
            [0.01, 0.04, 0.02],
            [0.03, 0.01, 0.01],
            [0.02, 0.05, 0.04],
            [0.04, 0.02, 0.03],
        ]
    )
    return LabeledMatrix(
        array,
        tuple(f"t{position}" for position in range(array.shape[0])),
        labels,
        "time",
        "instrument",
    )


def fixed_estimate(
    covariance: np.ndarray[Any, Any],
    *,
    labels: tuple[str, ...] = ("a", "b"),
    observation_count: int = 7,
    diagnostics: tuple[NumericalDiagnostic, ...] = (),
    psd_policy: PSDPolicy = PSDPolicy.RAISE,
) -> CovarianceEstimate:
    return build_covariance_estimate(
        np.asarray(covariance, dtype=np.float64),
        labels,
        observation_count=observation_count,
        diagnostics=diagnostics,
        psd_policy=psd_policy,
        tolerance=1e-10,
    )


class FixedEstimator:
    def __init__(self, estimate: CovarianceEstimate) -> None:
        self.result = estimate
        self.calls: list[LabeledMatrix] = []

    def estimate(self, matrix: LabeledMatrix) -> CovarianceEstimate:
        self.calls.append(matrix)
        return self.result


def test_zero_intensity_returns_base_covariance_exactly() -> None:
    matrix = returns()
    base = SampleCovariance(psd_policy=PSDPolicy.RAISE).estimate(matrix)
    shrunk = ShrinkageCovariance(
        shrinkage=0.0,
        base_estimator=SampleCovariance(psd_policy=PSDPolicy.RAISE),
        psd_policy=PSDPolicy.RAISE,
    ).estimate(matrix)

    np.testing.assert_array_equal(shrunk.covariance.values, base.covariance.values)
    assert shrunk.labels == ("a", "b", "c")


def test_full_diagonal_shrinkage_removes_cross_covariance() -> None:
    shrunk = ShrinkageCovariance(
        shrinkage=1.0,
        target=ShrinkageTarget.DIAGONAL,
    ).estimate(returns())

    np.testing.assert_array_equal(
        shrunk.covariance.values,
        np.diag(np.diag(shrunk.covariance.values)),
    )


def test_full_scaled_identity_shrinkage_has_equal_variances() -> None:
    shrunk = ShrinkageCovariance(
        shrinkage=1.0,
        target=ShrinkageTarget.SCALED_IDENTITY,
    ).estimate(returns())

    diagonal = np.diag(shrunk.covariance.values)
    np.testing.assert_array_equal(diagonal, np.repeat(diagonal[0], 3))
    np.testing.assert_array_equal(shrunk.covariance.values, np.diag(diagonal))


@pytest.mark.parametrize("intensity", [-0.01, 1.01])
def test_shrinkage_intensity_is_closed_unit_interval(intensity: float) -> None:
    with pytest.raises(DataValidationError, match="shrinkage"):
        ShrinkageCovariance(shrinkage=intensity)


@pytest.mark.parametrize(
    "intensity",
    [
        True,
        False,
        np.int64(1),
        np.float64(0.5),
        IntSubclass(1),
        FloatSubclass(0.5),
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_shrinkage_requires_finite_exact_builtin_number(intensity: object) -> None:
    with pytest.raises(DataValidationError, match="shrinkage") as captured:
        ShrinkageCovariance(shrinkage=cast("Any", intensity))

    assert captured.value.context["field"] == "shrinkage"


@pytest.mark.parametrize("intensity", [0, -0.0, 0.25, 1])
def test_valid_builtin_shrinkage_values_are_preserved(intensity: int | float) -> None:
    estimator = ShrinkageCovariance(shrinkage=intensity)

    assert estimator.shrinkage is intensity
    if type(intensity) is float and intensity == 0.0:
        assert np.signbit(estimator.shrinkage) == np.signbit(intensity)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target", "diagonal"),
        ("target", PSDPolicySubclass("diagonal")),
        ("psd_policy", "clip"),
        ("psd_policy", PSDPolicySubclass("clip")),
    ],
)
def test_enum_configuration_requires_exact_members(field: str, value: object) -> None:
    with pytest.raises(DataValidationError) as captured:
        ShrinkageCovariance(shrinkage=0.5, **{field: value})

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
        1.0000001e-2,
        float("nan"),
        float("inf"),
    ],
)
def test_tolerance_requires_exact_finite_bounded_builtin_number(tolerance: object) -> None:
    with pytest.raises(DataValidationError, match="tolerance") as captured:
        ShrinkageCovariance(shrinkage=0.5, tolerance=cast("Any", tolerance))

    assert captured.value.context["field"] == "tolerance"


@pytest.mark.parametrize("tolerance", [1e-300, 1e-10, 1e-2])
def test_valid_tolerance_is_preserved(tolerance: int | float) -> None:
    assert ShrinkageCovariance(shrinkage=0.5, tolerance=tolerance).tolerance is tolerance


@pytest.mark.parametrize(
    "maximum",
    [True, 1.0, np.int64(2), IntSubclass(2), 0, -1],
)
def test_max_dimension_requires_exact_positive_integer(maximum: object) -> None:
    with pytest.raises(DataValidationError, match="max_dimension") as captured:
        ShrinkageCovariance(shrinkage=0.5, max_dimension=cast("Any", maximum))

    assert captured.value.context["field"] == "max_dimension"


@pytest.mark.parametrize("base", [object(), None])
def test_base_estimator_must_expose_callable_estimate(base: object) -> None:
    with pytest.raises(DataValidationError, match="base_estimator") as captured:
        ShrinkageCovariance(shrinkage=0.5, base_estimator=cast("Any", base))

    assert captured.value.context["field"] == "base_estimator"


def test_noncallable_estimate_attribute_is_rejected() -> None:
    class NonCallable:
        estimate = 42

    with pytest.raises(DataValidationError, match="callable"):
        ShrinkageCovariance(shrinkage=0.5, base_estimator=cast("Any", NonCallable()))


def test_base_estimator_descriptor_memory_error_propagates_by_identity() -> None:
    error = MemoryError("descriptor memory")

    class Descriptor:
        @property
        def estimate(self) -> object:
            raise error

    with pytest.raises(MemoryError) as captured:
        ShrinkageCovariance(
            shrinkage=0.5,
            base_estimator=cast("Any", Descriptor()),
        )

    assert captured.value is error


def test_base_estimator_descriptor_ordinary_error_is_bounded_without_message() -> None:
    secret = "DO-NOT-LEAK-DESCRIPTOR-SECRET"

    class Descriptor:
        @property
        def estimate(self) -> object:
            raise RuntimeError(secret)

    with pytest.raises(DataValidationError, match="base_estimator") as captured:
        ShrinkageCovariance(
            shrinkage=0.5,
            base_estimator=cast("Any", Descriptor()),
        )

    assert captured.value.context == {
        "field": "base_estimator",
        "dtype": "Descriptor",
        "reason": "RuntimeError",
    }
    assert secret not in str(captured.value)
    assert secret not in repr(captured.value.context)


@pytest.mark.parametrize(
    "error",
    [KeyboardInterrupt("interrupt"), CustomBackendBaseError("descriptor base")],
)
def test_base_estimator_descriptor_base_exceptions_propagate(
    error: BaseException,
) -> None:
    class Descriptor:
        @property
        def estimate(self) -> object:
            raise error

    with pytest.raises(type(error)) as captured:
        ShrinkageCovariance(
            shrinkage=0.5,
            base_estimator=cast("Any", Descriptor()),
        )

    assert captured.value is error


def test_base_estimator_descriptor_is_deliberately_retrieved_once() -> None:
    class Descriptor:
        accesses = 0

        @property
        def estimate(self) -> object:
            self.accesses += 1
            return lambda matrix: matrix

    base = Descriptor()

    ShrinkageCovariance(
        shrinkage=0.5,
        base_estimator=cast("Any", base),
    )

    assert base.accesses == 1


def test_structural_estimator_protocol_and_frozen_slots_contract() -> None:
    estimator = ShrinkageCovariance(shrinkage=0.25)

    assert isinstance(estimator, RiskEstimator)
    assert not hasattr(estimator, "__dict__")
    with pytest.raises((FrozenInstanceError, AttributeError)):
        estimator.shrinkage = 0.5  # type: ignore[misc]
    assert pickle.loads(pickle.dumps(estimator)) == estimator


def test_wrong_base_signature_is_bounded() -> None:
    class WrongSignature:
        def estimate(self) -> CovarianceEstimate:
            raise AssertionError("unreachable")

    with pytest.raises(NumericalStabilityError, match="base estimator") as captured:
        ShrinkageCovariance(
            shrinkage=0.5,
            base_estimator=cast("Any", WrongSignature()),
        ).estimate(returns())

    assert captured.value.context == {
        "operation": "base_covariance_estimation",
        "reason": "TypeError",
    }


def test_ordinary_base_exception_is_bounded_without_sensitive_message() -> None:
    secret = "DO-NOT-LEAK-BASE-SECRET"

    class Failing:
        def estimate(self, matrix: LabeledMatrix) -> CovarianceEstimate:
            del matrix
            raise CustomBackendError(secret)

    with pytest.raises(NumericalStabilityError) as captured:
        ShrinkageCovariance(
            shrinkage=0.5,
            base_estimator=cast("Any", Failing()),
        ).estimate(returns())

    assert captured.value.context == {
        "operation": "base_covariance_estimation",
        "reason": "CustomBackendError",
    }
    assert secret not in str(captured.value)
    assert secret not in repr(captured.value.context)


@pytest.mark.parametrize("error", [MemoryError("memory"), CustomBackendBaseError("base")])
def test_memory_and_base_exceptions_propagate(error: BaseException) -> None:
    class Failing:
        def estimate(self, matrix: LabeledMatrix) -> CovarianceEstimate:
            del matrix
            raise error

    with pytest.raises(type(error)) as captured:
        ShrinkageCovariance(
            shrinkage=0.5,
            base_estimator=cast("Any", Failing()),
        ).estimate(returns())

    assert captured.value is error


def test_qamr_error_propagates_by_identity() -> None:
    error = DataValidationError("base failure", context={"safe": True})

    class Failing:
        def estimate(self, matrix: LabeledMatrix) -> CovarianceEstimate:
            del matrix
            raise error

    with pytest.raises(DataValidationError) as captured:
        ShrinkageCovariance(
            shrinkage=0.5,
            base_estimator=cast("Any", Failing()),
        ).estimate(returns())

    assert captured.value is error


@pytest.mark.parametrize("result", [None, object()])
def test_base_must_return_exact_covariance_estimate(result: object) -> None:
    class Malformed:
        def estimate(self, matrix: LabeledMatrix) -> object:
            del matrix
            return result

    with pytest.raises(DataValidationError, match="CovarianceEstimate") as captured:
        ShrinkageCovariance(
            shrinkage=0.5,
            base_estimator=cast("Any", Malformed()),
        ).estimate(returns())

    assert captured.value.context["field"] == "base_estimate"


def test_covariance_estimate_subclass_is_rejected() -> None:
    base = fixed_estimate(np.array([[2.0, 0.5], [0.5, 1.0]]))
    subclass = CovarianceEstimateSubclass(
        covariance=base.covariance,
        correlation=base.correlation,
        volatility=base.volatility,
        observation_count=base.observation_count,
        diagnostics=base.diagnostics,
    )

    with pytest.raises(DataValidationError, match="CovarianceEstimate"):
        ShrinkageCovariance(
            shrinkage=0.5,
            base_estimator=cast("Any", FixedEstimator(subclass)),
        ).estimate(returns([[1.0, 2.0], [2.0, 4.0]], labels=("a", "b")))


def test_base_labels_must_match_returns_exactly_and_in_order() -> None:
    base = fixed_estimate(
        np.array([[2.0, 0.5], [0.5, 1.0]]),
        labels=("b", "a"),
    )

    with pytest.raises(LabelAlignmentError, match="labels") as captured:
        ShrinkageCovariance(
            shrinkage=0.5,
            base_estimator=FixedEstimator(base),
        ).estimate(returns([[1.0, 2.0], [2.0, 4.0]], labels=("a", "b")))

    assert captured.value.context["reason"] == "base_labels_mismatch"


def test_returns_requires_exact_labeled_matrix_before_base_call() -> None:
    base = FixedEstimator(fixed_estimate(np.array([[1.0]]), labels=("solo",)))
    subclass = LabeledMatrixSubclass(
        [[1.0], [2.0]],
        ("t0", "t1"),
        ("solo",),
        "time",
        "instrument",
    )

    with pytest.raises(DataValidationError, match="exact LabeledMatrix"):
        ShrinkageCovariance(shrinkage=0.5, base_estimator=base).estimate(subclass)

    assert base.calls == []


def test_returns_hostile_type_is_rejected_without_attribute_access() -> None:
    class Hostile:
        def __getattribute__(self, name: str) -> object:
            if name == "__class__":
                return object.__getattribute__(self, name)
            raise AssertionError("hostile attributes must not be accessed")

    with pytest.raises(DataValidationError, match="exact LabeledMatrix"):
        ShrinkageCovariance(shrinkage=0.5).estimate(cast("Any", Hostile()))


def test_max_dimension_preflight_occurs_before_base_estimation() -> None:
    class Forbidden:
        def estimate(self, matrix: LabeledMatrix) -> CovarianceEstimate:
            del matrix
            raise AssertionError("base must not run")

    with pytest.raises(NumericalStabilityError, match="maximum dimension") as captured:
        ShrinkageCovariance(
            shrinkage=0.5,
            base_estimator=cast("Any", Forbidden()),
            max_dimension=1,
        ).estimate(returns([[1.0, 2.0], [2.0, 4.0]], labels=("a", "b")))

    assert captured.value.context == {"dimension": 2, "maximum": 1}


def test_wrapper_dimension_guard_applies_even_if_base_allows_more() -> None:
    base = SampleCovariance(max_dimension=10)

    with pytest.raises(NumericalStabilityError, match="maximum dimension"):
        ShrinkageCovariance(
            shrinkage=0.5,
            base_estimator=base,
            max_dimension=1,
        ).estimate(returns([[1.0, 2.0], [2.0, 4.0]], labels=("a", "b")))


def test_intermediate_shrinkage_matches_convex_hand_calculation() -> None:
    covariance = np.array([[4.0, 2.0], [2.0, 9.0]])
    base = fixed_estimate(covariance)

    estimate = ShrinkageCovariance(
        shrinkage=0.25,
        target=ShrinkageTarget.DIAGONAL,
        base_estimator=FixedEstimator(base),
        psd_policy=PSDPolicy.RAISE,
    ).estimate(returns([[1.0, 2.0], [2.0, 4.0]], labels=("a", "b")))

    np.testing.assert_allclose(
        estimate.covariance.values,
        np.array([[4.0, 1.5], [1.5, 9.0]]),
        rtol=0.0,
        atol=0.0,
    )


def test_intermediate_diagonal_shrinkage_preserves_source_diagonal_bit_exactly() -> None:
    covariance = np.array([[0.1, 0.02], [0.02, 0.3]])
    base = fixed_estimate(covariance)
    base_before = base.covariance.values

    estimate = ShrinkageCovariance(
        shrinkage=0.2,
        target=ShrinkageTarget.DIAGONAL,
        base_estimator=FixedEstimator(base),
        psd_policy=PSDPolicy.RAISE,
    ).estimate(returns([[1.0, 2.0], [2.0, 4.0]], labels=("a", "b")))

    np.testing.assert_array_equal(
        np.diag(estimate.covariance.values),
        np.diag(base_before),
    )
    assert estimate.covariance.values[0, 1] == pytest.approx(base_before[0, 1] * 0.8)
    assert estimate.covariance.values[1, 0] == pytest.approx(base_before[1, 0] * 0.8)
    np.testing.assert_array_equal(base.covariance.values, base_before)


def test_intermediate_diagonal_shrinkage_makes_subnormal_underflow_explicit() -> None:
    smallest_subnormal = float(np.nextafter(0.0, 1.0))
    covariance = np.array(
        [
            [0.1, smallest_subnormal],
            [smallest_subnormal, 0.2],
        ]
    )
    base = fixed_estimate(covariance)

    estimate = ShrinkageCovariance(
        shrinkage=0.5,
        target=ShrinkageTarget.DIAGONAL,
        base_estimator=FixedEstimator(base),
        psd_policy=PSDPolicy.RAISE,
    ).estimate(returns([[1.0, 2.0], [2.0, 4.0]], labels=("a", "b")))

    np.testing.assert_array_equal(
        np.diag(estimate.covariance.values),
        np.diag(base.covariance.values),
    )
    assert estimate.covariance.values[0, 1] == 0.0
    assert estimate.covariance.values[1, 0] == 0.0
    assert np.isfinite(estimate.correlation.values).all()
    assert np.all(estimate.volatility.values > 1e-12)


def test_scaled_identity_mean_does_not_overflow_extreme_finite_diagonal() -> None:
    covariance = np.diag(np.array([1.0e308, 1.0e308]))
    base = fixed_estimate(covariance)

    estimate = ShrinkageCovariance(
        shrinkage=1.0,
        target=ShrinkageTarget.SCALED_IDENTITY,
        base_estimator=FixedEstimator(base),
        psd_policy=PSDPolicy.RAISE,
    ).estimate(returns([[1.0, 2.0], [2.0, 4.0]], labels=("a", "b")))

    np.testing.assert_array_equal(estimate.covariance.values, covariance)
    assert np.isfinite(estimate.covariance.values).all()


def test_one_asset_scaled_identity_preserves_variance() -> None:
    base = fixed_estimate(np.array([[2.5]]), labels=("solo",))

    estimate = ShrinkageCovariance(
        shrinkage=1.0,
        target=ShrinkageTarget.SCALED_IDENTITY,
        base_estimator=FixedEstimator(base),
    ).estimate(returns([[1.0], [2.0]], labels=("solo",)))

    np.testing.assert_array_equal(estimate.covariance.values, np.array([[2.5]]))
    assert estimate.covariance.shape == (1, 1)


def test_diagnostics_observation_count_labels_axes_and_reconciliation_are_preserved() -> None:
    base_diagnostic = NumericalDiagnostic(
        code="base",
        severity=DiagnosticSeverity.INFO,
        message="base diagnostic",
    )
    base = fixed_estimate(
        np.array([[4.0, 2.0], [2.0, 9.0]]),
        observation_count=17,
        diagnostics=(base_diagnostic,),
    )

    estimate = ShrinkageCovariance(
        shrinkage=0.5,
        target=ShrinkageTarget.DIAGONAL,
        base_estimator=FixedEstimator(base),
    ).estimate(returns([[1.0, 2.0], [2.0, 4.0]], labels=("a", "b")))

    assert estimate.observation_count == 17
    assert estimate.labels == ("a", "b")
    assert estimate.covariance.row_labels == ("a", "b")
    assert estimate.covariance.column_labels == ("a", "b")
    assert estimate.covariance.row_name == "instrument"
    assert estimate.covariance.column_name == "instrument"
    assert estimate.diagnostics[:-1] == (base_diagnostic,)
    assert estimate.diagnostics[-1].code == "covariance_shrinkage"
    assert estimate.diagnostics[-1].severity is DiagnosticSeverity.INFO
    assert dict(estimate.diagnostics[-1].context) == {
        "shrinkage": 0.5,
        "target": "diagonal",
    }
    np.testing.assert_allclose(
        estimate.volatility.values,
        np.sqrt(np.diag(estimate.covariance.values)),
    )
    denominator = np.multiply.outer(estimate.volatility.values, estimate.volatility.values)
    np.testing.assert_allclose(
        estimate.correlation.values,
        estimate.covariance.values / denominator,
    )


def test_psd_policy_is_explicitly_applied_to_shrunk_covariance() -> None:
    labels = ("a", "b", "c")
    covariance = np.array(
        [
            [1.0, -0.9, -0.9],
            [-0.9, 1.0, -0.9],
            [-0.9, -0.9, 1.0],
        ]
    )
    indefinite = CovarianceEstimate(
        covariance=LabeledMatrix(
            covariance,
            labels,
            labels,
            "instrument",
            "instrument",
        ),
        correlation=LabeledMatrix(
            covariance,
            labels,
            labels,
            "instrument",
            "instrument",
        ),
        volatility=LabeledVector(np.ones(3), labels, "instrument"),
        observation_count=7,
    )

    with pytest.raises(NumericalStabilityError, match="positive semidefinite"):
        ShrinkageCovariance(
            shrinkage=0.0,
            base_estimator=FixedEstimator(indefinite),
            psd_policy=PSDPolicy.RAISE,
        ).estimate(returns(labels=labels))


@pytest.mark.parametrize(
    ("covariance", "match"),
    [
        (np.array([[1.0, np.nan], [np.nan, 1.0]]), "finite"),
        (np.array([[1.0, np.inf], [np.inf, 1.0]]), "finite"),
        (np.array([[-1.0, 0.0], [0.0, 1.0]]), "nonnegative"),
        (np.array([[1.0, 0.5], [0.1, 1.0]]), "symmetric"),
    ],
)
def test_malformed_base_covariance_is_rejected_safely(
    covariance: np.ndarray[Any, Any],
    match: str,
) -> None:
    base = fixed_estimate(np.eye(2))
    object.__setattr__(
        base,
        "covariance",
        LabeledMatrix(
            covariance,
            ("a", "b"),
            ("a", "b"),
            "instrument",
            "instrument",
        ),
    )

    with pytest.raises((DataValidationError, NumericalStabilityError), match=match):
        ShrinkageCovariance(
            shrinkage=0.5,
            base_estimator=FixedEstimator(base),
        ).estimate(returns([[1.0, 2.0], [2.0, 4.0]], labels=("a", "b")))


def test_hostile_covariance_value_export_is_bounded_without_sensitive_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = fixed_estimate(np.eye(2))
    secret = "DO-NOT-LEAK-COVARIANCE-SECRET"

    def hostile_values(_self: LabeledMatrix) -> np.ndarray[Any, Any]:
        raise CustomBackendError(secret)

    monkeypatch.setattr(LabeledMatrix, "values", property(hostile_values))

    with pytest.raises(NumericalStabilityError) as captured:
        ShrinkageCovariance(
            shrinkage=0.5,
            base_estimator=FixedEstimator(base),
        ).estimate(returns([[1.0, 2.0], [2.0, 4.0]], labels=("a", "b")))

    assert captured.value.context["operation"] == "covariance_shrinkage"
    assert captured.value.context["reason"] == "CustomBackendError"
    assert secret not in str(captured.value)


def test_input_and_base_estimate_are_not_mutated() -> None:
    matrix = returns([[1.0, 2.0], [2.0, 4.0]], labels=("a", "b"))
    base = fixed_estimate(np.array([[4.0, 2.0], [2.0, 9.0]]))
    matrix_before = matrix.values
    base_before = base.covariance.values
    base_copy = copy.deepcopy(base)

    ShrinkageCovariance(
        shrinkage=0.5,
        base_estimator=FixedEstimator(base),
    ).estimate(matrix)

    np.testing.assert_array_equal(matrix.values, matrix_before)
    np.testing.assert_array_equal(base.covariance.values, base_before)
    np.testing.assert_array_equal(base.covariance.values, base_copy.covariance.values)


def test_repeated_and_concurrent_calls_are_deterministic() -> None:
    matrix = returns()
    estimator = ShrinkageCovariance(
        shrinkage=0.4,
        target=ShrinkageTarget.SCALED_IDENTITY,
    )

    serial = [estimator.estimate(matrix).covariance.values for _ in range(3)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        concurrent = list(
            executor.map(
                lambda _: estimator.estimate(matrix).covariance.values,
                range(8),
            )
        )

    for result in serial[1:] + concurrent:
        np.testing.assert_array_equal(result, serial[0])


def test_kernel_ordinary_exception_is_bounded_without_sensitive_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qamr.risk import shrinkage as shrinkage_module

    secret = "DO-NOT-LEAK-KERNEL-SECRET"

    def fail(*args: object, **kwargs: object) -> np.ndarray[Any, Any]:
        del args, kwargs
        raise CustomBackendError(secret)

    monkeypatch.setattr(shrinkage_module, "_shrink_covariance", fail)
    with pytest.raises(NumericalStabilityError) as captured:
        ShrinkageCovariance(shrinkage=0.5).estimate(returns())

    assert captured.value.context == {
        "operation": "covariance_shrinkage",
        "reason": "CustomBackendError",
    }
    assert secret not in str(captured.value)


def test_kernel_qamr_error_and_memory_error_propagate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qamr.risk import shrinkage as shrinkage_module

    for error in (DataValidationError("kernel"), MemoryError("kernel")):

        def fail(
            *args: object,
            _error: BaseException = error,
            **kwargs: object,
        ) -> np.ndarray[Any, Any]:
            del args, kwargs
            raise _error

        monkeypatch.setattr(shrinkage_module, "_shrink_covariance", fail)
        with pytest.raises(type(error)) as captured:
            ShrinkageCovariance(shrinkage=0.5).estimate(returns())
        assert captured.value is error


def test_default_configuration_is_explicit() -> None:
    estimator = ShrinkageCovariance(shrinkage=0.5)

    assert estimator.target is ShrinkageTarget.DIAGONAL
    assert type(estimator.base_estimator) is SampleCovariance
    assert estimator.psd_policy is PSDPolicy.CLIP
    assert estimator.tolerance == 1e-10
    assert estimator.max_dimension == 2048


def test_no_shrinkage_intensity_or_target_is_inferred() -> None:
    with pytest.raises(TypeError):
        ShrinkageCovariance()  # type: ignore[call-arg]
