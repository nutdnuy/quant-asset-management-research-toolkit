import copy
import json
import pickle
from collections import UserDict
from collections.abc import Iterator, Mapping
from dataclasses import FrozenInstanceError, asdict
from datetime import date, datetime, timedelta
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import pytest

from qamr._types import JsonValue
from qamr.contracts import results as results_module
from qamr.contracts.arrays import LabeledMatrix, LabeledVector
from qamr.contracts.results import DiagnosticSeverity, NumericalDiagnostic
from qamr.errors import DataValidationError, LabelAlignmentError
from qamr.risk.estimates import CovarianceEstimate


class IntSubclass(int):
    pass


class TupleSubclass(tuple[NumericalDiagnostic, ...]):
    pass


class NumericalDiagnosticSubclass(NumericalDiagnostic):
    pass


class DictSubclass(dict[str, JsonValue]):
    pass


class CustomMapping(Mapping[str, JsonValue]):
    def __getitem__(self, key: str) -> JsonValue:
        if key != "value":
            raise KeyError(key)
        return 1

    def __iter__(self) -> Iterator[str]:
        return iter(("value",))

    def __len__(self) -> int:
        return 1


class StringSubclass(str):
    pass


class MaliciousStringSubclass(str):
    def strip(self, chars: str | None = None) -> str:
        del chars
        raise RuntimeError("strip must not be called")


class LabeledMatrixSubclass(LabeledMatrix):
    pass


class MaliciousLabeledMatrixSubclass(LabeledMatrix):
    def __getattribute__(self, name: str) -> object:
        if name in {"column_labels", "row_labels", "shape", "values"}:
            raise RuntimeError("matrix properties must not be accessed")
        return super().__getattribute__(name)


class LabeledVectorSubclass(LabeledVector):
    pass


class MaliciousLabeledVectorSubclass(LabeledVector):
    def __getattribute__(self, name: str) -> object:
        if name in {"labels", "shape", "values"}:
            raise RuntimeError("vector properties must not be accessed")
        return super().__getattribute__(name)


def square(values: object, labels: tuple[str, ...] = ("a", "b")) -> LabeledMatrix:
    return LabeledMatrix(
        np.asarray(values),
        labels,
        labels,
        "instrument",
        "instrument",
    )


def valid_estimate(**changes: object) -> CovarianceEstimate:
    arguments: dict[str, object] = {
        "covariance": square([[0.04, 0.01], [0.01, 0.09]]),
        "correlation": square([[1.0, 1 / 6], [1 / 6, 1.0]]),
        "volatility": LabeledVector(np.array([0.2, 0.3]), ("a", "b"), "instrument"),
        "observation_count": 40,
    }
    arguments.update(changes)
    return CovarianceEstimate(**arguments)  # type: ignore[arg-type]


def consistent_estimate(dtype: np.dtype[Any] | type[np.floating[Any]]) -> CovarianceEstimate:
    volatility_values = np.asarray([0.2, 0.3], dtype=dtype)
    correlation_values = np.asarray([[1.0, 0.25], [0.25, 1.0]], dtype=dtype)
    covariance_values = np.asarray(
        np.outer(volatility_values, volatility_values) * correlation_values,
        dtype=dtype,
    )
    return CovarianceEstimate(
        covariance=square(covariance_values),
        correlation=square(correlation_values),
        volatility=LabeledVector(volatility_values, ("a", "b"), "instrument"),
        observation_count=40,
    )


def test_diagnostic_severity_has_stable_string_values() -> None:
    assert DiagnosticSeverity.INFO.value == "info"
    assert DiagnosticSeverity.WARNING.value == "warning"


def test_numerical_diagnostic_preserves_deeply_immutable_context() -> None:
    weights: list[JsonValue] = [0.25, 0.75]
    details: dict[str, JsonValue] = {"weights": weights}
    context: dict[str, JsonValue] = {"details": details}
    diagnostic = NumericalDiagnostic(
        code="sample_covariance",
        severity=DiagnosticSeverity.INFO,
        message="estimated from complete observations",
        context=context,
    )

    weights[0] = 99.0
    details["new"] = True
    context["new"] = "value"

    immutable_details = cast(Any, diagnostic.context["details"])
    assert immutable_details["weights"] == (0.25, 0.75)
    with pytest.raises(TypeError):
        immutable_details["new"] = False
    with pytest.raises(TypeError):
        immutable_details["weights"][0] = 1.0
    with pytest.raises(TypeError):
        diagnostic.context["new"] = "mutation"  # type: ignore[index]


def test_numerical_diagnostic_context_mapping_is_stable() -> None:
    diagnostic = NumericalDiagnostic(
        code="stable",
        severity=DiagnosticSeverity.INFO,
        message="stable mapping",
    )

    assert diagnostic.context is diagnostic.context
    assert len(diagnostic.context) == 0
    assert dict(diagnostic.context) == {}


def test_numerical_diagnostic_as_dict_returns_fresh_json_snapshots() -> None:
    diagnostic = NumericalDiagnostic(
        code="stable",
        severity=DiagnosticSeverity.WARNING,
        message="stable snapshot",
        context={"nested": {"values": [1, 2]}},
    )

    first = diagnostic.as_dict()
    second = diagnostic.as_dict()
    first_context = cast(dict[str, JsonValue], first["context"])
    first_nested = cast(dict[str, JsonValue], first_context["nested"])
    first_values = cast(list[JsonValue], first_nested["values"])
    first_values.append(3)

    assert first == {
        "code": "stable",
        "severity": "warning",
        "message": "stable snapshot",
        "context": {"nested": {"values": [1, 2, 3]}},
    }
    assert second == {
        "code": "stable",
        "severity": "warning",
        "message": "stable snapshot",
        "context": {"nested": {"values": [1, 2]}},
    }
    assert diagnostic.as_dict() == second


def test_numerical_diagnostic_has_semantic_equality_and_canonical_hash() -> None:
    left = NumericalDiagnostic(
        code="stable",
        severity=DiagnosticSeverity.INFO,
        message="semantic value",
        context={"b": 2, "a": [1]},
    )
    right = NumericalDiagnostic(
        code="stable",
        severity=DiagnosticSeverity.INFO,
        message="semantic value",
        context={"a": [1], "b": 2},
    )

    assert left == right
    assert hash(left) == hash(right)


def test_immutable_context_equality_is_limited_to_same_wrapper_type() -> None:
    diagnostic = NumericalDiagnostic(
        code="stable",
        severity=DiagnosticSeverity.INFO,
        message="same-type equality only",
        context={"value": 1},
    )
    plain = {"value": 1}
    user_dict = UserDict(plain)

    assert diagnostic.context != plain
    assert plain != diagnostic.context
    assert diagnostic.context != user_dict
    # UserDict owns its left-hand comparison and coerces every Mapping to a
    # plain dict. The immutable wrapper intentionally makes no reciprocal claim.
    assert user_dict == diagnostic.context


def test_immutable_context_canonical_equality_preserves_json_scalar_types() -> None:
    reordered_left = NumericalDiagnostic(
        code="left",
        severity=DiagnosticSeverity.INFO,
        message="left",
        context={"b": 2, "a": 1},
    ).context
    reordered_right = NumericalDiagnostic(
        code="right",
        severity=DiagnosticSeverity.INFO,
        message="right",
        context={"a": 1, "b": 2},
    ).context
    distinct_scalars = [
        NumericalDiagnostic(
            code=f"value-{position}",
            severity=DiagnosticSeverity.INFO,
            message="typed scalar",
            context={"value": value},
        ).context
        for position, value in enumerate([True, 1, 1.0, -0.0])
    ]

    assert reordered_left == reordered_right
    assert hash(reordered_left) == hash(reordered_right)
    for position, left in enumerate(distinct_scalars):
        for right in distinct_scalars[position + 1 :]:
            assert left != right


def test_numerical_diagnostic_supports_deepcopy_pickle_and_safe_asdict() -> None:
    diagnostic = NumericalDiagnostic(
        code="stable",
        severity=DiagnosticSeverity.INFO,
        message="serializable value",
        context={"nested": {"values": [1, 2]}},
    )

    deepcopied = copy.deepcopy(diagnostic)
    restored = pickle.loads(pickle.dumps(diagnostic))
    dataclass_snapshot = asdict(diagnostic)

    assert deepcopied == diagnostic
    assert restored == diagnostic
    assert hash(deepcopied) == hash(diagnostic)
    assert hash(restored) == hash(diagnostic)
    assert dataclass_snapshot["context"] is diagnostic.context
    with pytest.raises(TypeError):
        cast(Any, dataclass_snapshot["context"])["new"] = 1
    assert diagnostic.as_dict()["context"] == {"nested": {"values": [1, 2]}}


@pytest.mark.parametrize(
    "context",
    [
        DictSubclass(value=1),
        UserDict({"value": 1}),
        MappingProxyType({"value": 1}),
        CustomMapping(),
    ],
)
def test_numerical_diagnostic_requires_exact_builtin_dict_context(
    context: object,
) -> None:
    with pytest.raises(DataValidationError) as captured:
        NumericalDiagnostic(
            code="invalid_context",
            severity=DiagnosticSeverity.WARNING,
            message="invalid outer context",
            context=context,  # type: ignore[arg-type]
        )

    assert captured.value.context["field"] == "context"
    assert captured.value.context["reason"] == "outer_not_builtin_dict"
    assert len(json.dumps(captured.value.as_dict(), allow_nan=False)) < 1_000


@pytest.mark.parametrize("field", ["code", "message"])
@pytest.mark.parametrize(
    "value",
    [
        StringSubclass("subclass"),
        MaliciousStringSubclass("malicious"),
    ],
)
def test_numerical_diagnostic_requires_exact_builtin_string_fields(
    field: str,
    value: str,
) -> None:
    arguments: dict[str, object] = {
        "code": "valid",
        "severity": DiagnosticSeverity.INFO,
        "message": "valid",
    }
    arguments[field] = value

    with pytest.raises(DataValidationError, match=field) as captured:
        NumericalDiagnostic(**arguments)  # type: ignore[arg-type]

    assert captured.value.context["field"] == field
    assert len(json.dumps(captured.value.as_dict(), allow_nan=False)) < 1_000


@pytest.mark.parametrize(
    "context",
    [
        {StringSubclass("key"): 1},
        {MaliciousStringSubclass("key"): 1},
        {"nested": {StringSubclass("key"): 1}},
        {"nested": {MaliciousStringSubclass("key"): 1}},
    ],
)
def test_numerical_diagnostic_requires_exact_builtin_string_context_keys(
    context: dict[str, JsonValue],
) -> None:
    with pytest.raises(DataValidationError) as captured:
        NumericalDiagnostic(
            code="invalid_context",
            severity=DiagnosticSeverity.WARNING,
            message="invalid context key",
            context=context,
        )

    assert captured.value.context["field"] == "context"
    assert captured.value.context["reason"] == "non_string_key"
    assert len(json.dumps(captured.value.as_dict(), allow_nan=False)) < 1_000


@pytest.mark.parametrize("field", ["code", "message"])
@pytest.mark.parametrize("value", ["", "   ", 1])
def test_numerical_diagnostic_requires_nonblank_strings(field: str, value: object) -> None:
    arguments: dict[str, object] = {
        "code": "valid",
        "severity": DiagnosticSeverity.INFO,
        "message": "valid",
    }
    arguments[field] = value

    with pytest.raises(DataValidationError, match=field) as captured:
        NumericalDiagnostic(**arguments)  # type: ignore[arg-type]

    assert captured.value.context["field"] == field


def test_diagnostic_string_limits_are_documented() -> None:
    assert results_module._MAX_DIAGNOSTIC_CODE_LENGTH == 128
    assert results_module._MAX_DIAGNOSTIC_CODE_BYTES == 512
    assert results_module._MAX_DIAGNOSTIC_MESSAGE_LENGTH == 4_096
    assert results_module._MAX_DIAGNOSTIC_MESSAGE_BYTES == 16_384


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("code", "c" * 129, "maximum_length_exceeded"),
        ("code", "😀" * 129, "maximum_length_exceeded"),
        ("message", "m" * 4_097, "maximum_length_exceeded"),
        ("message", "😀" * 4_097, "maximum_length_exceeded"),
    ],
)
def test_numerical_diagnostic_bounds_code_and_message(
    field: str,
    value: str,
    reason: str,
) -> None:
    arguments: dict[str, object] = {
        "code": "valid",
        "severity": DiagnosticSeverity.INFO,
        "message": "valid",
    }
    arguments[field] = value

    with pytest.raises(DataValidationError, match=field) as captured:
        NumericalDiagnostic(**arguments)  # type: ignore[arg-type]

    assert captured.value.context["field"] == field
    assert captured.value.context["reason"] == reason
    assert len(json.dumps(captured.value.as_dict(), allow_nan=False)) < 1_000


def test_numerical_diagnostic_accepts_exact_string_boundaries() -> None:
    diagnostic = NumericalDiagnostic(
        code="c" * 128,
        severity=DiagnosticSeverity.INFO,
        message="m" * 4_096,
    )

    assert len(diagnostic.code) == 128
    assert len(diagnostic.message) == 4_096


@pytest.mark.parametrize("severity", ["info", 1, None])
def test_numerical_diagnostic_requires_actual_severity_enum(severity: object) -> None:
    with pytest.raises(DataValidationError, match="severity") as captured:
        NumericalDiagnostic(
            code="valid",
            severity=severity,  # type: ignore[arg-type]
            message="valid",
        )

    assert captured.value.context["field"] == "severity"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("code", type("C" * 5_000, (str,), {})("code")),
        ("message", type("M" * 5_000, (str,), {})("message")),
        ("severity", type("S" * 5_000, (), {})()),
        ("context", type("X" * 5_000, (dict,), {})()),
    ],
)
def test_diagnostic_type_errors_bound_adversarial_type_names(
    field: str,
    value: object,
) -> None:
    arguments: dict[str, object] = {
        "code": "valid",
        "severity": DiagnosticSeverity.INFO,
        "message": "valid",
        "context": {},
    }
    arguments[field] = value

    with pytest.raises(DataValidationError) as captured:
        NumericalDiagnostic(**arguments)  # type: ignore[arg-type]

    assert captured.value.context["field"] == field
    assert len(cast(str, captured.value.context["dtype"])) <= 64
    assert len(json.dumps(captured.value.as_dict(), allow_nan=False)) < 1_000


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        ((1, 2), "unsupported_type"),
        ({1, 2}, "unsupported_type"),
        (date(2026, 1, 1), "unsupported_type"),
        (object(), "unsupported_type"),
        (float("nan"), "not_finite"),
        (float("inf"), "not_finite"),
    ],
)
def test_numerical_diagnostic_context_requires_strict_json(
    value: object,
    reason: str,
) -> None:
    with pytest.raises(DataValidationError) as captured:
        NumericalDiagnostic(
            code="invalid_context",
            severity=DiagnosticSeverity.WARNING,
            message="invalid context",
            context={"value": value},  # type: ignore[dict-item]
        )

    assert captured.value.context["field"] == "context"
    assert captured.value.context["reason"] == reason
    assert len(json.dumps(captured.value.as_dict(), allow_nan=False)) < 1_000


@pytest.mark.parametrize("key", ["", "   ", 1])
def test_numerical_diagnostic_context_rejects_invalid_keys(key: object) -> None:
    with pytest.raises(DataValidationError) as captured:
        NumericalDiagnostic(
            code="invalid_context",
            severity=DiagnosticSeverity.WARNING,
            message="invalid context",
            context={"nested": {key: 1}},  # type: ignore[dict-item]
        )

    assert captured.value.context["field"] == "context"
    assert captured.value.context["reason"] in {"blank_key", "non_string_key"}


def test_numerical_diagnostic_context_rejects_aliases() -> None:
    shared: list[JsonValue] = [1]

    with pytest.raises(DataValidationError) as captured:
        NumericalDiagnostic(
            code="invalid_context",
            severity=DiagnosticSeverity.WARNING,
            message="invalid context",
            context={"left": shared, "right": shared},
        )

    assert captured.value.context["reason"] == "aliased_container"


def test_numerical_diagnostic_context_rejects_cycles() -> None:
    cyclic: list[JsonValue] = []
    cyclic.append(cyclic)

    with pytest.raises(DataValidationError) as captured:
        NumericalDiagnostic(
            code="invalid_context",
            severity=DiagnosticSeverity.WARNING,
            message="invalid context",
            context={"cycle": cyclic},
        )

    assert captured.value.context["reason"] == "cyclic"


def test_numerical_diagnostic_context_rejects_excessive_depth_without_recursion_error() -> None:
    nested: Any = "leaf"
    for _ in range(1_500):
        nested = {"child": nested}

    with pytest.raises(DataValidationError) as captured:
        NumericalDiagnostic(
            code="invalid_context",
            severity=DiagnosticSeverity.WARNING,
            message="invalid context",
            context={"nested": nested},
        )

    assert captured.value.context["reason"] == "maximum_depth_exceeded"
    assert len(json.dumps(captured.value.as_dict(), allow_nan=False)) < 1_000


def test_numerical_diagnostic_is_frozen() -> None:
    diagnostic = NumericalDiagnostic(
        code="stable",
        severity=DiagnosticSeverity.INFO,
        message="stable diagnostic",
    )

    with pytest.raises(FrozenInstanceError):
        diagnostic.code = "changed"


def test_covariance_estimate_preserves_complete_labelled_result() -> None:
    diagnostic = NumericalDiagnostic(
        code="sample_covariance",
        severity=DiagnosticSeverity.INFO,
        message="estimated from complete observations",
        context={"ddof": 1},
    )
    estimate = valid_estimate(diagnostics=(diagnostic,))

    assert estimate.labels == ("a", "b")
    assert estimate.observation_count == 40
    assert estimate.diagnostics == (diagnostic,)
    assert estimate.diagnostics[0].context["ddof"] == 1


def test_covariance_estimate_supports_deepcopy_and_pickle_with_diagnostics() -> None:
    diagnostic = NumericalDiagnostic(
        code="stable",
        severity=DiagnosticSeverity.INFO,
        message="serializable estimate",
        context={"method": "sample"},
    )
    estimate = valid_estimate(diagnostics=(diagnostic,))

    deepcopied = copy.deepcopy(estimate)
    restored = pickle.loads(pickle.dumps(estimate))

    assert deepcopied is not estimate
    assert restored is not estimate
    assert deepcopied.diagnostics == estimate.diagnostics
    assert restored.diagnostics == estimate.diagnostics
    assert np.array_equal(deepcopied.covariance.values, estimate.covariance.values)
    assert np.array_equal(restored.correlation.values, estimate.correlation.values)


def test_covariance_estimate_uses_identity_equality_and_is_unhashable() -> None:
    estimate = valid_estimate()
    same_fields = CovarianceEstimate(
        covariance=estimate.covariance,
        correlation=estimate.correlation,
        volatility=estimate.volatility,
        observation_count=estimate.observation_count,
        diagnostics=estimate.diagnostics,
    )

    assert estimate == estimate
    assert estimate != same_fields
    with pytest.raises(TypeError):
        hash(estimate)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (
            "covariance",
            LabeledMatrixSubclass(
                np.array([[0.04, 0.01], [0.01, 0.09]]),
                ("a", "b"),
                ("a", "b"),
                "instrument",
                "instrument",
            ),
        ),
        (
            "correlation",
            MaliciousLabeledMatrixSubclass(
                np.eye(2),
                ("a", "b"),
                ("a", "b"),
                "instrument",
                "instrument",
            ),
        ),
        (
            "volatility",
            LabeledVectorSubclass(np.array([0.2, 0.3]), ("a", "b"), "instrument"),
        ),
        (
            "volatility",
            MaliciousLabeledVectorSubclass(
                np.array([0.2, 0.3]),
                ("a", "b"),
                "instrument",
            ),
        ),
    ],
)
def test_covariance_estimate_requires_exact_builtin_labelled_container_types(
    field: str,
    value: object,
) -> None:
    with pytest.raises(DataValidationError, match=field) as captured:
        valid_estimate(**{field: value})

    assert captured.value.context["field"] == field
    assert len(cast(str, captured.value.context["dtype"])) <= 64
    assert len(json.dumps(captured.value.as_dict(), allow_nan=False)) < 1_000


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("correlation", square([[1.0]], ("b",))),
        ("volatility", LabeledVector(np.array([1.0]), ("b",), "instrument")),
        (
            "covariance",
            LabeledMatrix(
                np.ones((1, 2)),
                ("a",),
                ("a", "b"),
                "instrument",
                "instrument",
            ),
        ),
        (
            "correlation",
            LabeledMatrix(
                np.ones((1, 2)),
                ("a",),
                ("a", "b"),
                "instrument",
                "instrument",
            ),
        ),
    ],
)
def test_covariance_estimate_rejects_misaligned_or_nonsquare_labels(
    field: str,
    value: object,
) -> None:
    with pytest.raises(LabelAlignmentError, match="risk estimate labels"):
        valid_estimate(**{field: value})


def test_covariance_estimate_requires_nonempty_asset_universe() -> None:
    with pytest.raises(LabelAlignmentError, match="non-empty") as captured:
        CovarianceEstimate(
            covariance=square(np.empty((0, 0)), ()),
            correlation=square(np.empty((0, 0)), ()),
            volatility=LabeledVector(np.array([]), (), "asset"),
            observation_count=1,
        )

    assert captured.value.context == {"reason": "empty_asset_universe"}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (
            "covariance",
            LabeledMatrix(
                np.array([[0.04, 0.01], [0.01, 0.09]]),
                ("a", "b"),
                ("a", "b"),
                "asset",
                "instrument",
            ),
        ),
        (
            "correlation",
            LabeledMatrix(
                np.array([[1.0, 1 / 6], [1 / 6, 1.0]]),
                ("a", "b"),
                ("a", "b"),
                "instrument",
                "asset",
            ),
        ),
        (
            "volatility",
            LabeledVector(np.array([0.2, 0.3]), ("a", "b"), "asset"),
        ),
    ],
)
def test_covariance_estimate_requires_consistent_risk_axis_names(
    field: str,
    value: object,
) -> None:
    with pytest.raises(LabelAlignmentError, match="axis names") as captured:
        valid_estimate(**{field: value})

    assert captured.value.context["reason"] == "axis_names"


def test_covariance_estimate_accepts_consistent_generic_risk_axis_name() -> None:
    estimate = CovarianceEstimate(
        covariance=LabeledMatrix(
            np.array([[0.04, 0.01], [0.01, 0.09]]),
            ("a", "b"),
            ("a", "b"),
            "asset",
            "asset",
        ),
        correlation=LabeledMatrix(
            np.array([[1.0, 1 / 6], [1 / 6, 1.0]]),
            ("a", "b"),
            ("a", "b"),
            "asset",
            "asset",
        ),
        volatility=LabeledVector(np.array([0.2, 0.3]), ("a", "b"), "asset"),
        observation_count=40,
    )

    assert estimate.labels == ("a", "b")


@pytest.mark.parametrize("observation_count", [0, -1, True, 1.0, np.int64(1)])
def test_covariance_estimate_requires_actual_positive_int(observation_count: object) -> None:
    with pytest.raises(DataValidationError, match="observation count") as captured:
        valid_estimate(observation_count=observation_count)

    assert captured.value.context["field"] == "observation_count"


def test_covariance_estimate_rejects_int_subclass_observation_count() -> None:
    with pytest.raises(DataValidationError, match="observation count") as captured:
        valid_estimate(observation_count=IntSubclass(40))

    assert captured.value.context == {
        "field": "observation_count",
        "dtype": "IntSubclass",
    }
    assert len(json.dumps(captured.value.as_dict(), allow_nan=False)) < 1_000


@pytest.mark.parametrize("diagnostics", [[], (object(),), ("invalid",)])
def test_covariance_estimate_requires_tuple_of_diagnostics(diagnostics: object) -> None:
    with pytest.raises(DataValidationError, match="diagnostics") as captured:
        valid_estimate(diagnostics=diagnostics)

    assert captured.value.context["field"] == "diagnostics"


def test_covariance_estimate_rejects_tuple_subclass_diagnostics() -> None:
    diagnostics = TupleSubclass()

    with pytest.raises(DataValidationError, match="diagnostics") as captured:
        valid_estimate(diagnostics=diagnostics)

    assert captured.value.context == {
        "field": "diagnostics",
        "dtype": "TupleSubclass",
    }
    assert len(json.dumps(captured.value.as_dict(), allow_nan=False)) < 1_000


def test_covariance_estimate_rejects_diagnostic_subclass_members() -> None:
    diagnostic = NumericalDiagnosticSubclass(
        code="subclass",
        severity=DiagnosticSeverity.INFO,
        message="subclass must not cross the boundary",
    )

    with pytest.raises(DataValidationError, match="diagnostics") as captured:
        valid_estimate(diagnostics=(diagnostic,))

    assert captured.value.context == {
        "field": "diagnostics",
        "position": 0,
        "dtype": "NumericalDiagnosticSubclass",
    }
    assert len(json.dumps(captured.value.as_dict(), allow_nan=False)) < 1_000


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        (
            "observation_count",
            type("I" * 5_000, (int,), {})(40),
        ),
        (
            "diagnostics",
            type("T" * 5_000, (tuple,), {})(),
        ),
        (
            "diagnostics",
            (
                type("D" * 5_000, (NumericalDiagnostic,), {})(
                    code="subclass",
                    severity=DiagnosticSeverity.INFO,
                    message="subclass with adversarial type name",
                ),
            ),
        ),
    ],
)
def test_exact_type_errors_bound_adversarial_subclass_names(
    field: str,
    invalid_value: object,
) -> None:
    with pytest.raises(DataValidationError) as captured:
        valid_estimate(**{field: invalid_value})

    assert captured.value.context["field"] == field
    assert len(cast(str, captured.value.context["dtype"])) <= 64
    assert len(json.dumps(captured.value.as_dict(), allow_nan=False)) < 1_000


@pytest.mark.parametrize(
    ("field", "values"),
    [
        ("covariance", [[True, False], [False, True]]),
        ("correlation", [[1 + 0j, 0j], [0j, 1 + 0j]]),
        ("volatility", ["0.2", "0.3"]),
        ("covariance", [[date(2026, 1, 1), date(2026, 1, 2)]] * 2),
        ("correlation", [[datetime(2026, 1, 1), datetime(2026, 1, 2)]] * 2),
        ("volatility", [timedelta(days=1), timedelta(days=2)]),
        ("covariance", [[object(), object()], [object(), object()]]),
        ("correlation", [[1.0, np.nan], [np.nan, 1.0]]),
        ("volatility", [0.2, np.inf]),
    ],
)
def test_covariance_estimate_requires_finite_real_numeric_values(
    field: str,
    values: object,
) -> None:
    replacement: object
    if field == "volatility":
        replacement = LabeledVector(np.asarray(values), ("a", "b"), "instrument")
    else:
        replacement = square(values)

    with pytest.raises(DataValidationError, match="finite real numeric") as captured:
        valid_estimate(**{field: replacement})

    assert captured.value.context["field"] == field
    assert set(captured.value.context) <= {"field", "dtype", "reason"}


@pytest.mark.parametrize("extreme", [np.iinfo(np.int64).min, np.iinfo(np.int64).max])
def test_covariance_estimate_rejects_extreme_integer_correlations(
    extreme: np.int64,
) -> None:
    correlation = square(np.array([[1, extreme], [extreme, 1]], dtype=np.int64))

    with pytest.raises(DataValidationError, match=r"\[-1, 1\]") as captured:
        valid_estimate(correlation=correlation)

    assert captured.value.context == {
        "field": "correlation",
        "reason": "outside_unit_interval",
    }


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        (
            "covariance",
            square(
                np.array(
                    [[np.iinfo(np.int64).min, 0], [0, 1]],
                    dtype=np.int64,
                )
            ),
        ),
        (
            "volatility",
            LabeledVector(
                np.array([np.iinfo(np.int64).min, 1], dtype=np.int64),
                ("a", "b"),
                "instrument",
            ),
        ),
    ],
)
def test_covariance_estimate_rejects_extreme_negative_integer_diagonals(
    field: str,
    replacement: object,
) -> None:
    with pytest.raises(DataValidationError) as captured:
        valid_estimate(**{field: replacement})

    assert captured.value.context["field"] == field
    assert len(json.dumps(captured.value.as_dict(), allow_nan=False)) < 1_000


@pytest.mark.parametrize(
    ("field", "covariance_dtype", "correlation_dtype", "volatility_dtype", "reason"),
    [
        (
            "covariance",
            np.float64,
            np.float16,
            np.float64,
            "negative_diagonal",
        ),
        (
            "covariance",
            np.float64,
            np.float64,
            np.float16,
            "negative_diagonal",
        ),
        (
            "correlation",
            np.float16,
            np.float64,
            np.float64,
            "diagonal_not_one",
        ),
        (
            "correlation",
            np.float64,
            np.float64,
            np.float16,
            "diagonal_not_one",
        ),
        (
            "volatility",
            np.float16,
            np.float64,
            np.float64,
            "negative",
        ),
        (
            "volatility",
            np.float64,
            np.float16,
            np.float64,
            "negative",
        ),
    ],
)
def test_float16_in_another_field_does_not_relax_float64_validation(
    field: str,
    covariance_dtype: type[np.floating[Any]],
    correlation_dtype: type[np.floating[Any]],
    volatility_dtype: type[np.floating[Any]],
    reason: str,
) -> None:
    covariance_values = np.diag(np.asarray([0.0, 0.09], dtype=covariance_dtype))
    correlation_values = np.eye(2, dtype=correlation_dtype)
    volatility_values = np.asarray([0.0, 0.3], dtype=volatility_dtype)
    if field == "covariance":
        covariance_values[0, 0] = -1e-7
    elif field == "correlation":
        correlation_values[0, 0] = 1.0 + 1e-7
    else:
        volatility_values[0] = -1e-7

    with pytest.raises(DataValidationError) as captured:
        CovarianceEstimate(
            covariance=square(covariance_values),
            correlation=square(correlation_values),
            volatility=LabeledVector(volatility_values, ("a", "b"), "instrument"),
            observation_count=40,
        )

    assert captured.value.context == {"field": field, "reason": reason}


def test_float16_covariance_does_not_relax_float64_correlation_bound() -> None:
    with pytest.raises(DataValidationError, match=r"\[-1, 1\]") as captured:
        CovarianceEstimate(
            covariance=square(np.ones((2, 2), dtype=np.float16)),
            correlation=square(
                np.asarray([[1.0, 1.0 + 1e-7], [1.0 + 1e-7, 1.0]], dtype=np.float64)
            ),
            volatility=LabeledVector(
                np.ones(2, dtype=np.float64),
                ("a", "b"),
                "instrument",
            ),
            observation_count=40,
        )

    assert captured.value.context == {
        "field": "correlation",
        "reason": "outside_unit_interval",
    }


def test_float16_correlation_keeps_its_own_inside_unit_bound_with_float64_peers() -> None:
    estimate = CovarianceEstimate(
        covariance=square(np.ones((2, 2), dtype=np.float64)),
        correlation=square(np.asarray([[1.0, 1.001], [1.001, 1.0]], dtype=np.float16)),
        volatility=LabeledVector(
            np.ones(2, dtype=np.float64),
            ("a", "b"),
            "instrument",
        ),
        observation_count=40,
    )

    assert estimate.correlation.values[0, 1] > 1.0


@pytest.mark.parametrize(
    ("field", "inside_value"),
    [
        ("covariance", -1e-7),
        ("correlation", 1.001),
        ("volatility", -1e-7),
    ],
)
def test_float16_field_keeps_its_own_inside_boundary_with_float64_peers(
    field: str,
    inside_value: float,
) -> None:
    covariance_values = np.diag(np.asarray([0.0, 0.09], dtype=np.float64))
    correlation_values = np.eye(2, dtype=np.float64)
    volatility_values = np.asarray([0.0, 0.3], dtype=np.float64)
    if field == "covariance":
        covariance_values = covariance_values.astype(np.float16)
        covariance_values[0, 0] = inside_value
    elif field == "correlation":
        correlation_values = correlation_values.astype(np.float16)
        correlation_values[0, 0] = inside_value
    else:
        volatility_values = volatility_values.astype(np.float16)
        volatility_values[0] = inside_value

    estimate = CovarianceEstimate(
        covariance=square(covariance_values),
        correlation=square(correlation_values),
        volatility=LabeledVector(volatility_values, ("a", "b"), "instrument"),
        observation_count=40,
    )

    assert estimate.labels == ("a", "b")


@pytest.mark.parametrize(
    ("field", "outside_value", "reason"),
    [
        ("covariance", -0.001, "negative_diagonal"),
        ("correlation", 1.01, "diagonal_not_one"),
        ("volatility", -0.001, "negative"),
    ],
)
def test_float16_field_keeps_its_own_outside_boundary_with_float64_peers(
    field: str,
    outside_value: float,
    reason: str,
) -> None:
    covariance_values = np.diag(np.asarray([0.0, 0.09], dtype=np.float64))
    correlation_values = np.eye(2, dtype=np.float64)
    volatility_values = np.asarray([0.0, 0.3], dtype=np.float64)
    if field == "covariance":
        covariance_values = covariance_values.astype(np.float16)
        covariance_values[0, 0] = outside_value
    elif field == "correlation":
        correlation_values = correlation_values.astype(np.float16)
        correlation_values[0, 0] = outside_value
    else:
        volatility_values = volatility_values.astype(np.float16)
        volatility_values[0] = outside_value

    with pytest.raises(DataValidationError) as captured:
        CovarianceEstimate(
            covariance=square(covariance_values),
            correlation=square(correlation_values),
            volatility=LabeledVector(volatility_values, ("a", "b"), "instrument"),
            observation_count=40,
        )

    assert captured.value.context == {"field": field, "reason": reason}


@pytest.mark.parametrize(
    ("dtype", "small_variance", "large_variance", "negative_volatility", "large_volatility"),
    [
        (np.float16, 0.01, 62_496.0, -0.1, 250.0),
        (np.float32, 1.0, 1e18, -1.0, 1e9),
        (np.float64, 1.0, 1e24, -1.0, 1e12),
    ],
)
def test_volatility_nonnegativity_uses_element_local_tolerance(
    dtype: type[np.floating[Any]],
    small_variance: float,
    large_variance: float,
    negative_volatility: float,
    large_volatility: float,
) -> None:
    with pytest.raises(DataValidationError, match="nonnegative") as captured:
        CovarianceEstimate(
            covariance=square(np.diag(np.asarray([small_variance, large_variance], dtype=dtype))),
            correlation=square(np.eye(2, dtype=dtype)),
            volatility=LabeledVector(
                np.asarray([negative_volatility, large_volatility], dtype=dtype),
                ("a", "b"),
                "instrument",
            ),
            observation_count=40,
        )

    assert captured.value.context == {
        "field": "volatility",
        "reason": "negative",
    }


@pytest.mark.parametrize(
    ("dtype", "negative_variance", "large_variance", "large_volatility"),
    [
        (np.float16, -0.01, 62_496.0, 250.0),
        (np.float32, -1.0, 1e18, 1e9),
        (np.float64, -1.0, 1e24, 1e12),
    ],
)
def test_covariance_diagonal_nonnegativity_uses_element_local_tolerance(
    dtype: type[np.floating[Any]],
    negative_variance: float,
    large_variance: float,
    large_volatility: float,
) -> None:
    with pytest.raises(DataValidationError, match="diagonal") as captured:
        CovarianceEstimate(
            covariance=square(
                np.diag(np.asarray([negative_variance, large_variance], dtype=dtype))
            ),
            correlation=square(np.eye(2, dtype=dtype)),
            volatility=LabeledVector(
                np.asarray([0.0, large_volatility], dtype=dtype),
                ("a", "b"),
                "instrument",
            ),
            observation_count=40,
        )

    assert captured.value.context == {
        "field": "covariance",
        "reason": "negative_diagonal",
    }


@pytest.mark.parametrize(
    ("dtype", "small_variance", "large_variance", "wrong_volatility", "large_volatility"),
    [
        (np.float16, 0.01, 62_496.0, 0.2, 250.0),
        (np.float32, 1.0, 1e18, 2.0, 1e9),
        (np.float64, 1.0, 1e24, 2.0, 1e12),
    ],
)
def test_volatility_reconciliation_uses_element_local_scales(
    dtype: type[np.floating[Any]],
    small_variance: float,
    large_variance: float,
    wrong_volatility: float,
    large_volatility: float,
) -> None:
    with pytest.raises(DataValidationError, match="covariance diagonal") as captured:
        CovarianceEstimate(
            covariance=square(np.diag(np.asarray([small_variance, large_variance], dtype=dtype))),
            correlation=square(np.eye(2, dtype=dtype)),
            volatility=LabeledVector(
                np.asarray([wrong_volatility, large_volatility], dtype=dtype),
                ("a", "b"),
                "instrument",
            ),
            observation_count=40,
        )

    assert captured.value.context == {
        "field": "volatility",
        "reason": "covariance_diagonal_mismatch",
    }


@pytest.mark.parametrize(
    ("dtype", "large_variance", "large_volatility"),
    [
        (np.float16, 62_496.0, 250.0),
        (np.float32, 1e18, 1e9),
        (np.float64, 1e24, 1e12),
    ],
)
def test_nonnegativity_accepts_dtype_zero_boundary_in_mixed_scale_results(
    dtype: type[np.floating[Any]],
    large_variance: float,
    large_volatility: float,
) -> None:
    tiny = np.finfo(dtype).smallest_subnormal
    estimate = CovarianceEstimate(
        covariance=square(np.diag(np.asarray([-tiny, large_variance], dtype=dtype))),
        correlation=square(np.eye(2, dtype=dtype)),
        volatility=LabeledVector(
            np.asarray([-tiny, large_volatility], dtype=dtype),
            ("a", "b"),
            "instrument",
        ),
        observation_count=40,
    )

    assert estimate.covariance.values[0, 0] == pytest.approx(-tiny)
    assert estimate.volatility.values[0] == pytest.approx(-tiny)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_covariance_estimate_accepts_self_consistent_lower_precision_values(
    dtype: type[np.floating[Any]],
) -> None:
    estimate = consistent_estimate(dtype)

    assert estimate.covariance.values.dtype == np.dtype(dtype)


@pytest.mark.parametrize(
    ("dtype", "correlation_delta"),
    [
        (np.float16, 0.02),
        (np.float32, 0.0001),
    ],
)
def test_dtype_aware_tolerance_rejects_nearby_true_correlation_mismatch(
    dtype: type[np.floating[Any]],
    correlation_delta: float,
) -> None:
    estimate = consistent_estimate(dtype)
    correlation_values = estimate.correlation.values
    correlation_values.flags.writeable = True
    correlation_values[0, 1] += correlation_delta
    correlation_values[1, 0] += correlation_delta

    with pytest.raises(DataValidationError, match="consistent") as captured:
        CovarianceEstimate(
            covariance=estimate.covariance,
            correlation=square(correlation_values),
            volatility=estimate.volatility,
            observation_count=40,
        )

    assert captured.value.context == {
        "field": "correlation",
        "reason": "covariance_mismatch",
    }


def test_covariance_estimate_reconciles_correlation_with_covariance() -> None:
    with pytest.raises(DataValidationError, match="consistent") as captured:
        valid_estimate(correlation=square([[1.0, 0.9], [0.9, 1.0]]))

    assert captured.value.context == {
        "field": "correlation",
        "reason": "covariance_mismatch",
    }


def test_covariance_estimate_accepts_zero_volatility_convention() -> None:
    estimate = CovarianceEstimate(
        covariance=square([[0.0, 0.0], [0.0, 0.09]]),
        correlation=square([[1.0, 0.0], [0.0, 1.0]]),
        volatility=LabeledVector(np.array([0.0, 0.3]), ("a", "b"), "instrument"),
        observation_count=40,
    )

    assert estimate.volatility.values[0] == pytest.approx(0.0)


def test_zero_volatility_tolerance_accepts_float16_subnormal_roundoff() -> None:
    tiny = np.finfo(np.float16).smallest_subnormal
    estimate = CovarianceEstimate(
        covariance=square(np.array([[0.0, tiny], [tiny, 100.0]], dtype=np.float16)),
        correlation=square(np.array([[1.0, tiny], [tiny, 1.0]], dtype=np.float16)),
        volatility=LabeledVector(
            np.array([0.0, 10.0], dtype=np.float16),
            ("a", "b"),
            "instrument",
        ),
        observation_count=40,
    )

    assert estimate.covariance.values[0, 1] == tiny


def test_float16_correlation_does_not_relax_float64_zero_covariance_entries() -> None:
    with pytest.raises(DataValidationError, match="zero-volatility") as captured:
        CovarianceEstimate(
            covariance=square(np.asarray([[0.0, 1e-7], [1e-7, 0.09]], dtype=np.float64)),
            correlation=square(np.eye(2, dtype=np.float16)),
            volatility=LabeledVector(
                np.asarray([0.0, 0.3], dtype=np.float64),
                ("a", "b"),
                "instrument",
            ),
            observation_count=40,
        )

    assert captured.value.context == {
        "field": "covariance",
        "reason": "zero_volatility_covariance",
    }


def test_float16_covariance_precision_applies_to_zero_covariance_entries() -> None:
    estimate = CovarianceEstimate(
        covariance=square(np.asarray([[0.0, 1e-7], [1e-7, 0.09]], dtype=np.float16)),
        correlation=square(np.eye(2, dtype=np.float64)),
        volatility=LabeledVector(
            np.asarray([0.0, 0.3], dtype=np.float64),
            ("a", "b"),
            "instrument",
        ),
        observation_count=40,
    )

    assert estimate.covariance.values[0, 1] == pytest.approx(1e-7)


def test_float16_covariance_does_not_relax_float64_zero_correlation_entries() -> None:
    with pytest.raises(DataValidationError, match="zero-volatility") as captured:
        CovarianceEstimate(
            covariance=square(np.diag(np.asarray([0.0, 0.09], dtype=np.float16))),
            correlation=square(np.asarray([[1.0, 1e-7], [1e-7, 1.0]], dtype=np.float64)),
            volatility=LabeledVector(
                np.asarray([0.0, 0.3], dtype=np.float64),
                ("a", "b"),
                "instrument",
            ),
            observation_count=40,
        )

    assert captured.value.context == {
        "field": "correlation",
        "reason": "zero_volatility_correlation",
    }


def test_float16_correlation_precision_applies_to_zero_correlation_entries() -> None:
    estimate = CovarianceEstimate(
        covariance=square(np.diag(np.asarray([0.0, 0.09], dtype=np.float64))),
        correlation=square(np.asarray([[1.0, 1e-7], [1e-7, 1.0]], dtype=np.float16)),
        volatility=LabeledVector(
            np.asarray([0.0, 0.3], dtype=np.float64),
            ("a", "b"),
            "instrument",
        ),
        observation_count=40,
    )

    assert estimate.correlation.values[0, 1] == pytest.approx(1e-7)


def test_zero_volatility_tolerance_is_not_scaled_by_unrelated_large_values() -> None:
    with pytest.raises(DataValidationError, match="zero-volatility") as captured:
        CovarianceEstimate(
            covariance=square(np.array([[0.0, 0.1], [0.1, 100.0]], dtype=np.float16)),
            correlation=square(np.eye(2, dtype=np.float16)),
            volatility=LabeledVector(
                np.array([0.0, 10.0], dtype=np.float16),
                ("a", "b"),
                "instrument",
            ),
            observation_count=40,
        )

    assert captured.value.context == {
        "field": "covariance",
        "reason": "zero_volatility_covariance",
    }


@pytest.mark.parametrize(
    ("covariance", "correlation", "field", "reason"),
    [
        (
            [[0.0, 0.001], [0.001, 0.09]],
            [[1.0, 0.0], [0.0, 1.0]],
            "covariance",
            "zero_volatility_covariance",
        ),
        (
            [[0.0, 0.0], [0.0, 0.09]],
            [[1.0, 0.1], [0.1, 1.0]],
            "correlation",
            "zero_volatility_correlation",
        ),
    ],
)
def test_covariance_estimate_rejects_inconsistent_zero_volatility_rows(
    covariance: list[list[float]],
    correlation: list[list[float]],
    field: str,
    reason: str,
) -> None:
    with pytest.raises(DataValidationError, match="zero-volatility") as captured:
        CovarianceEstimate(
            covariance=square(covariance),
            correlation=square(correlation),
            volatility=LabeledVector(np.array([0.0, 0.3]), ("a", "b"), "instrument"),
            observation_count=40,
        )

    assert captured.value.context == {"field": field, "reason": reason}


@pytest.mark.parametrize(
    ("field", "values", "message"),
    [
        ("covariance", [[0.04, 0.02], [0.01, 0.09]], "symmetric"),
        ("correlation", [[1.0, 0.2], [0.1, 1.0]], "symmetric"),
        ("covariance", [[-0.01, 0.0], [0.0, 0.09]], "diagonal"),
        ("correlation", [[0.99, 0.0], [0.0, 1.0]], "diagonal"),
        ("correlation", [[1.0, 1.01], [1.01, 1.0]], r"\[-1, 1\]"),
        ("volatility", [-0.2, 0.3], "nonnegative"),
        ("volatility", [0.21, 0.3], "covariance diagonal"),
    ],
)
def test_covariance_estimate_enforces_matrix_semantics(
    field: str,
    values: object,
    message: str,
) -> None:
    replacement = (
        LabeledVector(np.asarray(values), ("a", "b"), "instrument")
        if field == "volatility"
        else square(values)
    )

    with pytest.raises(DataValidationError, match=message) as captured:
        valid_estimate(**{field: replacement})

    assert captured.value.context["field"] == field


def test_covariance_estimate_allows_tiny_roundoff_at_documented_tolerances() -> None:
    estimate = valid_estimate(
        covariance=square([[0.04, 5e-13], [5e-13, -5e-13]]),
        correlation=square([[1.0 + 5e-13, 5e-13], [5e-13, 1.0 - 5e-13]]),
        volatility=LabeledVector(np.array([0.2 + 5e-13, 0.0]), ("a", "b"), "instrument"),
    )

    assert estimate.labels == ("a", "b")


def test_covariance_estimate_does_not_enforce_positive_semidefiniteness() -> None:
    values = np.array(
        [
            [1.0, 0.9, 0.9],
            [0.9, 1.0, -0.9],
            [0.9, -0.9, 1.0],
        ]
    )
    estimate = CovarianceEstimate(
        covariance=square(values, ("a", "b", "c")),
        correlation=square(values, ("a", "b", "c")),
        volatility=LabeledVector(np.ones(3), ("a", "b", "c"), "instrument"),
        observation_count=40,
    )

    assert np.linalg.eigvalsh(estimate.covariance.values).min() < 0.0


def test_covariance_estimate_is_frozen() -> None:
    estimate = valid_estimate()

    with pytest.raises(FrozenInstanceError):
        estimate.observation_count = 41
