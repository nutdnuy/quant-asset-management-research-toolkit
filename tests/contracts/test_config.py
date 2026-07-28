"""Tests for the immutable resolved research configuration."""

import dataclasses
import json
import pickle
from decimal import Decimal
from fractions import Fraction
from typing import Any, cast

import pytest

import qamr.contracts.config as config_module
from qamr.contracts.config import (
    PortfolioConstraints,
    ResearchConfig,
    TransactionCostConfig,
)
from qamr.contracts.dataset import MissingDataPolicy, ReturnConvention
from qamr.errors import DataValidationError, InfeasiblePortfolioError

_EXPECTED_MAX_COMPONENT_JSON_DEPTH = 100
_EXPECTED_MAX_COMPONENT_JSON_NODES = 2_000
_EXPECTED_MAX_COMPONENT_JSON_BYTES = 262_144
_EXPECTED_MAX_COMPONENT_KEY_LENGTH = 256
_EXPECTED_MAX_COMPONENT_STRING_LENGTH = 32_768


def test_research_config_serializes_to_deterministic_json_safe_shape() -> None:
    config = ResearchConfig(
        estimation_window=60,
        holding_period=5,
        rebalance_every=5,
        signal_availability_lag=1,
        execution_lag=1,
        return_convention=ReturnConvention.SIMPLE,
        missing_data_policy=MissingDataPolicy.RAISE,
        constraints=PortfolioConstraints(),
        transaction_costs=TransactionCostConfig(proportional_bps=7.5),
        annualization_factor=12.0,
        random_seed=42,
        components={"risk": {"name": "sample", "ddof": 1}},
    )

    assert config.to_dict() == {
        "annualization_factor": 12.0,
        "benchmark": None,
        "components": {"risk": {"ddof": 1, "name": "sample"}},
        "constraints": {
            "benchmark_relative": False,
            "gross_leverage": 1.0,
            "long_only": True,
            "max_weight": 1.0,
            "min_weight": 0.0,
            "net_exposure": 1.0,
        },
        "estimation_window": 60,
        "execution_lag": 1,
        "holding_period": 5,
        "missing_data_policy": "raise",
        "random_seed": 42,
        "rebalance_every": 5,
        "return_convention": "simple",
        "signal_availability_lag": 1,
        "transaction_costs": {
            "fixed_per_trade": 0.0,
            "proportional_bps": 7.5,
        },
    }


@pytest.mark.parametrize(
    "field",
    ["estimation_window", "holding_period", "rebalance_every"],
)
def test_research_config_requires_positive_windows(field: str) -> None:
    arguments = {"estimation_window": 60, field: 0}

    with pytest.raises(DataValidationError, match="positive") as exc_info:
        ResearchConfig(**arguments)

    assert exc_info.value.context["field"] == field


@pytest.mark.parametrize(
    "field",
    ["signal_availability_lag", "execution_lag"],
)
def test_research_config_rejects_negative_lags(field: str) -> None:
    arguments = {"estimation_window": 60, field: -1}

    with pytest.raises(DataValidationError) as exc_info:
        ResearchConfig(**arguments)

    assert exc_info.value.context["field"] == field


def test_long_only_constraints_reject_negative_minimum_weight() -> None:
    with pytest.raises(InfeasiblePortfolioError):
        PortfolioConstraints(long_only=True, min_weight=-0.1)


@pytest.mark.parametrize(
    ("field", "value"),
    [("proportional_bps", -0.1), ("fixed_per_trade", -0.1)],
)
def test_transaction_costs_reject_negative_values(field: str, value: float) -> None:
    with pytest.raises(DataValidationError) as exc_info:
        TransactionCostConfig(**{field: value})

    assert exc_info.value.context["field"] == field


@pytest.mark.parametrize("field", ["long_only", "benchmark_relative"])
@pytest.mark.parametrize("value", [0, 1, "true"])
def test_constraints_require_actual_booleans(field: str, value: object) -> None:
    with pytest.raises(DataValidationError) as exc_info:
        PortfolioConstraints(**{field: value})

    assert exc_info.value.context["field"] == field


@pytest.mark.parametrize(
    "field",
    ["min_weight", "max_weight", "gross_leverage", "net_exposure"],
)
@pytest.mark.parametrize("value", [True, "1.0", float("nan"), float("inf")])
def test_constraints_require_finite_real_optional_values(
    field: str,
    value: object,
) -> None:
    with pytest.raises(DataValidationError) as exc_info:
        PortfolioConstraints(**{field: value})

    assert exc_info.value.context["field"] == field
    json.dumps(exc_info.value.as_dict(), allow_nan=False)


@pytest.mark.parametrize(
    "arguments",
    [
        {"min_weight": 0.6, "max_weight": 0.5},
        {"gross_leverage": 0.0},
        {"gross_leverage": -1.0},
        {"gross_leverage": 1.0, "net_exposure": 1.1},
        {"gross_leverage": 1.0, "net_exposure": -1.1, "long_only": False},
        {"long_only": True, "net_exposure": -0.1},
    ],
)
def test_constraints_reject_infeasible_combinations(
    arguments: dict[str, object],
) -> None:
    with pytest.raises(InfeasiblePortfolioError):
        PortfolioConstraints(**arguments)


@pytest.mark.parametrize(
    "arguments",
    [
        {"long_only": True, "min_weight": None, "max_weight": -0.1},
        {
            "long_only": False,
            "min_weight": 0.0,
            "max_weight": 1.0,
            "net_exposure": -0.1,
        },
        {
            "long_only": False,
            "min_weight": -1.0,
            "max_weight": 0.0,
            "net_exposure": 0.1,
        },
        {
            "long_only": False,
            "min_weight": 0.0,
            "max_weight": 0.0,
            "net_exposure": 0.1,
        },
    ],
)
def test_constraints_reject_directionally_infeasible_bounds(
    arguments: dict[str, object],
) -> None:
    with pytest.raises(InfeasiblePortfolioError):
        PortfolioConstraints(**arguments)


@pytest.mark.parametrize(
    "arguments",
    [
        {
            "long_only": True,
            "min_weight": 0.0,
            "max_weight": 0.0,
            "net_exposure": 0.0,
        },
        {
            "long_only": False,
            "min_weight": 0.0,
            "max_weight": 1.0,
            "net_exposure": 0.0,
        },
        {
            "long_only": False,
            "min_weight": -1.0,
            "max_weight": 0.0,
            "net_exposure": 0.0,
        },
        {
            "long_only": False,
            "min_weight": -1.0,
            "max_weight": 1.0,
            "net_exposure": -0.1,
        },
        {
            "long_only": False,
            "min_weight": -1.0,
            "max_weight": 1.0,
            "net_exposure": 0.1,
        },
    ],
)
def test_constraints_accept_nearby_directional_boundaries(
    arguments: dict[str, object],
) -> None:
    assert PortfolioConstraints(**arguments).net_exposure == arguments["net_exposure"]


@pytest.mark.parametrize(
    "arguments",
    [
        {
            "long_only": False,
            "min_weight": 0.1,
            "max_weight": 1.0,
            "net_exposure": 0.0,
        },
        {
            "long_only": False,
            "min_weight": 0.1,
            "max_weight": 1.0,
            "net_exposure": -0.0,
        },
        {
            "long_only": False,
            "min_weight": 0.1,
            "max_weight": 1.0,
            "net_exposure": 0.05,
        },
        {
            "long_only": False,
            "min_weight": -1.0,
            "max_weight": -0.1,
            "net_exposure": 0.0,
        },
        {
            "long_only": False,
            "min_weight": -1.0,
            "max_weight": -0.1,
            "net_exposure": -0.0,
        },
        {
            "long_only": False,
            "min_weight": -1.0,
            "max_weight": -0.1,
            "net_exposure": -0.05,
        },
    ],
)
def test_constraints_reject_net_targets_inside_impossible_nonempty_gap(
    arguments: dict[str, object],
) -> None:
    with pytest.raises(InfeasiblePortfolioError):
        PortfolioConstraints(**arguments)


@pytest.mark.parametrize(
    "arguments",
    [
        {
            "long_only": False,
            "min_weight": 0.1,
            "max_weight": 1.0,
            "net_exposure": 0.1,
        },
        {
            "long_only": False,
            "min_weight": 0.1,
            "max_weight": 1.0,
            "net_exposure": 0.2,
        },
        {
            "long_only": False,
            "min_weight": -1.0,
            "max_weight": -0.1,
            "net_exposure": -0.1,
        },
        {
            "long_only": False,
            "min_weight": -1.0,
            "max_weight": -0.1,
            "net_exposure": -0.2,
        },
    ],
)
def test_constraints_accept_small_same_sign_net_boundaries(
    arguments: dict[str, object],
) -> None:
    constraints = PortfolioConstraints(**arguments)

    assert constraints.net_exposure == arguments["net_exposure"]


@pytest.mark.parametrize(
    "arguments",
    [
        {
            "long_only": False,
            "min_weight": 0.1,
            "max_weight": 1.0,
            "net_exposure": None,
            "gross_leverage": 0.05,
        },
        {
            "long_only": False,
            "min_weight": -1.0,
            "max_weight": -0.1,
            "net_exposure": None,
            "gross_leverage": 0.05,
        },
    ],
)
def test_constraints_reject_bounds_larger_than_gross_without_net_target(
    arguments: dict[str, object],
) -> None:
    with pytest.raises(InfeasiblePortfolioError):
        PortfolioConstraints(**arguments)


@pytest.mark.parametrize(
    "arguments",
    [
        {
            "long_only": False,
            "min_weight": 0.1,
            "max_weight": 1.0,
            "net_exposure": None,
            "gross_leverage": 0.1,
        },
        {
            "long_only": False,
            "min_weight": -1.0,
            "max_weight": -0.1,
            "net_exposure": None,
            "gross_leverage": 0.1,
        },
    ],
)
def test_constraints_accept_bounds_equal_to_gross_without_net_target(
    arguments: dict[str, object],
) -> None:
    constraints = PortfolioConstraints(**arguments)

    assert constraints.gross_leverage == 0.1


@pytest.mark.parametrize("field", ["proportional_bps", "fixed_per_trade"])
@pytest.mark.parametrize("value", [True, "1.0", float("nan"), float("inf")])
def test_costs_require_finite_real_values(field: str, value: object) -> None:
    with pytest.raises(DataValidationError) as exc_info:
        TransactionCostConfig(**{field: value})

    assert exc_info.value.context["field"] == field
    json.dumps(exc_info.value.as_dict(), allow_nan=False)


def test_numeric_overflow_is_reported_as_structured_validation_error() -> None:
    with pytest.raises(DataValidationError) as exc_info:
        TransactionCostConfig(proportional_bps=10**10_000)

    assert exc_info.value.context["field"] == "proportional_bps"
    json.dumps(exc_info.value.as_dict(), allow_nan=False)


@pytest.mark.parametrize(
    ("factory", "field"),
    [
        (
            lambda value: PortfolioConstraints(
                long_only=False,
                min_weight=value,
            ),
            "min_weight",
        ),
        (
            lambda value: TransactionCostConfig(proportional_bps=value),
            "proportional_bps",
        ),
        (
            lambda value: ResearchConfig(
                estimation_window=60,
                annualization_factor=value,
            ),
            "annualization_factor",
        ),
        (
            lambda value: ResearchConfig(
                estimation_window=60,
                components={"risk": {"value": value}},
            ),
            "components",
        ),
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        Fraction(-1, 10**1_000),
        Fraction(1, 2),
        Decimal("0.5"),
    ],
)
def test_numeric_contracts_reject_non_builtin_real_values(
    factory: Any,
    field: str,
    value: object,
) -> None:
    with pytest.raises(DataValidationError) as exc_info:
        factory(value)

    assert exc_info.value.context["field"] == field
    json.dumps(exc_info.value.as_dict(), allow_nan=False)


class _FloatSubclass(float):
    """Non-concrete numeric type used to verify the public numeric domain."""


@pytest.mark.parametrize(
    "factory",
    [
        lambda value: PortfolioConstraints(long_only=False, min_weight=value),
        lambda value: TransactionCostConfig(proportional_bps=value),
        lambda value: ResearchConfig(
            estimation_window=60,
            annualization_factor=value,
        ),
        lambda value: ResearchConfig(
            estimation_window=60,
            components={"risk": {"value": value}},
        ),
    ],
)
def test_numeric_contracts_reject_custom_float_subclasses(factory: Any) -> None:
    with pytest.raises(DataValidationError):
        factory(_FloatSubclass(0.5))


@pytest.mark.parametrize(
    "field",
    ["estimation_window", "holding_period", "rebalance_every"],
)
@pytest.mark.parametrize("value", [True, 1.0, "1"])
def test_research_windows_require_actual_integers(
    field: str,
    value: object,
) -> None:
    arguments = {"estimation_window": 60, field: value}

    with pytest.raises(DataValidationError) as exc_info:
        ResearchConfig(**arguments)

    assert exc_info.value.context["field"] == field


@pytest.mark.parametrize(
    "field",
    ["signal_availability_lag", "execution_lag"],
)
@pytest.mark.parametrize("value", [True, 1.0, "1"])
def test_research_lags_require_actual_integers(field: str, value: object) -> None:
    with pytest.raises(DataValidationError) as exc_info:
        ResearchConfig(estimation_window=60, **{field: value})

    assert exc_info.value.context["field"] == field


@pytest.mark.parametrize("value", [True, 1.0, "1"])
def test_random_seed_requires_actual_integer(value: object) -> None:
    with pytest.raises(DataValidationError) as exc_info:
        ResearchConfig(estimation_window=60, random_seed=value)

    assert exc_info.value.context["field"] == "random_seed"


@pytest.mark.parametrize(
    "value",
    [True, 0.0, -1.0, "12", float("nan"), float("inf")],
)
def test_annualization_factor_must_be_finite_positive_real(
    value: object,
) -> None:
    with pytest.raises(DataValidationError) as exc_info:
        ResearchConfig(estimation_window=60, annualization_factor=value)

    assert exc_info.value.context["field"] == "annualization_factor"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("return_convention", "simple"),
        ("missing_data_policy", "raise"),
    ],
)
def test_research_enums_require_runtime_enum_instances(
    field: str,
    value: object,
) -> None:
    with pytest.raises(DataValidationError) as exc_info:
        ResearchConfig(estimation_window=60, **{field: value})

    assert exc_info.value.context["field"] == field


@pytest.mark.parametrize("value", ["", "   ", 1])
def test_benchmark_must_be_none_or_nonblank_string(value: object) -> None:
    with pytest.raises(DataValidationError) as exc_info:
        ResearchConfig(estimation_window=60, benchmark=value)

    assert exc_info.value.context["field"] == "benchmark"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("constraints", {}),
        ("transaction_costs", {}),
    ],
)
def test_nested_configs_require_runtime_contract_instances(
    field: str,
    value: object,
) -> None:
    with pytest.raises(DataValidationError) as exc_info:
        ResearchConfig(estimation_window=60, **{field: value})

    assert exc_info.value.context["field"] == field


@pytest.mark.parametrize(
    "components",
    [
        [],
        {"": {"name": "sample"}},
        {"   ": {"name": "sample"}},
        {1: {"name": "sample"}},
        {"risk": []},
        {"risk": {1: "sample"}},
        {"risk": {"values": (1, 2)}},
        {"risk": {"values": {1, 2}}},
        {"risk": {"value": object()}},
        {"risk": {"value": float("nan")}},
        {"risk": {"value": float("inf")}},
        {"risk": {"nested": {"bad": (1, 2)}}},
        {"risk": {"nested": {1: "bad"}}},
    ],
)
def test_components_require_strict_standard_json_mappings(
    components: object,
) -> None:
    with pytest.raises(DataValidationError) as exc_info:
        ResearchConfig(estimation_window=60, components=components)

    assert exc_info.value.context["field"] == "components"
    json.dumps(exc_info.value.as_dict(), allow_nan=False)


@pytest.mark.parametrize("key", ["", "   "])
def test_components_reject_blank_nested_object_keys(key: str) -> None:
    with pytest.raises(DataValidationError) as exc_info:
        ResearchConfig(
            estimation_window=60,
            components={"risk": {"parameters": {key: 1}}},
        )

    assert exc_info.value.context["field"] == "components"
    serialized_error = json.dumps(exc_info.value.as_dict(), allow_nan=False)
    assert len(serialized_error) < 1_000


def test_components_reject_unrepresentable_integer_with_bounded_error() -> None:
    with pytest.raises(DataValidationError) as exc_info:
        ResearchConfig(
            estimation_window=60,
            components={"risk": {"extreme": 10**5_000}},
        )

    assert exc_info.value.context["field"] == "components"
    serialized_error = json.dumps(exc_info.value.as_dict(), allow_nan=False)
    assert len(serialized_error) < 1_000


@pytest.mark.parametrize("container", ["mapping", "list"])
def test_components_accept_documented_nesting_depth_with_fresh_snapshots(
    container: str,
) -> None:
    nested: Any = "leaf"
    for _ in range(100):
        nested = {"child": nested} if container == "mapping" else [nested]
    config = ResearchConfig(
        estimation_window=60,
        components={"risk": {"payload": nested}},
    )

    first = cast(Any, config.to_dict()["components"])["risk"]["payload"]
    for _ in range(100):
        first = first["child"] if container == "mapping" else first[0]
    assert first == "leaf"

    mutable = cast(Any, config.components)["risk"]["payload"]
    for _ in range(99):
        mutable = mutable["child"] if container == "mapping" else mutable[0]
    if container == "mapping":
        mutable["child"] = "changed"
    else:
        mutable[0] = "changed"

    fresh = cast(Any, config.components)["risk"]["payload"]
    for _ in range(100):
        fresh = fresh["child"] if container == "mapping" else fresh[0]
    assert fresh == "leaf"


@pytest.mark.parametrize("container", ["mapping", "list"])
@pytest.mark.parametrize("depth", [101, 1_500])
def test_components_reject_nesting_beyond_documented_limit_with_bounded_error(
    container: str,
    depth: int,
) -> None:
    nested: Any = "leaf"
    for _ in range(depth):
        nested = {"child": nested} if container == "mapping" else [nested]

    with pytest.raises(DataValidationError) as exc_info:
        ResearchConfig(
            estimation_window=60,
            components={"risk": {"payload": nested}},
        )

    assert exc_info.value.context["field"] == "components"
    assert exc_info.value.context["reason"] == "maximum_depth_exceeded"
    serialized_error = json.dumps(exc_info.value.as_dict(), allow_nan=False)
    assert len(serialized_error) < 1_000


def test_component_resource_budget_constants_are_documented_and_bounded() -> None:
    assert config_module._MAX_COMPONENT_JSON_DEPTH == _EXPECTED_MAX_COMPONENT_JSON_DEPTH
    assert config_module._MAX_COMPONENT_JSON_NODES == _EXPECTED_MAX_COMPONENT_JSON_NODES
    assert config_module._MAX_COMPONENT_JSON_BYTES == _EXPECTED_MAX_COMPONENT_JSON_BYTES
    assert config_module._MAX_COMPONENT_KEY_LENGTH == _EXPECTED_MAX_COMPONENT_KEY_LENGTH
    assert config_module._MAX_COMPONENT_STRING_LENGTH == _EXPECTED_MAX_COMPONENT_STRING_LENGTH


def test_components_accept_exact_key_and_string_length_boundaries() -> None:
    key = "k" * _EXPECTED_MAX_COMPONENT_KEY_LENGTH
    value = "v" * _EXPECTED_MAX_COMPONENT_STRING_LENGTH

    config = ResearchConfig(
        estimation_window=60,
        components={"risk": {key: value}},
    )

    assert cast(Any, config.components)["risk"][key] == value


def test_components_accept_exact_total_node_boundary() -> None:
    values = [None] * (_EXPECTED_MAX_COMPONENT_JSON_NODES - 3)

    config = ResearchConfig(
        estimation_window=60,
        components={"risk": {"values": values}},
    )

    assert len(cast(Any, config.components)["risk"]["values"]) == len(values)


@pytest.mark.parametrize(
    ("components", "reason"),
    [
        (
            {"risk": {"k" * (_EXPECTED_MAX_COMPONENT_KEY_LENGTH + 1): 1}},
            "maximum_key_length_exceeded",
        ),
        (
            {"risk": {" " * (_EXPECTED_MAX_COMPONENT_KEY_LENGTH + 1): 1}},
            "maximum_key_length_exceeded",
        ),
        (
            {"risk": {"value": "v" * (_EXPECTED_MAX_COMPONENT_STRING_LENGTH + 1)}},
            "maximum_string_length_exceeded",
        ),
        (
            {
                "risk": {
                    "values": [None] * (_EXPECTED_MAX_COMPONENT_JSON_NODES - 2),
                }
            },
            "maximum_nodes_exceeded",
        ),
        (
            {
                "risk": {
                    "values": [
                        "v" * _EXPECTED_MAX_COMPONENT_STRING_LENGTH
                        for _ in range(
                            _EXPECTED_MAX_COMPONENT_JSON_BYTES
                            // _EXPECTED_MAX_COMPONENT_STRING_LENGTH
                            + 1
                        )
                    ],
                }
            },
            "maximum_serialized_bytes_exceeded",
        ),
    ],
)
def test_components_reject_resource_budget_overruns_with_bounded_errors(
    components: object,
    reason: str,
) -> None:
    with pytest.raises(DataValidationError) as exc_info:
        ResearchConfig(estimation_window=60, components=components)

    assert exc_info.value.context["field"] == "components"
    assert exc_info.value.context["reason"] == reason
    serialized_error = json.dumps(exc_info.value.as_dict(), allow_nan=False)
    assert len(serialized_error) < 1_000


@pytest.mark.parametrize(
    "field",
    [
        "estimation_window",
        "holding_period",
        "rebalance_every",
        "signal_availability_lag",
        "execution_lag",
        "random_seed",
    ],
)
@pytest.mark.parametrize("sign", [-1, 1])
def test_research_integer_fields_reject_unrepresentable_values_with_bounded_error(
    field: str,
    sign: int,
) -> None:
    arguments = {"estimation_window": 60, field: sign * 10**5_000}

    with pytest.raises(DataValidationError) as exc_info:
        ResearchConfig(**arguments)

    assert exc_info.value.context["field"] == field
    serialized_error = json.dumps(exc_info.value.as_dict(), allow_nan=False)
    assert len(serialized_error) < 1_000


def test_components_are_deeply_isolated_from_caller_mutation() -> None:
    components = {
        "signal": {
            "name": "blend",
            "parameters": {"weights": [0.25, 0.75]},
        }
    }
    config = ResearchConfig(estimation_window=60, components=components)

    components["signal"]["parameters"]["weights"][0] = 999.0
    components["signal"]["name"] = "changed"

    assert config.to_dict()["components"] == {
        "signal": {
            "name": "blend",
            "parameters": {"weights": [0.25, 0.75]},
        }
    }


def test_components_property_returns_fresh_deep_snapshots() -> None:
    config = ResearchConfig(
        estimation_window=60,
        components={"risk": {"parameters": {"weights": [0.25, 0.75]}}},
    )

    first = config.components
    mutable_first = cast(Any, first)
    mutable_first["risk"]["parameters"]["weights"][0] = 999.0
    with pytest.raises(TypeError):
        mutable_first["new"] = {"name": "injected"}

    assert config.components == {"risk": {"parameters": {"weights": [0.25, 0.75]}}}


def test_to_dict_returns_fresh_deep_snapshots_with_sorted_component_keys() -> None:
    config = ResearchConfig(
        estimation_window=60,
        components={
            "zeta": {"z": 1, "a": {"z": 2, "a": [3]}},
            "alpha": {"name": "sample"},
        },
    )

    first = config.to_dict()
    first_components = cast(Any, first["components"])
    first_components["zeta"]["a"]["a"][0] = 999
    first["constraints"] = {}
    second = config.to_dict()

    second_components = cast(Any, second["components"])
    assert list(second_components) == ["alpha", "zeta"]
    assert list(second_components["zeta"]) == ["a", "z"]
    assert list(second_components["zeta"]["a"]) == ["a", "z"]
    assert second_components["zeta"]["a"]["a"] == [3]
    assert second["constraints"] == PortfolioConstraints().to_dict()


def test_config_contracts_are_frozen_and_slotted() -> None:
    config = ResearchConfig(estimation_window=60)

    assert dataclasses.is_dataclass(config)
    assert not hasattr(config, "__dict__")
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.random_seed = 9


def test_research_config_exposes_components_as_real_public_dataclass_field() -> None:
    field_names = [definition.name for definition in dataclasses.fields(ResearchConfig)]

    assert field_names == [
        "estimation_window",
        "holding_period",
        "rebalance_every",
        "signal_availability_lag",
        "execution_lag",
        "return_convention",
        "missing_data_policy",
        "constraints",
        "transaction_costs",
        "benchmark",
        "annualization_factor",
        "random_seed",
        "components",
    ]
    assert not any(name.startswith("_") for name in field_names)


def test_research_config_replace_preserves_isolation_and_accepts_components() -> None:
    original = ResearchConfig(
        estimation_window=60,
        components={"risk": {"weights": [0.25, 0.75]}},
    )

    replaced = dataclasses.replace(
        original,
        random_seed=7,
        components={"risk": {"weights": [1.0]}},
    )
    preserved = dataclasses.replace(original, random_seed=8)

    assert original.random_seed == 0
    assert original.to_dict()["components"] == {"risk": {"weights": [0.25, 0.75]}}
    assert replaced.random_seed == 7
    assert replaced.to_dict()["components"] == {"risk": {"weights": [1.0]}}
    assert preserved.random_seed == 8
    assert preserved.to_dict()["components"] == {"risk": {"weights": [0.25, 0.75]}}


def test_research_config_asdict_exposes_only_public_deeply_isolated_fields() -> None:
    config = ResearchConfig(
        estimation_window=60,
        components={"risk": {"weights": [0.25, 0.75]}},
    )

    snapshot = dataclasses.asdict(config)

    assert set(snapshot) == {definition.name for definition in dataclasses.fields(ResearchConfig)}
    assert "components" in snapshot
    assert not any(name.startswith("_") for name in snapshot)
    mutable_weights = cast(Any, snapshot["components"])["risk"]["weights"]
    mutable_weights[0] = 999.0
    assert config.to_dict()["components"] == {"risk": {"weights": [0.25, 0.75]}}


def test_dataclass_fields_support_exact_allowed_key_routing() -> None:
    supplied: dict[str, Any] = {
        "estimation_window": 60,
        "random_seed": 7,
        "components": {"risk": {"name": "sample"}},
        "unknown": "rejected",
    }
    allowed = {
        definition.name for definition in dataclasses.fields(ResearchConfig) if definition.init
    }
    routed = {key: value for key, value in supplied.items() if key in allowed}
    rejected = set(supplied) - allowed

    config = ResearchConfig(**routed)

    assert rejected == {"unknown"}
    assert config.random_seed == 7
    assert config.to_dict()["components"] == {"risk": {"name": "sample"}}


@pytest.mark.parametrize(
    ("left_value", "right_value"),
    [
        (True, 1),
        (1, 1.0),
        (-0.0, 0.0),
    ],
)
def test_component_and_config_equality_use_canonical_json_scalar_identity(
    left_value: object,
    right_value: object,
) -> None:
    left = ResearchConfig(
        estimation_window=60,
        components={"risk": {"value": left_value}},
    )
    right = ResearchConfig(
        estimation_window=60,
        components={"risk": {"value": right_value}},
    )

    assert left.components != right.components
    assert left != right
    assert len({left, right}) == 2


def test_canonical_component_equality_ignores_mapping_insertion_order() -> None:
    left = ResearchConfig(
        estimation_window=60,
        components={
            "zeta": {"b": 2, "a": 1},
            "alpha": {"name": "sample"},
        },
    )
    right = ResearchConfig(
        estimation_window=60,
        components={
            "alpha": {"name": "sample"},
            "zeta": {"a": 1, "b": 2},
        },
    )

    assert left.components == right.components
    assert left == right
    assert hash(left.components) == hash(right.components)
    assert hash(left) == hash(right)


def test_component_mapping_comparison_normalizes_valid_mappings_safely() -> None:
    config = ResearchConfig(
        estimation_window=60,
        components={"risk": {"value": True}},
    )

    assert config.components == {"risk": {"value": True}}
    assert config.components != {"risk": {"value": 1}}
    assert config.components != {"risk": {"value": object()}}
    assert config.components != {"risk": []}
    assert config.components != object()


def test_asdict_retains_immutable_mapping_and_to_dict_is_json_boundary() -> None:
    config = ResearchConfig(
        estimation_window=60,
        components={"risk": {"name": "sample"}},
    )

    dataclass_snapshot = dataclasses.asdict(config)

    assert dataclass_snapshot["components"] is config.components
    with pytest.raises(TypeError):
        json.dumps(dataclass_snapshot, allow_nan=False)
    assert json.loads(json.dumps(config.to_dict(), allow_nan=False)) == config.to_dict()


def test_research_config_supports_safe_pickle_round_trip() -> None:
    config = ResearchConfig(
        estimation_window=60,
        random_seed=7,
        components={"risk": {"weights": [0.25, 0.75]}},
    )

    restored = pickle.loads(pickle.dumps(config))

    assert restored == config
    assert hash(restored) == hash(config)
    assert restored.to_dict() == config.to_dict()
    with pytest.raises(TypeError):
        cast(Any, restored.components)["new"] = {"name": "injected"}
