"""Immutable, deterministic resolved research configuration."""

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from math import isfinite
from typing import cast

from qamr._types import JsonValue
from qamr.contracts.dataset import MissingDataPolicy, ReturnConvention
from qamr.errors import DataValidationError, InfeasiblePortfolioError

_MAX_JSON_INTEGER_BITS = 14_000
# Fixed component budgets keep validation deterministic and resource-bounded:
# nodes count the outer object and every JSON value; lengths count characters;
# bytes count the final canonical UTF-8 JSON representation.
_MAX_COMPONENT_JSON_DEPTH = 100
_MAX_COMPONENT_JSON_NODES = 2_000
_MAX_COMPONENT_JSON_BYTES = 262_144
_MAX_COMPONENT_KEY_LENGTH = 256
_MAX_COMPONENT_STRING_LENGTH = 32_768


def _require_finite_real(name: str, value: object) -> float:
    if type(value) is not int and type(value) is not float:
        raise DataValidationError(
            f"{name} must be a built-in int or float",
            context={"field": name, "dtype": type(value).__name__},
        )
    try:
        normalized = float(value)
    except OverflowError as error:
        raise DataValidationError(
            f"{name} must be representable as a finite float",
            context={"field": name, "reason": "not_representable"},
        ) from error
    if not isfinite(normalized):
        raise DataValidationError(
            f"{name} must be a finite real number",
            context={"field": name, "reason": "not_finite"},
        )
    return normalized


def _is_json_representable_integer(value: int) -> bool:
    if value.bit_length() > _MAX_JSON_INTEGER_BITS:
        return False
    try:
        json.dumps(value)
    except (OverflowError, ValueError):
        return False
    return True


def _require_json_representable_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataValidationError(
            f"{name} must be an integer",
            context={"field": name, "dtype": type(value).__name__},
        )
    if not _is_json_representable_integer(value):
        raise DataValidationError(
            f"{name} must be representable as a JSON integer",
            context={"field": name, "reason": "not_json_representable"},
        )
    return value


def _require_positive_integer(name: str, value: object) -> None:
    validated = _require_json_representable_integer(name, value)
    if validated <= 0:
        raise DataValidationError(
            f"{name} must be positive",
            context={"field": name, "reason": "not_positive"},
        )


def _require_nonnegative_integer(name: str, value: object) -> None:
    validated = _require_json_representable_integer(name, value)
    if validated < 0:
        raise DataValidationError(
            f"{name} must be nonnegative",
            context={"field": name, "reason": "negative"},
        )


def _require_actual_bool(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise DataValidationError(
            f"{name} must be a boolean",
            context={"field": name, "dtype": type(value).__name__},
        )


def _components_error(reason: str, path: list[str | int]) -> DataValidationError:
    bounded_path: list[JsonValue] = [
        segment[:40] if isinstance(segment, str) else segment for segment in path[:8]
    ]
    return DataValidationError(
        "components must contain only standard JSON values",
        context={
            "field": "components",
            "reason": reason,
            "path": bounded_path,
        },
    )


class _ComponentBudget:
    __slots__ = ("nodes", "string_bytes")

    def __init__(self) -> None:
        self.nodes = 1
        self.string_bytes = 0

    def consume_node(self, path: list[str | int]) -> None:
        self.nodes += 1
        if self.nodes > _MAX_COMPONENT_JSON_NODES:
            raise _components_error("maximum_nodes_exceeded", path)

    def consume_text(
        self,
        value: str,
        *,
        is_key: bool,
        path: list[str | int],
    ) -> None:
        maximum = _MAX_COMPONENT_KEY_LENGTH if is_key else _MAX_COMPONENT_STRING_LENGTH
        if len(value) > maximum:
            reason = "maximum_key_length_exceeded" if is_key else "maximum_string_length_exceeded"
            raise _components_error(reason, path)
        self.string_bytes += len(json.dumps(value).encode("utf-8"))
        if self.string_bytes > _MAX_COMPONENT_JSON_BYTES:
            raise _components_error("maximum_serialized_bytes_exceeded", path)


def _normalize_json(
    value: object,
    *,
    path: list[str | int],
    active_containers: set[int],
    depth: int,
    budget: _ComponentBudget,
) -> JsonValue:
    budget.consume_node(path)
    if value is None or type(value) is bool:
        return value
    if type(value) is str:
        budget.consume_text(value, is_key=False, path=path)
        return value
    if type(value) is int:
        if not _is_json_representable_integer(value):
            raise _components_error("integer_not_json_representable", path)
        return value
    if type(value) is float:
        if not isfinite(value):
            raise _components_error("not_finite", path)
        return value
    if type(value) is list:
        if depth > _MAX_COMPONENT_JSON_DEPTH:
            raise _components_error("maximum_depth_exceeded", path)
        container_id = id(value)
        if container_id in active_containers:
            raise _components_error("cyclic", path)
        active_containers.add(container_id)
        try:
            return [
                _normalize_json(
                    item,
                    path=[*path, position],
                    active_containers=active_containers,
                    depth=depth + 1,
                    budget=budget,
                )
                for position, item in enumerate(value)
            ]
        finally:
            active_containers.remove(container_id)
    if isinstance(value, Mapping):
        if depth > _MAX_COMPONENT_JSON_DEPTH:
            raise _components_error("maximum_depth_exceeded", path)
        container_id = id(value)
        if container_id in active_containers:
            raise _components_error("cyclic", path)
        active_containers.add(container_id)
        try:
            normalized: dict[str, JsonValue] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise _components_error("non_string_key", path)
                budget.consume_text(key, is_key=True, path=[*path, key])
                if not key.strip():
                    raise _components_error("blank_key", [*path, key])
                normalized[key] = _normalize_json(
                    item,
                    path=[*path, key],
                    active_containers=active_containers,
                    depth=depth + 1,
                    budget=budget,
                )
            return normalized
        finally:
            active_containers.remove(container_id)
    raise _components_error("unsupported_type", path)


def _normalize_components(value: object) -> dict[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise _components_error("outer_not_mapping", [])
    budget = _ComponentBudget()
    normalized: dict[str, JsonValue] = {}
    for name, component in value.items():
        if not isinstance(name, str):
            raise _components_error("non_string_component_name", [])
        budget.consume_text(name, is_key=True, path=[name])
        if not name.strip():
            raise _components_error("blank_component_name", [name])
        if not isinstance(component, Mapping):
            raise _components_error("component_not_mapping", [name])
        try:
            normalized[name] = _normalize_json(
                component,
                path=[name],
                active_containers=set(),
                depth=0,
                budget=budget,
            )
        except RecursionError as error:
            raise _components_error("maximum_depth_exceeded", [name]) from error
    return normalized


def _canonicalize_components(value: object) -> str:
    normalized_components = _normalize_components(value)
    try:
        canonical_json = json.dumps(
            normalized_components,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (OverflowError, RecursionError, ValueError) as error:
        raise _components_error("not_json_representable", []) from error
    if len(canonical_json.encode("utf-8")) > _MAX_COMPONENT_JSON_BYTES:
        raise _components_error("maximum_serialized_bytes_exceeded", [])
    return canonical_json


@dataclass(frozen=True, slots=True)
class PortfolioConstraints:
    """Resolved portfolio feasibility constraints."""

    long_only: bool = True
    min_weight: float | None = 0.0
    max_weight: float | None = 1.0
    gross_leverage: float | None = 1.0
    net_exposure: float | None = 1.0
    benchmark_relative: bool = False

    def __post_init__(self) -> None:
        _require_actual_bool("long_only", self.long_only)
        _require_actual_bool("benchmark_relative", self.benchmark_relative)
        for name in (
            "min_weight",
            "max_weight",
            "gross_leverage",
            "net_exposure",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _require_finite_real(name, value))
        if (
            self.min_weight is not None
            and self.max_weight is not None
            and self.min_weight > self.max_weight
        ):
            raise InfeasiblePortfolioError(
                "minimum weight cannot exceed maximum weight",
                context={"field": "min_weight", "reason": "above_maximum"},
            )
        if self.long_only and self.min_weight is not None and self.min_weight < 0:
            raise InfeasiblePortfolioError(
                "long-only minimum weight cannot be negative",
                context={"field": "min_weight", "value": self.min_weight},
            )
        if self.long_only and self.max_weight is not None and self.max_weight < 0:
            raise InfeasiblePortfolioError(
                "long-only maximum weight cannot be negative",
                context={"field": "max_weight", "value": self.max_weight},
            )
        if self.gross_leverage is not None and self.gross_leverage <= 0:
            raise InfeasiblePortfolioError(
                "gross leverage must be positive",
                context={"field": "gross_leverage", "value": self.gross_leverage},
            )
        if self.long_only and self.net_exposure is not None and self.net_exposure < 0:
            raise InfeasiblePortfolioError(
                "long-only net exposure cannot be negative",
                context={"field": "net_exposure", "value": self.net_exposure},
            )
        if (
            self.min_weight is not None
            and self.min_weight >= 0
            and self.net_exposure is not None
            and self.net_exposure < 0
        ):
            raise InfeasiblePortfolioError(
                "nonnegative weights cannot have negative net exposure",
                context={"field": "net_exposure", "reason": "negative_with_nonnegative_bounds"},
            )
        if (
            self.max_weight is not None
            and self.max_weight <= 0
            and self.net_exposure is not None
            and self.net_exposure > 0
        ):
            raise InfeasiblePortfolioError(
                "nonpositive weights cannot have positive net exposure",
                context={"field": "net_exposure", "reason": "positive_with_nonpositive_bounds"},
            )
        if (
            self.min_weight is not None
            and self.min_weight > 0
            and self.net_exposure is not None
            and self.net_exposure < self.min_weight
        ):
            raise InfeasiblePortfolioError(
                "net exposure cannot be below a positive per-position minimum",
                context={"field": "net_exposure", "reason": "below_positive_minimum"},
            )
        if (
            self.max_weight is not None
            and self.max_weight < 0
            and self.net_exposure is not None
            and self.net_exposure > self.max_weight
        ):
            raise InfeasiblePortfolioError(
                "net exposure cannot exceed a negative per-position maximum",
                context={"field": "net_exposure", "reason": "above_negative_maximum"},
            )
        if (
            self.net_exposure is None
            and self.gross_leverage is not None
            and self.min_weight is not None
            and self.min_weight > self.gross_leverage
        ):
            raise InfeasiblePortfolioError(
                "positive minimum weight cannot exceed gross leverage",
                context={"field": "min_weight", "reason": "exceeds_gross"},
            )
        if (
            self.net_exposure is None
            and self.gross_leverage is not None
            and self.max_weight is not None
            and self.max_weight < 0
            and abs(self.max_weight) > self.gross_leverage
        ):
            raise InfeasiblePortfolioError(
                "negative maximum weight magnitude cannot exceed gross leverage",
                context={"field": "max_weight", "reason": "exceeds_gross"},
            )
        if (
            self.gross_leverage is not None
            and self.net_exposure is not None
            and abs(self.net_exposure) > self.gross_leverage
        ):
            raise InfeasiblePortfolioError(
                "net exposure magnitude cannot exceed gross leverage",
                context={"field": "net_exposure", "reason": "exceeds_gross"},
            )

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a JSON-safe constraints snapshot."""
        return {
            "benchmark_relative": self.benchmark_relative,
            "gross_leverage": self.gross_leverage,
            "long_only": self.long_only,
            "max_weight": self.max_weight,
            "min_weight": self.min_weight,
            "net_exposure": self.net_exposure,
        }


@dataclass(frozen=True, slots=True)
class TransactionCostConfig:
    """Resolved transaction cost assumptions."""

    proportional_bps: float = 0.0
    fixed_per_trade: float = 0.0

    def __post_init__(self) -> None:
        for name in ("proportional_bps", "fixed_per_trade"):
            normalized = _require_finite_real(name, getattr(self, name))
            if normalized < 0:
                raise DataValidationError(
                    f"{name} must be nonnegative",
                    context={"field": name, "value": normalized},
                )
            object.__setattr__(self, name, normalized)

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a JSON-safe transaction-cost snapshot."""
        return {
            "fixed_per_trade": self.fixed_per_trade,
            "proportional_bps": self.proportional_bps,
        }


_DEFAULT_CONSTRAINTS = PortfolioConstraints()
_DEFAULT_TRANSACTION_COSTS = TransactionCostConfig()


class _ImmutableComponents(Mapping[str, Mapping[str, JsonValue]]):
    """Canonical immutable mapping whose nested reads are defensive snapshots."""

    __slots__ = ("_canonical_json",)
    _canonical_json: str

    def __init__(self, canonical_json: str) -> None:
        object.__setattr__(self, "_canonical_json", canonical_json)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("components are immutable")

    def __getitem__(self, key: str) -> Mapping[str, JsonValue]:
        materialized = self.to_dict()
        return cast(dict[str, JsonValue], materialized[key])

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    def __deepcopy__(self, memo: dict[int, object]) -> "_ImmutableComponents":
        return self

    def __repr__(self) -> str:
        return repr(self.to_dict())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _ImmutableComponents):
            return self._canonical_json == other._canonical_json
        if not isinstance(other, Mapping):
            return NotImplemented
        try:
            return self._canonical_json == _canonicalize_components(other)
        except Exception:
            return False

    def __hash__(self) -> int:
        return hash(self._canonical_json)

    def __reduce__(self) -> tuple[type["_ImmutableComponents"], tuple[str]]:
        return (type(self), (self._canonical_json,))

    def to_dict(self) -> dict[str, JsonValue]:
        """Materialize a fresh normal JSON object."""
        return cast(dict[str, JsonValue], json.loads(self._canonical_json))


@dataclass(frozen=True, slots=True, init=False)
class ResearchConfig:
    """One resolved configuration shared by all research workflows."""

    estimation_window: int
    holding_period: int
    rebalance_every: int
    signal_availability_lag: int
    execution_lag: int
    return_convention: ReturnConvention
    missing_data_policy: MissingDataPolicy
    constraints: PortfolioConstraints
    transaction_costs: TransactionCostConfig
    benchmark: str | None
    annualization_factor: float | None
    random_seed: int
    components: Mapping[str, Mapping[str, JsonValue]]

    def __init__(
        self,
        estimation_window: int,
        holding_period: int = 1,
        rebalance_every: int = 1,
        signal_availability_lag: int = 0,
        execution_lag: int = 1,
        return_convention: ReturnConvention = ReturnConvention.SIMPLE,
        missing_data_policy: MissingDataPolicy = MissingDataPolicy.RAISE,
        constraints: PortfolioConstraints = _DEFAULT_CONSTRAINTS,
        transaction_costs: TransactionCostConfig = _DEFAULT_TRANSACTION_COSTS,
        benchmark: str | None = None,
        annualization_factor: float | None = None,
        random_seed: int = 0,
        components: Mapping[str, Mapping[str, JsonValue]] | None = None,
    ) -> None:
        _require_positive_integer("estimation_window", estimation_window)
        _require_positive_integer("holding_period", holding_period)
        _require_positive_integer("rebalance_every", rebalance_every)
        _require_nonnegative_integer("signal_availability_lag", signal_availability_lag)
        _require_nonnegative_integer("execution_lag", execution_lag)
        _require_json_representable_integer("random_seed", random_seed)
        if not isinstance(return_convention, ReturnConvention):
            raise DataValidationError(
                "return_convention must be a ReturnConvention",
                context={
                    "field": "return_convention",
                    "dtype": type(return_convention).__name__,
                },
            )
        if not isinstance(missing_data_policy, MissingDataPolicy):
            raise DataValidationError(
                "missing_data_policy must be a MissingDataPolicy",
                context={
                    "field": "missing_data_policy",
                    "dtype": type(missing_data_policy).__name__,
                },
            )
        if not isinstance(constraints, PortfolioConstraints):
            raise DataValidationError(
                "constraints must be a PortfolioConstraints",
                context={
                    "field": "constraints",
                    "dtype": type(constraints).__name__,
                },
            )
        if not isinstance(transaction_costs, TransactionCostConfig):
            raise DataValidationError(
                "transaction_costs must be a TransactionCostConfig",
                context={
                    "field": "transaction_costs",
                    "dtype": type(transaction_costs).__name__,
                },
            )
        if benchmark is not None:
            if not isinstance(benchmark, str):
                raise DataValidationError(
                    "benchmark must be a string or None",
                    context={
                        "field": "benchmark",
                        "dtype": type(benchmark).__name__,
                    },
                )
            if not benchmark.strip():
                raise DataValidationError(
                    "benchmark must not be blank",
                    context={"field": "benchmark"},
                )
        normalized_annualization: float | None = None
        if annualization_factor is not None:
            normalized_annualization = _require_finite_real(
                "annualization_factor",
                annualization_factor,
            )
            if normalized_annualization <= 0:
                raise DataValidationError(
                    "annualization_factor must be positive",
                    context={
                        "field": "annualization_factor",
                        "value": normalized_annualization,
                    },
                )
        components_value = {} if components is None else components
        components_json = _canonicalize_components(components_value)
        object.__setattr__(self, "estimation_window", estimation_window)
        object.__setattr__(self, "holding_period", holding_period)
        object.__setattr__(self, "rebalance_every", rebalance_every)
        object.__setattr__(self, "signal_availability_lag", signal_availability_lag)
        object.__setattr__(self, "execution_lag", execution_lag)
        object.__setattr__(self, "return_convention", return_convention)
        object.__setattr__(self, "missing_data_policy", missing_data_policy)
        object.__setattr__(self, "constraints", constraints)
        object.__setattr__(self, "transaction_costs", transaction_costs)
        object.__setattr__(self, "benchmark", benchmark)
        object.__setattr__(self, "annualization_factor", normalized_annualization)
        object.__setattr__(self, "random_seed", random_seed)
        object.__setattr__(self, "components", _ImmutableComponents(components_json))

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the fresh JSON persistence representation.

        ``dataclasses.asdict`` intentionally retains the immutable components
        mapping wrapper; callers needing standard JSON containers use this method.
        """
        return {
            "annualization_factor": self.annualization_factor,
            "benchmark": self.benchmark,
            "components": cast(_ImmutableComponents, self.components).to_dict(),
            "constraints": self.constraints.to_dict(),
            "estimation_window": self.estimation_window,
            "execution_lag": self.execution_lag,
            "holding_period": self.holding_period,
            "missing_data_policy": self.missing_data_policy.value,
            "random_seed": self.random_seed,
            "rebalance_every": self.rebalance_every,
            "return_convention": self.return_convention.value,
            "signal_availability_lag": self.signal_availability_lag,
            "transaction_costs": self.transaction_costs.to_dict(),
        }
