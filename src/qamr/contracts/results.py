"""Structured, immutable calculation diagnostics."""

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import TypeAlias, cast

from qamr._types import JsonScalar, JsonValue
from qamr.errors import DataValidationError

_MAX_JSON_INTEGER_BITS = 14_000
# These match the resolved-configuration boundary so all embedded JSON has
# deterministic limits and validation stops before Python's recursion limit.
_MAX_CONTEXT_JSON_DEPTH = 100
_MAX_CONTEXT_JSON_NODES = 2_000
_MAX_CONTEXT_JSON_BYTES = 262_144
_MAX_CONTEXT_KEY_LENGTH = 256
_MAX_CONTEXT_STRING_LENGTH = 32_768
# Diagnostic text is kept bounded in both characters and UTF-8 bytes. The byte
# limits reflect the maximum four-byte UTF-8 encoding at each character cap.
_MAX_DIAGNOSTIC_CODE_LENGTH = 128
_MAX_DIAGNOSTIC_CODE_BYTES = 512
_MAX_DIAGNOSTIC_MESSAGE_LENGTH = 4_096
_MAX_DIAGNOSTIC_MESSAGE_BYTES = 16_384

ReadOnlyJsonValue: TypeAlias = (
    JsonScalar | tuple["ReadOnlyJsonValue", ...] | Mapping[str, "ReadOnlyJsonValue"]
)
_MISSING_ERROR_VALUE = object()
_DEFAULT_DIAGNOSTIC_CONTEXT: dict[str, JsonValue] = {}


def _bounded_type_name(value: object) -> str:
    return type(value).__name__[:64]


def _context_error(
    reason: str,
    path: list[str | int],
    *,
    invalid_value: object = _MISSING_ERROR_VALUE,
) -> DataValidationError:
    bounded_path: list[JsonValue] = [
        segment[:40] if isinstance(segment, str) else segment for segment in path[:8]
    ]
    context: dict[str, JsonValue] = {
        "field": "context",
        "reason": reason,
        "path": bounded_path,
    }
    if invalid_value is not _MISSING_ERROR_VALUE:
        context["dtype"] = _bounded_type_name(invalid_value)
    return DataValidationError(
        "diagnostic context must contain only standard JSON values",
        context=context,
    )


class _ContextBudget:
    __slots__ = ("nodes", "string_bytes")

    def __init__(self) -> None:
        self.nodes = 1
        self.string_bytes = 0

    def consume_node(self, path: list[str | int]) -> None:
        self.nodes += 1
        if self.nodes > _MAX_CONTEXT_JSON_NODES:
            raise _context_error("maximum_nodes_exceeded", path)

    def consume_text(
        self,
        value: str,
        *,
        is_key: bool,
        path: list[str | int],
    ) -> None:
        maximum = _MAX_CONTEXT_KEY_LENGTH if is_key else _MAX_CONTEXT_STRING_LENGTH
        if len(value) > maximum:
            reason = "maximum_key_length_exceeded" if is_key else "maximum_string_length_exceeded"
            raise _context_error(reason, path)
        self.string_bytes += len(json.dumps(value).encode("utf-8"))
        if self.string_bytes > _MAX_CONTEXT_JSON_BYTES:
            raise _context_error("maximum_serialized_bytes_exceeded", path)


def _normalize_json(
    value: object,
    *,
    path: list[str | int],
    active_containers: set[int],
    seen_containers: set[int],
    depth: int,
    budget: _ContextBudget,
) -> JsonValue:
    budget.consume_node(path)
    if value is None or type(value) is bool:
        return value
    if type(value) is str:
        budget.consume_text(value, is_key=False, path=path)
        return value
    if type(value) is int:
        if value.bit_length() > _MAX_JSON_INTEGER_BITS:
            raise _context_error("integer_not_json_representable", path)
        return value
    if type(value) is float:
        if not isfinite(value):
            raise _context_error("not_finite", path)
        return value
    if type(value) is list or type(value) is dict:
        if depth > _MAX_CONTEXT_JSON_DEPTH:
            raise _context_error("maximum_depth_exceeded", path)
        container_id = id(value)
        if container_id in active_containers:
            raise _context_error("cyclic", path)
        if container_id in seen_containers:
            raise _context_error("aliased_container", path)
        seen_containers.add(container_id)
        active_containers.add(container_id)
        try:
            if type(value) is list:
                return [
                    _normalize_json(
                        item,
                        path=[*path, position],
                        active_containers=active_containers,
                        seen_containers=seen_containers,
                        depth=depth + 1,
                        budget=budget,
                    )
                    for position, item in enumerate(cast(list[object], value))
                ]
            normalized: dict[str, JsonValue] = {}
            for key, item in cast(dict[object, object], value).items():
                if type(key) is not str:
                    raise _context_error("non_string_key", path, invalid_value=key)
                budget.consume_text(key, is_key=True, path=[*path, key])
                if not key.strip():
                    raise _context_error("blank_key", [*path, key])
                normalized[key] = _normalize_json(
                    item,
                    path=[*path, key],
                    active_containers=active_containers,
                    seen_containers=seen_containers,
                    depth=depth + 1,
                    budget=budget,
                )
            return normalized
        finally:
            active_containers.remove(container_id)
    raise _context_error("unsupported_type", path, invalid_value=value)


def _to_read_only_json(value: JsonValue) -> ReadOnlyJsonValue:
    if isinstance(value, dict):
        canonical_json = json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return _ImmutableJsonMapping(canonical_json)
    if isinstance(value, list):
        return tuple(_to_read_only_json(nested_value) for nested_value in value)
    return value


class _ImmutableJsonMapping(Mapping[str, ReadOnlyJsonValue]):
    """Immutable JSON object with canonical, same-wrapper value semantics."""

    __slots__ = ("_canonical_json",)
    _canonical_json: str

    def __init__(self, canonical_json: str) -> None:
        object.__setattr__(self, "_canonical_json", canonical_json)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("diagnostic context is immutable")

    def __getitem__(self, key: str) -> ReadOnlyJsonValue:
        materialized = self.to_dict()
        return _to_read_only_json(materialized[key])

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    def __deepcopy__(self, memo: dict[int, object]) -> "_ImmutableJsonMapping":
        return self

    def __reduce__(self) -> tuple[type["_ImmutableJsonMapping"], tuple[str]]:
        return (type(self), (self._canonical_json,))

    def __repr__(self) -> str:
        return repr(self.to_dict())

    def __eq__(self, other: object) -> bool:
        if type(other) is not _ImmutableJsonMapping:
            return False
        assert isinstance(other, _ImmutableJsonMapping)
        return self._canonical_json == other._canonical_json

    def __hash__(self) -> int:
        return hash(self._canonical_json)

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a fresh ordinary JSON object."""
        return cast(dict[str, JsonValue], json.loads(self._canonical_json))


def _immutable_context(value: object) -> _ImmutableJsonMapping:
    if type(value) is not dict:
        raise _context_error("outer_not_builtin_dict", [], invalid_value=value)
    budget = _ContextBudget()
    normalized = _normalize_json(
        value,
        path=[],
        active_containers=set(),
        seen_containers=set(),
        depth=0,
        budget=budget,
    )
    assert isinstance(normalized, dict)
    try:
        serialized = json.dumps(
            normalized,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (OverflowError, RecursionError, ValueError) as error:
        raise _context_error("not_json_representable", []) from error
    if len(serialized.encode("utf-8")) > _MAX_CONTEXT_JSON_BYTES:
        raise _context_error("maximum_serialized_bytes_exceeded", [])
    return _ImmutableJsonMapping(serialized)


def _require_bounded_nonblank_string(
    name: str,
    value: object,
    *,
    maximum_length: int,
    maximum_bytes: int,
) -> None:
    if type(value) is not str:
        raise DataValidationError(
            f"diagnostic {name} must be a string",
            context={"field": name, "dtype": _bounded_type_name(value)},
        )
    if not value.strip():
        raise DataValidationError(
            f"diagnostic {name} must not be blank",
            context={"field": name},
        )
    if len(value) > maximum_length:
        raise DataValidationError(
            f"diagnostic {name} exceeds its maximum length",
            context={
                "field": name,
                "reason": "maximum_length_exceeded",
                "maximum_length": maximum_length,
            },
        )
    if len(value.encode("utf-8")) > maximum_bytes:
        raise DataValidationError(
            f"diagnostic {name} exceeds its maximum UTF-8 size",
            context={
                "field": name,
                "reason": "maximum_bytes_exceeded",
                "maximum_bytes": maximum_bytes,
            },
        )


class DiagnosticSeverity(str, Enum):
    """Supported severity levels for numerical diagnostics."""

    INFO = "info"
    WARNING = "warning"


@dataclass(frozen=True, slots=True, init=False)
class NumericalDiagnostic:
    """A hashable diagnostic value; use :meth:`as_dict` for JSON persistence.

    ``dataclasses.asdict`` is safe but retains the immutable context wrapper.
    """

    code: str
    severity: DiagnosticSeverity
    message: str
    context: Mapping[str, ReadOnlyJsonValue] = field(repr=False)

    def __init__(
        self,
        code: str,
        severity: DiagnosticSeverity,
        message: str,
        context: dict[str, JsonValue] = _DEFAULT_DIAGNOSTIC_CONTEXT,
    ) -> None:
        _require_bounded_nonblank_string(
            "code",
            code,
            maximum_length=_MAX_DIAGNOSTIC_CODE_LENGTH,
            maximum_bytes=_MAX_DIAGNOSTIC_CODE_BYTES,
        )
        if type(severity) is not DiagnosticSeverity:
            raise DataValidationError(
                "diagnostic severity must be a DiagnosticSeverity",
                context={
                    "field": "severity",
                    "dtype": _bounded_type_name(severity),
                },
            )
        _require_bounded_nonblank_string(
            "message",
            message,
            maximum_length=_MAX_DIAGNOSTIC_MESSAGE_LENGTH,
            maximum_bytes=_MAX_DIAGNOSTIC_MESSAGE_BYTES,
        )
        immutable_context = _immutable_context(context)
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "context", immutable_context)

    def as_dict(self) -> dict[str, JsonValue]:
        """Return a fresh deterministic JSON representation."""
        context = cast(_ImmutableJsonMapping, self.context)
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "context": context.to_dict(),
        }
