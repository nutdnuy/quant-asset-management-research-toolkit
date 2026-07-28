"""Structured public exceptions for QAMR."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import ClassVar, TypeAlias

from qamr._types import JsonScalar, JsonValue

_ReadOnlyJsonValue: TypeAlias = (
    JsonScalar | tuple["_ReadOnlyJsonValue", ...] | Mapping[str, "_ReadOnlyJsonValue"]
)


def _freeze_json(value: JsonValue) -> _ReadOnlyJsonValue:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(nested_value) for key, nested_value in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(nested_value) for nested_value in value)
    return value


def _materialize_json(value: _ReadOnlyJsonValue) -> JsonValue:
    if isinstance(value, Mapping):
        return {key: _materialize_json(nested_value) for key, nested_value in value.items()}
    if isinstance(value, tuple):
        return [_materialize_json(nested_value) for nested_value in value]
    return value


class QAMRError(Exception):
    """Base class for structured QAMR exceptions."""

    code: ClassVar[str] = "qamr_error"

    def __init__(
        self,
        message: str,
        *,
        context: Mapping[str, JsonValue] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        supplied_context = {} if context is None else context
        self.context: Mapping[str, _ReadOnlyJsonValue] = MappingProxyType(
            {key: _freeze_json(value) for key, value in supplied_context.items()}
        )

    def as_dict(self) -> dict[str, JsonValue]:
        """Return a JSON-safe representation of the error."""
        return {
            "error": self.code,
            "message": self.message,
            "context": {key: _materialize_json(value) for key, value in self.context.items()},
        }


class DataValidationError(QAMRError):
    """Raised when input data fails validation."""

    code = "data_validation"


class LabelAlignmentError(QAMRError):
    """Raised when labeled inputs cannot be aligned safely."""

    code = "label_alignment"


class InsufficientHistoryError(QAMRError):
    """Raised when an operation lacks sufficient historical observations."""

    code = "insufficient_history"


class LookaheadRiskError(QAMRError):
    """Raised when an operation risks using unavailable future information."""

    code = "lookahead_risk"


class NumericalStabilityError(QAMRError):
    """Raised when numerical stability requirements are not met."""

    code = "numerical_stability"


class InfeasiblePortfolioError(QAMRError):
    """Raised when portfolio constraints have no feasible solution."""

    code = "infeasible_portfolio"


class ReconciliationError(QAMRError):
    """Raised when calculated values cannot be reconciled."""

    code = "reconciliation"
