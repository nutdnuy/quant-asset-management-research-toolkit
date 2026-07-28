"""Canonical, validation-only input contract for research workflows."""

import re
from collections.abc import Hashable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from qamr._types import JsonValue
from qamr.contracts.arrays import LabeledMatrix, LabeledPanel, LabeledVector
from qamr.errors import DataValidationError, LabelAlignmentError

_RESERVED_SIGNAL_FIELDS = frozenset(
    {
        "future_return",
        "future_returns",
        "forward_return",
        "forward_returns",
    }
)


class ReturnConvention(str, Enum):
    """Supported interpretations of return observations."""

    SIMPLE = "simple"
    LOG = "log"


class MissingDataPolicy(str, Enum):
    """Supported workflow policies for missing observations."""

    RAISE = "raise"
    DROP_OBSERVATION = "drop_observation"


def _require_optional_nonblank_string(name: str, value: object) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise DataValidationError(
            f"{name} must be a string or None",
            context={
                "field": name,
                "dtype": type(value).__name__,
            },
        )
    if not value.strip():
        raise DataValidationError(
            f"{name} must not be blank",
            context={"field": name},
        )


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    """Explicit conventions needed to interpret a research dataset."""

    frequency: str
    timezone: str | None
    currency: str | None = None
    calendar: str | None = None
    return_convention: ReturnConvention | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.frequency, str):
            raise DataValidationError(
                "frequency must be a string",
                context={
                    "field": "frequency",
                    "dtype": type(self.frequency).__name__,
                },
            )
        if not self.frequency.strip():
            raise DataValidationError(
                "frequency must not be blank",
                context={"field": "frequency"},
            )
        _require_optional_nonblank_string("timezone", self.timezone)
        _require_optional_nonblank_string("currency", self.currency)
        _require_optional_nonblank_string("calendar", self.calendar)
        if self.return_convention is not None and not isinstance(
            self.return_convention, ReturnConvention
        ):
            raise DataValidationError(
                "return_convention must be a ReturnConvention",
                context={
                    "field": "return_convention",
                    "dtype": type(self.return_convention).__name__,
                },
            )


@dataclass(frozen=True, slots=True)
class InputProvenance:
    """Origin and optional content-addressed identity of input data."""

    source: str = "unspecified"
    content_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, str):
            raise DataValidationError(
                "source must be a string",
                context={
                    "field": "source",
                    "dtype": type(self.source).__name__,
                },
            )
        if not self.source.strip():
            raise DataValidationError(
                "source must not be blank",
                context={"field": "source"},
            )
        if self.content_fingerprint is None:
            return
        if not isinstance(self.content_fingerprint, str):
            raise DataValidationError(
                "content fingerprint must be a string",
                context={
                    "field": "content_fingerprint",
                    "dtype": type(self.content_fingerprint).__name__,
                },
            )
        algorithm, separator, digest = self.content_fingerprint.partition(":")
        if separator != ":" or algorithm != "sha256":
            raise DataValidationError(
                "content fingerprint must use sha256",
                context={
                    "field": "content_fingerprint",
                    "algorithm": algorithm[:32],
                    "reason": "algorithm",
                },
            )
        if re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
            raise DataValidationError(
                "content fingerprint has an invalid sha256 digest",
                context={
                    "field": "content_fingerprint",
                    "algorithm": "sha256",
                    "reason": "digest_format",
                },
            )


def _same_axes(left: LabeledMatrix, right: LabeledMatrix) -> bool:
    """Return whether both matrices have exactly equal axes in equal order."""
    return (
        left.row_labels == right.row_labels
        and left.column_labels == right.column_labels
        and left.row_name == right.row_name
        and left.column_name == right.column_name
    )


def _require_real_numeric(
    name: str,
    data: LabeledMatrix | LabeledPanel | LabeledVector,
) -> None:
    """Require real, non-boolean observations, permitting NaN but not infinity."""
    values = data.values
    context = {"field": name, "dtype": str(values.dtype)}
    is_real_numeric = values.dtype.kind in {"i", "u", "f"}
    if not is_real_numeric:
        raise DataValidationError(
            "data must have real numeric dtype",
            context=context,
        )
    if np.isinf(values).any():
        raise DataValidationError(
            "data must not contain infinity",
            context=context,
        )


def _require_primary_axes(
    name: str,
    matrix: LabeledMatrix,
    primary_name: str,
    primary: LabeledMatrix,
) -> None:
    if not _same_axes(matrix, primary):
        raise LabelAlignmentError(
            f"{name} must match primary axes exactly",
            context={"field": name, "primary": primary_name},
        )


def _bounded_label_sample(labels: list[Hashable]) -> list[JsonValue]:
    sample: list[JsonValue] = []
    sample.extend(sorted(str(label)[:40] for label in labels)[:5])
    return sample


def _require_ordered_subset(
    name: str,
    axis: str,
    supplied_labels: tuple[Hashable, ...],
    primary_labels: tuple[Hashable, ...],
) -> None:
    primary_label_set = set(primary_labels)
    unknown = [label for label in supplied_labels if label not in primary_label_set]
    if unknown:
        raise LabelAlignmentError(
            f"{name} has unknown {axis} labels; expected an ordered subset",
            context={
                "field": name,
                "axis": axis,
                "reason": "unknown_labels",
                "unknown_count": len(unknown),
                "unknown_sample": _bounded_label_sample(unknown),
            },
        )
    positions = {label: position for position, label in enumerate(primary_labels)}
    supplied_positions = [positions[label] for label in supplied_labels]
    if supplied_positions != sorted(supplied_positions):
        raise LabelAlignmentError(
            f"{name} must use an ordered {axis} subset",
            context={
                "field": name,
                "axis": axis,
                "reason": "out_of_order",
            },
        )


def _require_panel_axes(
    name: str,
    panel: LabeledPanel,
    primary: LabeledMatrix,
) -> None:
    _require_ordered_subset(
        name,
        "time",
        panel.time_labels,
        primary.row_labels,
    )
    _require_ordered_subset(
        name,
        "instrument",
        panel.instrument_labels,
        primary.column_labels,
    )


def _require_benchmark_return_axis(
    benchmark_returns: LabeledVector,
    primary_name: str,
    primary: LabeledMatrix,
) -> None:
    reason: str | None = None
    if benchmark_returns.labels != primary.row_labels:
        reason = "labels"
    elif benchmark_returns.axis_name != primary.row_name:
        reason = "axis_name"
    if reason is not None:
        raise LabelAlignmentError(
            "benchmark_returns must match the primary time axis exactly",
            context={
                "field": "benchmark_returns",
                "primary": primary_name,
                "axis": "time",
                "reason": reason,
            },
        )


def _reject_reserved_signal_fields(signals: LabeledPanel) -> None:
    offending = {
        label.casefold()
        for label in signals.field_labels
        if isinstance(label, str) and label.casefold() in _RESERVED_SIGNAL_FIELDS
    }
    if offending:
        offending_fields: list[JsonValue] = []
        offending_fields.extend(sorted(offending))
        raise DataValidationError(
            "signals contain a reserved future-return field",
            context={
                "field": "signals",
                "offending_fields": offending_fields,
            },
        )


@dataclass(frozen=True, slots=True)
class ResearchDataset:
    """The sole public, pre-aligned input accepted by research workflows."""

    metadata: DatasetMetadata
    returns: LabeledMatrix | None = None
    prices: LabeledMatrix | None = None
    signals: LabeledPanel | None = None
    future_returns: LabeledMatrix | None = None
    universe: LabeledMatrix | None = None
    benchmark_returns: LabeledVector | None = None
    benchmark_weights: LabeledMatrix | None = None
    groups: LabeledPanel | None = None
    exposures: LabeledPanel | None = None
    provenance: InputProvenance = field(default_factory=InputProvenance)

    def __post_init__(self) -> None:
        if self.returns is None and self.prices is None:
            raise DataValidationError(
                "dataset requires returns or prices",
                context={"required_fields": ["returns", "prices"]},
            )

        if self.returns is not None and self.metadata.return_convention is None:
            raise DataValidationError(
                "returns require a declared return convention",
                context={"field": "returns"},
            )
        if self.future_returns is not None and self.metadata.return_convention is None:
            raise DataValidationError(
                "future returns require a declared return convention",
                context={"field": "future_returns"},
            )
        if self.benchmark_returns is not None and self.metadata.return_convention is None:
            raise DataValidationError(
                "benchmark returns require a declared return convention",
                context={"field": "benchmark_returns"},
            )

        numeric_matrices = (
            ("returns", self.returns),
            ("prices", self.prices),
            ("future_returns", self.future_returns),
            ("benchmark_weights", self.benchmark_weights),
        )
        for name, matrix in numeric_matrices:
            if matrix is not None:
                _require_real_numeric(name, matrix)
        if self.benchmark_returns is not None:
            _require_real_numeric("benchmark_returns", self.benchmark_returns)
        if self.signals is not None:
            _require_real_numeric("signals", self.signals)
        if self.exposures is not None:
            _require_real_numeric("exposures", self.exposures)

        if (
            self.returns is not None
            and self.prices is not None
            and not _same_axes(self.returns, self.prices)
        ):
            raise LabelAlignmentError(
                "returns and prices must have identical axes",
                context={"left": "returns", "right": "prices"},
            )

        primary_name = "returns" if self.returns is not None else "prices"
        primary = self.returns if self.returns is not None else self.prices
        if primary is None:
            raise AssertionError("validated dataset must have primary data")

        aligned_matrices = (
            ("future_returns", self.future_returns),
            ("universe", self.universe),
            ("benchmark_weights", self.benchmark_weights),
        )
        for name, matrix in aligned_matrices:
            if matrix is not None:
                _require_primary_axes(name, matrix, primary_name, primary)
        if self.benchmark_returns is not None:
            _require_benchmark_return_axis(
                self.benchmark_returns,
                primary_name,
                primary,
            )

        if self.universe is not None:
            universe_dtype = self.universe.values.dtype
            if universe_dtype != np.dtype(np.bool_):
                raise DataValidationError(
                    "universe must have exactly boolean dtype",
                    context={
                        "field": "universe",
                        "dtype": str(universe_dtype),
                    },
                )

        instrument_panels = (
            ("signals", self.signals),
            ("groups", self.groups),
            ("exposures", self.exposures),
        )
        if self.signals is not None:
            _reject_reserved_signal_fields(self.signals)
        for name, panel in instrument_panels:
            if panel is not None:
                _require_panel_axes(name, panel, primary)
