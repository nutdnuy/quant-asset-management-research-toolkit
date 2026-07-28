"""Immutable labelled NumPy containers for the public boundary.

Labels are expected to be stable immutable hashables. Object-dtype cells are
deep-copied at input and export boundaries and must themselves be hashable.
"""

from collections.abc import Hashable, Iterable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from qamr._types import JsonValue
from qamr.errors import DataValidationError


def _unique_labels(
    labels: Iterable[Hashable],
    axis_name: str,
) -> tuple[Hashable, ...]:
    """Copy and validate stable immutable labels for one axis."""
    copied_labels = tuple(labels)
    seen: set[Hashable] = set()
    for position, label in enumerate(copied_labels):
        try:
            hash(label)
        except TypeError as error:
            raise DataValidationError(
                "labels must be hashable",
                context={"axis": axis_name, "position": position},
            ) from error
        if label != label:
            raise DataValidationError(
                "labels must be reflexive",
                context={"axis": axis_name, "position": position},
            )
        if label in seen:
            raise DataValidationError(
                "labels must be unique",
                context={"axis": axis_name, "duplicate": str(label)},
            )
        seen.add(label)
    return copied_labels


def _copy_object_array(values: NDArray[Any]) -> NDArray[Any]:
    """Deep-copy hashable object cells without exposing their representations."""
    copied_values: NDArray[Any] = np.empty(values.shape, dtype=object)
    for index in np.ndindex(values.shape):
        position: list[JsonValue] = [int(axis_position) for axis_position in index]
        try:
            cell = deepcopy(values[index])
        except Exception as error:
            raise DataValidationError(
                "object cells must be safely copyable",
                context={"position": position, "reason": "not_copyable"},
            ) from error
        try:
            hash(cell)
        except TypeError as error:
            raise DataValidationError(
                "object cells must be hashable",
                context={"position": position, "reason": "unhashable"},
            ) from error
        copied_values[index] = cell
    return copied_values


def _copy_array(values: Any) -> NDArray[Any]:
    try:
        copied_values = np.array(values, copy=True)
    except (TypeError, ValueError) as error:
        raise DataValidationError(
            "values could not be converted to an array",
            context={"reason": type(error).__name__},
        ) from error
    if copied_values.dtype == object:
        return _copy_object_array(copied_values)
    return copied_values


def _immutable_array(values: Any, dimensions: int) -> NDArray[Any]:
    copied_values = _copy_array(values)
    if copied_values.ndim != dimensions:
        raise DataValidationError(
            "array has the wrong number of dimensions",
            context={"expected": dimensions, "actual": copied_values.ndim},
        )
    copied_values.flags.writeable = False
    return copied_values


def _public_array_snapshot(values: NDArray[Any]) -> NDArray[Any]:
    snapshot = _copy_array(values)
    snapshot.flags.writeable = False
    return snapshot


@dataclass(frozen=True, slots=True, init=False, eq=False)
class LabeledVector:
    """One-dimensional immutable values with one labelled axis."""

    _values: NDArray[Any] = field(repr=False)
    labels: tuple[Hashable, ...]
    axis_name: str

    __hash__: ClassVar[None] = None  # type: ignore[assignment]

    def __init__(
        self,
        values: Any,
        labels: Iterable[Hashable],
        axis_name: str,
    ) -> None:
        copied_values = _immutable_array(values, dimensions=1)
        copied_labels = _unique_labels(labels, axis_name)
        actual_shape: list[JsonValue] = [int(size) for size in copied_values.shape]
        expected_shape: list[JsonValue] = [len(copied_labels)]
        if actual_shape != expected_shape:
            raise DataValidationError(
                "shape does not match labels",
                context={
                    "shape": actual_shape,
                    "expected": expected_shape,
                },
            )
        object.__setattr__(self, "_values", copied_values)
        object.__setattr__(self, "labels", copied_labels)
        object.__setattr__(self, "axis_name", axis_name)

    @property
    def values(self) -> NDArray[Any]:
        """Return an isolated, read-only snapshot of the vector values."""
        return _public_array_snapshot(self._values)

    @property
    def shape(self) -> tuple[int]:
        """Return the vector shape."""
        return (int(self._values.shape[0]),)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class LabeledMatrix:
    """Two-dimensional immutable values with labelled rows and columns."""

    _values: NDArray[Any] = field(repr=False)
    row_labels: tuple[Hashable, ...]
    column_labels: tuple[Hashable, ...]
    row_name: str
    column_name: str

    __hash__: ClassVar[None] = None  # type: ignore[assignment]

    def __init__(
        self,
        values: Any,
        row_labels: Iterable[Hashable],
        column_labels: Iterable[Hashable],
        row_name: str,
        column_name: str,
    ) -> None:
        copied_values = _immutable_array(values, dimensions=2)
        copied_row_labels = _unique_labels(row_labels, row_name)
        copied_column_labels = _unique_labels(column_labels, column_name)
        actual_shape: list[JsonValue] = [int(size) for size in copied_values.shape]
        expected_shape: list[JsonValue] = [
            len(copied_row_labels),
            len(copied_column_labels),
        ]
        if actual_shape != expected_shape:
            raise DataValidationError(
                "shape does not match labels",
                context={
                    "shape": actual_shape,
                    "expected": expected_shape,
                },
            )
        object.__setattr__(self, "_values", copied_values)
        object.__setattr__(self, "row_labels", copied_row_labels)
        object.__setattr__(self, "column_labels", copied_column_labels)
        object.__setattr__(self, "row_name", row_name)
        object.__setattr__(self, "column_name", column_name)

    @property
    def values(self) -> NDArray[Any]:
        """Return an isolated, read-only snapshot of the matrix values."""
        return _public_array_snapshot(self._values)

    @property
    def shape(self) -> tuple[int, int]:
        """Return the matrix shape."""
        return (int(self._values.shape[0]), int(self._values.shape[1]))


@dataclass(frozen=True, slots=True, init=False, eq=False)
class LabeledPanel:
    """Three-dimensional immutable values with time, instrument, and field axes."""

    _values: NDArray[Any] = field(repr=False)
    time_labels: tuple[Hashable, ...]
    instrument_labels: tuple[Hashable, ...]
    field_labels: tuple[Hashable, ...]

    __hash__: ClassVar[None] = None  # type: ignore[assignment]

    def __init__(
        self,
        values: Any,
        time_labels: Iterable[Hashable],
        instrument_labels: Iterable[Hashable],
        field_labels: Iterable[Hashable],
    ) -> None:
        copied_values = _immutable_array(values, dimensions=3)
        copied_time_labels = _unique_labels(time_labels, "time")
        copied_instrument_labels = _unique_labels(
            instrument_labels,
            "instrument",
        )
        copied_field_labels = _unique_labels(field_labels, "field")
        actual_shape: list[JsonValue] = [int(size) for size in copied_values.shape]
        expected_shape: list[JsonValue] = [
            len(copied_time_labels),
            len(copied_instrument_labels),
            len(copied_field_labels),
        ]
        if actual_shape != expected_shape:
            raise DataValidationError(
                "shape does not match labels",
                context={
                    "shape": actual_shape,
                    "expected": expected_shape,
                },
            )
        object.__setattr__(self, "_values", copied_values)
        object.__setattr__(self, "time_labels", copied_time_labels)
        object.__setattr__(
            self,
            "instrument_labels",
            copied_instrument_labels,
        )
        object.__setattr__(self, "field_labels", copied_field_labels)

    @property
    def values(self) -> NDArray[Any]:
        """Return an isolated, read-only snapshot of the panel values."""
        return _public_array_snapshot(self._values)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return the panel shape."""
        return (
            int(self._values.shape[0]),
            int(self._values.shape[1]),
            int(self._values.shape[2]),
        )
