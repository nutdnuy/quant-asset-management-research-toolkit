"""Lossless optional Pandas conversion into canonical research contracts."""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Hashable, Iterable
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

from qamr._types import JsonValue
from qamr.contracts.arrays import LabeledMatrix, LabeledPanel, LabeledVector
from qamr.contracts.dataset import DatasetMetadata, InputProvenance, ResearchDataset
from qamr.errors import DataValidationError, LabelAlignmentError, QAMRError

# Panel conversion creates a dense canonical tensor. These deterministic limits
# cap both shape explosion and its conservative in-memory footprint.
_MAX_PANEL_CELLS = 10_000_000
_MAX_PANEL_ESTIMATED_BYTES = 256 * 1024 * 1024
_MIN_NUMERIC_PANEL_BYTES_PER_CELL = 8
_OBJECT_PANEL_BYTES_PER_CELL = 64
_MAX_EXCEPTION_CHAIN_DEPTH = 8


@dataclass(frozen=True, slots=True)
class PandasResearchInput:
    """Optional Pandas objects supplied to :class:`PandasAdapter`."""

    returns: pd.DataFrame | None = None
    prices: pd.DataFrame | None = None
    signals: pd.DataFrame | None = None
    future_returns: pd.DataFrame | None = None
    universe: pd.DataFrame | None = None
    benchmark_returns: pd.Series[Any] | None = None
    benchmark_weights: pd.DataFrame | None = None
    groups: pd.DataFrame | None = None
    exposures: pd.DataFrame | None = None


def _validation_error(
    message: str,
    name: str,
    **details: JsonValue,
) -> DataValidationError:
    context: dict[str, JsonValue] = {"field": name}
    context.update(details)
    return DataValidationError(message, context=context)


def _unexpected_adaptation_error(
    component: str,
    error: Exception,
) -> DataValidationError:
    return DataValidationError(
        "pandas adaptation encountered an unexpected validation failure",
        context={
            "field": component,
            "component": component,
            "reason": "unexpected_exception",
            "type": type(error).__name__[:80],
        },
    )


def _memory_error_in_chain(error: BaseException) -> MemoryError | None:
    pending: list[tuple[BaseException, int]] = [(error, 0)]
    seen: set[int] = set()
    while pending:
        current, depth = pending.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        if isinstance(current, MemoryError):
            return current
        if depth >= _MAX_EXCEPTION_CHAIN_DEPTH:
            continue
        cause = current.__cause__
        context = current.__context__
        if isinstance(cause, BaseException):
            pending.append((cause, depth + 1))
        if isinstance(context, BaseException):
            pending.append((context, depth + 1))
    return None


def _reraise_memory_error_from_chain(error: QAMRError) -> None:
    memory_error = _memory_error_in_chain(error)
    if memory_error is not None:
        raise memory_error


def _label_is_reflexive(label: object) -> bool:
    if label is pd.NA or label is pd.NaT:
        return False
    try:
        comparison = label == label
    except MemoryError:
        raise
    except Exception:
        return False
    return isinstance(comparison, bool | np.bool_) and bool(comparison)


def _validate_labels(labels: Iterable[Hashable], name: str, axis: str) -> None:
    for position, label in enumerate(labels):
        try:
            hash(label)
        except MemoryError:
            raise
        except Exception as error:
            raise _validation_error(
                f"{name} contains invalid labels",
                name,
                axis=axis,
                position=position,
                reason="unhashable",
                type=type(label).__name__[:80],
            ) from error
        if not _label_is_reflexive(label):
            raise _validation_error(
                f"{name} contains invalid labels",
                name,
                axis=axis,
                position=position,
                reason="non_reflexive",
                type=type(label).__name__[:80],
            )


def _validate_axis(
    values: pd.Index[Any],
    name: str,
    axis: str,
    duplicate_message: str,
) -> None:
    if isinstance(values, pd.MultiIndex):
        for level_position, level in enumerate(values.levels):
            _validate_labels(
                level,
                name,
                f"{axis}.level_{level_position}",
            )
    _validate_labels(values, name, axis)
    try:
        has_duplicates = values.has_duplicates
    except MemoryError:
        raise
    except Exception as error:
        raise _validation_error(
            f"{name} label duplicate validation failed",
            name,
            axis=axis,
            reason="duplicate_check_failed",
            type=type(error).__name__[:80],
        ) from error
    if has_duplicates:
        raise _validation_error(
            duplicate_message,
            name,
            axis=axis,
        )


def _validate_frame_axes(frame: pd.DataFrame, name: str) -> None:
    _validate_axis(
        frame.index,
        name,
        "index",
        f"{name} contains duplicate observations",
    )
    _validate_axis(
        frame.columns,
        name,
        "columns",
        f"{name} contains duplicate columns",
    )


def _dtype_reason(dtype: Any) -> str:
    kind = getattr(dtype, "kind", None)
    if is_bool_dtype(dtype):
        return "boolean"
    if kind == "c":
        return "complex"
    if kind == "M":
        return "datetime"
    if kind == "m":
        return "timedelta"
    if isinstance(dtype, pd.CategoricalDtype):
        return "category"
    if kind in {"O", "U", "S"}:
        return "strings" if pd.api.types.is_string_dtype(dtype) else "object"
    return "non_numeric"


def _numeric_target_dtype(
    dtypes: Iterable[Any],
    name: str,
    *,
    requires_missing: bool,
) -> np.dtype[Any]:
    numpy_dtypes: list[np.dtype[Any]] = []
    for dtype in dtypes:
        kind = getattr(dtype, "kind", None)
        if kind not in {"i", "u", "f"}:
            raise _validation_error(
                f"{name} must be numeric",
                name,
                dtype=str(dtype)[:80],
                reason=_dtype_reason(dtype),
            )
        numpy_dtype = getattr(dtype, "numpy_dtype", dtype)
        try:
            numpy_dtypes.append(np.dtype(numpy_dtype))
        except TypeError as error:
            raise _validation_error(
                f"{name} must be numeric",
                name,
                dtype=str(dtype)[:80],
                reason="unsupported_dtype",
            ) from error
    if not numpy_dtypes:
        return np.dtype(np.float64)
    target = np.dtype(np.result_type(*numpy_dtypes))
    if requires_missing and target.kind in {"i", "u"}:
        target = np.dtype(np.float64)
    if target.kind not in {"i", "u", "f"}:
        raise _validation_error(
            f"{name} must be numeric",
            name,
            dtype=str(target)[:80],
            reason="unsupported_common_dtype",
        )
    return target


def _is_missing_scalar(value: Any) -> bool:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, bool | np.bool_) and bool(missing)


def _is_exact_numeric_conversion(original: object, converted: object) -> bool:
    if isinstance(original, bool):
        return False
    if isinstance(original, Integral):
        if isinstance(converted, Integral):
            return int(converted) == int(original)
        if isinstance(converted, Real):
            converted_float = float(converted)
            return (
                math.isfinite(converted_float)
                and converted_float.is_integer()
                and int(converted_float) == int(original)
            )
        return False
    if isinstance(original, Real) and isinstance(converted, Real):
        try:
            return bool(converted == original)
        except (TypeError, ValueError):
            return False
    return False


def _convert_numeric_scalar(
    original: object,
    target: np.dtype[Any],
    name: str,
    position: tuple[int, ...],
) -> object:
    position_context: list[JsonValue] = list(position)
    try:
        converted = np.asarray(original, dtype=target).item()
    except (OverflowError, TypeError, ValueError) as error:
        raise _validation_error(
            f"{name} conversion would be lossy",
            name,
            reason="lossy_conversion",
            position=position_context,
            source_type=type(original).__name__[:80],
            target_dtype=str(target)[:80],
        ) from error
    if not _is_exact_numeric_conversion(original, converted):
        raise _validation_error(
            f"{name} conversion would be lossy",
            name,
            reason="lossy_conversion",
            position=position_context,
            source_type=type(original).__name__[:80],
            target_dtype=str(target)[:80],
        )
    return converted


def _numeric_values(frame: pd.DataFrame, name: str) -> np.ndarray[Any, Any]:
    original_values = tuple(
        tuple(frame.iat[row, column] for column in range(frame.shape[1]))
        for row in range(frame.shape[0])
    )
    has_missing = any(
        _is_missing_scalar(value) for row_values in original_values for value in row_values
    )
    target = _numeric_target_dtype(
        frame.dtypes,
        name,
        requires_missing=has_missing,
    )
    values = np.empty(frame.shape, dtype=target)
    for row, row_values in enumerate(original_values):
        for column, original in enumerate(row_values):
            if _is_missing_scalar(original):
                values[row, column] = np.nan
                continue
            values[row, column] = _convert_numeric_scalar(
                original,
                target,
                name,
                (row, column),
            )
    if np.isinf(values).any():
        raise _validation_error(
            f"{name} must not contain infinity",
            name,
            dtype=str(values.dtype)[:80],
        )
    return values


def _numeric_series_values(series: pd.Series[Any], name: str) -> np.ndarray[Any, Any]:
    original_values = tuple(series.iloc[position] for position in range(len(series)))
    has_missing = any(_is_missing_scalar(value) for value in original_values)
    target = _numeric_target_dtype(
        [series.dtype],
        name,
        requires_missing=has_missing,
    )
    values = np.empty(series.shape, dtype=target)
    for position, original in enumerate(original_values):
        if _is_missing_scalar(original):
            values[position] = np.nan
            continue
        values[position] = _convert_numeric_scalar(
            original,
            target,
            name,
            (position,),
        )
    if np.isinf(values).any():
        raise _validation_error(
            f"{name} must not contain infinity",
            name,
            dtype=str(values.dtype)[:80],
        )
    return values


def _numeric_panel_values(
    frame: pd.DataFrame,
    name: str,
    times: tuple[Hashable, ...],
    instruments: tuple[Hashable, ...],
) -> np.ndarray[Any, Any]:
    original_values = tuple(
        tuple(
            frame.iat[observation, field_position] for field_position in range(len(frame.columns))
        )
        for observation in range(len(frame.index))
    )
    has_cartesian_gaps = len(frame.index) < len(times) * len(instruments)
    has_missing = has_cartesian_gaps or any(
        _is_missing_scalar(value)
        for observation_values in original_values
        for value in observation_values
    )
    target = _numeric_target_dtype(
        frame.dtypes,
        name,
        requires_missing=has_missing,
    )
    shape = (len(times), len(instruments), len(frame.columns))
    values = np.empty(shape, dtype=target)
    if has_missing:
        values.fill(np.nan)
    time_positions = {label: position for position, label in enumerate(times)}
    instrument_positions = {label: position for position, label in enumerate(instruments)}
    for observation, observation_values in enumerate(original_values):
        time, instrument = frame.index[observation]
        time_position = time_positions[time]
        instrument_position = instrument_positions[instrument]
        for field_position, original in enumerate(observation_values):
            if _is_missing_scalar(original):
                continue
            values[time_position, instrument_position, field_position] = _convert_numeric_scalar(
                original,
                target,
                name,
                (time_position, instrument_position, field_position),
            )
    if np.isinf(values).any():
        raise _validation_error(
            f"{name} must not contain infinity",
            name,
            dtype=str(values.dtype)[:80],
        )
    return values


def _object_panel_values(
    frame: pd.DataFrame,
    times: tuple[Hashable, ...],
    instruments: tuple[Hashable, ...],
) -> np.ndarray[Any, Any]:
    shape = (len(times), len(instruments), len(frame.columns))
    values = np.empty(shape, dtype=object)
    values.fill(np.nan)
    time_positions = {label: position for position, label in enumerate(times)}
    instrument_positions = {label: position for position, label in enumerate(instruments)}
    original_values = tuple(
        tuple(
            frame.iat[observation, field_position] for field_position in range(len(frame.columns))
        )
        for observation in range(len(frame.index))
    )
    for observation, observation_values in enumerate(original_values):
        time, instrument = frame.index[observation]
        time_position = time_positions[time]
        instrument_position = instrument_positions[instrument]
        for field_position, original in enumerate(observation_values):
            values[time_position, instrument_position, field_position] = original
    return values


def _numeric_matrix(frame: pd.DataFrame, name: str) -> LabeledMatrix:
    _validate_frame_axes(frame, name)
    try:
        return LabeledMatrix(
            values=_numeric_values(frame, name),
            row_labels=tuple(frame.index),
            column_labels=tuple(frame.columns),
            row_name="time",
            column_name="instrument",
        )
    except QAMRError as error:
        _reraise_memory_error_from_chain(error)
        raise
    except MemoryError:
        raise
    except Exception as error:
        raise _unexpected_adaptation_error(name, error) from error


def _validate_ordered_panel_subset(
    name: str,
    axis: str,
    labels: tuple[Hashable, ...],
    primary_labels: tuple[Hashable, ...],
) -> None:
    primary_positions = {label: position for position, label in enumerate(primary_labels)}
    unknown_count = sum(label not in primary_positions for label in labels)
    if unknown_count:
        raise LabelAlignmentError(
            f"{name} {axis} labels must be an ordered subset of primary axes",
            context={
                "field": name,
                "axis": axis,
                "reason": "unknown_labels",
                "unknown_count": unknown_count,
            },
        )
    positions = [primary_positions[label] for label in labels]
    if positions != sorted(positions):
        raise LabelAlignmentError(
            f"{name} {axis} labels must be an ordered subset of primary axes",
            context={
                "field": name,
                "axis": axis,
                "reason": "out_of_order",
            },
        )


def _validate_panel_alignment(
    name: str,
    primary: pd.DataFrame,
    times: tuple[Hashable, ...],
    instruments: tuple[Hashable, ...],
) -> None:
    _validate_ordered_panel_subset(
        name,
        "time",
        times,
        tuple(primary.index),
    )
    _validate_ordered_panel_subset(
        name,
        "instrument",
        instruments,
        tuple(primary.columns),
    )


def _validate_panel_budget(
    name: str,
    time_count: int,
    instrument_count: int,
    field_count: int,
    bytes_per_cell: int,
) -> None:
    cells = time_count * instrument_count * field_count
    estimated_bytes = cells * bytes_per_cell
    if cells > _MAX_PANEL_CELLS or estimated_bytes > _MAX_PANEL_ESTIMATED_BYTES:
        raise _validation_error(
            f"{name} exceeds the panel materialization budget",
            name,
            dimensions={
                "time": time_count,
                "instrument": instrument_count,
                "field": field_count,
            },
            cells=cells,
            estimated_bytes=estimated_bytes,
            limits={
                "cells": _MAX_PANEL_CELLS,
                "estimated_bytes": _MAX_PANEL_ESTIMATED_BYTES,
            },
        )


def _group_panel_error(
    name: str,
    error: Exception,
) -> DataValidationError:
    context: dict[str, JsonValue] = {
        "field": name,
        "reason": "invalid_object_cell",
        "type": type(error).__name__[:80],
    }
    message = f"{name} contains an invalid object cell"
    if isinstance(error, DataValidationError):
        message = error.message
        reason = error.context.get("reason")
        if isinstance(reason, str):
            context["reason"] = reason[:80]
        position = error.context.get("position")
        if isinstance(position, tuple):
            bounded_position: list[JsonValue] = []
            bounded_position.extend(int(value) for value in position[:3] if isinstance(value, int))
            context["position"] = bounded_position
    return DataValidationError(message, context=context)


def _panel(
    frame: pd.DataFrame,
    name: str,
    *,
    numeric: bool,
    primary: pd.DataFrame,
) -> LabeledPanel:
    _validate_frame_axes(frame, name)
    if not isinstance(frame.index, pd.MultiIndex) or frame.index.nlevels != 2:
        raise _validation_error(
            f"{name} must use a two-level time/instrument index",
            name,
            expected_levels=2,
        )
    if list(frame.index.names) != ["time", "instrument"]:
        raise _validation_error(
            f"{name} index levels must be named time and instrument",
            name,
            expected_names=["time", "instrument"],
            observed_names=[
                None if value is None else str(value)[:80] for value in frame.index.names
            ],
        )
    times = tuple(pd.unique(frame.index.get_level_values("time")))
    instruments = tuple(pd.unique(frame.index.get_level_values("instrument")))
    _validate_labels(times, name, "time")
    _validate_labels(instruments, name, "instrument")
    _validate_panel_alignment(name, primary, times, instruments)
    if numeric:
        target = _numeric_target_dtype(
            frame.dtypes,
            name,
            requires_missing=False,
        )
        bytes_per_cell = max(
            int(target.itemsize),
            _MIN_NUMERIC_PANEL_BYTES_PER_CELL,
        )
        _validate_panel_budget(
            name,
            len(times),
            len(instruments),
            len(frame.columns),
            bytes_per_cell,
        )
        values = _numeric_panel_values(frame, name, times, instruments)
    else:
        _validate_panel_budget(
            name,
            len(times),
            len(instruments),
            len(frame.columns),
            _OBJECT_PANEL_BYTES_PER_CELL,
        )
        values = _object_panel_values(frame, times, instruments)
    try:
        return LabeledPanel(
            values=values.reshape(
                len(times),
                len(instruments),
                len(frame.columns),
            ),
            time_labels=times,
            instrument_labels=instruments,
            field_labels=tuple(frame.columns),
        )
    except QAMRError as error:
        _reraise_memory_error_from_chain(error)
        if numeric:
            raise
        raise _group_panel_error(name, error) from error
    except MemoryError:
        raise
    except Exception as error:
        if not numeric:
            raise _group_panel_error(name, error) from error
        raise _unexpected_adaptation_error(name, error) from error


def _vector(series: pd.Series[Any], name: str) -> LabeledVector:
    _validate_axis(
        series.index,
        name,
        "index",
        f"{name} contains duplicate observations",
    )
    try:
        return LabeledVector(
            _numeric_series_values(series, name),
            tuple(series.index),
            "time",
        )
    except QAMRError as error:
        _reraise_memory_error_from_chain(error)
        raise
    except MemoryError:
        raise
    except Exception as error:
        raise _unexpected_adaptation_error(name, error) from error


def _universe(frame: pd.DataFrame) -> LabeledMatrix:
    name = "universe"
    _validate_frame_axes(frame, name)
    if frame.isna().to_numpy().any():
        raise _validation_error(
            "universe contains missing values",
            name,
            reason="missing",
        )
    invalid_dtypes = [
        (position, dtype) for position, dtype in enumerate(frame.dtypes) if not is_bool_dtype(dtype)
    ]
    if invalid_dtypes:
        sample: list[JsonValue] = [
            {
                "position": position,
                "dtype": str(dtype)[:80],
            }
            for position, dtype in invalid_dtypes[:5]
        ]
        raise _validation_error(
            "universe must be boolean",
            name,
            invalid_dtype_count=len(invalid_dtypes),
            invalid_dtype_sample=sample,
            invalid_dtype_sample_truncated=len(invalid_dtypes) > len(sample),
        )
    values = frame.to_numpy(dtype=bool, copy=True)
    try:
        return LabeledMatrix(
            values,
            tuple(frame.index),
            tuple(frame.columns),
            "time",
            "instrument",
        )
    except QAMRError as error:
        _reraise_memory_error_from_chain(error)
        raise
    except MemoryError:
        raise
    except Exception as error:
        raise _unexpected_adaptation_error(name, error) from error


_FRAME_FIELDS = (
    "returns",
    "prices",
    "signals",
    "future_returns",
    "universe",
    "benchmark_weights",
    "groups",
    "exposures",
)


def _validate_component_types(source: PandasResearchInput) -> None:
    for name in _FRAME_FIELDS:
        value = getattr(source, name)
        if value is not None and not isinstance(value, pd.DataFrame):
            raise _validation_error(
                f"{name} has the wrong pandas type",
                name,
                expected="DataFrame",
                actual=type(value).__name__[:80],
            )
    benchmark_returns = source.benchmark_returns
    if benchmark_returns is not None and not isinstance(
        benchmark_returns,
        pd.Series,
    ):
        raise _validation_error(
            "benchmark_returns has the wrong pandas type",
            "benchmark_returns",
            expected="Series",
            actual=type(benchmark_returns).__name__[:80],
        )


def _time_index(value: pd.DataFrame | pd.Series[Any], *, panel: bool) -> pd.Index[Any]:
    if panel and isinstance(value.index, pd.MultiIndex):
        return value.index.get_level_values(0)
    return value.index


def _is_datetime_label(label: object) -> bool:
    return isinstance(label, dt.datetime | dt.date | np.datetime64)


def _observed_timezone(
    index: pd.Index[Any],
    name: str,
) -> tuple[bool, str | None]:
    if not isinstance(index, pd.DatetimeIndex):
        datetime_flags = [_is_datetime_label(label) for label in index]
        datetime_count = sum(datetime_flags)
        if datetime_count == 0:
            return False, None
        if datetime_count != len(index):
            raise _validation_error(
                f"{name} time axis must not mix datetime and generic labels",
                name,
                reason="mixed_datetime_and_generic",
                datetime_count=datetime_count,
                label_count=len(index),
            )
        awareness: list[bool] = []
        timezones: list[str] = []
        for label in index:
            timestamp = pd.Timestamp(label)
            is_aware = timestamp.tzinfo is not None and timestamp.utcoffset() is not None
            awareness.append(is_aware)
            if is_aware:
                timezones.append(str(timestamp.tzinfo)[:80])
        aware_count = sum(awareness)
        if 0 < aware_count < len(awareness):
            raise _validation_error(
                f"{name} time axis must not mix aware and naive labels",
                name,
                reason="mixed_aware_and_naive",
                aware_count=aware_count,
                naive_count=len(awareness) - aware_count,
            )
        unique_timezones = sorted(set(timezones))
        if len(unique_timezones) > 1:
            timezone_sample: list[JsonValue] = []
            timezone_sample.extend(unique_timezones[:5])
            raise _validation_error(
                f"{name} time axis contains mixed timezones",
                name,
                reason="mixed_timezones",
                timezone_count=len(unique_timezones),
                timezone_sample=timezone_sample,
                timezone_sample_truncated=len(unique_timezones) > 5,
            )
        return True, unique_timezones[0] if unique_timezones else None
    return True, None if index.tz is None else str(index.tz)[:80]


@dataclass(frozen=True, slots=True)
class PandasAdapter:
    """Adapt Pandas containers without implicit sorting, casting, or filling."""

    metadata: DatasetMetadata
    provenance: InputProvenance = field(default_factory=InputProvenance)

    def _validate_contract_types(self) -> None:
        if not isinstance(self.metadata, DatasetMetadata):
            raise _validation_error(
                "metadata must be DatasetMetadata",
                "metadata",
                actual=type(self.metadata).__name__[:80],
            )
        if not isinstance(self.provenance, InputProvenance):
            raise _validation_error(
                "provenance must be InputProvenance",
                "provenance",
                actual=type(self.provenance).__name__[:80],
            )

    def _validate_timezones(self, source: PandasResearchInput) -> None:
        panel_fields = {"signals", "groups", "exposures"}
        for name in (*_FRAME_FIELDS, "benchmark_returns"):
            value = getattr(source, name)
            if value is None:
                continue
            is_datetime, observed = _observed_timezone(
                _time_index(value, panel=name in panel_fields),
                name,
            )
            if not is_datetime:
                continue
            declared = self.metadata.timezone
            if observed != declared:
                raise _validation_error(
                    "timezone does not match declared metadata",
                    name,
                    declared=None if declared is None else declared[:80],
                    observed=None if observed is None else observed[:80],
                )

    def adapt(self, source: PandasResearchInput) -> ResearchDataset:
        """Convert a validated Pandas input bundle to a canonical dataset."""
        try:
            return self._adapt(source)
        except QAMRError as error:
            _reraise_memory_error_from_chain(error)
            raise
        except MemoryError:
            raise
        except Exception as error:
            raise _unexpected_adaptation_error("adapter", error) from error

    def _adapt(self, source: PandasResearchInput) -> ResearchDataset:
        self._validate_contract_types()
        if not isinstance(source, PandasResearchInput):
            raise _validation_error(
                "source must be PandasResearchInput",
                "source",
                actual=type(source).__name__[:80],
            )
        _validate_component_types(source)
        self._validate_timezones(source)
        primary = source.returns if source.returns is not None else source.prices
        if primary is None:
            raise _validation_error(
                "dataset requires returns or prices",
                "source",
                required_fields=["returns", "prices"],
            )

        try:
            return ResearchDataset(
                returns=(
                    _numeric_matrix(source.returns, "returns")
                    if source.returns is not None
                    else None
                ),
                prices=(
                    _numeric_matrix(source.prices, "prices") if source.prices is not None else None
                ),
                signals=(
                    _panel(
                        source.signals,
                        "signals",
                        numeric=True,
                        primary=primary,
                    )
                    if source.signals is not None
                    else None
                ),
                future_returns=(
                    _numeric_matrix(source.future_returns, "future_returns")
                    if source.future_returns is not None
                    else None
                ),
                universe=(_universe(source.universe) if source.universe is not None else None),
                benchmark_returns=(
                    _vector(source.benchmark_returns, "benchmark_returns")
                    if source.benchmark_returns is not None
                    else None
                ),
                benchmark_weights=(
                    _numeric_matrix(
                        source.benchmark_weights,
                        "benchmark_weights",
                    )
                    if source.benchmark_weights is not None
                    else None
                ),
                groups=(
                    _panel(
                        source.groups,
                        "groups",
                        numeric=False,
                        primary=primary,
                    )
                    if source.groups is not None
                    else None
                ),
                exposures=(
                    _panel(
                        source.exposures,
                        "exposures",
                        numeric=True,
                        primary=primary,
                    )
                    if source.exposures is not None
                    else None
                ),
                metadata=self.metadata,
                provenance=self.provenance,
            )
        except QAMRError as error:
            _reraise_memory_error_from_chain(error)
            raise
        except MemoryError:
            raise
        except Exception as error:
            raise _unexpected_adaptation_error("dataset", error) from error
