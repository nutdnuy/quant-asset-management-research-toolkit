from __future__ import annotations

import dataclasses
import datetime as dt
import json
import subprocess
import sys
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

import qamr.contracts.pandas_adapter as pandas_adapter_module
from qamr.contracts.dataset import DatasetMetadata, InputProvenance, ReturnConvention
from qamr.contracts.interfaces import DataAdapter
from qamr.contracts.pandas_adapter import PandasAdapter, PandasResearchInput
from qamr.errors import DataValidationError, LabelAlignmentError


def adapter(*, timezone: str | None = "UTC") -> PandasAdapter:
    return PandasAdapter(
        metadata=DatasetMetadata(
            frequency="business-day",
            timezone=timezone,
            currency="USD",
            calendar="generic-weekday",
            return_convention=ReturnConvention.SIMPLE,
        ),
        provenance=InputProvenance(
            source="synthetic-pandas-fixture",
            content_fingerprint=f"sha256:{'01' * 32}",
        ),
    )


def primary(
    *,
    index: pd.Index[Any] | None = None,
    columns: pd.Index[Any] | list[Any] | None = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        [[0.01, 0.02], [0.03, 0.04]],
        index=["t0", "t1"] if index is None else index,
        columns=["a", "b"] if columns is None else columns,
    )


def panel_index(
    tuples: list[tuple[Any, Any]] | None = None,
    *,
    names: list[str | None] | None = None,
) -> pd.MultiIndex:
    observations = (
        [("t0", "a"), ("t0", "b"), ("t1", "a"), ("t1", "b")] if tuples is None else tuples
    )
    return pd.MultiIndex.from_tuples(
        observations,
        names=["time", "instrument"] if names is None else names,
    )


def assert_structured(error: DataValidationError) -> None:
    json.dumps(error.as_dict())
    assert error.code == "data_validation"
    assert "field" in error.context


def _raise_runtime_hash(_self: object) -> int:
    raise RuntimeError("malicious hash")


MaliciousHashLabel = type(
    "malicious_label_type_" + ("x" * 1_000),
    (),
    {
        "__hash__": _raise_runtime_hash,
        "__eq__": lambda self, other: self is other,
    },
)


class DuplicateEngineHashFailure:
    def __init__(self) -> None:
        self.calls = 0

    def __hash__(self) -> int:
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("duplicate engine hash")
        return 1

    def __eq__(self, other: object) -> bool:
        return self is other


class NthHashFailure:
    def __init__(self, failure_at: int | None) -> None:
        self.failure_at = failure_at
        self.calls = 0

    def __hash__(self) -> int:
        self.calls += 1
        if self.failure_at is not None and self.calls >= self.failure_at:
            raise RuntimeError("stateful hash failure")
        return 1

    def __eq__(self, other: object) -> bool:
        return self is other


class MemoryHashLabel:
    def __hash__(self) -> int:
        raise MemoryError

    def __eq__(self, other: object) -> bool:
        return self is other


class MemoryReflexiveLabel:
    def __hash__(self) -> int:
        return 1

    def __eq__(self, other: object) -> bool:
        raise MemoryError


class DuplicateEngineMemoryLabel:
    def __init__(self) -> None:
        self.calls = 0

    def __hash__(self) -> int:
        self.calls += 1
        if self.calls > 1:
            raise MemoryError
        return 1

    def __eq__(self, other: object) -> bool:
        return self is other


class MemoryCopyGroupCell:
    def __hash__(self) -> int:
        return 1

    def __deepcopy__(self, memo: dict[int, object]) -> object:
        raise MemoryError


class MemoryHashGroupCell:
    def __hash__(self) -> int:
        raise MemoryError

    def __deepcopy__(self, memo: dict[int, object]) -> object:
        return self


class StopAdaptation(BaseException):
    pass


class StopHashLabel:
    def __hash__(self) -> int:
        raise StopAdaptation

    def __eq__(self, other: object) -> bool:
        return self is other


class HostileTextError(RuntimeError):
    def __repr__(self) -> str:
        raise AssertionError("repr must not be called")

    def __str__(self) -> str:
        raise AssertionError("str must not be called")


class InspectionBombFrame(pd.DataFrame):
    def isna(self) -> pd.DataFrame:
        raise AssertionError("isna called before panel budget")

    def to_numpy(self, *args: object, **kwargs: object) -> np.ndarray[Any, Any]:
        raise AssertionError("to_numpy called before panel budget")


class UncopyableGroupCell:
    def __hash__(self) -> int:
        return 1

    def __deepcopy__(self, memo: dict[int, object]) -> object:
        raise RuntimeError("copy disabled")


class ExplodingHashGroupCell:
    def __hash__(self) -> int:
        raise RuntimeError("cell hash disabled")

    def __deepcopy__(self, memo: dict[int, object]) -> object:
        return self


def test_wide_returns_preserve_timestamp_columns_nan_and_timezone() -> None:
    index = pd.date_range("2026-01-01", periods=2, tz="UTC")
    returns = pd.DataFrame(
        [[0.01, np.nan], [0.02, 0.03]],
        index=index,
        columns=["asset-b", "asset-a"],
    )

    dataset = adapter().adapt(PandasResearchInput(returns=returns))

    assert dataset.returns is not None
    assert dataset.returns.row_labels == tuple(index)
    assert dataset.returns.column_labels == ("asset-b", "asset-a")
    assert np.isnan(dataset.returns.values[0, 1])
    assert dataset.metadata.timezone == "UTC"


def test_adapter_structurally_satisfies_runtime_protocol() -> None:
    assert isinstance(adapter(), DataAdapter)


def test_input_and_adapter_are_frozen_slotted_dataclasses() -> None:
    source = PandasResearchInput(returns=primary())
    assert source.__slots__
    assert adapter().__slots__
    with pytest.raises(dataclasses.FrozenInstanceError):
        source.returns = None  # type: ignore[misc]


def test_long_signal_frame_becomes_time_instrument_field_panel() -> None:
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-01-01", tz="UTC"), "a"),
            (pd.Timestamp("2026-01-01", tz="UTC"), "b"),
            (pd.Timestamp("2026-01-02", tz="UTC"), "a"),
            (pd.Timestamp("2026-01-02", tz="UTC"), "b"),
        ],
        names=["time", "instrument"],
    )
    signals = pd.DataFrame(
        {"value": [1.0, 2.0, 3.0, 4.0], "quality": [0.9, 0.8, 0.7, 0.6]},
        index=index,
    )
    returns = primary(
        index=pd.date_range("2026-01-01", periods=2, tz="UTC"),
    )

    dataset = adapter().adapt(PandasResearchInput(returns=returns, signals=signals))

    assert dataset.signals is not None
    assert dataset.signals.shape == (2, 2, 2)
    assert dataset.signals.instrument_labels == ("a", "b")
    assert dataset.signals.field_labels == ("value", "quality")
    np.testing.assert_array_equal(
        dataset.signals.values[:, :, 0],
        [[1.0, 2.0], [3.0, 4.0]],
    )


def test_panel_preserves_first_observed_order_and_missing_cells() -> None:
    index = panel_index([("t1", "b"), ("t0", "a"), ("t1", "a")])
    signals = pd.DataFrame(
        {"z": [1.0, 2.0, 3.0], "a": [4.0, 5.0, 6.0]},
        index=index,
    )

    dataset = adapter().adapt(
        PandasResearchInput(
            returns=primary(index=pd.Index(["t1", "t0"]), columns=["b", "a"]),
            signals=signals,
        )
    )

    assert dataset.signals is not None
    assert dataset.signals.time_labels == ("t1", "t0")
    assert dataset.signals.instrument_labels == ("b", "a")
    assert dataset.signals.field_labels == ("z", "a")
    assert dataset.signals.shape == (2, 2, 2)
    assert np.isnan(dataset.signals.values[1, 0, :]).all()


def test_panel_alignment_is_rejected_before_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals = pd.DataFrame(
        {"value": [1.0]},
        index=panel_index([("unknown-time", "unknown-instrument")]),
    )

    def fail_if_materialized(*args: object, **kwargs: object) -> np.ndarray[Any, Any]:
        raise AssertionError("panel materialization happened before alignment")

    monkeypatch.setattr(
        pandas_adapter_module,
        "_numeric_panel_values",
        fail_if_materialized,
    )

    with pytest.raises(LabelAlignmentError, match="ordered subset"):
        adapter().adapt(PandasResearchInput(returns=primary(), signals=signals))


def test_sparse_panel_rejects_cartesian_materialization_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    axis_size = 101
    times = [f"t{position:03d}" for position in range(axis_size)]
    instruments = [f"a{position:03d}" for position in range(axis_size)]
    returns = pd.DataFrame(
        np.zeros((axis_size, axis_size)),
        index=times,
        columns=instruments,
    )
    signals = pd.DataFrame(
        {"value": np.ones(axis_size)},
        index=panel_index(list(zip(times, instruments, strict=True))),
    )
    monkeypatch.setattr(pandas_adapter_module, "_MAX_PANEL_CELLS", 10_000)
    monkeypatch.setattr(
        pandas_adapter_module,
        "_MAX_PANEL_ESTIMATED_BYTES",
        1_000_000,
    )

    with pytest.raises(DataValidationError, match="materialization budget") as captured:
        adapter().adapt(PandasResearchInput(returns=returns, signals=signals))

    assert set(captured.value.context) == {
        "field",
        "dimensions",
        "cells",
        "estimated_bytes",
        "limits",
    }
    assert captured.value.context["cells"] == axis_size * axis_size
    assert len(json.dumps(captured.value.as_dict())) < 500


def test_panel_materialization_budget_exact_and_near_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals = pd.DataFrame(
        {"value": [1.0, 2.0, 3.0, 4.0]},
        index=panel_index(),
    )
    monkeypatch.setattr(pandas_adapter_module, "_MAX_PANEL_CELLS", 4)
    monkeypatch.setattr(pandas_adapter_module, "_MAX_PANEL_ESTIMATED_BYTES", 32)

    dataset = adapter().adapt(PandasResearchInput(returns=primary(), signals=signals))
    assert dataset.signals is not None

    monkeypatch.setattr(pandas_adapter_module, "_MAX_PANEL_CELLS", 3)
    with pytest.raises(DataValidationError, match="materialization budget"):
        adapter().adapt(PandasResearchInput(returns=primary(), signals=signals))

    monkeypatch.setattr(pandas_adapter_module, "_MAX_PANEL_CELLS", 4)
    monkeypatch.setattr(pandas_adapter_module, "_MAX_PANEL_ESTIMATED_BYTES", 31)
    with pytest.raises(DataValidationError, match="materialization budget"):
        adapter().adapt(PandasResearchInput(returns=primary(), signals=signals))


def test_over_budget_numeric_panel_uses_only_cheap_metadata_inspection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = panel_index([("t0", "a"), ("t1", "b")])
    signals = InspectionBombFrame(
        np.ones((2, 20)),
        index=index,
        columns=[f"field-{position:02d}" for position in range(20)],
    )
    monkeypatch.setattr(pandas_adapter_module, "_MAX_PANEL_CELLS", 10)
    monkeypatch.setattr(
        pandas_adapter_module,
        "_MAX_PANEL_ESTIMATED_BYTES",
        10_000,
    )

    original_converter = pandas_adapter_module._convert_numeric_scalar

    def fail_if_converted(*args: object, **kwargs: object) -> object:
        if len(args) >= 3 and args[2] == "signals":
            raise AssertionError("scalar conversion called before panel budget")
        return original_converter(*args, **kwargs)

    def fail_if_materialized(*args: object, **kwargs: object) -> np.ndarray[Any, Any]:
        raise AssertionError("materializer called before panel budget")

    monkeypatch.setattr(
        pandas_adapter_module,
        "_convert_numeric_scalar",
        fail_if_converted,
    )
    monkeypatch.setattr(
        pandas_adapter_module,
        "_numeric_panel_values",
        fail_if_materialized,
    )

    with pytest.raises(DataValidationError, match="materialization budget"):
        adapter().adapt(PandasResearchInput(returns=primary(), signals=signals))


def test_over_budget_object_panel_does_not_materialize_or_convert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = panel_index([("t0", "a"), ("t1", "b")])
    groups = InspectionBombFrame(
        [["x"] * 20, ["y"] * 20],
        index=index,
        columns=[f"field-{position:02d}" for position in range(20)],
    )
    monkeypatch.setattr(pandas_adapter_module, "_MAX_PANEL_CELLS", 10)
    monkeypatch.setattr(
        pandas_adapter_module,
        "_MAX_PANEL_ESTIMATED_BYTES",
        10_000,
    )

    def fail_if_materialized(*args: object, **kwargs: object) -> np.ndarray[Any, Any]:
        raise AssertionError("object materializer called before panel budget")

    monkeypatch.setattr(
        pandas_adapter_module,
        "_object_panel_values",
        fail_if_materialized,
    )

    with pytest.raises(DataValidationError, match="materialization budget"):
        adapter().adapt(PandasResearchInput(returns=primary(), groups=groups))


def test_duplicate_observation_is_rejected_before_reshaping() -> None:
    signals = pd.DataFrame(
        {"value": [1.0, 2.0]},
        index=panel_index([("t0", "a"), ("t0", "a")]),
    )

    with pytest.raises(DataValidationError, match="duplicate observations"):
        adapter().adapt(PandasResearchInput(returns=primary(), signals=signals))


def test_panel_requires_semantic_two_level_index() -> None:
    wrong_level_count = pd.DataFrame(
        {"value": [1.0, 2.0]},
        index=pd.Index(["t0", "t1"]),
    )
    wrong_names = pd.DataFrame(
        {"value": [1.0]},
        index=panel_index([("t0", "a")], names=["instrument", "time"]),
    )

    with pytest.raises(DataValidationError, match="two-level"):
        adapter().adapt(PandasResearchInput(returns=primary(), signals=wrong_level_count))
    with pytest.raises(DataValidationError, match="named time and instrument"):
        adapter().adapt(PandasResearchInput(returns=primary(), signals=wrong_names))


def test_duplicate_wide_axes_are_rejected() -> None:
    duplicate_columns = pd.DataFrame(
        [[0.0, 0.1]],
        index=["t0"],
        columns=["a", "a"],
    )
    duplicate_rows = pd.DataFrame([[0.0], [0.1]], index=["t0", "t0"], columns=["a"])

    with pytest.raises(DataValidationError, match="duplicate columns"):
        adapter().adapt(PandasResearchInput(returns=duplicate_columns))
    with pytest.raises(DataValidationError, match="duplicate observations"):
        adapter().adapt(PandasResearchInput(returns=duplicate_rows))


def test_multiindex_duplicate_wide_axis_is_rejected() -> None:
    columns = pd.MultiIndex.from_tuples([("x", "a"), ("x", "a")])
    returns = pd.DataFrame([[0.0, 0.1]], index=["t0"], columns=columns)

    with pytest.raises(DataValidationError, match="duplicate columns"):
        adapter().adapt(PandasResearchInput(returns=returns))


@pytest.mark.parametrize("axis", ["index", "columns"])
def test_malicious_label_hash_is_structured_and_bounded(axis: str) -> None:
    label = MaliciousHashLabel()
    malicious_axis = pd.Index(
        np.asarray([label], dtype=object),
        dtype=object,
    )
    returns = pd.DataFrame(
        [[0.0]],
        index=malicious_axis if axis == "index" else ["t0"],
        columns=malicious_axis if axis == "columns" else ["a"],
    )

    with pytest.raises(DataValidationError, match="invalid labels") as captured:
        adapter().adapt(PandasResearchInput(returns=returns))

    payload = captured.value.as_dict()
    assert set(payload["context"]) == {
        "field",
        "axis",
        "position",
        "reason",
        "type",
    }
    assert len(json.dumps(payload)) < 500
    assert len(captured.value.context["type"]) <= 80


def test_malicious_multiindex_level_hash_is_structured() -> None:
    label = MaliciousHashLabel()
    malicious_level = pd.Index(
        np.asarray([label], dtype=object),
        dtype=object,
    )
    index = pd.MultiIndex(
        levels=[malicious_level, pd.Index(["a"])],
        codes=[[0], [0]],
        names=["time", "instrument"],
        verify_integrity=False,
    )
    signals = pd.DataFrame({"value": [1.0]}, index=index)
    returns = pd.DataFrame([[0.0]], index=["t0"], columns=["a"])

    with pytest.raises(DataValidationError, match="invalid labels") as captured:
        adapter().adapt(PandasResearchInput(returns=returns, signals=signals))

    assert captured.value.context["axis"] == "index.level_0"
    assert len(json.dumps(captured.value.as_dict())) < 500


def test_duplicate_engine_hash_failure_is_structured() -> None:
    label = DuplicateEngineHashFailure()
    index = pd.Index(np.asarray([label], dtype=object), dtype=object)
    returns = pd.DataFrame([[0.0]], index=index, columns=["a"])
    label.calls = 0

    with pytest.raises(DataValidationError) as captured:
        adapter().adapt(PandasResearchInput(returns=returns))

    payload = captured.value.as_dict()
    assert payload["error"] == "data_validation"
    assert len(json.dumps(payload)) < 500
    context = payload["context"]
    assert context["field"] in {"returns", "adapter", "dataset"}
    assert isinstance(context["reason"], str)
    assert "axis" in context or ("component" in context and "type" in context)


@pytest.mark.parametrize("axis", ["index", "columns"])
@pytest.mark.parametrize("failure_at", [2, 3, 4])
def test_stateful_hash_failures_never_escape_matrix_adaptation(
    axis: str,
    failure_at: int,
) -> None:
    label = NthHashFailure(failure_at)
    hostile_axis = pd.Index(np.asarray([label], dtype=object), dtype=object)
    returns = pd.DataFrame(
        [[0.0]],
        index=hostile_axis if axis == "index" else ["t0"],
        columns=hostile_axis if axis == "columns" else ["a"],
    )
    label.calls = 0

    with pytest.raises(DataValidationError) as captured:
        adapter().adapt(PandasResearchInput(returns=returns))

    payload = captured.value.as_dict()
    assert len(json.dumps(payload)) < 500
    assert isinstance(payload["context"]["type"], str)
    assert len(payload["context"]["type"]) <= 80


@pytest.mark.parametrize("failure_at", [8, 12])
def test_stateful_hash_failures_never_escape_multiindex_adaptation(
    failure_at: int,
) -> None:
    label = NthHashFailure(failure_at)
    instrument_level = pd.Index(
        np.asarray([label], dtype=object),
        dtype=object,
    )
    index = pd.MultiIndex(
        levels=[pd.Index(["t0"]), instrument_level],
        codes=[[0], [0]],
        names=["time", "instrument"],
        verify_integrity=False,
    )
    returns = pd.DataFrame([[0.0]], index=["t0"], columns=instrument_level)
    signals = pd.DataFrame({"value": [1.0]}, index=index)
    label.calls = 0

    with pytest.raises(DataValidationError) as captured:
        adapter().adapt(PandasResearchInput(returns=returns, signals=signals))

    payload = captured.value.as_dict()
    assert len(json.dumps(payload)) < 500
    assert isinstance(payload["context"]["type"], str)
    assert len(payload["context"]["type"]) <= 80


def test_stable_custom_hash_labels_adapt_successfully() -> None:
    label = NthHashFailure(None)
    index = pd.Index(np.asarray([label], dtype=object), dtype=object)
    returns = pd.DataFrame([[0.0]], index=index, columns=["a"])

    dataset = adapter().adapt(PandasResearchInput(returns=returns))

    assert dataset.returns is not None
    assert dataset.returns.row_labels == (label,)


@pytest.mark.parametrize(
    "label",
    [
        MemoryHashLabel(),
        MemoryReflexiveLabel(),
    ],
)
def test_label_hash_and_reflexivity_memory_errors_propagate(
    label: object,
) -> None:
    index = pd.Index(np.asarray([label], dtype=object), dtype=object)
    returns = pd.DataFrame([[0.0]], index=index, columns=["a"])

    with pytest.raises(MemoryError):
        adapter().adapt(PandasResearchInput(returns=returns))


def test_duplicate_engine_memory_error_propagates() -> None:
    label = DuplicateEngineMemoryLabel()
    index = pd.Index(np.asarray([label], dtype=object), dtype=object)
    returns = pd.DataFrame([[0.0]], index=index, columns=["a"])
    label.calls = 0

    with pytest.raises(MemoryError):
        adapter().adapt(PandasResearchInput(returns=returns))


def test_custom_base_exception_is_not_wrapped() -> None:
    label = StopHashLabel()
    index = pd.Index(np.asarray([label], dtype=object), dtype=object)
    returns = pd.DataFrame([[0.0]], index=index, columns=["a"])

    with pytest.raises(StopAdaptation):
        adapter().adapt(PandasResearchInput(returns=returns))


@pytest.mark.parametrize(
    "cell",
    [
        MemoryCopyGroupCell(),
        MemoryHashGroupCell(),
    ],
)
def test_group_cell_memory_errors_propagate(cell: object) -> None:
    groups = pd.DataFrame(
        {"group": [cell]},
        index=panel_index([("t0", "a")]),
    )
    returns = pd.DataFrame([[0.0]], index=["t0"], columns=["a"])

    with pytest.raises(MemoryError):
        adapter().adapt(PandasResearchInput(returns=returns, groups=groups))


def test_late_dataset_memory_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_memory_error(*args: object, **kwargs: object) -> object:
        raise MemoryError

    monkeypatch.setattr(
        pandas_adapter_module,
        "ResearchDataset",
        raise_memory_error,
    )

    with pytest.raises(MemoryError):
        adapter().adapt(PandasResearchInput(returns=primary()))


def test_missing_universe_value_is_not_filled() -> None:
    universe = pd.DataFrame([[True, None], [False, True]], index=["t0", "t1"], columns=["a", "b"])

    with pytest.raises(DataValidationError, match="universe contains missing values"):
        adapter().adapt(PandasResearchInput(returns=primary(), universe=universe))


@pytest.mark.parametrize(
    "values",
    [
        [[1, 0], [0, 1]],
        [["True", "False"], ["False", "True"]],
    ],
)
def test_non_boolean_universe_is_not_silently_cast(values: list[list[Any]]) -> None:
    universe = pd.DataFrame(values, index=["t0", "t1"], columns=["a", "b"])

    with pytest.raises(DataValidationError, match="universe must be boolean"):
        adapter().adapt(PandasResearchInput(returns=primary(), universe=universe))


def test_boolean_universe_preserves_values() -> None:
    universe = pd.DataFrame(
        [[True, False], [False, True]],
        index=["t0", "t1"],
        columns=["a", "b"],
        dtype=bool,
    )
    dataset = adapter().adapt(PandasResearchInput(returns=primary(), universe=universe))

    assert dataset.universe is not None
    assert dataset.universe.values.dtype == np.dtype(np.bool_)
    np.testing.assert_array_equal(dataset.universe.values, universe.to_numpy())


@pytest.mark.parametrize("mixed_builtin_column", [False, True])
def test_nullable_boolean_universe_is_converted_losslessly(
    mixed_builtin_column: bool,
) -> None:
    columns: dict[str, pd.Series[Any]] = {
        "a": pd.Series(
            [True, False],
            index=["t0", "t1"],
            dtype="boolean",
        ),
        "b": pd.Series(
            [False, True],
            index=["t0", "t1"],
            dtype=bool if mixed_builtin_column else "boolean",
        ),
    }
    universe = pd.DataFrame(columns)

    dataset = adapter().adapt(PandasResearchInput(returns=primary(), universe=universe))

    assert dataset.universe is not None
    assert dataset.universe.values.dtype == np.dtype(np.bool_)
    np.testing.assert_array_equal(
        dataset.universe.values,
        [[True, False], [False, True]],
    )


def test_nullable_boolean_universe_still_rejects_missing_values() -> None:
    universe = pd.DataFrame(
        {
            "a": pd.Series(
                [True, pd.NA],
                index=["t0", "t1"],
                dtype="boolean",
            ),
            "b": pd.Series(
                [False, True],
                index=["t0", "t1"],
                dtype="boolean",
            ),
        }
    )

    with pytest.raises(DataValidationError, match="universe contains missing values"):
        adapter().adapt(PandasResearchInput(returns=primary(), universe=universe))


def test_universe_dtype_error_context_is_bounded_for_many_columns() -> None:
    column_count = 10_000
    universe = pd.DataFrame(
        [["not-bool"] * column_count],
        index=["t0"],
        columns=[f"asset-{position:05d}" for position in range(column_count)],
    )

    with pytest.raises(DataValidationError, match="universe must be boolean") as captured:
        adapter().adapt(PandasResearchInput(returns=primary(), universe=universe))

    payload = json.dumps(captured.value.as_dict(), sort_keys=True)
    assert len(payload) < 1_000
    assert captured.value.context["invalid_dtype_count"] == column_count
    assert len(captured.value.context["invalid_dtype_sample"]) <= 5


@pytest.mark.parametrize(
    ("value", "label"),
    [
        ("1.2", "strings"),
        (True, "boolean"),
        (1 + 2j, "complex"),
        (pd.Timestamp("2026-01-01"), "datetime"),
        (pd.Timedelta(days=1), "timedelta"),
    ],
)
def test_returns_reject_non_real_numeric_dtypes(value: Any, label: str) -> None:
    returns = pd.DataFrame([[value]], index=["t0"], columns=["a"])

    with pytest.raises(DataValidationError, match="returns must be numeric") as captured:
        adapter().adapt(PandasResearchInput(returns=returns))

    assert captured.value.context["reason"] == label
    assert_structured(captured.value)


@pytest.mark.parametrize(
    "returns",
    [
        pd.DataFrame(
            {
                "large": pd.Series(
                    [2**53 + 1, pd.NA],
                    index=["t0", "t1"],
                    dtype="Int64",
                )
            },
            index=["t0", "t1"],
        ),
        pd.DataFrame(
            {
                "large": pd.Series(
                    [2**53 + 1, 2**53 + 1],
                    index=["t0", "t1"],
                    dtype=np.int64,
                ),
                "float": pd.Series(
                    [1.5, 2.5],
                    index=["t0", "t1"],
                    dtype=np.float64,
                ),
            },
            index=["t0", "t1"],
        ),
        pd.DataFrame(
            {
                "unsigned": pd.Series(
                    [2**63 + 1, 2**63 + 2],
                    index=["t0", "t1"],
                    dtype=np.uint64,
                ),
                "signed": pd.Series(
                    [-1, -2],
                    index=["t0", "t1"],
                    dtype=np.int64,
                ),
            },
            index=["t0", "t1"],
        ),
    ],
)
def test_matrix_rejects_lossy_numeric_conversion(returns: pd.DataFrame) -> None:
    with pytest.raises(
        DataValidationError,
        match="conversion would be lossy",
    ) as captured:
        adapter().adapt(PandasResearchInput(returns=returns))

    assert captured.value.context["reason"] == "lossy_conversion"
    assert_structured(captured.value)


def test_benchmark_series_rejects_lossy_nullable_integer_conversion() -> None:
    benchmark = pd.Series([2**53 + 1, pd.NA], index=["t0", "t1"], dtype="Int64")

    with pytest.raises(DataValidationError, match="conversion would be lossy"):
        adapter().adapt(
            PandasResearchInput(
                returns=primary(),
                benchmark_returns=benchmark,
            )
        )


def test_numeric_panel_rejects_lossy_conversion_when_cartesian_gap_requires_nan() -> None:
    index = panel_index([("t0", "a"), ("t0", "b"), ("t1", "a")])
    signals = pd.DataFrame({"value": pd.Series([2**53 + 1, 1, 2], index=index, dtype="Int64")})

    with pytest.raises(DataValidationError, match="conversion would be lossy"):
        adapter().adapt(PandasResearchInput(returns=primary(), signals=signals))


def test_safe_nullable_and_mixed_numeric_values_preserve_identity_and_missing() -> None:
    returns = pd.DataFrame(
        {
            "integer": pd.Series(
                [2**53, pd.NA],
                index=["t0", "t1"],
                dtype="Int64",
            ),
            "float": pd.Series(
                [1.5, 2.5],
                index=["t0", "t1"],
                dtype=np.float64,
            ),
        },
        index=["t0", "t1"],
    )

    dataset = adapter().adapt(PandasResearchInput(returns=returns))

    assert dataset.returns is not None
    assert dataset.returns.values[0, 0] == 2**53
    assert np.isnan(dataset.returns.values[1, 0])
    assert dataset.returns.values[0, 1] == 1.5


def test_numeric_panels_reject_non_numeric_dtypes() -> None:
    for field in ("signals", "exposures"):
        frame = pd.DataFrame(
            {"value": pd.Categorical(["1", "2", "3", "4"])},
            index=panel_index(),
        )
        source = PandasResearchInput(returns=primary(), **{field: frame})
        with pytest.raises(DataValidationError, match=f"{field} must be numeric"):
            adapter().adapt(source)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("returns", np.inf),
        ("prices", -np.inf),
        ("future_returns", np.inf),
        ("benchmark_weights", -np.inf),
        ("benchmark_returns", np.inf),
    ],
)
def test_numeric_wide_components_reject_infinity(field: str, value: float) -> None:
    supplied: dict[str, Any] = {"returns": primary()}
    if field == "benchmark_returns":
        supplied[field] = pd.Series([value, 0.0], index=["t0", "t1"])
    else:
        supplied[field] = pd.DataFrame(
            [[value, 0.0], [0.0, 0.0]],
            index=["t0", "t1"],
            columns=["a", "b"],
        )

    with pytest.raises(DataValidationError, match="must not contain infinity"):
        adapter().adapt(PandasResearchInput(**supplied))


def test_numeric_panels_reject_infinity() -> None:
    for field in ("signals", "exposures"):
        frame = pd.DataFrame(
            {"value": [np.inf, 2.0, 3.0, 4.0]},
            index=panel_index(),
        )
        source = PandasResearchInput(returns=primary(), **{field: frame})
        with pytest.raises(DataValidationError, match="must not contain infinity"):
            adapter().adapt(source)


def test_groups_may_be_non_numeric() -> None:
    groups = pd.DataFrame(
        {"sector": ["tech", "bank", "tech", "bank"]},
        index=panel_index(),
    )

    dataset = adapter().adapt(PandasResearchInput(returns=primary(), groups=groups))

    assert dataset.groups is not None
    np.testing.assert_array_equal(
        dataset.groups.values[:, :, 0],
        [["tech", "bank"], ["tech", "bank"]],
    )


@pytest.mark.parametrize(
    ("dtype", "large_value"),
    [
        (np.int64, 2**53 + 1),
        (np.uint64, 2**63 + 1),
    ],
)
def test_group_integer_ids_survive_cartesian_gaps_without_promotion(
    dtype: type[np.generic],
    large_value: int,
) -> None:
    index = panel_index([("t0", "a"), ("t0", "b"), ("t1", "a")])
    groups = pd.DataFrame(
        {
            "group_id": pd.Series(
                [large_value, 2, 3],
                index=index,
                dtype=dtype,
            )
        }
    )
    original = groups.iat[0, 0]

    dataset = adapter().adapt(PandasResearchInput(returns=primary(), groups=groups))

    assert dataset.groups is not None
    converted = dataset.groups.values
    assert converted.dtype == np.dtype(object)
    assert converted[0, 0, 0] == large_value
    assert type(converted[0, 0, 0]) is type(original)
    assert np.isnan(converted[1, 1, 0])


def test_group_mixed_string_and_integer_fields_preserve_cells_and_order() -> None:
    index = panel_index([("t0", "a"), ("t0", "b"), ("t1", "a")])
    groups = pd.DataFrame(
        {
            "label": pd.Series(["large", "small", "other"], index=index),
            "group_id": pd.Series([2**53 + 1, 2, 3], index=index, dtype=np.int64),
        }
    )
    original_label = groups.iat[0, 0]
    original_id = groups.iat[0, 1]

    dataset = adapter().adapt(PandasResearchInput(returns=primary(), groups=groups))

    assert dataset.groups is not None
    assert dataset.groups.field_labels == ("label", "group_id")
    converted = dataset.groups.values
    assert converted[0, 0, 0] == "large"
    assert type(converted[0, 0, 0]) is type(original_label)
    assert converted[0, 0, 1] == 2**53 + 1
    assert type(converted[0, 0, 1]) is type(original_id)
    assert all(pd.isna(cell) for cell in converted[1, 1, :])


@pytest.mark.parametrize(
    "cell",
    [
        ["mutable"],
        UncopyableGroupCell(),
        ExplodingHashGroupCell(),
    ],
)
def test_invalid_group_cells_raise_structured_validation(cell: object) -> None:
    groups = pd.DataFrame(
        {"group": [cell]},
        index=panel_index([("t0", "a")]),
    )
    returns = pd.DataFrame([[0.0]], index=["t0"], columns=["a"])

    with pytest.raises(DataValidationError) as captured:
        adapter().adapt(PandasResearchInput(returns=returns, groups=groups))

    assert captured.value.context["field"] == "groups"
    assert len(json.dumps(captured.value.as_dict())) < 500


def test_benchmark_series_becomes_time_vector() -> None:
    benchmark = pd.Series([0.01, np.nan], index=["t0", "t1"], name="benchmark")

    dataset = adapter().adapt(PandasResearchInput(returns=primary(), benchmark_returns=benchmark))

    assert dataset.benchmark_returns is not None
    assert dataset.benchmark_returns.labels == ("t0", "t1")
    assert dataset.benchmark_returns.axis_name == "time"
    assert np.isnan(dataset.benchmark_returns.values[1])


@pytest.mark.parametrize(
    ("field", "wrong"),
    [
        ("returns", pd.Series([0.1])),
        ("prices", np.array([[1.0]])),
        ("signals", {"value": [1.0]}),
        ("future_returns", pd.Series([0.1])),
        ("universe", pd.Series([True])),
        ("benchmark_returns", pd.DataFrame([[0.1]])),
        ("benchmark_weights", pd.Series([1.0])),
        ("groups", pd.Series(["x"])),
        ("exposures", [[1.0]]),
    ],
)
def test_wrong_component_types_raise_structured_validation(
    field: str,
    wrong: object,
) -> None:
    supplied: dict[str, Any] = {"returns": primary(), field: wrong}

    with pytest.raises(DataValidationError, match="has the wrong pandas type") as captured:
        adapter().adapt(PandasResearchInput(**supplied))

    assert captured.value.context["field"] == field
    assert_structured(captured.value)


def test_wrong_source_type_raises_structured_validation() -> None:
    with pytest.raises(DataValidationError, match="source must be PandasResearchInput") as captured:
        adapter().adapt(object())  # type: ignore[arg-type]

    assert_structured(captured.value)


def test_wrong_adapter_contract_types_raise_structured_validation() -> None:
    invalid_metadata = PandasAdapter(metadata=object())  # type: ignore[arg-type]
    with pytest.raises(DataValidationError, match="metadata must be DatasetMetadata") as captured:
        invalid_metadata.adapt(PandasResearchInput(returns=primary()))
    assert_structured(captured.value)

    invalid_provenance = PandasAdapter(
        metadata=adapter().metadata,
        provenance=object(),  # type: ignore[arg-type]
    )
    with pytest.raises(DataValidationError, match="provenance must be InputProvenance") as captured:
        invalid_provenance.adapt(PandasResearchInput(returns=primary()))
    assert_structured(captured.value)


@pytest.mark.parametrize(
    "field",
    [
        "returns",
        "prices",
        "future_returns",
        "universe",
        "benchmark_returns",
        "benchmark_weights",
        "signals",
        "groups",
        "exposures",
    ],
)
def test_timezone_metadata_is_checked_for_every_component(field: str) -> None:
    mismatched_time = pd.date_range("2026-01-01", periods=2, tz="Asia/Bangkok")
    supplied: dict[str, Any] = {"returns": primary()}
    if field in {"signals", "groups", "exposures"}:
        index = pd.MultiIndex.from_product(
            [mismatched_time, ["a", "b"]],
            names=["time", "instrument"],
        )
        value: Any = ["x"] * 4 if field == "groups" else [1.0] * 4
        supplied[field] = pd.DataFrame({"value": value}, index=index)
    elif field == "benchmark_returns":
        supplied[field] = pd.Series([0.0, 0.0], index=mismatched_time)
    else:
        value = [[True, True], [True, True]] if field == "universe" else [[0.0, 0.0], [0.0, 0.0]]
        supplied[field] = pd.DataFrame(value, index=mismatched_time, columns=["a", "b"])

    with pytest.raises(DataValidationError, match="timezone does not match") as captured:
        adapter().adapt(PandasResearchInput(**supplied))

    assert captured.value.context["field"] == field
    assert_structured(captured.value)


def test_naive_datetime_axis_requires_declared_timezone_none() -> None:
    naive = primary(index=pd.date_range("2026-01-01", periods=2))

    with pytest.raises(DataValidationError, match="timezone does not match"):
        adapter().adapt(PandasResearchInput(returns=naive))

    dataset = adapter(timezone=None).adapt(PandasResearchInput(returns=naive))
    assert dataset.returns is not None


def test_zoneinfo_timezone_string_matches_metadata_deterministically() -> None:
    index = pd.date_range(
        "2026-01-01",
        periods=2,
        tz=ZoneInfo("Asia/Bangkok"),
    )
    dataset = adapter(timezone="Asia/Bangkok").adapt(
        PandasResearchInput(returns=primary(index=index))
    )
    assert dataset.returns is not None


def object_time_index(values: list[object]) -> pd.Index[Any]:
    index = pd.Index(np.asarray(values, dtype=object), dtype=object)
    assert index.dtype == np.dtype(object)
    return index


def test_object_datetime_axis_timezone_must_match_metadata() -> None:
    index = object_time_index(
        [
            pd.Timestamp("2026-01-01", tz="UTC"),
            pd.Timestamp("2026-01-02", tz="UTC"),
        ]
    )

    with pytest.raises(DataValidationError, match="timezone does not match"):
        adapter(timezone="Asia/Bangkok").adapt(PandasResearchInput(returns=primary(index=index)))


def test_object_datetime_axis_rejects_mixed_aware_timezones() -> None:
    index = object_time_index(
        [
            pd.Timestamp("2026-01-01", tz="UTC"),
            pd.Timestamp("2026-01-02", tz="Asia/Bangkok"),
        ]
    )

    with pytest.raises(DataValidationError, match="mixed timezones") as captured:
        adapter().adapt(PandasResearchInput(returns=primary(index=index)))

    assert_structured(captured.value)


def test_object_datetime_axis_rejects_mixed_naive_and_aware_labels() -> None:
    index = object_time_index(
        [
            dt.datetime(2026, 1, 1),
            dt.datetime(2026, 1, 2, tzinfo=dt.UTC),
        ]
    )

    with pytest.raises(DataValidationError, match="mix aware and naive"):
        adapter().adapt(PandasResearchInput(returns=primary(index=index)))


@pytest.mark.parametrize("field", ["benchmark_returns", "signals"])
def test_object_datetime_timezone_validation_applies_to_vector_and_panel(
    field: str,
) -> None:
    times = object_time_index(
        [
            pd.Timestamp("2026-01-01", tz="Asia/Bangkok"),
            pd.Timestamp("2026-01-02", tz="Asia/Bangkok"),
        ]
    )
    supplied: dict[str, Any] = {"returns": primary()}
    if field == "benchmark_returns":
        supplied[field] = pd.Series([0.0, 0.0], index=times)
    else:
        index = pd.MultiIndex.from_product(
            [times, ["a", "b"]],
            names=["time", "instrument"],
        )
        supplied[field] = pd.DataFrame({"value": [1.0] * 4}, index=index)

    with pytest.raises(DataValidationError, match="timezone does not match"):
        adapter().adapt(PandasResearchInput(**supplied))


def test_object_time_axis_rejects_mixed_datetime_and_generic_labels() -> None:
    index = object_time_index([pd.Timestamp("2026-01-01", tz="UTC"), "t1"])

    with pytest.raises(DataValidationError, match="mix datetime and generic"):
        adapter().adapt(PandasResearchInput(returns=primary(index=index)))


def test_timezone_mismatch_context_is_bounded_and_json_safe() -> None:
    index = pd.date_range("2026-01-01", periods=2, tz="UTC")

    with pytest.raises(DataValidationError) as captured:
        adapter(timezone="x" * 1_000).adapt(PandasResearchInput(returns=primary(index=index)))

    assert len(captured.value.context["declared"]) <= 80
    json.dumps(captured.value.as_dict())


def test_non_datetime_labels_do_not_claim_timezone_validation() -> None:
    dataset = adapter().adapt(PandasResearchInput(returns=primary()))
    assert dataset.returns is not None


def test_caller_mutation_after_adapt_does_not_change_dataset() -> None:
    returns = primary()
    signals = pd.DataFrame(
        {"value": [1.0, 2.0, 3.0, 4.0]},
        index=panel_index(),
    )
    dataset = adapter().adapt(PandasResearchInput(returns=returns, signals=signals))

    returns.iloc[0, 0] = 999.0
    signals.iloc[0, 0] = 999.0

    assert dataset.returns is not None
    assert dataset.signals is not None
    assert dataset.returns.values[0, 0] == 0.01
    assert dataset.signals.values[0, 0, 0] == 1.0


def test_invalid_axis_labels_raise_structured_validation() -> None:
    returns = pd.DataFrame([[0.0]], index=[pd.NA], columns=["a"])

    with pytest.raises(DataValidationError, match="invalid labels") as captured:
        adapter().adapt(PandasResearchInput(returns=returns))

    assert_structured(captured.value)


def test_stateful_hash_failure_at_benchmark_vector_boundary_is_structured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    label = NthHashFailure(None)
    index = pd.Index(np.asarray([label], dtype=object), dtype=object)
    benchmark = pd.Series([0.0], index=index)
    original_vector = pandas_adapter_module.LabeledVector

    def fail_inside_vector(
        values: object,
        labels: object,
        axis_name: str,
    ) -> object:
        label.failure_at = label.calls + 1
        return original_vector(values, labels, axis_name)

    monkeypatch.setattr(
        pandas_adapter_module,
        "LabeledVector",
        fail_inside_vector,
    )

    with pytest.raises(DataValidationError) as captured:
        adapter().adapt(
            PandasResearchInput(
                returns=pd.DataFrame([[0.0]], index=["t0"], columns=["a"]),
                benchmark_returns=benchmark,
            )
        )

    payload = captured.value.as_dict()
    assert payload["context"]["component"] == "benchmark_returns"
    assert payload["context"]["type"] == "RuntimeError"
    assert len(json.dumps(payload)) < 500


def test_stateful_hash_failure_at_long_panel_boundary_is_structured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    label = NthHashFailure(None)
    columns = pd.Index(np.asarray([label], dtype=object), dtype=object)
    signals = pd.DataFrame(
        [[1.0]],
        index=panel_index([("t0", "a")]),
        columns=columns,
    )
    original_panel = pandas_adapter_module.LabeledPanel

    def fail_inside_panel(*args: object, **kwargs: object) -> object:
        label.failure_at = label.calls + 1
        return original_panel(*args, **kwargs)

    monkeypatch.setattr(
        pandas_adapter_module,
        "LabeledPanel",
        fail_inside_panel,
    )

    with pytest.raises(DataValidationError) as captured:
        adapter().adapt(
            PandasResearchInput(
                returns=pd.DataFrame([[0.0]], index=["t0"], columns=["a"]),
                signals=signals,
            )
        )

    payload = captured.value.as_dict()
    assert payload["context"]["component"] == "signals"
    assert payload["context"]["type"] == "RuntimeError"
    assert len(json.dumps(payload)) < 500


def test_stateful_hash_failure_at_final_dataset_boundary_is_structured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    label = NthHashFailure(None)
    instruments = pd.Index(np.asarray([label], dtype=object), dtype=object)
    returns = pd.DataFrame([[0.0]], index=["t0"], columns=instruments)
    signals = pd.DataFrame(
        {"value": [1.0]},
        index=pd.MultiIndex(
            levels=[pd.Index(["t0"]), instruments],
            codes=[[0], [0]],
            names=["time", "instrument"],
            verify_integrity=False,
        ),
    )
    original_dataset = pandas_adapter_module.ResearchDataset

    def fail_inside_dataset(*args: object, **kwargs: object) -> object:
        label.failure_at = label.calls + 1
        return original_dataset(*args, **kwargs)

    monkeypatch.setattr(
        pandas_adapter_module,
        "ResearchDataset",
        fail_inside_dataset,
    )

    with pytest.raises(DataValidationError) as captured:
        adapter().adapt(PandasResearchInput(returns=returns, signals=signals))

    payload = captured.value.as_dict()
    assert payload["context"]["component"] == "dataset"
    assert payload["context"]["type"] == "RuntimeError"
    assert len(json.dumps(payload)) < 500


def test_exception_chain_cycle_and_hostile_text_terminate_safely() -> None:
    cycle = HostileTextError()
    cycle.__cause__ = cycle
    cycle.__context__ = cycle

    assert pandas_adapter_module._memory_error_in_chain(cycle) is None


def test_exception_chain_depth_bound_is_intentional() -> None:
    root = RuntimeError()
    current: BaseException = root
    for _ in range(pandas_adapter_module._MAX_EXCEPTION_CHAIN_DEPTH):
        nested = RuntimeError()
        current.__cause__ = nested
        current = nested
    beyond_limit = MemoryError()
    current.__cause__ = beyond_limit

    assert pandas_adapter_module._memory_error_in_chain(root) is None


def test_embedded_memory_error_within_chain_is_found_and_reraised() -> None:
    wrapped = DataValidationError("wrapped")
    intermediate = HostileTextError()
    memory_error = MemoryError()
    wrapped.__cause__ = intermediate
    intermediate.__context__ = memory_error

    assert pandas_adapter_module._memory_error_in_chain(wrapped) is memory_error
    with pytest.raises(MemoryError) as captured:
        pandas_adapter_module._reraise_memory_error_from_chain(wrapped)
    assert captured.value is memory_error


def test_qamr_error_without_memory_is_preserved_by_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = DataValidationError(
        "sentinel",
        context={"field": "returns"},
    )

    def raise_sentinel(*args: object, **kwargs: object) -> object:
        raise sentinel

    monkeypatch.setattr(
        pandas_adapter_module,
        "_numeric_matrix",
        raise_sentinel,
    )

    with pytest.raises(DataValidationError) as captured:
        adapter().adapt(PandasResearchInput(returns=primary()))

    assert captured.value is sentinel


def test_keyboard_interrupt_propagates_from_component_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupt_component(*args: object, **kwargs: object) -> object:
        raise KeyboardInterrupt

    monkeypatch.setattr(
        pandas_adapter_module,
        "LabeledMatrix",
        interrupt_component,
    )

    with pytest.raises(KeyboardInterrupt):
        adapter().adapt(PandasResearchInput(returns=primary()))


def test_custom_base_exception_propagates_from_final_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def stop_final_boundary(*args: object, **kwargs: object) -> object:
        raise StopAdaptation

    monkeypatch.setattr(
        pandas_adapter_module,
        "ResearchDataset",
        stop_final_boundary,
    )

    with pytest.raises(StopAdaptation):
        adapter().adapt(PandasResearchInput(returns=primary()))


def test_core_dataset_import_does_not_import_optional_pandas_modules() -> None:
    probe = (
        "import json, sys;"
        "from qamr.contracts.dataset import ResearchDataset;"
        "print(json.dumps({'name': ResearchDataset.__name__,"
        "'pandas': 'pandas' in sys.modules,"
        "'adapter': 'qamr.contracts.pandas_adapter' in sys.modules}))"
    )

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {
        "name": "ResearchDataset",
        "pandas": False,
        "adapter": False,
    }
