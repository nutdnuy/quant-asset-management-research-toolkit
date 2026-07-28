from dataclasses import FrozenInstanceError
from datetime import date
from typing import Any

import numpy as np
import pytest

from qamr.contracts.arrays import LabeledMatrix, LabeledPanel, LabeledVector
from qamr.errors import DataValidationError


def _make_vector(values: Any) -> LabeledVector:
    return LabeledVector(values, ("a", "b"), "instrument")


def _make_matrix(values: Any) -> LabeledMatrix:
    return LabeledMatrix(values, ("t0", "t1"), ("a", "b"), "time", "instrument")


def _make_panel(values: Any) -> LabeledPanel:
    return LabeledPanel(values, ("t0", "t1"), ("a", "b"), ("value",))


_CONTAINER_CASES = [
    pytest.param(_make_vector, (2,), "labels", id="vector"),
    pytest.param(_make_matrix, (2, 2), "row_labels", id="matrix"),
    pytest.param(_make_panel, (2, 2, 1), "time_labels", id="panel"),
]


class _StableHashableBox:
    def __init__(self, value: list[str]) -> None:
        self.value = value


def test_labeled_matrix_preserves_labels_and_copies_values() -> None:
    source = np.array([[0.01, 0.02], [0.03, 0.04]])
    matrix = LabeledMatrix(
        values=source,
        row_labels=(date(2026, 1, 1), date(2026, 1, 2)),
        column_labels=("asset-a", "asset-b"),
        row_name="time",
        column_name="instrument",
    )
    source[0, 0] = 99.0

    assert matrix.shape == (2, 2)
    assert matrix.values[0, 0] == pytest.approx(0.01)
    assert matrix.row_labels[0] == date(2026, 1, 1)
    assert matrix.column_labels == ("asset-a", "asset-b")
    assert matrix.values.flags.writeable is False


def test_labeled_vector_rejects_duplicate_labels() -> None:
    with pytest.raises(DataValidationError, match="labels must be unique") as captured:
        LabeledVector(np.array([1.0, 2.0]), ("asset-a", "asset-a"), "instrument")

    assert captured.value.context == {"axis": "instrument", "duplicate": "asset-a"}


def test_labeled_matrix_rejects_shape_mismatch() -> None:
    with pytest.raises(DataValidationError, match="shape does not match labels"):
        LabeledMatrix(
            np.ones((2, 3)),
            row_labels=("t0", "t1"),
            column_labels=("a", "b"),
            row_name="time",
            column_name="instrument",
        )


def test_labeled_panel_has_time_instrument_field_axes() -> None:
    panel = LabeledPanel(
        np.arange(12.0).reshape(2, 3, 2),
        time_labels=("t0", "t1"),
        instrument_labels=("a", "b", "c"),
        field_labels=("value", "quality"),
    )

    assert panel.shape == (2, 3, 2)
    assert panel.field_labels == ("value", "quality")


def test_unhashable_label_is_rejected_with_axis_context() -> None:
    with pytest.raises(DataValidationError, match="labels must be hashable") as captured:
        LabeledVector(np.array([1.0]), (["asset-a"],), "instrument")  # type: ignore[arg-type]

    assert captured.value.context == {"axis": "instrument", "position": 0}


@pytest.mark.parametrize(("factory", "shape", "_label_field"), _CONTAINER_CASES)
def test_source_mutation_is_isolated_for_every_container(
    factory: Any,
    shape: tuple[int, ...],
    _label_field: str,
) -> None:
    source = np.arange(float(np.prod(shape))).reshape(shape)
    container = factory(source)
    source.flat[0] = 99.0

    assert container.values.flat[0] == pytest.approx(0.0)


@pytest.mark.parametrize(("factory", "shape", "_label_field"), _CONTAINER_CASES)
def test_public_values_are_read_only_defensive_snapshots(
    factory: Any,
    shape: tuple[int, ...],
    _label_field: str,
) -> None:
    container = factory(np.arange(float(np.prod(shape))).reshape(shape))
    snapshot = container.values

    assert snapshot.flags.writeable is False
    assert snapshot.base is None

    snapshot.flags.writeable = True
    writable_view = snapshot.reshape(-1)
    writable_view[0] = 99.0

    assert container.values.flat[0] == pytest.approx(0.0)


@pytest.mark.parametrize(("factory", "shape", "label_field"), _CONTAINER_CASES)
def test_container_fields_are_frozen(
    factory: Any,
    shape: tuple[int, ...],
    label_field: str,
) -> None:
    container = factory(np.zeros(shape))

    with pytest.raises(FrozenInstanceError):
        setattr(container, label_field, ())


@pytest.mark.parametrize(("factory", "shape", "_label_field"), _CONTAINER_CASES)
def test_object_string_and_date_cells_are_supported(
    factory: Any,
    shape: tuple[int, ...],
    _label_field: str,
) -> None:
    source = np.full(shape, "stable", dtype=object)
    source.flat[0] = date(2026, 1, 1)

    container = factory(source)

    assert container.values.flat[0] == date(2026, 1, 1)
    assert container.values.flat[-1] == "stable"


@pytest.mark.parametrize(("factory", "shape", "_label_field"), _CONTAINER_CASES)
def test_nested_mutable_object_cell_is_rejected(
    factory: Any,
    shape: tuple[int, ...],
    _label_field: str,
) -> None:
    source = np.full(shape, "stable", dtype=object)
    source.flat[0] = ["mutable"]

    with pytest.raises(DataValidationError, match="object cells must be hashable") as captured:
        factory(source)

    assert captured.value.as_dict()["context"] == {
        "position": [0] * len(shape),
        "reason": "unhashable",
    }


def test_hashable_custom_object_cells_are_defensively_copied() -> None:
    source = np.empty(2, dtype=object)
    source[0] = _StableHashableBox(["original"])
    source[1] = "stable"
    vector = _make_vector(source)

    source[0].value.append("source mutation")
    exported = vector.values
    exported[0].value.append("snapshot mutation")

    assert vector.values[0].value == ["original"]


@pytest.mark.parametrize(("factory", "shape", "_label_field"), _CONTAINER_CASES)
def test_comparison_and_hash_policy_is_deliberate(
    factory: Any,
    shape: tuple[int, ...],
    _label_field: str,
) -> None:
    left = factory(np.zeros(shape))
    right = factory(np.zeros(shape))

    assert left == left
    assert left != right
    with pytest.raises(TypeError):
        hash(left)


@pytest.mark.parametrize(("factory", "_shape", "_label_field"), _CONTAINER_CASES)
def test_ragged_values_raise_structured_validation_error(
    factory: Any,
    _shape: tuple[int, ...],
    _label_field: str,
) -> None:
    with pytest.raises(DataValidationError, match="could not be converted"):
        factory([[1.0], [2.0, 3.0]])


def test_non_reflexive_label_is_rejected_with_axis_context() -> None:
    with pytest.raises(DataValidationError, match="labels must be reflexive") as captured:
        LabeledVector(np.array([1.0]), (float("nan"),), "instrument")

    assert captured.value.context == {"axis": "instrument", "position": 0}
