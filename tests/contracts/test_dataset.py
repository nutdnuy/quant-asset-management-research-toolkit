import numpy as np
import pytest

from qamr.contracts.arrays import LabeledMatrix, LabeledPanel, LabeledVector
from qamr.contracts.dataset import (
    DatasetMetadata,
    InputProvenance,
    MissingDataPolicy,
    ResearchDataset,
    ReturnConvention,
)
from qamr.errors import DataValidationError, LabelAlignmentError


def matrix(
    values: object,
    columns: tuple[str, ...] = ("a", "b"),
) -> LabeledMatrix:
    return LabeledMatrix(values, ("t0", "t1"), columns, "time", "instrument")


def metadata(
    convention: ReturnConvention | None = ReturnConvention.SIMPLE,
) -> DatasetMetadata:
    return DatasetMetadata(
        frequency="business-day",
        timezone="UTC",
        calendar="generic-weekday",
        return_convention=convention,
    )


def test_returns_dataset_preserves_convention_missing_values_and_provenance() -> None:
    returns = matrix([[0.01, np.nan], [0.02, -0.01]])
    provenance = InputProvenance(
        source="fixture",
        content_fingerprint=f"sha256:{'a' * 64}",
    )

    dataset = ResearchDataset(
        metadata=metadata(ReturnConvention.LOG),
        returns=returns,
        provenance=provenance,
    )

    assert dataset.metadata.return_convention is ReturnConvention.LOG
    assert np.isnan(dataset.returns.values[0, 1])  # type: ignore[union-attr]
    assert dataset.provenance.content_fingerprint == f"sha256:{'a' * 64}"


def test_dataset_requires_returns_or_prices() -> None:
    with pytest.raises(DataValidationError, match="returns or prices") as captured:
        ResearchDataset(metadata=metadata())

    assert captured.value.as_dict()["context"] == {
        "required_fields": ["returns", "prices"],
    }


def test_return_data_requires_declared_return_convention() -> None:
    no_convention = metadata(None)

    with pytest.raises(DataValidationError, match="return convention") as returns_error:
        ResearchDataset(metadata=no_convention, returns=matrix(np.zeros((2, 2))))

    with pytest.raises(DataValidationError, match="return convention") as future_error:
        ResearchDataset(
            metadata=no_convention,
            prices=matrix(np.ones((2, 2))),
            future_returns=matrix(np.zeros((2, 2))),
        )

    assert returns_error.value.context == {"field": "returns"}
    assert future_error.value.context == {"field": "future_returns"}


def test_returns_and_prices_require_exactly_identical_axes() -> None:
    returns = matrix(np.zeros((2, 2)))
    reordered_prices = matrix(np.ones((2, 2)), columns=("b", "a"))

    with pytest.raises(LabelAlignmentError, match="identical axes") as captured:
        ResearchDataset(
            metadata=metadata(),
            returns=returns,
            prices=reordered_prices,
        )

    assert captured.value.context == {"left": "returns", "right": "prices"}


def test_universe_requires_primary_axes_and_boolean_dtype() -> None:
    returns = matrix(np.zeros((2, 2)))
    reordered = matrix(
        np.array([[True, True], [True, False]]),
        columns=("b", "a"),
    )
    floats = matrix(np.ones((2, 2)))

    with pytest.raises(LabelAlignmentError, match="primary axes") as alignment_error:
        ResearchDataset(
            metadata=metadata(),
            returns=returns,
            universe=reordered,
        )

    with pytest.raises(DataValidationError, match="boolean dtype") as dtype_error:
        ResearchDataset(
            metadata=metadata(),
            returns=returns,
            universe=floats,
        )

    assert alignment_error.value.context == {
        "field": "universe",
        "primary": "returns",
    }
    assert dtype_error.value.context == {
        "field": "universe",
        "dtype": "float64",
    }


def test_signal_panel_may_use_subset_of_primary_instruments() -> None:
    signals = LabeledPanel(
        np.array([[[1.0]]]),
        time_labels=("t0",),
        instrument_labels=("a",),
        field_labels=("value",),
    )

    dataset = ResearchDataset(
        metadata=metadata(),
        returns=matrix(np.zeros((2, 2))),
        signals=signals,
    )

    assert dataset.signals is signals


def test_signal_panel_rejects_reserved_future_return_field() -> None:
    signals = LabeledPanel(
        np.array([[[0.01]]]),
        time_labels=("t0",),
        instrument_labels=("a",),
        field_labels=("future_returns",),
    )

    with pytest.raises(DataValidationError, match="reserved") as captured:
        ResearchDataset(
            metadata=metadata(),
            returns=matrix(np.zeros((2, 2))),
            signals=signals,
        )

    assert captured.value.as_dict()["context"] == {
        "field": "signals",
        "offending_fields": ["future_returns"],
    }


def test_dataset_enum_values_are_stable_strings() -> None:
    assert ReturnConvention.SIMPLE.value == "simple"
    assert ReturnConvention.LOG.value == "log"
    assert MissingDataPolicy.RAISE.value == "raise"
    assert MissingDataPolicy.DROP_OBSERVATION.value == "drop_observation"


@pytest.mark.parametrize("frequency", ["", " \t "])
def test_metadata_rejects_blank_frequency(frequency: str) -> None:
    with pytest.raises(DataValidationError, match="frequency must not be blank") as captured:
        DatasetMetadata(frequency=frequency, timezone="UTC")

    assert captured.value.context == {"field": "frequency"}


def test_input_provenance_validates_source_and_sha256_algorithm() -> None:
    with pytest.raises(DataValidationError, match="source must not be blank") as source_error:
        InputProvenance(source=" \t ")

    with pytest.raises(DataValidationError, match="must use sha256") as fingerprint_error:
        InputProvenance(
            source="fixture",
            content_fingerprint="md5:0123456789abcdef",
        )

    valid = InputProvenance(
        source="fixture",
        content_fingerprint=f"sha256:{'a' * 64}",
    )

    assert source_error.value.context == {"field": "source"}
    assert fingerprint_error.value.context == {
        "field": "content_fingerprint",
        "algorithm": "md5",
        "reason": "algorithm",
    }
    assert valid.content_fingerprint == f"sha256:{'a' * 64}"


@pytest.mark.parametrize("field_name", ["returns", "prices"])
def test_primary_numeric_matrix_rejects_nonnumeric_dtype(field_name: str) -> None:
    nonnumeric = matrix([["x", "y"], ["z", "w"]])

    with pytest.raises(DataValidationError, match="numeric dtype") as captured:
        ResearchDataset(metadata=metadata(), **{field_name: nonnumeric})

    assert captured.value.context == {
        "field": field_name,
        "dtype": str(nonnumeric.values.dtype),
    }


@pytest.mark.parametrize("infinite_value", [np.inf, -np.inf], ids=["positive", "negative"])
def test_numeric_matrix_rejects_infinity(infinite_value: float) -> None:
    returns = matrix([[0.01, infinite_value], [0.02, np.nan]])

    with pytest.raises(DataValidationError, match="must not contain infinity") as captured:
        ResearchDataset(metadata=metadata(), returns=returns)

    assert captured.value.context == {
        "field": "returns",
        "dtype": "float64",
    }


def test_returns_and_prices_reject_different_row_axes() -> None:
    returns = matrix(np.zeros((2, 2)))
    prices = LabeledMatrix(
        np.ones((2, 2)),
        row_labels=("t1", "t0"),
        column_labels=("a", "b"),
        row_name="time",
        column_name="instrument",
    )

    with pytest.raises(LabelAlignmentError, match="identical axes") as captured:
        ResearchDataset(
            metadata=metadata(),
            returns=returns,
            prices=prices,
        )

    assert captured.value.context == {"left": "returns", "right": "prices"}


def test_future_returns_require_exact_primary_axes() -> None:
    with pytest.raises(LabelAlignmentError, match="primary axes") as captured:
        ResearchDataset(
            metadata=metadata(),
            returns=matrix(np.zeros((2, 2))),
            future_returns=matrix(np.zeros((2, 2)), columns=("b", "a")),
        )

    assert captured.value.context == {
        "field": "future_returns",
        "primary": "returns",
    }


def test_benchmark_weights_require_exact_primary_axes() -> None:
    with pytest.raises(LabelAlignmentError, match="primary axes") as captured:
        ResearchDataset(
            metadata=metadata(),
            prices=matrix(np.ones((2, 2))),
            benchmark_weights=matrix(np.full((2, 2), 0.5), columns=("b", "a")),
        )

    assert captured.value.context == {
        "field": "benchmark_weights",
        "primary": "prices",
    }


@pytest.mark.parametrize("panel_field", ["signals", "groups", "exposures"])
def test_panel_instruments_reject_unknown_primary_instrument(panel_field: str) -> None:
    panel = LabeledPanel(
        np.array([[[1.0]]]),
        time_labels=("t0",),
        instrument_labels=("unknown",),
        field_labels=("value",),
    )

    with pytest.raises(LabelAlignmentError, match="subset") as captured:
        ResearchDataset(
            metadata=metadata(),
            returns=matrix(np.zeros((2, 2))),
            **{panel_field: panel},
        )

    assert captured.value.as_dict()["context"] == {
        "field": panel_field,
        "axis": "instrument",
        "reason": "unknown_labels",
        "unknown_count": 1,
        "unknown_sample": ["unknown"],
    }


@pytest.mark.parametrize(
    "reserved_field",
    [
        "future_return",
        "future_returns",
        "forward_return",
        "forward_returns",
        "FuTuRe_ReTuRn",
    ],
)
def test_reserved_signal_fields_are_case_insensitive(reserved_field: str) -> None:
    signals = LabeledPanel(
        np.array([[[0.01]]]),
        time_labels=("t0",),
        instrument_labels=("a",),
        field_labels=(reserved_field,),
    )

    with pytest.raises(DataValidationError, match="reserved") as captured:
        ResearchDataset(
            metadata=metadata(),
            returns=matrix(np.zeros((2, 2))),
            signals=signals,
        )

    assert captured.value.as_dict()["context"] == {
        "field": "signals",
        "offending_fields": [reserved_field.casefold()],
    }


def test_metadata_accepts_none_timezone() -> None:
    dataset_metadata = DatasetMetadata(
        frequency="business-day",
        timezone=None,
    )

    assert dataset_metadata.timezone is None


@pytest.mark.parametrize(
    ("field_name", "value"),
    [("frequency", None), ("frequency", 5), ("source", None), ("source", 5)],
)
def test_metadata_and_provenance_reject_non_string_text(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(DataValidationError, match="must be a string") as captured:
        if field_name == "frequency":
            DatasetMetadata(
                frequency=value,  # type: ignore[arg-type]
                timezone="UTC",
            )
        else:
            InputProvenance(source=value)  # type: ignore[arg-type]

    assert captured.value.context == {
        "field": field_name,
        "dtype": type(value).__name__,
    }


@pytest.mark.parametrize(
    "fingerprint",
    [
        "sha256:",
        "sha256:not-hex",
        f"sha256:{'a' * 63}",
        f"sha256:{'a' * 65}",
    ],
    ids=["empty", "nonhex", "short", "oversized"],
)
def test_provenance_rejects_invalid_sha256_digest_format(fingerprint: str) -> None:
    with pytest.raises(DataValidationError, match="digest") as captured:
        InputProvenance(source="fixture", content_fingerprint=fingerprint)

    assert captured.value.context == {
        "field": "content_fingerprint",
        "algorithm": "sha256",
        "reason": "digest_format",
    }


def test_provenance_accepts_exact_sha256_fingerprint() -> None:
    fingerprint = f"sha256:{'aB' * 32}"

    provenance = InputProvenance(
        source="fixture",
        content_fingerprint=fingerprint,
    )

    assert provenance.content_fingerprint == fingerprint


def test_provenance_rejects_non_exact_sha256_prefix_case() -> None:
    with pytest.raises(DataValidationError, match="must use sha256") as captured:
        InputProvenance(
            source="fixture",
            content_fingerprint=f"SHA256:{'a' * 64}",
        )

    assert captured.value.context == {
        "field": "content_fingerprint",
        "algorithm": "SHA256",
        "reason": "algorithm",
    }


@pytest.mark.parametrize(
    ("values", "expected_dtype"),
    [
        (np.ones((2, 2), dtype=bool), "bool"),
        (np.ones((2, 2), dtype=np.complex128), "complex128"),
    ],
    ids=["boolean", "complex"],
)
def test_numeric_matrices_require_real_non_boolean_values(
    values: np.ndarray[tuple[int, int], np.dtype[np.generic]],
    expected_dtype: str,
) -> None:
    with pytest.raises(DataValidationError, match="real numeric dtype") as captured:
        ResearchDataset(metadata=metadata(), returns=matrix(values))

    assert captured.value.context == {
        "field": "returns",
        "dtype": expected_dtype,
    }


@pytest.mark.parametrize("panel_field", ["signals", "exposures"])
@pytest.mark.parametrize(
    ("values", "expected_dtype"),
    [
        (np.array([[["value"]]]), "<U5"),
        (np.ones((1, 1, 1), dtype=np.complex128), "complex128"),
        (np.ones((1, 1, 1), dtype=bool), "bool"),
    ],
    ids=["string", "complex", "boolean"],
)
def test_numerical_panels_require_real_non_boolean_values(
    panel_field: str,
    values: np.ndarray[tuple[int, int, int], np.dtype[np.generic]],
    expected_dtype: str,
) -> None:
    panel = LabeledPanel(
        values,
        time_labels=("t0",),
        instrument_labels=("a",),
        field_labels=("value",),
    )

    with pytest.raises(DataValidationError, match="real numeric dtype") as captured:
        ResearchDataset(
            metadata=metadata(),
            returns=matrix(np.zeros((2, 2))),
            **{panel_field: panel},
        )

    assert captured.value.context == {
        "field": panel_field,
        "dtype": expected_dtype,
    }


@pytest.mark.parametrize("panel_field", ["signals", "exposures"])
def test_numerical_panels_reject_infinity(panel_field: str) -> None:
    panel = LabeledPanel(
        np.array([[[np.inf]]]),
        time_labels=("t0",),
        instrument_labels=("a",),
        field_labels=("value",),
    )

    with pytest.raises(DataValidationError, match="must not contain infinity") as captured:
        ResearchDataset(
            metadata=metadata(),
            returns=matrix(np.zeros((2, 2))),
            **{panel_field: panel},
        )

    assert captured.value.context == {
        "field": panel_field,
        "dtype": "float64",
    }


def test_groups_may_contain_categorical_strings() -> None:
    groups = LabeledPanel(
        np.array([[["sector"]]]),
        time_labels=("t0",),
        instrument_labels=("a",),
        field_labels=("classification",),
    )

    dataset = ResearchDataset(
        metadata=metadata(),
        returns=matrix(np.zeros((2, 2))),
        groups=groups,
    )

    assert dataset.groups is groups


def test_matrix_alignment_includes_semantic_axis_names() -> None:
    prices = LabeledMatrix(
        np.ones((2, 2)),
        row_labels=("t0", "t1"),
        column_labels=("a", "b"),
        row_name="date",
        column_name="instrument",
    )

    with pytest.raises(LabelAlignmentError, match="identical axes") as captured:
        ResearchDataset(
            metadata=metadata(),
            returns=matrix(np.zeros((2, 2))),
            prices=prices,
        )

    assert captured.value.context == {"left": "returns", "right": "prices"}


def test_benchmark_weight_alignment_includes_semantic_axis_names() -> None:
    weights = LabeledMatrix(
        np.full((2, 2), 0.5),
        row_labels=("t0", "t1"),
        column_labels=("a", "b"),
        row_name="time",
        column_name="asset",
    )

    with pytest.raises(LabelAlignmentError, match="primary axes") as captured:
        ResearchDataset(
            metadata=metadata(),
            prices=matrix(np.ones((2, 2))),
            benchmark_weights=weights,
        )

    assert captured.value.context == {
        "field": "benchmark_weights",
        "primary": "prices",
    }


def benchmark_vector(
    values: object,
    labels: tuple[str, ...] = ("t0", "t1"),
    axis_name: str = "time",
) -> LabeledVector:
    return LabeledVector(values, labels, axis_name)


def test_benchmark_returns_require_declared_return_convention() -> None:
    with pytest.raises(DataValidationError, match="return convention") as captured:
        ResearchDataset(
            metadata=metadata(None),
            prices=matrix(np.ones((2, 2))),
            benchmark_returns=benchmark_vector([0.01, 0.02]),
        )

    assert captured.value.context == {"field": "benchmark_returns"}


@pytest.mark.parametrize(
    ("values", "expected_dtype"),
    [
        (["x", "y"], "<U1"),
        (np.ones(2, dtype=np.complex128), "complex128"),
        (np.ones(2, dtype=bool), "bool"),
    ],
    ids=["string", "complex", "boolean"],
)
def test_benchmark_returns_require_real_non_boolean_values(
    values: object,
    expected_dtype: str,
) -> None:
    with pytest.raises(DataValidationError, match="real numeric dtype") as captured:
        ResearchDataset(
            metadata=metadata(),
            returns=matrix(np.zeros((2, 2))),
            benchmark_returns=benchmark_vector(values),
        )

    assert captured.value.context == {
        "field": "benchmark_returns",
        "dtype": expected_dtype,
    }


@pytest.mark.parametrize("infinite_value", [np.inf, -np.inf])
def test_benchmark_returns_reject_infinity(infinite_value: float) -> None:
    with pytest.raises(DataValidationError, match="must not contain infinity") as captured:
        ResearchDataset(
            metadata=metadata(),
            returns=matrix(np.zeros((2, 2))),
            benchmark_returns=benchmark_vector([0.01, infinite_value]),
        )

    assert captured.value.context == {
        "field": "benchmark_returns",
        "dtype": "float64",
    }


@pytest.mark.parametrize(
    "labels",
    [("t1", "t0"), ("t0", "unknown")],
    ids=["reordered", "unknown"],
)
def test_benchmark_returns_require_exact_primary_labels(
    labels: tuple[str, ...],
) -> None:
    with pytest.raises(LabelAlignmentError, match="primary time axis") as captured:
        ResearchDataset(
            metadata=metadata(),
            returns=matrix(np.zeros((2, 2))),
            benchmark_returns=benchmark_vector([0.01, 0.02], labels=labels),
        )

    assert captured.value.context == {
        "field": "benchmark_returns",
        "primary": "returns",
        "axis": "time",
        "reason": "labels",
    }


def test_benchmark_returns_require_primary_axis_name() -> None:
    with pytest.raises(LabelAlignmentError, match="primary time axis") as captured:
        ResearchDataset(
            metadata=metadata(),
            returns=matrix(np.zeros((2, 2))),
            benchmark_returns=benchmark_vector(
                [0.01, 0.02],
                axis_name="date",
            ),
        )

    assert captured.value.context == {
        "field": "benchmark_returns",
        "primary": "returns",
        "axis": "time",
        "reason": "axis_name",
    }


def test_valid_benchmark_returns_preserve_nan() -> None:
    benchmark_returns = benchmark_vector([0.01, np.nan])

    dataset = ResearchDataset(
        metadata=metadata(),
        returns=matrix(np.zeros((2, 2))),
        benchmark_returns=benchmark_returns,
    )

    assert dataset.benchmark_returns is benchmark_returns
    assert np.isnan(benchmark_returns.values[1])


@pytest.mark.parametrize("panel_field", ["signals", "groups", "exposures"])
def test_panels_reject_unknown_time_labels(panel_field: str) -> None:
    panel = LabeledPanel(
        np.ones((1, 1, 1)),
        time_labels=("unknown",),
        instrument_labels=("a",),
        field_labels=("value",),
    )

    with pytest.raises(LabelAlignmentError, match="unknown time") as captured:
        ResearchDataset(
            metadata=metadata(),
            returns=matrix(np.zeros((2, 2))),
            **{panel_field: panel},
        )

    assert captured.value.as_dict()["context"] == {
        "field": panel_field,
        "axis": "time",
        "reason": "unknown_labels",
        "unknown_count": 1,
        "unknown_sample": ["unknown"],
    }


@pytest.mark.parametrize("panel_field", ["signals", "groups", "exposures"])
def test_panels_reject_reversed_time_subsets(panel_field: str) -> None:
    values: object = (
        np.array([[["group"]], [["group"]]]) if panel_field == "groups" else np.ones((2, 1, 1))
    )
    panel = LabeledPanel(
        values,
        time_labels=("t1", "t0"),
        instrument_labels=("a",),
        field_labels=("value",),
    )

    with pytest.raises(LabelAlignmentError, match="ordered time") as captured:
        ResearchDataset(
            metadata=metadata(),
            returns=matrix(np.zeros((2, 2))),
            **{panel_field: panel},
        )

    assert captured.value.context == {
        "field": panel_field,
        "axis": "time",
        "reason": "out_of_order",
    }


@pytest.mark.parametrize("panel_field", ["signals", "groups", "exposures"])
def test_panels_reject_reversed_instrument_subsets(panel_field: str) -> None:
    values: object = (
        np.array([[["group"], ["group"]]]) if panel_field == "groups" else np.ones((1, 2, 1))
    )
    panel = LabeledPanel(
        values,
        time_labels=("t0",),
        instrument_labels=("b", "a"),
        field_labels=("value",),
    )

    with pytest.raises(LabelAlignmentError, match="ordered instrument") as captured:
        ResearchDataset(
            metadata=metadata(),
            returns=matrix(np.zeros((2, 2))),
            **{panel_field: panel},
        )

    assert captured.value.context == {
        "field": panel_field,
        "axis": "instrument",
        "reason": "out_of_order",
    }


@pytest.mark.parametrize("panel_field", ["signals", "groups", "exposures"])
def test_panels_allow_sparse_ordered_subsets(panel_field: str) -> None:
    values: object = np.array([[["group"]]]) if panel_field == "groups" else np.array([[[np.nan]]])
    panel = LabeledPanel(
        values,
        time_labels=("t1",),
        instrument_labels=("b",),
        field_labels=("value",),
    )

    dataset = ResearchDataset(
        metadata=metadata(),
        returns=matrix(np.zeros((2, 2))),
        **{panel_field: panel},
    )

    assert getattr(dataset, panel_field) is panel


@pytest.mark.parametrize(
    ("values", "expected_dtype"),
    [
        (
            np.array([[1, 2], [3, 4]], dtype="timedelta64[D]"),
            "timedelta64[D]",
        ),
        (
            np.array(
                [
                    ["2026-01-01", "2026-01-02"],
                    ["2026-01-03", "2026-01-04"],
                ],
                dtype="datetime64[D]",
            ),
            "datetime64[D]",
        ),
    ],
    ids=["timedelta", "datetime"],
)
def test_matrix_numerical_fields_reject_temporal_dtypes(
    values: np.ndarray[tuple[int, int], np.dtype[np.generic]],
    expected_dtype: str,
) -> None:
    with pytest.raises(DataValidationError, match="real numeric dtype") as captured:
        ResearchDataset(metadata=metadata(), prices=matrix(values))

    assert captured.value.context == {
        "field": "prices",
        "dtype": expected_dtype,
    }


def test_panel_numerical_fields_reject_timedelta_dtype() -> None:
    signals = LabeledPanel(
        np.array([[[1]]], dtype="timedelta64[D]"),
        time_labels=("t0",),
        instrument_labels=("a",),
        field_labels=("value",),
    )

    with pytest.raises(DataValidationError, match="real numeric dtype") as captured:
        ResearchDataset(
            metadata=metadata(),
            returns=matrix(np.zeros((2, 2))),
            signals=signals,
        )

    assert captured.value.context == {
        "field": "signals",
        "dtype": "timedelta64[D]",
    }


def test_benchmark_returns_reject_timedelta_dtype() -> None:
    benchmark_returns = benchmark_vector(
        np.array([1, 2], dtype="timedelta64[D]"),
    )

    with pytest.raises(DataValidationError, match="real numeric dtype") as captured:
        ResearchDataset(
            metadata=metadata(),
            returns=matrix(np.zeros((2, 2))),
            benchmark_returns=benchmark_returns,
        )

    assert captured.value.context == {
        "field": "benchmark_returns",
        "dtype": "timedelta64[D]",
    }


@pytest.mark.parametrize("field_name", ["timezone", "currency", "calendar"])
def test_optional_metadata_text_accepts_none(field_name: str) -> None:
    supplied: dict[str, object] = {
        "frequency": "business-day",
        "timezone": "UTC",
        field_name: None,
    }

    dataset_metadata = DatasetMetadata(**supplied)  # type: ignore[arg-type]

    assert getattr(dataset_metadata, field_name) is None


@pytest.mark.parametrize("field_name", ["timezone", "currency", "calendar"])
@pytest.mark.parametrize(
    ("value", "message", "expected_context"),
    [
        (" \t ", "must not be blank", None),
        (5, "must be a string or None", {"dtype": "int"}),
    ],
    ids=["blank", "non-string"],
)
def test_optional_metadata_text_rejects_invalid_runtime_values(
    field_name: str,
    value: object,
    message: str,
    expected_context: dict[str, str] | None,
) -> None:
    supplied: dict[str, object] = {
        "frequency": "business-day",
        "timezone": "UTC",
        field_name: value,
    }

    with pytest.raises(DataValidationError, match=message) as captured:
        DatasetMetadata(**supplied)  # type: ignore[arg-type]

    context = {"field": field_name}
    if expected_context is not None:
        context.update(expected_context)
    assert captured.value.context == context


def test_metadata_rejects_string_return_convention() -> None:
    with pytest.raises(DataValidationError, match="ReturnConvention") as captured:
        DatasetMetadata(
            frequency="business-day",
            timezone="UTC",
            return_convention="simple",  # type: ignore[arg-type]
        )

    assert captured.value.context == {
        "field": "return_convention",
        "dtype": "str",
    }
