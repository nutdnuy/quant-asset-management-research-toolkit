from collections.abc import Mapping, Sequence

import pytest

from qamr._types import JsonValue
from qamr.errors import (
    DataValidationError,
    InfeasiblePortfolioError,
    InsufficientHistoryError,
    LabelAlignmentError,
    LookaheadRiskError,
    NumericalStabilityError,
    QAMRError,
    ReconciliationError,
)


def test_structured_error_serializes_without_input_payloads() -> None:
    error = LabelAlignmentError(
        "instrument labels differ",
        context={"left_count": 2, "right_count": 3, "axis": "instrument"},
    )

    assert error.code == "label_alignment"
    assert str(error) == "instrument labels differ"
    assert error.as_dict() == {
        "error": "label_alignment",
        "message": "instrument labels differ",
        "context": {"left_count": 2, "right_count": 3, "axis": "instrument"},
    }


def test_error_context_is_copied_and_read_only() -> None:
    supplied = {"field": "returns"}
    error = DataValidationError("invalid field", context=supplied)
    supplied["field"] = "mutated"

    assert error.context["field"] == "returns"
    with pytest.raises(TypeError):
        error.context["field"] = "cannot mutate"  # type: ignore[index]


def test_nested_context_is_isolated_from_supplied_containers() -> None:
    labels: list[JsonValue] = ["A", "B"]
    metadata: dict[str, JsonValue] = {"source": "input"}
    details: dict[str, JsonValue] = {"labels": labels, "metadata": metadata}
    error = DataValidationError("invalid field", context={"details": details})

    labels.append("C")
    metadata["source"] = "mutated"

    assert error.as_dict()["context"] == {
        "details": {
            "labels": ["A", "B"],
            "metadata": {"source": "input"},
        }
    }


def test_nested_error_context_is_read_only() -> None:
    error = DataValidationError(
        "invalid field",
        context={"details": {"labels": ["A", "B"], "metadata": {"source": "input"}}},
    )

    details = error.context["details"]
    assert isinstance(details, Mapping)
    labels = details["labels"]
    metadata = details["metadata"]
    assert isinstance(labels, Sequence)
    assert not isinstance(labels, str)
    assert isinstance(metadata, Mapping)

    with pytest.raises(TypeError):
        metadata["source"] = "mutated"  # type: ignore[index]
    with pytest.raises(TypeError):
        labels[0] = "mutated"  # type: ignore[index]


def test_as_dict_returns_fresh_nested_containers() -> None:
    error = DataValidationError(
        "invalid field",
        context={"details": {"labels": ["A", "B"], "metadata": {"source": "input"}}},
    )
    first = error.as_dict()
    first_context = first["context"]
    assert isinstance(first_context, dict)
    first_details = first_context["details"]
    assert isinstance(first_details, dict)
    first_labels = first_details["labels"]
    first_metadata = first_details["metadata"]
    assert isinstance(first_labels, list)
    assert isinstance(first_metadata, dict)

    first_labels.append("C")
    first_metadata["source"] = "mutated"

    assert error.as_dict()["context"] == {
        "details": {
            "labels": ["A", "B"],
            "metadata": {"source": "input"},
        }
    }


@pytest.mark.parametrize(
    ("error_type", "code"),
    [
        (QAMRError, "qamr_error"),
        (DataValidationError, "data_validation"),
        (LabelAlignmentError, "label_alignment"),
        (InsufficientHistoryError, "insufficient_history"),
        (LookaheadRiskError, "lookahead_risk"),
        (NumericalStabilityError, "numerical_stability"),
        (InfeasiblePortfolioError, "infeasible_portfolio"),
        (ReconciliationError, "reconciliation"),
    ],
)
def test_error_code_is_stable(
    error_type: type[QAMRError],
    code: str,
) -> None:
    assert error_type("message").code == code
