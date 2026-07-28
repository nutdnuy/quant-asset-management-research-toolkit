from typing import assert_type, get_type_hints

import numpy as np

from qamr._types import JsonValue
from qamr.contracts.arrays import LabeledMatrix, LabeledVector
from qamr.contracts.dataset import DatasetMetadata, ResearchDataset, ReturnConvention
from qamr.contracts.interfaces import DataAdapter, RiskEstimator
from qamr.contracts.results import DiagnosticSeverity, NumericalDiagnostic
from qamr.risk.estimates import CovarianceEstimate


class DictAdapter:
    def adapt(self, source: dict[str, object]) -> ResearchDataset:
        del source
        return ResearchDataset(
            returns=LabeledMatrix(
                np.array([[0.0]]),
                ("t0",),
                ("a",),
                "time",
                "instrument",
            ),
            metadata=DatasetMetadata(
                frequency="event",
                timezone=None,
                return_convention=ReturnConvention.SIMPLE,
            ),
        )


class MissingAdapterMethod:
    pass


class ConformingEstimator:
    def estimate(self, returns: LabeledMatrix) -> CovarianceEstimate:
        labels = returns.column_labels
        size = len(labels)
        return CovarianceEstimate(
            covariance=LabeledMatrix(
                np.eye(size),
                labels,
                labels,
                "instrument",
                "instrument",
            ),
            correlation=LabeledMatrix(
                np.eye(size),
                labels,
                labels,
                "instrument",
                "instrument",
            ),
            volatility=LabeledVector(
                np.ones(size),
                labels,
                "instrument",
            ),
            observation_count=1,
        )


class MissingEstimatorMethod:
    pass


class NonCallableAdapterAttribute:
    adapt = 1


class NonCallableEstimatorAttribute:
    estimate = 1


def accepts_adapter(adapter: DataAdapter[dict[str, object]]) -> None:
    assert_type(adapter.adapt({}), ResearchDataset)


def accepts_estimator(estimator: RiskEstimator) -> None:
    returns = LabeledMatrix(np.array([[0.0]]), ("t0",), ("a",), "time", "instrument")
    assert_type(estimator.estimate(returns), CovarianceEstimate)


def accepts_diagnostic_context(context: dict[str, JsonValue]) -> None:
    diagnostic = NumericalDiagnostic(
        code="static",
        severity=DiagnosticSeverity.INFO,
        message="static constructor input",
        context=context,
    )
    assert_type(diagnostic, NumericalDiagnostic)


def test_structural_data_adapter_protocol() -> None:
    adapter = DictAdapter()

    accepts_adapter(adapter)

    assert isinstance(adapter, DataAdapter)
    assert not isinstance(MissingAdapterMethod(), DataAdapter)


def test_structural_risk_estimator_protocol() -> None:
    estimator = ConformingEstimator()

    accepts_estimator(estimator)

    assert isinstance(estimator, RiskEstimator)
    assert not isinstance(MissingEstimatorMethod(), RiskEstimator)


def test_risk_estimator_annotations_resolve_at_runtime() -> None:
    hints = get_type_hints(RiskEstimator.estimate)

    assert hints["return"] is CovarianceEstimate
    assert hints["returns"] is LabeledMatrix


def test_runtime_protocol_checks_only_require_named_attributes() -> None:
    assert isinstance(NonCallableAdapterAttribute(), DataAdapter)
    assert isinstance(NonCallableEstimatorAttribute(), RiskEstimator)
    assert "does not validate callability or signatures" in (DataAdapter.__doc__ or "")
    assert "does not validate callability or signatures" in (RiskEstimator.__doc__ or "")
