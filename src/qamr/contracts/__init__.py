"""Stable public research contracts."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qamr.contracts.arrays import LabeledMatrix, LabeledPanel, LabeledVector
    from qamr.contracts.config import (
        PortfolioConstraints,
        ResearchConfig,
        TransactionCostConfig,
    )
    from qamr.contracts.dataset import (
        DatasetMetadata,
        InputProvenance,
        MissingDataPolicy,
        ResearchDataset,
        ReturnConvention,
    )
    from qamr.contracts.interfaces import DataAdapter, RiskEstimator
    from qamr.contracts.results import DiagnosticSeverity, NumericalDiagnostic

__all__ = [
    "DataAdapter",
    "DatasetMetadata",
    "DiagnosticSeverity",
    "InputProvenance",
    "LabeledMatrix",
    "LabeledPanel",
    "LabeledVector",
    "MissingDataPolicy",
    "NumericalDiagnostic",
    "PortfolioConstraints",
    "ResearchConfig",
    "ResearchDataset",
    "ReturnConvention",
    "RiskEstimator",
    "TransactionCostConfig",
]

_EXPORTS = {
    "DataAdapter": ("qamr.contracts.interfaces", "DataAdapter"),
    "DatasetMetadata": ("qamr.contracts.dataset", "DatasetMetadata"),
    "DiagnosticSeverity": ("qamr.contracts.results", "DiagnosticSeverity"),
    "InputProvenance": ("qamr.contracts.dataset", "InputProvenance"),
    "LabeledMatrix": ("qamr.contracts.arrays", "LabeledMatrix"),
    "LabeledPanel": ("qamr.contracts.arrays", "LabeledPanel"),
    "LabeledVector": ("qamr.contracts.arrays", "LabeledVector"),
    "MissingDataPolicy": ("qamr.contracts.dataset", "MissingDataPolicy"),
    "NumericalDiagnostic": ("qamr.contracts.results", "NumericalDiagnostic"),
    "PortfolioConstraints": ("qamr.contracts.config", "PortfolioConstraints"),
    "ResearchConfig": ("qamr.contracts.config", "ResearchConfig"),
    "ResearchDataset": ("qamr.contracts.dataset", "ResearchDataset"),
    "ReturnConvention": ("qamr.contracts.dataset", "ReturnConvention"),
    "RiskEstimator": ("qamr.contracts.interfaces", "RiskEstimator"),
    "TransactionCostConfig": ("qamr.contracts.config", "TransactionCostConfig"),
}


def __getattr__(name: str) -> object:
    """Load a public contract without eagerly coupling contract and risk modules."""
    export = _EXPORTS.get(name)
    if export is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = export
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Advertise public contracts without forcing their modules to load."""
    return sorted(set(globals()).union(__all__))
