"""Small structural interfaces used by built-in workflows."""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from qamr.contracts.arrays import LabeledMatrix
from qamr.contracts.dataset import ResearchDataset
from qamr.risk.estimates import CovarianceEstimate

SourceT = TypeVar("SourceT", contravariant=True)


@runtime_checkable
class DataAdapter(Protocol[SourceT]):
    """Convert a source into a dataset.

    A runtime check does not validate callability or signatures; it only
    verifies attribute presence.
    """

    def adapt(self, source: SourceT) -> ResearchDataset:
        """Convert a source without changing labels or missing values."""
        ...


@runtime_checkable
class RiskEstimator(Protocol):
    """Estimate labelled risk from returns.

    A runtime check does not validate callability or signatures; it only
    verifies attribute presence.
    """

    def estimate(self, returns: LabeledMatrix) -> CovarianceEstimate:
        """Estimate labelled risk from time-by-instrument returns."""
        ...
