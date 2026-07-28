"""Stable public risk-estimation API."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qamr.risk.estimates import CovarianceEstimate
    from qamr.risk.ewma import EWMACovariance
    from qamr.risk.matrices import (
        PSDPolicy,
        apply_psd_policy,
        correlation_to_covariance,
        covariance_to_correlation,
    )
    from qamr.risk.sample import SampleCovariance
    from qamr.risk.shrinkage import ShrinkageCovariance, ShrinkageTarget
    from qamr.risk.spectral import SpectralDenoisedCovariance

__all__ = [
    "CovarianceEstimate",
    "EWMACovariance",
    "PSDPolicy",
    "SampleCovariance",
    "ShrinkageCovariance",
    "ShrinkageTarget",
    "SpectralDenoisedCovariance",
    "apply_psd_policy",
    "correlation_to_covariance",
    "covariance_to_correlation",
]

_EXPORTS = {
    "CovarianceEstimate": ("qamr.risk.estimates", "CovarianceEstimate"),
    "EWMACovariance": ("qamr.risk.ewma", "EWMACovariance"),
    "PSDPolicy": ("qamr.risk.matrices", "PSDPolicy"),
    "SampleCovariance": ("qamr.risk.sample", "SampleCovariance"),
    "ShrinkageCovariance": ("qamr.risk.shrinkage", "ShrinkageCovariance"),
    "ShrinkageTarget": ("qamr.risk.shrinkage", "ShrinkageTarget"),
    "SpectralDenoisedCovariance": (
        "qamr.risk.spectral",
        "SpectralDenoisedCovariance",
    ),
    "apply_psd_policy": ("qamr.risk.matrices", "apply_psd_policy"),
    "correlation_to_covariance": (
        "qamr.risk.matrices",
        "correlation_to_covariance",
    ),
    "covariance_to_correlation": (
        "qamr.risk.matrices",
        "covariance_to_correlation",
    ),
}


def __getattr__(name: str) -> object:
    """Load a public estimator without creating package-initialization cycles."""
    export = _EXPORTS.get(name)
    if export is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = export
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Advertise public estimators without forcing their modules to load."""
    return sorted(set(globals()).union(__all__))
