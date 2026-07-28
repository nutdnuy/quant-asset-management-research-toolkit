"""Stable public portfolio-allocation API."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qamr.allocation.hierarchical import (
        condensed_correlation_distance,
        herc_weights,
        hrp_weights,
    )
    from qamr.allocation.risk import portfolio_volatility, risk_contributions
    from qamr.allocation.weights import equal_weights, inverse_volatility_weights

__all__ = [
    "condensed_correlation_distance",
    "equal_weights",
    "herc_weights",
    "hrp_weights",
    "inverse_volatility_weights",
    "portfolio_volatility",
    "risk_contributions",
]

_EXPORTS = {
    "condensed_correlation_distance": (
        "qamr.allocation.hierarchical",
        "condensed_correlation_distance",
    ),
    "equal_weights": ("qamr.allocation.weights", "equal_weights"),
    "herc_weights": ("qamr.allocation.hierarchical", "herc_weights"),
    "hrp_weights": ("qamr.allocation.hierarchical", "hrp_weights"),
    "inverse_volatility_weights": (
        "qamr.allocation.weights",
        "inverse_volatility_weights",
    ),
    "portfolio_volatility": ("qamr.allocation.risk", "portfolio_volatility"),
    "risk_contributions": ("qamr.allocation.risk", "risk_contributions"),
}


def __getattr__(name: str) -> object:
    """Load a public allocator without eagerly importing implementation modules."""
    export = _EXPORTS.get(name)
    if export is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = export
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Advertise public allocation members without loading them."""
    return sorted(set(globals()).union(__all__))
