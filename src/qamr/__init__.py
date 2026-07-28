"""Generic quant asset-management research components."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("quant-asset-management-research-toolkit")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__"]
