from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import qamr.contracts as contracts
import qamr.risk as risk
from qamr.contracts import (
    DataAdapter,
    DatasetMetadata,
    DiagnosticSeverity,
    InputProvenance,
    LabeledMatrix,
    LabeledPanel,
    LabeledVector,
    MissingDataPolicy,
    NumericalDiagnostic,
    PortfolioConstraints,
    ResearchConfig,
    ResearchDataset,
    ReturnConvention,
    RiskEstimator,
    TransactionCostConfig,
)
from qamr.risk import (
    CovarianceEstimate,
    EWMACovariance,
    PSDPolicy,
    SampleCovariance,
    ShrinkageCovariance,
    ShrinkageTarget,
    SpectralDenoisedCovariance,
    apply_psd_policy,
    correlation_to_covariance,
    covariance_to_correlation,
)

CONTRACT_EXPORTS = {
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
}
RISK_EXPORTS = {
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
}


def test_public_contract_and_risk_names_are_importable() -> None:
    expected = {
        DataAdapter,
        DatasetMetadata,
        DiagnosticSeverity,
        InputProvenance,
        LabeledMatrix,
        LabeledPanel,
        LabeledVector,
        MissingDataPolicy,
        NumericalDiagnostic,
        PortfolioConstraints,
        ResearchConfig,
        ResearchDataset,
        ReturnConvention,
        RiskEstimator,
        TransactionCostConfig,
        CovarianceEstimate,
        EWMACovariance,
        PSDPolicy,
        SampleCovariance,
        ShrinkageCovariance,
        ShrinkageTarget,
        SpectralDenoisedCovariance,
        apply_psd_policy,
        correlation_to_covariance,
        covariance_to_correlation,
    }

    assert len(expected) == 25


def test_public_all_lists_are_exact_and_have_no_duplicates() -> None:
    assert set(contracts.__all__) == CONTRACT_EXPORTS
    assert len(contracts.__all__) == len(CONTRACT_EXPORTS)
    assert set(risk.__all__) == RISK_EXPORTS
    assert len(risk.__all__) == len(RISK_EXPORTS)


def test_importing_contracts_does_not_import_optional_pandas() -> None:
    source_root = Path(__file__).parents[1] / "src"
    code = """
import builtins
import sys

sys.path.insert(0, sys.argv[1])
real_import = builtins.__import__

def block_pandas(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "pandas" or name.startswith("pandas."):
        raise AssertionError(f"unexpected optional pandas import: {name}")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = block_pandas
import qamr.contracts
assert "pandas" not in sys.modules
assert not any(name.startswith("pandas.") for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code, str(source_root)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_contract_annotations_resolve_before_risk_package_is_imported() -> None:
    source_root = Path(__file__).parents[1] / "src"
    code = """
import builtins
import sys
import typing

sys.path.insert(0, sys.argv[1])
real_import = builtins.__import__

def block_pandas(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "pandas" or name.startswith("pandas."):
        raise AssertionError(f"unexpected optional pandas import: {name}")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = block_pandas
import qamr.contracts as contracts
hints = typing.get_type_hints(contracts.RiskEstimator.estimate)
from qamr.risk.estimates import CovarianceEstimate
assert hints["return"] is CovarianceEstimate
assert "pandas" not in sys.modules
assert not any(name.startswith("pandas.") for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code, str(source_root)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_public_module_directories_advertise_exports_without_loading_them() -> None:
    source_root = Path(__file__).parents[1] / "src"
    code = """
import sys

sys.path.insert(0, sys.argv[1])
import qamr.contracts as contracts
import qamr.risk as risk

assert "qamr.contracts.interfaces" not in sys.modules
assert "qamr.risk.estimates" not in sys.modules
contract_names = dir(contracts)
risk_names = dir(risk)
assert contract_names == sorted(set(contract_names))
assert risk_names == sorted(set(risk_names))
assert set(contracts.__all__).issubset(contract_names)
assert set(risk.__all__).issubset(risk_names)
assert dir(contracts) == contract_names
assert dir(risk) == risk_names
assert "qamr.contracts.interfaces" not in sys.modules
assert "qamr.risk.estimates" not in sys.modules
assert "pandas" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code, str(source_root)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_inspect_getmembers_resolves_public_exports_without_pandas() -> None:
    source_root = Path(__file__).parents[1] / "src"
    code = """
import builtins
import inspect
import sys

sys.path.insert(0, sys.argv[1])
real_import = builtins.__import__

def block_pandas(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "pandas" or name.startswith("pandas."):
        raise AssertionError(f"unexpected optional pandas import: {name}")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = block_pandas
import qamr.contracts as contracts
import qamr.risk as risk

contract_members = dict(inspect.getmembers(contracts))
risk_members = dict(inspect.getmembers(risk))
assert set(contracts.__all__).issubset(contract_members)
assert set(risk.__all__).issubset(risk_members)
assert contract_members["RiskEstimator"].__name__ == "RiskEstimator"
assert risk_members["CovarianceEstimate"].__name__ == "CovarianceEstimate"
assert "pandas" not in sys.modules
assert not any(name.startswith("pandas.") for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code, str(source_root)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
