from importlib.metadata import version

import qamr


def test_package_version_is_exposed() -> None:
    assert qamr.__version__ == version("quant-asset-management-research-toolkit")
    assert qamr.__version__ == "0.1.0"
