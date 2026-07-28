from __future__ import annotations

import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

FORBIDDEN_ARCHIVE_PARTS = {
    ".coverage",
    ".env",
    ".hypothesis",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".worktrees",
    "__pycache__",
    "artifacts",
    "build",
    "dist",
    "results",
    "secrets",
    "tests",
}
FORBIDDEN_TEXT = (
    b"/Users/" + b"nuthdanai",
    b".worktrees/" + b"qamr-v1",
)


def test_sdist_contains_only_portable_release_sources(tmp_path: Path) -> None:
    repository = Path(__file__).parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--sdist",
            "--outdir",
            str(tmp_path),
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

    archives = list(tmp_path.glob("*.tar.gz"))
    assert len(archives) == 1
    with tarfile.open(archives[0], mode="r:gz") as archive:
        members = archive.getmembers()
        relative_names: set[PurePosixPath] = set()
        for member in members:
            path = PurePosixPath(member.name)
            assert not path.is_absolute()
            assert ".." not in path.parts
            relative = PurePosixPath(*path.parts[1:])
            relative_names.add(relative)
            assert FORBIDDEN_ARCHIVE_PARTS.isdisjoint(relative.parts)
            if member.isfile():
                extracted = archive.extractfile(member)
                assert extracted is not None
                contents = extracted.read()
                assert not any(marker in contents for marker in FORBIDDEN_TEXT)

    assert PurePosixPath("README.md") in relative_names
    assert PurePosixPath("pyproject.toml") in relative_names
    assert PurePosixPath("src/qamr/py.typed") in relative_names
    assert any(name.parts[:3] == ("src", "qamr", "contracts") for name in relative_names)
    assert any(name.parts[:3] == ("src", "qamr", "risk") for name in relative_names)
