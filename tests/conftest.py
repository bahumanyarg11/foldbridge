"""Shared fixtures for foldbridge tests.

Collection is robust to a half-built package: if ``import foldbridge`` fails (e.g. because
``foldbridge/__init__.py`` imports a module such as ``score`` that does not exist yet), we
install a bare namespace stub for the package so that ``foldbridge.structure`` and
``foldbridge.spec`` can still be imported directly.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent / "data"
EXAMPLES = ROOT / "examples"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FOLDBRIDGE_IMPORT_ERROR: Exception | None = None
try:
    import foldbridge  # noqa: F401
except Exception as exc:  # ImportError, or anything raised by a half-written module
    FOLDBRIDGE_IMPORT_ERROR = exc
    stub = types.ModuleType("foldbridge")
    stub.__path__ = [str(ROOT / "foldbridge")]  # type: ignore[attr-defined]
    stub.__file__ = str(ROOT / "foldbridge" / "__init__.py")
    sys.modules["foldbridge"] = stub


def optional_module(name: str):
    """Import ``foldbridge.<name>`` or return None if it is missing / broken."""
    try:
        return importlib.import_module(f"foldbridge.{name}")
    except Exception:
        sys.modules.pop(f"foldbridge.{name}", None)
        return None


ENTRIES = ["1BRS", "2PTC", "1A2K", "4HHB", "1CRN"]


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA


@pytest.fixture(scope="session")
def examples_dir() -> Path:
    return EXAMPLES


def _path(entry: str, ext: str) -> Path:
    p = DATA / f"{entry}.{ext}"
    if not p.exists():
        pytest.skip(f"fixture {p.name} missing (download from https://files.rcsb.org/download/{p.name})")
    return p


@pytest.fixture(scope="session")
def pdb_path():
    """Factory: pdb_path('1BRS') -> tests/data/1BRS.pdb"""
    return lambda entry: _path(entry, "pdb")


@pytest.fixture(scope="session")
def cif_path():
    """Factory: cif_path('1BRS') -> tests/data/1BRS.cif"""
    return lambda entry: _path(entry, "cif")


@pytest.fixture(scope="session")
def brs_pdb() -> Path:
    return _path("1BRS", "pdb")


@pytest.fixture(scope="session")
def brs_cif() -> Path:
    return _path("1BRS", "cif")


@pytest.fixture(scope="session")
def hhb_pdb() -> Path:
    return _path("4HHB", "pdb")


@pytest.fixture(scope="session")
def hhb_cif() -> Path:
    return _path("4HHB", "cif")


@pytest.fixture(scope="session")
def crn_pdb() -> Path:
    return _path("1CRN", "pdb")


@pytest.fixture(scope="session")
def ptc_pdb() -> Path:
    return _path("2PTC", "pdb")


def seqres(path: Path) -> dict[str, list[str]]:
    """Chain -> list of 3-letter residue names from PDB SEQRES records."""
    out: dict[str, list[str]] = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("SEQRES"):
            out.setdefault(line[11], []).extend(line[19:].split())
    return out
