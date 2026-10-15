"""Regression test: foldbridge DockQ-lite vs stored official DockQ v2 values.

The fixtures (validation/fixtures/) are 10 decoys drawn from the 180-decoy validation set in
validation/dockq_agreement.csv (built by validation/run_dockq_validation.py), spanning DockQ
0.01-0.98 and all CAPRI classes over three natives: 1BRS A/D, 2PTC E/I (insertion codes
184A/188A/221A) and 1A2K D/A. Official values come from DockQ 2.1.3; DockQ is NOT needed here.

Tolerances (justified in docs/VALIDATION.md from the full 180-decoy run, after the
insertion-code fix in foldbridge.score):
  * DockQ  |d| < 1e-4 -- max observed |d| = 1.7e-6 over all 180 decoys and 2.9e-7 over these
    10 fixtures (float32 PDB coordinates vs float64). 1e-4 is ~60x the full-run worst case and
    200x below the 0.02 that would start to move decoys across CAPRI class boundaries. Before the
    fix, 2PTC_32_graded was off by 1.3e-4 and fails this.
  * Fnat   |d| < 1e-3 -- observed 0 on all 180 decoys; one contact is >= 1/nat_total ~ 0.015,
    so any contact-definition regression fails this.
  * iRMSD / LRMSD |d| < 1e-3 A -- observed max 3.1e-6 / 6.9e-5 A (fixtures: 3.1e-6 / 5.2e-5 A).
    Pre-fix, the 2PTC insertion-code collapse gave iRMSD errors of 0.038 A (2PTC_55_swapped) and
    1.6e-3 A, LRMSD 5.8e-3 A (2PTC_32_graded), so a regression of that fix fails here.
  * CAPRI class identical (unless the official DockQ sits within 1e-4 of a class boundary).
"""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "validation" / "fixtures"
OFFICIAL = FIX / "official_dockq_v2.json"

TOL = {"dockq": 1e-4, "fnat": 1e-3, "irmsd": 1e-3, "lrmsd": 1e-3}
CAPRI = [(0.23, "incorrect"), (0.49, "acceptable"), (0.80, "medium"), (math.inf, "high")]

score_mod = pytest.importorskip("foldbridge.score")

if not OFFICIAL.exists():  # pragma: no cover
    pytest.skip("validation fixtures missing; run validation/run_dockq_validation.py", allow_module_level=True)

DECOYS = json.loads(OFFICIAL.read_text())["decoys"]


def _capri(d: float) -> str:
    return next(name for thr, name in CAPRI if d < thr)


def _gunzip(src: Path, dst: Path) -> Path:
    dst.write_bytes(gzip.decompress(src.read_bytes()))
    return dst


@pytest.fixture(scope="module")
def unpacked(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("dockq_fixtures")
    for p in FIX.glob("*.pdb.gz"):
        _gunzip(p, d / p.name[:-3])
    return d


def test_fixture_set_spans_all_capri_classes():
    assert len(DECOYS) == 10
    assert {d["capri_official"] for d in DECOYS} == {"incorrect", "acceptable", "medium", "high"}
    assert {d["native"] for d in DECOYS} == {"1BRS", "2PTC", "1A2K"}


@pytest.mark.parametrize("dec", DECOYS, ids=[d["id"] for d in DECOYS])
def test_dockq_lite_matches_official_dockq_v2(dec, unpacked):
    model = unpacked / f"{dec['id']}.pdb"
    native = unpacked / f"{dec['native']}_native.pdb"
    got = score_mod.score_structure(model, reference=native)
    for m, tol in TOL.items():
        assert m in got, f"{m} missing from score_structure output"
        exp = dec[f"{m}_official"]
        assert abs(got[m] - exp) < tol, f"{dec['id']} {m}: foldbridge {got[m]:.5f} vs DockQ v2 {exp:.5f}"
    off = dec["dockq_official"]
    near_boundary = any(abs(off - thr) < TOL["dockq"] for thr, _ in CAPRI[:-1])
    if not near_boundary:
        assert _capri(got["dockq"]) == dec["capri_official"]


@pytest.mark.parametrize("dec", DECOYS[:4], ids=[d["id"] for d in DECOYS[:4]])
def test_dockq_lite_direct_call(dec, unpacked):
    """dockq_lite(model, ref, receptor, ligand) gives the same numbers as score_structure."""
    from foldbridge.structure import read_structure

    model = read_structure(unpacked / f"{dec['id']}.pdb")
    native = read_structure(unpacked / f"{dec['native']}_native.pdb")
    got = score_mod.dockq_lite(model, native, dec["receptor"], dec["ligand"])
    assert abs(got["dockq"] - dec["dockq_official"]) < TOL["dockq"]
