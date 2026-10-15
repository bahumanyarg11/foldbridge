"""End-to-end scoring of real crystal complexes with foldbridge.score.

Skipped (with a reason) if ``foldbridge/score.py`` is not present. While the package is
being built concurrently, set ``FOLDBRIDGE_WAIT_FOR_SCORE=10`` to poll up to 10 x 30 s for
the module to appear before deciding to skip.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pytest
from conftest import ROOT, optional_module

from foldbridge.structure import read_structure

SCORE_PY = ROOT / "foldbridge" / "score.py"


def _wait_for_score(polls: int, interval: float = 30.0) -> bool:
    for k in range(polls + 1):
        if SCORE_PY.exists():
            return True
        if k < polls:
            time.sleep(interval)
    return SCORE_PY.exists()


_wait_for_score(int(os.environ.get("FOLDBRIDGE_WAIT_FOR_SCORE", "0")))
score = optional_module("score") if SCORE_PY.exists() else None
pytestmark = pytest.mark.skipif(
    score is None or not hasattr(score, "score_structure"),
    reason="foldbridge/score.py (score_structure) not available yet",
)


@pytest.fixture(scope="module")
def brs(brs_pdb):
    return read_structure(brs_pdb)


@pytest.fixture(scope="module")
def brs_ad(brs):
    return score.score_structure(brs, chains=["A", "D"])


def _translated(st, chain: str, shift):
    out = st.subset(np.ones(len(st), dtype=bool))
    out.coords = out.coords.copy()
    out.coords[out.chain_ids == chain] += np.asarray(shift, dtype=np.float32)
    return out


def test_barnase_barstar_interface(brs_ad):
    r = brs_ad
    assert r["n_chains"] == 2
    assert r["n_residues"] == 108 + 87          # observed barnase A + barstar D residues
    assert r["n_interface_residues"] > 10
    assert r["n_contacts"] > 10


def test_barnase_barstar_no_clashes(brs_ad):
    assert brs_ad["clash_count"] <= 1
    assert brs_ad["clash_fraction"] < 0.005


def test_barnase_barstar_buried_sasa(brs_ad):
    # literature: ~1590 A^2 total buried on complex formation (Buckle et al. 1994)
    assert 1200 <= brs_ad["bsa"] <= 2000
    assert brs_ad["bsa"] == pytest.approx(brs_ad["bsa_A"] + brs_ad["bsa_D"], rel=1e-6)
    assert brs_ad["bsa_A"] > 0 and brs_ad["bsa_D"] > 0


def test_bsa_cif_matches_pdb(brs_ad, brs_cif):
    r = score.score_structure(read_structure(brs_cif), chains=["A", "D"])
    assert r["bsa"] == pytest.approx(brs_ad["bsa"], rel=1e-6)
    assert r["n_interface_residues"] == brs_ad["n_interface_residues"]


def test_dockq_self_is_one(brs):
    r = score.score_structure(brs, chains=["A", "D"], reference=brs)
    assert r["dockq"] == pytest.approx(1.0, abs=1e-6)
    assert r["fnat"] == pytest.approx(1.0)
    assert r["irmsd"] < 1e-3 and r["lrmsd"] < 1e-3
    assert r["rmsd"] < 1e-3


def test_dockq_rigid_motion_of_whole_complex_is_one(brs, brs_pdb):
    # moving both chains together must not change DockQ (superposition-invariant)
    moved = _translated(_translated(brs, "A", [3, -7, 11]), "D", [3, -7, 11])
    r = score.score_structure(moved, chains=["A", "D"], reference=brs)
    assert r["dockq"] == pytest.approx(1.0, abs=1e-4)


def test_dockq_translated_partner_much_lower(brs):
    moved = _translated(brs, "D", [10.0, 0.0, 0.0])
    r = score.score_structure(moved, chains=["A", "D"], reference=brs)
    assert r["dockq"] < 0.3
    assert r["fnat"] < 0.2
    assert r["lrmsd"] == pytest.approx(10.0, abs=0.05)   # pure 10 A translation of barstar
    assert r["irmsd"] > 2.0
    assert r["n_interface_residues"] < 10
    assert r["bsa"] < 500


def test_ipsae_from_synthetic_pae(brs):
    sub = brs.subset(brs.chain_mask("A") | brs.chain_mask("D"))
    n_tok = int(sub.rep_mask().sum())
    good = score.score_structure(sub, pae=np.full((n_tok, n_tok), 1.0))
    bad = score.score_structure(sub, pae=np.full((n_tok, n_tok), 30.0))
    assert good["ipsae"] > 0.8
    assert bad["ipsae"] == pytest.approx(0.0)
    assert good["pae_interchain"] == pytest.approx(1.0)


def test_pdockq_tracks_plddt(brs):
    sub = brs.subset(brs.chain_mask("A") | brs.chain_mask("D"))
    hi, lo = sub.subset(np.ones(len(sub), bool)), sub.subset(np.ones(len(sub), bool))
    hi.bfactor = np.full(len(sub), 90.0, dtype=np.float32)
    lo.bfactor = np.full(len(sub), 30.0, dtype=np.float32)
    hi.experimental = lo.experimental = False  # pretend these B-factors are pLDDT
    p_hi = score.score_structure(hi)["pdockq"]
    p_lo = score.score_structure(lo)["pdockq"]
    assert 0.018 <= p_lo < p_hi <= 0.742
    assert p_hi > 0.3


def test_trypsin_bpti(ptc_pdb):
    r = score.score_structure(ptc_pdb, chains=["E", "I"])
    assert r["n_interface_residues"] > 10
    assert r["clash_count"] == 0
    assert 1000 <= r["bsa"] <= 2000            # ~1400 A^2 for trypsin-BPTI


def test_hemoglobin_four_chains(hhb_cif):
    r = score.score_structure(hhb_cif)
    assert r["n_chains"] == 4
    assert r["n_interface_residues"] > 50
    assert r["bsa"] > 3000


def test_hemoglobin_metal_coordination_not_clash(hhb_cif):
    # HEM FE - His NE2 (~2.0-2.1 A in 4HHB) is coordination, not a clash
    s = read_structure(hhb_cif)
    keep = ~((s.chain_ids == "D") & np.isin(s.res_ids, [4, 6, 66]))   # genuine close contacts in 4HHB
    r = score.score_structure(s.subset(keep))
    assert r["clash_count"] == 0
