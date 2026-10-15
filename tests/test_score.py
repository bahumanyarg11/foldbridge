"""Tests for foldbridge.score on synthetic, in-code structures (no files, no network)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from foldbridge.backends.base import ModelResult
from foldbridge.score import (
    _atom_keys,
    _grid_pairs,
    _residue_contacts,
    dockq_lite,
    ipsae,
    kabsch_rmsd,
    load_pae,
    neighbor_pairs,
    pdockq,
    score_result,
    score_structure,
    shrake_rupley,
)
from foldbridge.structure import Structure

# backbone + CB offsets relative to CA for an idealised extended strand along x
OFFSETS = {
    "N": (-1.2, 0.6, 0.0),
    "CA": (0.0, 0.0, 0.0),
    "C": (1.2, 0.6, 0.0),
    "O": (1.2, 1.8, 0.0),
    "CB": (0.0, -1.0, 1.1),
}


def make_chain(cid: str, n: int, shift=(0.0, 0.0, 0.0), plddt: float = 80.0, start: int = 1):
    rows = []
    for k in range(n):
        ca = np.array([3.8 * k, 0.0, 0.0]) + np.asarray(shift)
        resn = "GLY" if k == n // 2 else "ALA"
        for an, off in OFFSETS.items():
            if resn == "GLY" and an == "CB":
                continue
            rows.append((ca + off, an, an[0], resn, start + k, cid, plddt))
    return rows


def build(*chains) -> Structure:
    rows = [r for ch in chains for r in ch]
    return Structure(
        coords=np.array([r[0] for r in rows], dtype=np.float32),
        atom_names=np.array([r[1] for r in rows], dtype=object),
        elements=np.array([r[2] for r in rows], dtype=object),
        res_names=np.array([r[3] for r in rows], dtype=object),
        res_ids=np.array([r[4] for r in rows], dtype=np.int64),
        chain_ids=np.array([r[5] for r in rows], dtype=object),
        bfactor=np.array([r[6] for r in rows], dtype=np.float32),
        hetero=np.zeros(len(rows), dtype=bool),
    )


def rigid(st: Structure, R: np.ndarray, t) -> Structure:
    out = st.subset(np.ones(len(st), dtype=bool))
    out.coords = (st.coords.astype(np.float64) @ R.T + np.asarray(t)).astype(np.float32)
    return out


def rot(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


N = 10


@pytest.fixture
def contact():
    return build(make_chain("A", N, plddt=80.0), make_chain("B", N, shift=(0, 0, 5.5), plddt=60.0))


@pytest.fixture
def apart():
    return build(make_chain("A", N), make_chain("B", N, shift=(0, 0, 40.0)))


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------

def test_grid_pairs_match_bruteforce():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 30, size=(400, 3))
    Y = rng.uniform(0, 30, size=(300, 3))
    D = np.linalg.norm(X[:, None] - Y[None], axis=2)
    i, j = _grid_pairs(X, Y, 4.0)
    assert set(zip(i.tolist(), j.tolist())) == set(zip(*np.nonzero(D < 4.0)))
    si, sj = neighbor_pairs(X, 4.0, use_scipy=False)
    DX = np.linalg.norm(X[:, None] - X[None], axis=2)
    bi, bj = np.nonzero(np.triu(DX < 4.0, k=1))
    assert set(zip(si.tolist(), sj.tolist())) == set(zip(bi.tolist(), bj.tolist()))


def test_sasa_isolated_and_two_spheres():
    r, p = 1.7, 1.4
    a = shrake_rupley(np.zeros((1, 3)), np.array([r]))
    assert a[0] == pytest.approx(4 * math.pi * (r + p) ** 2)
    d = 4.0
    s = shrake_rupley(np.array([[0, 0, 0], [d, 0, 0]], float), np.array([r, r]))
    R = r + p
    h = R - d / 2  # cap height for equal spheres
    expected = 4 * math.pi * R ** 2 - 2 * math.pi * R * h
    assert s[0] == pytest.approx(expected, rel=0.05)
    assert s[0] == pytest.approx(s[1], rel=0.03)  # spiral points are not mirror-symmetric


def test_kabsch_rmsd_rigid_copy_is_zero():
    rng = np.random.default_rng(1)
    P = rng.normal(size=(50, 3)) * 10
    Q = P @ rot([1, 2, 3], 0.7).T + [5, -3, 2]
    assert kabsch_rmsd(P, Q) == pytest.approx(0.0, abs=1e-6)
    assert kabsch_rmsd(P, Q + rng.normal(scale=0.5, size=Q.shape)) > 0.2


def test_pdockq_formula():
    assert pdockq(80.0, 0) == 0.0
    x = 80.0 * math.log10(50)
    assert pdockq(80.0, 50) == pytest.approx(0.724 / (1 + math.exp(-0.052 * (x - 152.611))) + 0.018)
    assert 0.018 < pdockq(90.0, 200) < 0.742


def test_ipsae_good_vs_bad_and_bounds():
    tc = np.array(["A"] * N + ["B"] * N)
    inter = tc[:, None] != tc[None, :]
    good = np.where(inter, 1.0, 0.5)
    bad = np.where(inter, 25.0, 0.5)
    g, b = ipsae(good, tc, "A", "B"), ipsae(bad, tc, "A", "B")
    d0 = 1.24 * (27 - 15) ** (1 / 3) - 1.8  # N0 = 10 is floored to 27
    assert g == pytest.approx(1 / (1 + (1.0 / d0) ** 2))
    assert b == 0.0
    assert 0.0 <= g <= 1.0


def test_load_pae_formats(tmp_path):
    M = np.arange(16, dtype=float).reshape(4, 4)
    np.save(tmp_path / "p.npy", M)
    np.savez(tmp_path / "p.npz", pae=M)
    (tmp_path / "p.json").write_text(f'[{{"predicted_aligned_error": {M.tolist()}}}]')
    for f in ("p.npy", "p.npz", "p.json"):
        np.testing.assert_allclose(load_pae(tmp_path / f), M)
    np.testing.assert_allclose(load_pae(M[None]), M)


# ---------------------------------------------------------------------------
# score_structure
# ---------------------------------------------------------------------------

def test_counts_and_plddt(contact):
    s = score_structure(contact)
    assert s["n_chains"] == 2 and s["n_residues"] == 2 * N and s["n_tokens"] == 2 * N
    assert s["plddt"] == pytest.approx(70.0)
    assert s["plddt_A"] == pytest.approx(80.0) and s["plddt_B"] == pytest.approx(60.0)
    assert 60.0 <= s["plddt_interface"] <= 80.0


def test_plddt_scale_autodetect(contact):
    st = contact.subset(np.ones(len(contact), bool))
    st.bfactor = (contact.bfactor / 100.0).astype(np.float32)
    assert score_structure(st)["plddt"] == pytest.approx(70.0, rel=1e-5)


def test_no_plddt_when_bfactors_zero(contact):
    st = contact.subset(np.ones(len(contact), bool))
    st.bfactor = np.zeros(len(st), np.float32)
    s = score_structure(st)
    assert "plddt" not in s and "pdockq" not in s and "pdockq_contacts" in s


def test_interface_contact_vs_apart(contact, apart):
    c, a = score_structure(contact), score_structure(apart)
    assert c["n_contacts"] > 0 and c["n_interface_residues"] >= N
    assert a["n_contacts"] == 0 and a["n_interface_residues"] == 0
    assert c["bsa"] > 100.0 and c["bsa_A"] > 0 and c["bsa_B"] > 0
    assert c["bsa"] == pytest.approx(c["bsa_A"] + c["bsa_B"])
    assert a["bsa"] == pytest.approx(0.0)
    assert c["pdockq"] > a["pdockq"] == 0.0
    assert c["clash_count"] == 0 and a["clash_count"] == 0
    assert c["clash_fraction"] == 0.0


def test_bsa_matches_definition(contact):
    """bsa == SASA(A) + SASA(B) - SASA(AB) computed independently."""
    from foldbridge.score import _radii
    X = contact.coords.astype(float)
    rad = _radii(contact.elements)
    A = contact.chain_ids == "A"
    tot = shrake_rupley(X, rad).sum()
    sa = shrake_rupley(X[A], rad[A]).sum()
    sb = shrake_rupley(X[~A], rad[~A]).sum()
    assert score_structure(contact)["bsa"] == pytest.approx(sa + sb - tot, rel=1e-6)


def test_clashes_overlap():
    st = build(make_chain("A", N), make_chain("B", N, shift=(0.3, 0.2, 0.4)))
    s = score_structure(st)
    assert s["clash_count"] >= N
    assert 0.5 < s["clash_fraction"] <= 1.0
    # intra-chain sequence neighbours (C(i)-N(i+1) = 1.4 A) are not clashes
    assert score_structure(build(make_chain("A", N)))["clash_count"] == 0


def test_pae_metrics(contact):
    tc = np.array(["A"] * N + ["B"] * N)
    inter = tc[:, None] != tc[None, :]
    good = np.where(inter, 2.0, 0.5)
    bad = np.where(inter, 25.0, 0.5)
    g, b = score_structure(contact, pae=good), score_structure(contact, pae=bad)
    assert g["ipae"] == pytest.approx(2.0) and b["ipae"] == pytest.approx(25.0)
    assert g["pae_interchain"] == pytest.approx(2.0)
    assert g["ipsae"] > 0.1 and b["ipsae"] == 0.0
    assert g["ipsae_A_B"] == g["ipsae"]


def test_ipae_only_counts_contacts():
    # A contacts only the first 3 residues of a long B; PAE is low on contacts only
    st = build(make_chain("A", 3), make_chain("B", N, shift=(0, 0, 5.5)))
    pae = np.full((3 + N, 3 + N), 30.0)
    pae[:3, 3:6] = pae[3:6, :3] = 1.0
    s = score_structure(st, pae=pae)
    assert s["ipae"] < 10.0 < s["pae_interchain"]


def test_pae_size_mismatch_is_skipped(contact):
    with pytest.warns(UserWarning):
        s = score_structure(contact, pae=np.zeros((5, 5)))
    assert "ipae" not in s and "plddt" in s


def test_reference_rigid_copy_gives_dockq_one(contact):
    ref = rigid(contact, rot([0.3, 1, 0.2], 1.1), [10, -4, 7])
    s = score_structure(contact, reference=ref)
    assert s["rmsd"] == pytest.approx(0.0, abs=1e-3)
    assert s["n_aligned"] == 2 * N
    assert s["fnat"] == pytest.approx(1.0)
    assert s["irmsd"] == pytest.approx(0.0, abs=1e-3)
    assert s["lrmsd"] == pytest.approx(0.0, abs=1e-3)
    assert s["dockq"] == pytest.approx(1.0, abs=1e-4)


def test_reference_displaced_ligand(contact):
    ref = contact.subset(np.ones(len(contact), bool))
    moved = ref.coords.copy()
    moved[ref.chain_ids == "B"] += np.array([0, 0, 3.0], np.float32)  # B pushed 3 A away
    ref.coords = moved
    s = score_structure(contact, reference=ref)
    assert s["lrmsd"] == pytest.approx(3.0, abs=1e-3)
    assert 0.0 < s["irmsd"] < 3.0
    assert s["fnat"] < 1.0
    assert 0.0 < s["dockq"] < 0.9
    d = dockq_lite(contact, ref, "A", "B")
    assert d["dockq"] == pytest.approx(s["dockq"])


def test_chains_subset():
    st = build(make_chain("A", N), make_chain("B", N, shift=(0, 0, 5.5)), make_chain("C", N, shift=(0, 0, 60)))
    tc = np.array(["A"] * N + ["B"] * N + ["C"] * N)
    pae = np.where(tc[:, None] != tc[None, :], 3.0, 0.5)
    full = score_structure(st, pae=pae)
    sub = score_structure(st, pae=pae, chains=["A", "C"])
    assert full["n_chains"] == 3 and "ipsae_B_C" in full
    assert sub["n_chains"] == 2 and sub["n_contacts"] == 0 and "ipsae_A_C" in sub
    assert sub["n_residues"] == 2 * N


def test_deterministic(contact):
    pae = np.where(np.arange(2 * N)[:, None] // N != np.arange(2 * N)[None] // N, 4.0, 1.0)
    assert score_structure(contact, pae=pae) == score_structure(contact, pae=pae)


def test_file_roundtrip_and_score_result(contact, tmp_path):
    p = tmp_path / "model.pdb"
    contact.write_pdb(p)
    tc = np.array(["A"] * N + ["B"] * N)
    np.savez(tmp_path / "pae.npz", pae=np.where(tc[:, None] != tc[None, :], 3.0, 0.5))
    res = ModelResult(backend="boltz", path=p, seed=7, sample=2,
                      scores={"iptm": 0.8, "ptm": 0.7, "plddt": 0.75}, pae_path=tmp_path / "pae.npz")
    row = score_result(res, reference=p)
    assert row["backend"] == "boltz" and row["seed"] == 7 and row["sample"] == 2
    assert row["id"] == "boltz_s7_m2" and row["path"] == str(p)
    assert row["iptm"] == 0.8
    assert row["plddt"] == pytest.approx(70.0, rel=1e-3) and row["plddt_backend"] == 0.75
    assert row["dockq"] == pytest.approx(1.0, abs=1e-3)
    assert "ipsae" in row and "bsa" in row


# ---------------------------------------------------------------------------
# insertion codes (2PTC: trypsin E carries 184/184A, 188/188A, 221/221A)
# ---------------------------------------------------------------------------

DATA = Path(__file__).resolve().parent / "data"


@pytest.fixture(scope="module")
def ptc():
    from foldbridge.structure import read_structure

    p = DATA / "2PTC.pdb"
    if not p.exists():  # pragma: no cover
        pytest.skip("tests/data/2PTC.pdb missing")
    return read_structure(p)


def _take(st: Structure, idx: np.ndarray) -> Structure:
    return Structure(st.coords[idx], st.atom_names[idx], st.elements[idx], st.res_names[idx],
                     st.res_ids[idx], st.chain_ids[idx], st.bfactor[idx], st.hetero[idx], st.source,
                     st.ins_codes[idx])


def test_2ptc_insertion_codes_are_distinct_residues(ptc):
    keys = ptc.residue_keys()
    assert ("E", 184, "") in keys and ("E", 184, "A") in keys
    ca = _atom_keys(ptc, ptc.ca_mask())
    assert len(ca) == int(ptc.ca_mask().sum()) == 281  # 184/184A etc. no longer collapse
    assert ptc.res_names[ca[("E", 184, "", "CA")]] == "TYR"
    assert ptc.res_names[ca[("E", 184, "A", "CA")]] == "GLY"
    assert len(_atom_keys(ptc, np.ones(len(ptc), bool))) == len(ptc)
    if_res = _residue_contacts(ptc, "E", "I", 10.0)
    assert any(a == (184, "A") for a, _ in if_res)  # 184A is an interface residue
    assert all(isinstance(a, tuple) and isinstance(b, tuple) for a, b in if_res)


def test_2ptc_self_dockq_is_exactly_one(ptc):
    s = score_structure(ptc, reference=ptc)
    assert s["dockq"] == 1.0 and s["fnat"] == 1.0
    assert s["n_aligned"] == 281.0 and s["rmsd"] < 1e-6
    # same atoms written in reverse order: matching must go by (chain, resid, icode, atom)
    rev = _take(ptc, np.arange(len(ptc))[::-1])
    s2 = score_structure(rev, reference=ptc)
    assert s2["dockq"] == 1.0 and s2["fnat"] == 1.0 and s2["n_aligned"] == 281.0


def test_2ptc_moving_184A_does_not_touch_184(ptc):
    """Displacing only residue 184A must register in iRMSD (184A is an interface residue);
    with (chain, resid) keys 184A was shadowed by 184 and the move was invisible."""
    m = ptc.subset(np.ones(len(ptc), bool))
    sel = (m.chain_ids == "E") & (m.res_ids == 184) & (m.ins_codes == "A")
    coords = m.coords.copy()
    coords[sel] += np.array([3.0, 0.0, 0.0], np.float32)
    m.coords = coords
    s = score_structure(m, reference=ptc)
    assert s["irmsd"] > 0.05 and s["dockq"] < 1.0
    assert s["n_aligned"] == 281.0


# ---------------------------------------------------------------------------
# chain selection errors, experimental structures, reference coverage, int counts
# ---------------------------------------------------------------------------

def test_missing_chains_raise(contact, tmp_path):
    p = tmp_path / "m.pdb"
    contact.write_pdb(p)
    with pytest.raises(ValueError, match=r"^chains E,I not found in m\.pdb \(available: A, B\)$"):
        score_structure(p, chains=["E", "I"])
    with pytest.raises(ValueError, match=r"^chain X not found in m\.pdb \(available: A, B\)$"):
        score_structure(p, chains=["A", "X"])  # partially missing ids are an error too
    assert score_structure(p, chains=["B", "A"])["n_chains"] == 2


def test_experimental_structure_has_no_plddt(brs_pdb, brs_cif):
    from foldbridge.score import EXPERIMENTAL_NOTE

    for p in (brs_pdb, brs_cif):
        s = score_structure(p, chains=["A", "D"])
        assert s["note"] == EXPERIMENTAL_NOTE == "experimental structure: B-factors are not pLDDT"
        assert not any(k.startswith("plddt") for k in s), p
        assert "pdockq" not in s and s["n_interface_residues"] > 10 and s["bsa"] > 0
    from foldbridge.structure import read_structure

    st = read_structure(brs_pdb)
    st.experimental = False  # same coordinates read as a "prediction": B-factors used as pLDDT
    s = score_structure(st, chains=["A", "D"])
    assert "plddt" in s and "pdockq" in s and "note" not in s


def test_counts_are_ints(contact):
    s = score_structure(contact, reference=contact)
    for k in ("n_atoms", "n_residues", "n_tokens", "n_chains", "n_contacts", "n_interface_residues",
              "clash_count", "n_clash_residues", "pdockq_contacts", "n_aligned"):
        assert type(s[k]) is int, k
    assert isinstance(s["clash_fraction"], float) and isinstance(s["bsa"], float)


def test_reference_coverage_warning(contact):
    import warnings as _w

    ref = contact.subset(np.ones(len(contact), bool))
    ref.chain_ids = np.where(ref.chain_ids == "A", "X", "Y").astype(object)  # chain ids don't match
    with pytest.warns(UserWarning, match=r"reference matched only 0 of 20 residues; check chain ids/numbering"):
        score_structure(contact, reference=ref)
    half = contact.subset(contact.chain_ids == "A")
    with _w.catch_warnings():
        _w.simplefilter("error")  # 10 of 20 = 50 %: no warning
        score_structure(contact, reference=half)
