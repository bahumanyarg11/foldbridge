"""MD prep: numpy fallback, ligand export, restraints, optional OpenMM path."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from foldbridge import mdprep
from foldbridge.mdprep import low_plddt_residues, plddt_values, prepare_for_md, write_restraints
from foldbridge.structure import Structure, read_structure

BACKBONE = {"N": (-1.2, 0.8, 0.0), "CA": (0.0, 0.0, 0.0), "C": (1.2, 0.8, 0.0), "O": (1.2, 2.0, 0.0),
            "CB": (0.0, -1.0, 1.2)}
RES = ["MET", "LYS", "ALA", "GLU", "LEU", "ARG", "SER", "ASP", "VAL", "LYS", "GLY", "THR"]


def make_structure(chains=("A", "B"), n=12, start=1, low_tail=2, ligand=True, water=True,
                   gap_after=None, scale=1.0) -> Structure:
    rows = []
    for ci, ch in enumerate(chains):
        y0, flip = -7.0 * ci, (-1 if ci % 2 else 1)
        rid = start
        for i in range(n):
            if gap_after is not None and i == gap_after:
                rid += 3
            rn = RES[i % len(RES)]
            b = (40.0 if i >= n - low_tail else 85.0) * scale
            ca = np.array([3.8 * i, y0, 0.0])
            for an, off in BACKBONE.items():
                if an == "CB" and rn == "GLY":
                    continue
                o = np.array(off) * np.array([1, flip, 1])
                rows.append((ca + o, an, an[0], rn, rid, ch, b, False))
            rid += 1
    if ligand:
        for k, an in enumerate(["C1", "C2", "O1"]):
            rows.append((np.array([10.0 + 1.4 * k, -3.5, 6.0]), an, an[0], "LIG", 900, "A", 0.0, True))
    if water:
        rows.append((np.array([0.0, 20.0, 0.0]), "O", "O", "HOH", 1000, "A", 0.0, True))
    coords, names, elems, rn, ri, ch, bf, het = zip(*rows)
    return Structure(np.asarray(coords, np.float32), np.asarray(names, object), np.asarray(elems, object),
                     np.asarray(rn, object), np.asarray(ri, np.int64), np.asarray(ch, object),
                     np.asarray(bf, np.float32), np.asarray(het, bool), "synthetic")


@pytest.fixture
def pdb(tmp_path) -> Path:
    p = tmp_path / "model.pdb"
    make_structure(start=5).write_pdb(p)
    return p


@pytest.fixture
def no_openmm(monkeypatch):
    monkeypatch.setattr(mdprep, "openmm_available", lambda: False)


def test_fallback_cleanup(pdb, tmp_path, no_openmm):
    out = tmp_path / "md"
    rep = prepare_for_md(pdb, out)
    assert rep["openmm"] is False
    assert (out / "prepared.pdb").is_file() and (out / "system_report.json").is_file()
    assert json.loads((out / "system_report.json").read_text())["openmm"] is False
    st = read_structure(out / "prepared.pdb")
    assert set(st.res_names) <= set(RES)  # ligand + water dropped
    assert rep["waters_removed"] == 1 and "removed" in rep["ligand_note"]
    assert rep["atoms"] == len(st)
    assert [c["id"] for c in rep["chains"]] == ["A", "B"]
    assert all(c["residues"] == 12 for c in rep["chains"])
    # renumbered from 1 (input started at 5)
    assert st.res_ids[st.chain_mask("A")].min() == 1 and st.res_ids.max() == 12
    # charge: per chain K x2, R, E, D -> +3 -2 = +1; two chains -> +2
    assert rep["charge_estimate"] == 2
    low = rep["low_plddt_residues"]
    assert len(low) == 4 and {r["resid"] for r in low} == {11, 12}
    assert {r["orig_resid"] for r in low} == {15, 16}
    assert rep["n_low_plddt_residues"] == 4
    assert rep["missing_residues"] == []
    assert any("openmm" in w for w in rep["warnings"])
    assert rep["restraint_files"] == []  # no .itp without hydrogens: indices would be wrong


def test_fallback_restraints_as_residue_table(pdb, tmp_path, no_openmm):
    """Fallback writes (chain, resid, icode, atom) rows + a warning naming full paths, never an .itp."""
    out = tmp_path / "md"
    rep = prepare_for_md(pdb, out)
    assert not list(out.glob("*.itp"))
    tsv = Path(rep["restraint_atoms_file"])
    assert tsv.is_absolute() and tsv == (out / "restraints_confident.tsv").resolve()
    rows = [ln.split("\t") for ln in tsv.read_text().splitlines()]
    assert rows[0][:4] == ["chain", "resid", "icode", "atom_name"]
    body = rows[1:]
    assert len(body) == 20 and {r[3] for r in body} == {"CA"}  # 2 chains x 10 confident residues
    assert body[0][:3] == ["A", "1", ""] and body[0][6] == "5"  # renumbered; orig_resid kept
    w = [x for x in rep["warnings"] if "pdb2gmx" in x]
    assert len(w) == 1 and str(tsv) in w[0] and str((out / "prepared.pdb").resolve()) in w[0]
    assert "hydrogens" in w[0] and "genrestr" in w[0]
    assert Path(rep["prepared_pdb"]).is_absolute() and Path(rep["report"]).is_absolute()


def test_fallback_keeps_insertion_codes_apart(tmp_path, no_openmm):
    """52 and 52A are two residues: renumbered separately, low-pLDDT entries keep their icode."""
    rows = []
    for k, (rid, ic, rn, b) in enumerate(((51, "", "ALA", 90.0), (52, "", "TYR", 30.0), (52, "A", "GLY", 35.0),
                                          (53, "", "SER", 90.0))):
        for an, off in BACKBONE.items():
            if an == "CB" and rn == "GLY":
                continue
            rows.append((np.array([3.8 * k, 0, 0]) + np.array(off), an, an[0], rn, rid, "A", b, False, ic))
    coords, names, elems, rn, ri, ch, bf, het, ic = zip(*rows)
    st = Structure(np.asarray(coords, np.float32), np.asarray(names, object), np.asarray(elems, object),
                   np.asarray(rn, object), np.asarray(ri, np.int64), np.asarray(ch, object),
                   np.asarray(bf, np.float32), np.asarray(het, bool), "synthetic", np.asarray(ic, object))
    p = tmp_path / "ic.pdb"
    st.write_pdb(p)
    rep = prepare_for_md(p, tmp_path / "md")
    assert rep["chains"][0]["residues"] == 4
    prepared = read_structure(tmp_path / "md" / "prepared.pdb")
    assert sorted(set(prepared.res_ids.tolist())) == [1, 2, 3, 4]
    low = sorted((r["resid"], r["orig_resid"], r["orig_icode"], r["resname"]) for r in rep["low_plddt_residues"])
    assert low == [(2, 52, "", "TYR"), (3, 52, "A", "GLY")]
    keys = {(r["chain"], r["resid"], r["icode"]) for r in low_plddt_residues(st)}
    assert keys == {("A", 52, ""), ("A", 52, "A")}


def test_restraint_files_per_chain(pdb, tmp_path):
    """OpenMM-path helper: one .itp per chain, CA indices within the chain's moleculetype."""
    rep = {"restraint_files": mdprep._restraint_files(read_structure(pdb), tmp_path, 70.0)}
    assert len(rep["restraint_files"]) == 2
    itp = Path(rep["restraint_files"][0]).read_text()
    assert "[ position_restraints ]" in itp and "#ifdef POSRES_CONFIDENT" in itp
    body = [ln for ln in itp.splitlines() if ln.strip() and ln.split()[0].isdigit()]
    assert len(body) == 10  # 12 residues, last 2 low pLDDT
    # first CA of chain is atom 2 (N, CA, ...)
    assert body[0].split()[:2] == ["2", "1"]


def test_keep_ligands_exports(pdb, tmp_path, no_openmm, monkeypatch):
    real = mdprep._have
    monkeypatch.setattr(mdprep, "_have", lambda m: False if m.startswith(("openmmforcefields", "openff")) else real(m))
    rep = prepare_for_md(pdb, tmp_path / "md", keep_ligands=True)
    assert len(rep["ligands"]) == 1
    lig = rep["ligands"][0]
    assert lig["resname"] == "LIG" and lig["atoms"] == 3 and Path(lig["pdb"]).is_file()
    assert "parameterisation" in rep["ligand_note"]
    assert "LIG" not in set(read_structure(tmp_path / "md" / "prepared.pdb").res_names)


def test_solvate_without_openmm_warns(pdb, tmp_path, no_openmm):
    rep = prepare_for_md(pdb, tmp_path / "md", solvate=True)
    assert rep["solvated"] is False
    assert any("solvation" in w for w in rep["warnings"])


def test_gap_detection(tmp_path, no_openmm):
    p = tmp_path / "gap.pdb"
    make_structure(chains=("A",), gap_after=6, ligand=False, water=False).write_pdb(p)
    rep = prepare_for_md(p, tmp_path / "md")
    gaps = rep["missing_residues"]
    assert len(gaps) == 1 and gaps[0]["n_missing"] == 3 and gaps[0]["after_resid"] == 6


def test_write_restraints_direct(tmp_path):
    st = make_structure(chains=("A",), ligand=False, water=False)
    text = write_restraints(st, plddt_cutoff=70, out_path=tmp_path / "posre.itp", force_constant=500)
    assert (tmp_path / "posre.itp").read_text() == text
    body = [ln.split() for ln in text.splitlines() if ln.strip() and ln.split()[0].isdigit()]
    ca_idx = np.flatnonzero(st.ca_mask())[:10] + 1
    assert [int(b[0]) for b in body] == list(ca_idx)
    assert all(b[1:] == ["1", "500", "500", "500"] for b in body)
    # path input + 0-1 pLDDT scale
    p = tmp_path / "s.pdb"
    make_structure(chains=("A",), ligand=False, water=False, scale=0.01).write_pdb(p)
    assert "10 of 12" in write_restraints(p)
    # stricter cutoff -> none
    assert "0 of 12" in write_restraints(st, plddt_cutoff=90)


def test_no_plddt_column():
    st = make_structure(chains=("A",), ligand=False, water=False, scale=0.0)
    vals, note = plddt_values(st)
    assert vals is None and "zero" in note
    assert low_plddt_residues(st) == []
    assert "0 of 12" in write_restraints(st)


def test_errors(tmp_path, no_openmm):
    with pytest.raises(FileNotFoundError):
        prepare_for_md(tmp_path / "missing.pdb", tmp_path / "o")
    p = tmp_path / "lig_only.pdb"
    make_structure(chains=(), ligand=True, water=False).write_pdb(p)
    with pytest.raises(ValueError):
        prepare_for_md(p, tmp_path / "o")


def test_openmm_path(pdb, tmp_path):
    pytest.importorskip("openmm")
    pytest.importorskip("pdbfixer")
    rep = prepare_for_md(pdb, tmp_path / "md", ph=7.0)
    assert rep["openmm"] is True and rep["hydrogens_added"] is True
    st = read_structure(tmp_path / "md" / "prepared.pdb")
    assert (st.elements == "H").any()
    assert rep["charge_estimate"] == 2
    assert len(rep["low_plddt_residues"]) == 4


def test_copy_bfactor_respects_insertion_codes(tmp_path):
    """184 and 184A are different residues: each keeps its own pLDDT in the prepared PDB."""
    rows = []
    for rid, ic, rn, b in ((183, "", "ALA", 90.0), (184, "", "TYR", 80.0), (184, "A", "GLY", 30.0),
                           (185, "", "SER", 70.0)):
        x = 3.8 * len(rows) / 4
        for an, off in BACKBONE.items():
            if an == "CB" and rn == "GLY":
                continue
            rows.append((np.array([x, 0, 0]) + np.array(off), an, an[0], rn, rid, "E", b, False, ic))
    coords, names, elems, rn, ri, ch, bf, het, ic = zip(*rows)
    src = Structure(np.asarray(coords, np.float32), np.asarray(names, object), np.asarray(elems, object),
                    np.asarray(rn, object), np.asarray(ri, np.int64), np.asarray(ch, object),
                    np.asarray(bf, np.float32), np.asarray(het, bool), "synthetic", np.asarray(ic, object))
    out = tmp_path / "prepared.pdb"
    blank = src.subset(np.ones(len(src), bool))
    blank.bfactor = np.zeros(len(src), np.float32)  # what OpenMM writes
    blank.write_pdb(out)
    assert {line[26] for line in out.read_text().splitlines() if line.startswith("ATOM")} == {" ", "A"}
    mdprep._copy_bfactor(out, src)
    got = read_structure(out).per_residue_bfactor()
    assert got[("E", 184, "")] == pytest.approx(80.0)
    assert got[("E", 184, "A")] == pytest.approx(30.0)
    assert got[("E", 183, "")] == pytest.approx(90.0) and got[("E", 185, "")] == pytest.approx(70.0)


def test_experimental_structure_bfactors_not_plddt(tmp_path, no_openmm):
    """Crystal B-factors are not pLDDT: no low-pLDDT list, no restraints."""
    src = Path(__file__).parent / "data" / "1CRN.pdb"
    st = read_structure(src)
    assert st.experimental
    vals, note = plddt_values(st)
    assert vals is None and "experimental" in note
    rep = prepare_for_md(src, tmp_path / "md")
    assert rep["low_plddt_residues"] == [] and rep["restraint_files"] == []
    assert "restraint_atoms_file" not in rep and not (tmp_path / "md" / "restraints_confident.tsv").exists()
    assert "experimental" in rep["plddt_source"]


def test_1brs_crystal_bfactors_not_low_plddt(tmp_path, no_openmm):
    st = read_structure(Path(__file__).parent / "data" / "1BRS.pdb")
    assert st.experimental and low_plddt_residues(st) == []
    assert "0 of" in write_restraints(st)  # no pLDDT -> nothing restrained
    rep = prepare_for_md(Path(__file__).parent / "data" / "1BRS.pdb", tmp_path / "md")
    assert rep["n_low_plddt_residues"] == 0 and rep["restraint_files"] == []
    assert not any("pdb2gmx" in w for w in rep["warnings"])
