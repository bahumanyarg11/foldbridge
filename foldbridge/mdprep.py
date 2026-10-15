"""Hand a predicted structure to molecular dynamics.

``prepare_for_md`` writes ``prepared.pdb`` + ``system_report.json``. When the
B-factor column holds pLDDT, confident CA atoms (pLDDT >= cutoff) are marked
for position restraints: as GROMACS ``posre_confident_<chain>.itp`` files on the
OpenMM path (hydrogens already present, so atom numbers are final), or as
``restraints_confident.tsv`` keyed by (chain, resid, icode, atom name) in the
numpy fallback, whose hydrogen-free PDB would give wrong atom numbers once
pdb2gmx adds hydrogens.

* With ``openmm`` + ``pdbfixer`` importable: missing residues/atoms are found
  and rebuilt, non-standard residues replaced, hydrogens added at ``ph``, and
  optionally the system is solvated (amber14-all + TIP3P, 0.15 M NaCl,
  neutralised). The net charge comes from the amber14 force field.
* Otherwise a pure-numpy cleanup runs (drop waters/altlocs, split ligands,
  renumber residues from 1 per chain) and the report says ``"openmm": false``.

Ligands never go into the force-field system automatically: they need bond
orders + parameters (GAFF/SMIRNOFF via openmmforcefields, or acpype/CGenFF for
GROMACS). With ``keep_ligands=True`` they are written to ``ligands/`` as PDB
(and SDF when RDKit is available) and the report says what to do next.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np

from .structure import Structure, read_structure

LOW_PLDDT = 50.0
POSITIVE = {"ARG": 1, "LYS": 1, "HIP": 1}
NEGATIVE = {"ASP": -1, "GLU": -1}
CA_CA_BREAK = 4.2  # Angstrom; consecutive CA farther apart than this = chain break


def _have(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def openmm_available() -> bool:
    return _have("openmm") and _have("pdbfixer")


# ---- pLDDT helpers ---------------------------------------------------------------

def plddt_values(structure: Structure) -> tuple[np.ndarray | None, str]:
    """B-factors on a 0-100 pLDDT scale, or (None, reason) if they don't look like pLDDT."""
    b = np.asarray(structure.bfactor, dtype=float)
    if b.size == 0:
        return None, "empty structure"
    if getattr(structure, "experimental", False):
        return None, "experimental structure: B-factors are crystallographic, not pLDDT"
    if np.allclose(b, 0.0):
        return None, "B-factor column is all zero (no pLDDT)"
    if b.min() < 0 or b.max() > 100.0:
        return None, "B-factor range is outside 0-100; not a pLDDT column"
    if b.max() <= 1.0:
        return b * 100.0, "pLDDT read from B-factor (0-1 scale, multiplied by 100)"
    return b, "pLDDT read from B-factor (0-100 scale)"


def low_plddt_residues(structure: Structure, cutoff: float = LOW_PLDDT) -> list[dict[str, Any]]:
    vals, _ = plddt_values(structure)
    if vals is None:
        return []
    poly = structure.polymer_mask()
    out: dict[tuple[str, int, str], dict[str, Any]] = {}
    acc: dict[tuple[str, int, str], list[float]] = {}
    for i in np.flatnonzero(poly):
        k = (str(structure.chain_ids[i]), int(structure.res_ids[i]), str(structure.ins_codes[i]))
        acc.setdefault(k, []).append(vals[i])
        out.setdefault(k, {"chain": k[0], "resid": k[1], "icode": k[2], "resname": str(structure.res_names[i])})
    res = []
    for k, v in acc.items():
        m = float(np.mean(v))
        if m < cutoff:
            res.append({**out[k], "plddt": round(m, 2)})
    return res


def write_restraints(structure: Structure | str | Path, plddt_cutoff: float = 70.0,
                     out_path: str | Path | None = None, force_constant: float = 1000.0,
                     define: str = "POSRES_CONFIDENT") -> str:
    """GROMACS ``[ position_restraints ]`` block for CA atoms with pLDDT >= cutoff.

    Atom indices are 1-based positions in ``structure`` as given, so pass a
    single chain whose atom order matches the GROMACS moleculetype (one file
    per chain for multi-chain systems; ``prepare_for_md`` does this). Returns
    the .itp text and writes it to ``out_path`` when given.
    """
    st = read_structure(structure) if isinstance(structure, (str, Path)) else structure
    vals, note = plddt_values(st)
    ca = st.ca_mask()
    idx = np.flatnonzero(ca & (vals >= plddt_cutoff)) if vals is not None else np.array([], int)
    fc = f"{force_constant:g}"
    lines = [
        f"; foldbridge position restraints: CA atoms with pLDDT >= {plddt_cutoff:g}",
        f"; {len(idx)} of {int(ca.sum())} CA atoms restrained; {note}",
        "; indices are 1-based atom numbers within this structure/moleculetype",
        f"#ifdef {define}" if define else "",
        "[ position_restraints ]",
        ";  ai  funct       fcx       fcy       fcz",
    ]
    lines = [ln for ln in lines if ln != ""]
    for i in idx:
        lines.append(f"{i + 1:6d}     1 {fc:>9} {fc:>9} {fc:>9}")
    if define:
        lines.append("#endif")
    text = "\n".join(lines) + "\n"
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(text)
    return text


# ---- structure-level helpers -------------------------------------------------------

def _residue_groups(st: Structure, mask: np.ndarray) -> list[np.ndarray]:
    """Atom-index arrays, one per (chain, resid, icode, resname) run within mask."""
    groups: list[list[int]] = []
    last = None
    for i in np.flatnonzero(mask):
        k = (st.chain_ids[i], int(st.res_ids[i]), st.ins_codes[i], st.res_names[i])
        if k != last:
            groups.append([])
            last = k
        groups[-1].append(int(i))
    return [np.asarray(g) for g in groups]


def _gaps(st: Structure) -> list[dict[str, Any]]:
    """Chain breaks from residue numbering jumps or long CA-CA distances."""
    out = []
    ca = np.flatnonzero(st.ca_mask())
    for prev, cur in zip(ca[:-1], ca[1:]):
        if st.chain_ids[prev] != st.chain_ids[cur]:
            continue
        r0, r1 = int(st.res_ids[prev]), int(st.res_ids[cur])
        d = float(np.linalg.norm(st.coords[cur] - st.coords[prev]))
        if r1 - r0 > 1 or d > CA_CA_BREAK:
            out.append({"chain": str(st.chain_ids[cur]), "after_resid": r0, "before_resid": r1,
                        "n_missing": max(r1 - r0 - 1, 0), "ca_ca_distance": round(d, 2)})
    return out


def _count_charge(res_names: list[str], his_protonated: int = 0) -> int:
    q = sum(POSITIVE.get(r, 0) + NEGATIVE.get(r, 0) for r in res_names)
    return int(q + his_protonated)


def _split_ligands(st: Structure, out_dir: Path) -> list[dict[str, Any]]:
    lig = st.ligand_mask() & ~st.water_mask()
    if not lig.any():
        return []
    lig_dir = out_dir / "ligands"
    lig_dir.mkdir(parents=True, exist_ok=True)
    rdkit = _have("rdkit")
    out = []
    for g in _residue_groups(st, lig):
        sub = st.subset(np.isin(np.arange(len(st)), g))
        rn, ch, ri = str(sub.res_names[0]), str(sub.chain_ids[0]), int(sub.res_ids[0])
        stem = f"{rn}_{ch}{ri}"
        pdb = lig_dir / f"{stem}.pdb"
        sub.write_pdb(pdb)
        entry: dict[str, Any] = {"resname": rn, "chain": ch, "resid": ri, "atoms": len(sub),
                                 "pdb": str(pdb)}
        if rdkit:
            try:
                from rdkit import Chem  # type: ignore

                mol = Chem.MolFromPDBFile(str(pdb), removeHs=False, sanitize=False)
                if mol is not None:
                    sdf = lig_dir / f"{stem}.sdf"
                    w = Chem.SDWriter(str(sdf))
                    w.write(mol)
                    w.close()
                    entry["sdf"] = str(sdf)
                    entry["sdf_note"] = "bond orders perceived from geometry only; assign from SMILES"
            except Exception as e:  # pragma: no cover - rdkit optional
                entry["sdf_error"] = str(e)
        out.append(entry)
    return out


def _ligand_note(ligs: list[dict[str, Any]], keep: bool) -> str | None:
    if not ligs:
        return None
    if not keep:
        return f"{len(ligs)} ligand(s) removed (use keep_ligands/--keep-ligands to export them)"
    if _have("openmmforcefields") or _have("openff.toolkit"):
        return ("ligands written to ligands/ and NOT included in prepared.pdb; openmmforcefields/openff "
                "is installed: register a GAFFTemplateGenerator/SMIRNOFFTemplateGenerator with the "
                "ligand SMILES and combine with the protein via openmm.app.Modeller")
    return ("ligands written to ligands/ and NOT included in prepared.pdb; they need "
            "parameterisation (install openmmforcefields + openff-toolkit, or use acpype/CGenFF "
            "for GROMACS) before they can be simulated")


def _chain_summary(st: Structure) -> list[dict[str, Any]]:
    out = []
    for c in st.chains:
        m = st.chain_mask(c)
        sub = st.subset(m)
        out.append({"id": str(c), "atoms": int(m.sum()), "residues": len(sub.residue_keys()),
                    "sequence": st.sequence(c)})
    return out


def _restraint_atom_table(st: Structure, orig: Structure, out_dir: Path, plddt_cutoff: float) -> Path | None:
    """TSV of the CA atoms to restrain, keyed by residue identity rather than atom index.

    ``st`` is the renumbered structure written to prepared.pdb; ``orig`` the same atoms with
    the input numbering (``orig_resid`` / ``orig_icode`` columns).
    """
    vals, _ = plddt_values(orig)  # orig keeps the input's `experimental` flag
    if vals is None:
        return None
    idx = np.flatnonzero(st.ca_mask() & st.polymer_mask() & (vals >= plddt_cutoff))
    p = out_dir / "restraints_confident.tsv"
    lines = ["chain\tresid\ticode\tatom_name\tresname\tplddt\torig_resid\torig_icode"]
    for i in idx:
        lines.append(f"{st.chain_ids[i]}\t{int(st.res_ids[i])}\t{st.ins_codes[i]}\t{st.atom_names[i]}\t"
                     f"{st.res_names[i]}\t{vals[i]:.2f}\t{int(orig.res_ids[i])}\t{orig.ins_codes[i]}")
    p.write_text("\n".join(lines) + "\n")
    return p


def _restraint_files(st: Structure, out_dir: Path, plddt_cutoff: float) -> list[str]:
    vals, _ = plddt_values(st)
    if vals is None:
        return []
    files = []
    poly = st.polymer_mask()
    for c in st.chains:
        m = st.chain_mask(c) & poly
        if not (m & st.ca_mask()).any():
            continue
        p = (out_dir / f"posre_confident_{c}.itp").resolve()
        write_restraints(st.subset(st.chain_mask(c)), plddt_cutoff=plddt_cutoff, out_path=p,
                         define="POSRES_CONFIDENT")
        files.append(str(p))
    return files


# ---- fallback (numpy only) ---------------------------------------------------------------

def _fallback(st: Structure, out_dir: Path, report: dict[str, Any], plddt_cutoff: float) -> Structure:
    poly = st.polymer_mask() & ~st.water_mask()
    clean = st.subset(poly)
    # renumber residues 1..n per chain; insertion codes are part of the residue key (52 != 52A)
    new_ids = clean.res_ids.copy()
    mapping: dict[tuple[str, int, str], int] = {}
    for c in clean.chains:
        n = 0
        last = None
        for i in np.flatnonzero(clean.chain_mask(c)):
            k = (str(c), int(clean.res_ids[i]), str(clean.ins_codes[i]))
            if k != last:
                n += 1
                mapping[k] = n
                last = k
            new_ids[i] = n
    renum = Structure(clean.coords, clean.atom_names, clean.elements, clean.res_names, new_ids,
                      clean.chain_ids, clean.bfactor, clean.hetero, clean.source)  # ins_codes cleared
    out_pdb = out_dir / "prepared.pdb"
    renum.write_pdb(out_pdb)

    low = low_plddt_residues(clean)
    for r in low:
        r["orig_resid"], r["orig_icode"] = r["resid"], r["icode"]
        r["resid"], r["icode"] = mapping[(r["chain"], r["orig_resid"], r["orig_icode"])], ""
    res_names = [str(renum.res_names[g[0]]) for g in _residue_groups(renum, np.ones(len(renum), bool))]
    report.update({
        "prepared_pdb": str(out_pdb.resolve()),
        "atoms": len(renum),
        "chains": _chain_summary(renum),
        "charge_estimate": _count_charge(res_names),
        "charge_method": "residue count at pH ~7 (K,R +1; D,E -1; HIS neutral; termini cancel)",
        "missing_residues": _gaps(st),
        "missing_residues_method": "residue-numbering gaps / CA-CA > 4.2 A (not rebuilt)",
        "hydrogens_added": False,
        "solvated": False,
        "renumbered": True,
        "low_plddt_residues": low,
        "restraint_files": [],
    })
    report["warnings"].append(
        f"openmm/pdbfixer not installed: {out_pdb.resolve()} was cleaned only (no missing atoms/residues "
        "rebuilt, no hydrogens, no solvent). Install with `pip install foldbridge[md]` "
        "(or `conda install -c conda-forge openmm pdbfixer`).")
    atoms_file = _restraint_atom_table(renum, clean, out_dir, plddt_cutoff)
    if atoms_file is not None:
        n = sum(1 for _ in atoms_file.open()) - 1
        report["restraint_atoms_file"] = str(atoms_file.resolve())
        report["warnings"].append(
            f"no GROMACS position-restraint .itp written: {out_pdb.resolve()} has no hydrogens, so its "
            "atom numbers will not match the topology after pdb2gmx adds them. The "
            f"{n} CA atoms with pLDDT >= {plddt_cutoff:g} are listed by (chain, resid, icode, atom name) "
            f"in {atoms_file.resolve()}; regenerate restraints from the processed structure (e.g. make an "
            "index group of these atoms and run `gmx genrestr`), or install openmm + pdbfixer and rerun "
            "prep-md, which writes ready-to-use posre_confident_<chain>.itp files.")
    if report.get("solvate_requested"):
        report["warnings"].append("solvation requested but skipped (needs openmm)")
    return renum


# ---- OpenMM path ---------------------------------------------------------------------------

def _openmm(src: Path, st: Structure, out_dir: Path, report: dict[str, Any], ph: float,
            solvate: bool, padding_nm: float, plddt_cutoff: float) -> Structure:
    import openmm  # type: ignore
    from openmm import app, unit  # type: ignore
    from pdbfixer import PDBFixer  # type: ignore

    try:
        fixer = PDBFixer(filename=str(src))
        report["pdbfixer_input"] = str(src)
    except Exception as e:  # unusual mmCIF etc. -> feed a clean PDB written by us
        tmp = out_dir / "_input_clean.pdb"
        st.subset(~st.water_mask()).write_pdb(tmp)
        fixer = PDBFixer(filename=str(tmp))
        report["pdbfixer_input"] = str(tmp)
        report["warnings"].append(f"pdbfixer could not read {src.resolve()} directly ({e}); "
                                  f"used a cleaned PDB {tmp.resolve()}")

    fixer.findMissingResidues()
    chains = list(fixer.topology.chains())
    missing = []
    for (ci, ri), names in sorted(fixer.missingResidues.items()):
        chain = chains[ci]
        nres = len(list(chain.residues()))
        terminal = ri == 0 or ri == nres
        missing.append({"chain": chain.id, "insert_index": ri, "residues": list(names),
                        "terminal": terminal, "rebuilt": not terminal})
    # standard practice: don't model dangling termini, do rebuild internal gaps
    for (ci, ri) in list(fixer.missingResidues):
        nres = len(list(chains[ci].residues()))
        if ri == 0 or ri == nres:
            del fixer.missingResidues[(ci, ri)]

    fixer.findNonstandardResidues()
    nonstd = [{"chain": r.chain.id, "resid": r.id, "resname": r.name, "replaced_with": rep}
              for r, rep in fixer.nonstandardResidues]
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    n_missing_atoms = sum(len(v) for v in fixer.missingAtoms.values())
    n_missing_term = sum(len(v) for v in fixer.missingTerminals.values())
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)

    topology, positions = fixer.topology, fixer.positions
    ff = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")

    def net_charge(top, pos) -> float | None:
        try:
            system = ff.createSystem(top, nonbondedMethod=app.NoCutoff, constraints=None)
            for f in system.getForces():
                if isinstance(f, openmm.NonbondedForce):
                    return float(sum(f.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
                                     for i in range(f.getNumParticles())))
        except Exception as e:
            report["warnings"].append(f"amber14 could not parameterise the protein: {e}")
        return None

    q = net_charge(topology, positions)
    his = sum(1 for r in topology.residues() if r.name == "HIS"
              and {"HD1", "HE2"} <= {a.name for a in r.atoms()})
    res_names = [r.name for r in topology.residues()]
    report["charge_estimate"] = round(q) if q is not None else _count_charge(res_names, his)
    report["charge_method"] = ("amber14 force-field partial charges" if q is not None
                               else "residue count after protonation")

    if solvate:
        modeller = app.Modeller(topology, positions)
        modeller.addSolvent(ff, model="tip3p", padding=padding_nm * unit.nanometer,
                            ionicStrength=0.15 * unit.molar, neutralize=True)
        topology, positions = modeller.topology, modeller.positions
        n_wat = sum(1 for r in topology.residues() if r.name == "HOH")
        n_ions = sum(1 for r in topology.residues() if r.name in {"NA", "CL"})
        report.update({"waters": n_wat, "ions": n_ions, "padding_nm": padding_nm,
                       "ionic_strength_M": 0.15, "water_model": "tip3p"})
        qs = net_charge(topology, positions)
        report["system_charge"] = round(qs) if qs is not None else None

    out_pdb = out_dir / "prepared.pdb"
    with open(out_pdb, "w") as fh:
        app.PDBFile.writeFile(topology, positions, fh, keepIds=True)
    _copy_bfactor(out_pdb, st)

    prepared = read_structure(out_pdb)
    report.update({
        "prepared_pdb": str(out_pdb.resolve()),
        "atoms": topology.getNumAtoms(),
        "chains": _chain_summary(prepared.subset(prepared.polymer_mask())),
        "missing_residues": missing,
        "chain_breaks": _gaps(st),  # numbering jumps / long CA-CA: invisible to PDBFixer without SEQRES
        "missing_atoms_added": n_missing_atoms,
        "missing_terminal_atoms_added": n_missing_term,
        "nonstandard_residues": nonstd,
        "hydrogens_added": True,
        "solvated": bool(solvate),
        "renumbered": False,
        "forcefield": ["amber14-all.xml", "amber14/tip3p.xml"],
        "openmm_version": getattr(openmm, "__version__", None) or openmm.version.version,
        "low_plddt_residues": low_plddt_residues(st),
        # B-factors were copied from the input: only meaningful as pLDDT if the input's were
        "restraint_files": (_restraint_files(prepared.subset(prepared.polymer_mask()), out_dir, plddt_cutoff)
                            if plddt_values(st)[0] is not None else []),
    })
    return prepared


def _copy_bfactor(pdb_path: Path, src: Structure) -> None:
    """Carry per-residue pLDDT from the input into the B-factor column of an OpenMM-written PDB."""
    per_res = src.per_residue_bfactor()
    if not per_res:
        return
    out = []
    for line in pdb_path.read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 66:
            try:
                key = (line[21].strip() or "A", int(line[22:26]), line[26].strip())
            except ValueError:
                key = None
            b = per_res.get(key) if key else None
            if b is not None:
                line = f"{line[:60]}{b:6.2f}{line[66:]}"
        out.append(line)
    pdb_path.write_text("\n".join(out) + "\n")


# ---- entry point -----------------------------------------------------------------------------

def prepare_for_md(path: str | Path, out_dir: str | Path, ph: float = 7.0, keep_ligands: bool = False,
                   solvate: bool = False, padding_nm: float = 1.0, plddt_cutoff: float = 70.0,
                   use_openmm: bool | None = None) -> dict[str, Any]:
    """Clean / protonate / (optionally) solvate ``path`` into ``out_dir``. Returns the report dict.

    ``use_openmm=None`` auto-detects; False forces the numpy fallback.
    """
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(f"structure not found: {src}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    st = read_structure(src)
    if len(st) == 0:
        raise ValueError(f"no atoms read from {src}")
    if not st.polymer_mask().any():
        raise ValueError(f"{src} has no protein/nucleic-acid residues to prepare")

    have_mm = openmm_available() if use_openmm is None else bool(use_openmm and openmm_available())
    vals, plddt_note = plddt_values(st)
    report: dict[str, Any] = {
        "input": str(src.resolve()),
        "openmm": have_mm,
        "ph": ph,
        "solvate_requested": bool(solvate),
        "keep_ligands": bool(keep_ligands),
        "plddt_source": plddt_note,
        "plddt_low_cutoff": LOW_PLDDT,
        "restraint_plddt_cutoff": plddt_cutoff,
        "low_plddt_note": ("residues with pLDDT < 50 are likely disordered: candidates for removal, "
                           "restraint-free treatment, or extra sampling; CA atoms with pLDDT >= "
                           f"{plddt_cutoff:g} are position-restrained in "
                           + ("posre_confident_<chain>.itp (#define POSRES_CONFIDENT)" if have_mm else
                              "restraints_confident.tsv (regenerate the .itp after pdb2gmx)")),
        "warnings": [],
    }
    if use_openmm and not have_mm:
        report["warnings"].append("openmm requested but not importable; used fallback")

    ligs = _split_ligands(st, out) if keep_ligands else []
    n_lig = len(_residue_groups(st, st.ligand_mask() & ~st.water_mask()))
    report["ligands"] = ligs
    note = _ligand_note(ligs if keep_ligands else [None] * n_lig, keep_ligands)  # type: ignore[list-item]
    if note:
        report["ligand_note"] = note
    report["waters_removed"] = len(_residue_groups(st, st.water_mask()))

    if have_mm:
        _openmm(src, st, out, report, ph, solvate, padding_nm, plddt_cutoff)
    else:
        _fallback(st, out, report, plddt_cutoff)
    report["n_low_plddt_residues"] = len(report.get("low_plddt_residues", []))
    rp = (out / "system_report.json").resolve()
    report["report"] = str(rp)
    rp.write_text(json.dumps(report, indent=2, default=str))
    return report
