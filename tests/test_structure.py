"""Reader / writer / mask tests for foldbridge.structure on real RCSB entries."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import ENTRIES, seqres

from foldbridge.structure import (
    THREE2ONE,
    Structure,
    _cif_tokens,
    read_cif,
    read_pdb,
    read_structure,
)

# ---------------------------------------------------------------------------
# PDB vs mmCIF consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entry", ENTRIES)
def test_pdb_and_cif_agree(entry, pdb_path, cif_path):
    p = read_structure(pdb_path(entry))
    c = read_structure(cif_path(entry))
    hp, hc = p.subset(p.heavy_mask()), c.subset(c.heavy_mask())
    assert len(hp) == len(hc) > 0
    assert p.chains == c.chains
    for ch in p.chains:
        assert p.sequence(ch) == c.sequence(ch), ch
    # same atoms in the same order
    np.testing.assert_allclose(hp.coords, hc.coords, atol=1e-3)
    assert list(hp.atom_names) == list(hc.atom_names)
    assert list(hp.res_names) == list(hc.res_names)
    assert list(hp.res_ids) == list(hc.res_ids)
    assert list(hp.elements) == list(hc.elements)
    assert list(hp.hetero) == list(hc.hetero)
    assert list(hp.ins_codes) == list(hc.ins_codes)


def test_read_structure_dispatch_and_sniff(brs_pdb, tmp_path, crn_pdb):
    assert read_structure(brs_pdb).source.endswith(".pdb")
    # unknown suffix -> content sniffing (PDB text)
    pdb_noext = tmp_path / "model.dat"
    pdb_noext.write_text(brs_pdb.read_text())
    assert len(read_structure(pdb_noext)) == len(read_structure(brs_pdb))
    # a small CIF whose _atom_site loop starts within the first 2000 chars sniffs fine
    small = tmp_path / "small.txt"
    small.write_text(NA_CIF)
    assert len(read_structure(small)) == 9


def test_read_structure_sniffs_real_cif_without_suffix(brs_cif, brs_pdb, tmp_path):
    cif_noext = tmp_path / "model.txt"
    cif_noext.write_text(brs_cif.read_text())
    assert len(read_structure(cif_noext)) == len(read_structure(brs_pdb))
    # a data_ block whose _atom_site loop is far down is still recognised as mmCIF
    padded = tmp_path / "padded.dat"
    padded.write_text("\n\n" + NA_CIF.replace("#\nloop_\n_atom_site", "#\n" + "#" * 5000 + "\nloop_\n_atom_site", 1))
    assert len(read_structure(padded)) == 9


def test_altloc_only_first_kept(brs_pdb, brs_cif):
    # 1BRS barstar D Ser14 has CB/OG in altlocs A and B; only A must survive.
    for path in (brs_pdb, brs_cif):
        s = read_structure(path)
        m = (s.chain_ids == "D") & (s.res_ids == 14)
        names = list(s.atom_names[m])
        assert names.count("CB") == 1 and names.count("OG") == 1, (path.name, names)
        cb = s.coords[m][names.index("CB")]
        np.testing.assert_allclose(cb, [32.742, 22.579, 1.536], atol=1e-3)  # altloc A coords


# ---------------------------------------------------------------------------
# sequences vs SEQRES
# ---------------------------------------------------------------------------

BARNASE = "AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR"
BARSTAR = "KKAVINGEQIRSISDLHQTLKKELALPEYYGENLDALWDALTGWVEYPLVLEWRQFEQSKQLTENGAESVLQVFREAKAEGADITIILS"


@pytest.mark.parametrize("entry", ["1BRS", "4HHB", "1CRN"])
def test_sequence_matches_seqres_by_residue_number(entry, pdb_path):
    """For entries numbered 1..N along SEQRES, every observed CA residue must match SEQRES[resid-1]."""
    path = pdb_path(entry)
    s = read_structure(path)
    sr = {ch: "".join(THREE2ONE.get(r, "X") for r in v) for ch, v in seqres(path).items()}
    assert set(s.chains) <= set(sr)
    for ch in s.chains:
        obs = s.sequence(ch)
        m = s.chain_mask(ch) & s.ca_mask()
        ids = s.res_ids[m]
        assert len(obs) == len(ids) > 0
        for aa, rid in zip(obs, ids):
            assert sr[ch][rid - 1] == aa, (entry, ch, rid)


def test_sequence_known_values(brs_pdb, hhb_pdb, crn_pdb, ptc_pdb):
    brs = read_structure(brs_pdb)
    assert brs.sequence("B") == BARNASE            # fully resolved copy
    assert brs.sequence("F") == BARSTAR
    assert brs.sequence("A") == BARNASE[2:]        # Ala1/Gln2 missing (REMARK 465)
    assert brs.sequence("D") == BARSTAR[:63] + BARSTAR[65:]  # Glu64/Asn65 missing
    crn = read_structure(crn_pdb)
    assert crn.sequence("A") == "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN"
    hhb = read_structure(hhb_pdb)
    assert [len(hhb.sequence(c)) for c in "ABCD"] == [141, 146, 141, 146]
    ptc = read_structure(ptc_pdb)   # trypsin E (chymotrypsinogen numbering) + BPTI I
    sr = {ch: "".join(THREE2ONE[r] for r in v) for ch, v in seqres(ptc_pdb).items()}
    assert ptc.sequence("E") == sr["E"] and ptc.sequence("I") == sr["I"]
    assert ptc.sequence("Z") == ""


# ---------------------------------------------------------------------------
# masks on haemoglobin
# ---------------------------------------------------------------------------


def test_masks_hemoglobin(hhb_pdb):
    s = read_structure(hhb_pdb)
    n = len(s)
    prot, lig, wat = s.protein_mask(), s.ligand_mask(), s.water_mask()
    # partition: every atom is exactly one of protein / ligand / water (no NA here)
    assert np.all(prot.astype(int) + lig.astype(int) + wat.astype(int) == 1)
    assert prot.sum() + lig.sum() + wat.sum() == n
    assert wat.sum() == 221

    # four haem groups, one per globin chain, 43 heavy atoms each
    hem = s.res_names == "HEM"
    hem_res = sorted({(str(c), int(r)) for c, r in zip(s.chain_ids[hem], s.res_ids[hem])})
    assert hem_res == [("A", 142), ("B", 148), ("C", 142), ("D", 148)]
    assert hem.sum() == 4 * 43
    assert np.all(lig[hem]) and np.all(s.hetero[hem])
    assert set(s.res_names[lig]) == {"HEM", "PO4"}
    fe = hem & (s.atom_names == "FE")
    assert fe.sum() == 4 and set(s.elements[fe]) == {"FE"}

    # CA: one per residue, protein only
    ca = s.ca_mask()
    assert ca.sum() == 141 * 2 + 146 * 2
    assert not np.any(ca & lig)

    # representative atoms: CA for protein, all ligand heavy atoms, never water
    rep = s.rep_mask()
    assert not np.any(rep & wat)
    assert np.all(rep[ca])
    assert np.all(rep[hem])
    assert rep.sum() == ca.sum() + lig.sum()

    # polymer mask excludes hetero groups and water
    poly = s.polymer_mask()
    assert not np.any(poly & (lig | wat))
    assert poly.sum() == prot.sum()

    # chain masks partition the structure
    assert sum(s.chain_mask(c).sum() for c in s.chains) == n
    assert s.chain_mask("Q").sum() == 0
    assert np.all(s.heavy_mask())  # no hydrogens in 4HHB


def test_calcium_ion_not_a_ca_atom(ptc_pdb):
    # 2PTC has a Ca2+ ion: res name CA, atom name CA -> ligand, not an alpha carbon
    s = read_structure(ptc_pdb)
    ion = s.res_names == "CA"
    assert ion.sum() == 1
    assert not s.ca_mask()[ion].any()
    assert s.ligand_mask()[ion].all()
    assert s.elements[ion][0] == "CA"


def test_residue_keys_and_bfactor(crn_pdb):
    s = read_structure(crn_pdb)
    keys = s.residue_keys()
    assert keys == [("A", i, "") for i in range(1, 47)]
    pb = s.per_residue_bfactor()
    assert set(pb) == set(keys)
    m = (s.chain_ids == "A") & (s.res_ids == 1)
    assert pb[("A", 1, "")] == pytest.approx(float(s.bfactor[m].mean()))
    # legacy (chain, resid) lookups still resolve to the residue without insertion code
    assert ("A", 1) in pb and pb[("A", 1)] == pb.get(("A", 1)) == pb[("A", 1, "")]
    assert pb.get(("A", 999)) is None and ("A", 999) not in pb


def test_insertion_codes_give_distinct_residues(ptc_pdb, cif_path, tmp_path):
    s = read_structure(ptc_pdb)
    e = s.subset(s.chain_mask("E") & s.protein_mask())
    assert len(e.residue_keys()) == int(e.ca_mask().sum()) == 223
    assert len(e.per_residue_bfactor()) == 223
    keys = e.residue_keys()
    for rid in (184, 188, 221):
        assert ("E", rid, "") in keys and ("E", rid, "A") in keys
        assert keys.index(("E", rid, "A")) == keys.index(("E", rid, "")) - 1  # 184A precedes 184
    # mmCIF pdbx_PDB_ins_code agrees with PDB column 27
    c = read_structure(cif_path("2PTC"))
    assert list(c.subset(c.heavy_mask()).ins_codes) == list(s.subset(s.heavy_mask()).ins_codes)
    # write -> read keeps insertion codes at column 27
    out = tmp_path / "ptc.pdb"
    e.write_pdb(out)
    assert list(read_pdb(out).ins_codes) == list(e.ins_codes)
    line = next(l for l in out.read_text().splitlines() if l[17:20] == "GLY" and l[22:27] == " 184A")
    assert line[26] == "A"


def test_ins_codes_default_and_subset():
    s = _toy(["A", "A", "B"])
    assert list(s.ins_codes) == ["", "", ""]
    assert s.residue_keys() == [("A", 1, ""), ("A", 2, ""), ("B", 3, "")]
    t = Structure(s.coords, s.atom_names, s.elements, s.res_names, np.array([1, 1, 1]), s.chain_ids,
                  s.bfactor, s.hetero, "", np.array(["", "A", ""], dtype=object))
    assert t.residue_keys() == [("A", 1, ""), ("A", 1, "A"), ("B", 1, "")]
    assert list(t.subset(np.array([False, True, True])).ins_codes) == ["A", ""]
    with pytest.raises(ValueError):
        Structure(s.coords, s.atom_names, s.elements, s.res_names, s.res_ids, s.chain_ids,
                  s.bfactor, s.hetero, "", np.array([""], dtype=object))


def test_subset_preserves_fields(brs_pdb):
    s = read_structure(brs_pdb)
    m = s.chain_mask("A") | s.chain_mask("D")
    sub = s.subset(m)
    assert len(sub) == int(m.sum())
    assert sub.chains == ["A", "D"]
    assert sub.source == s.source
    np.testing.assert_array_equal(sub.coords, s.coords[m])
    assert sub.sequence("A") == s.sequence("A")


# ---------------------------------------------------------------------------
# writer round trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entry,fmt", [("4HHB", "pdb"), ("4HHB", "cif"), ("1BRS", "cif"), ("2PTC", "pdb")])
def test_write_pdb_roundtrip(entry, fmt, pdb_path, cif_path, tmp_path):
    src = (pdb_path if fmt == "pdb" else cif_path)(entry)
    s = read_structure(src)
    out = tmp_path / "rt.pdb"
    s.write_pdb(out)
    t = read_pdb(out)
    assert len(t) == len(s)
    assert np.abs(t.coords - s.coords).max() < 1e-3
    assert list(t.atom_names) == list(s.atom_names)
    assert list(t.res_names) == list(s.res_names)
    assert list(t.res_ids) == list(s.res_ids)
    assert list(t.chain_ids) == list(s.chain_ids)
    assert list(t.elements) == list(s.elements)
    assert list(t.hetero) == list(s.hetero)
    np.testing.assert_allclose(t.bfactor, s.bfactor, atol=0.01)
    # fixed-column sanity: every ATOM/HETATM line is >= 78 chars and coordinates sit at 31-54
    for line in out.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            assert len(line) >= 78
            float(line[30:38]); float(line[38:46]); float(line[46:54])


def _toy(chain_ids: list[str]) -> Structure:
    n = len(chain_ids)
    return Structure(
        coords=np.arange(n * 3, dtype=np.float32).reshape(n, 3),
        atom_names=np.array(["CA"] * n, dtype=object),
        elements=np.array(["C"] * n, dtype=object),
        res_names=np.array(["GLY"] * n, dtype=object),
        res_ids=np.arange(1, n + 1),
        chain_ids=np.array(chain_ids, dtype=object),
        bfactor=np.full(n, 90.0, dtype=np.float32),
        hetero=np.zeros(n, dtype=bool),
    )


def test_write_pdb_multichar_chain_ids_not_merged(tmp_path):
    s = _toy(["A", "A2"])
    out = tmp_path / "toy.pdb"
    mapping = s.write_pdb(out)
    assert mapping == {"A": "A", "A2": "B"}
    assert read_pdb(out).chains == ["A", "B"]
    # deterministic, skips letters already taken, keeps 1-char ids untouched
    s2 = _toy(["AA", "B", "AB", "A"])
    assert s2.write_pdb(tmp_path / "t2.pdb") == {"AA": "C", "B": "B", "AB": "D", "A": "A"}
    assert read_pdb(tmp_path / "t2.pdb").chains == ["C", "B", "D", "A"]
    assert s2.write_pdb(tmp_path / "t3.pdb") == s2.pdb_chain_map()


def test_write_pdb_too_many_chains(tmp_path):
    ok = _toy([f"C{i}" for i in range(62)])
    m = ok.write_pdb(tmp_path / "ok.pdb")
    assert len(set(m.values())) == 62 and all(len(v) == 1 for v in m.values())
    with pytest.raises(ValueError, match="chains"):
        _toy([f"C{i}" for i in range(63)]).write_pdb(tmp_path / "bad.pdb")


# ---------------------------------------------------------------------------
# mmCIF tokenizer / nucleic acids
# ---------------------------------------------------------------------------


def test_cif_tokens_quotes():
    assert _cif_tokens("ATOM 1 C \"C1'\" . DA") == ["ATOM", "1", "C", "C1'", ".", "DA"]
    assert _cif_tokens("'a b' c") == ["a b", "c"]
    assert _cif_tokens('"it\'s" x') == ["it's", "x"]
    # a quote followed by non-space does not close the token (CIF 1.1 rule)
    assert _cif_tokens("'C1'' next") == ["C1'", "next"]
    assert _cif_tokens("  spaced   tokens\t") == ["spaced", "tokens"]


NA_CIF = """data_TOY
#
loop_
_entity.id
_entity.type
1 polymer
2 non-polymer
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
ATOM   1  P P     . DA A 1 1.000 2.000 3.000 1.00 10.0 1 B 1
ATOM   2  O "O5'" . DA A 1 1.500 2.000 3.000 1.00 11.0 1 B 1
ATOM   3  C "C1'" . DA A 1 2.000 2.500 3.000 1.00 12.0 1 B 1
ATOM   4  N N9    . DA A 1 2.500 2.500 3.500 1.00 13.0 1 B 1
ATOM   5  P P     . DG A 2 5.000 2.000 3.000 1.00 14.0 2 B 1
ATOM   6  C 'C1'' . DG A 2 6.000 2.500 3.000 1.00 15.0 2 B 1
ATOM   7  C "C1'" . G  C 1 9.000 9.000 9.000 1.00 16.0 1 R 1
HETATM 8  MG MG   . MG D . 0.000 0.000 0.000 1.00 20.0 101 B 1
HETATM 9  O O     . HOH E . 4.000 4.000 4.000 1.00 30.0 201 B 1
ATOM   10 P P     . DA A 1 7.000 7.000 7.000 1.00 10.0 1 B 2
ATOM   11 C "C1'" . DA A 1 7.000 7.000 7.000 1.00 10.0 1 B 2
#
loop_
_atom_type.symbol
C
"""


def test_cif_nucleic_acid_quoted_atom_names(tmp_path):
    p = tmp_path / "toy.cif"
    p.write_text(NA_CIF)
    s = read_cif(p)
    assert len(s) == 9                                   # model 2 dropped
    assert list(s.atom_names[:4]) == ["P", "O5'", "C1'", "N9"]
    assert s.atom_names[5] == "C1'"
    assert s.chains == ["B", "R"]                        # auth_asym_id preferred
    assert list(s.res_ids) == [1, 1, 1, 1, 2, 2, 1, 101, 201]  # auth_seq_id preferred
    assert s.elements[7] == "MG" and s.hetero[7] and not s.hetero[0]
    np.testing.assert_allclose(s.coords[2], [2.0, 2.5, 3.0])
    assert s.bfactor[6] == pytest.approx(16.0)
    # one representative atom (C1') per nucleotide, plus the Mg ion, no water
    rep = s.rep_mask()
    assert list(s.atom_names[rep]) == ["C1'", "C1'", "C1'", "MG"]
    assert s.polymer_mask().sum() == 7
    assert s.water_mask().sum() == 1
    assert s.ca_mask().sum() == 0 and s.sequence("B") == ""
    # write -> read keeps the primed names
    out = tmp_path / "toy.pdb"
    s.write_pdb(out)
    t = read_pdb(out)
    assert list(t.atom_names) == list(s.atom_names)
    np.testing.assert_allclose(t.coords, s.coords, atol=1e-3)


def test_cif_without_atom_site_raises(tmp_path):
    p = tmp_path / "empty.cif"
    p.write_text("data_X\n#\nloop_\n_entity.id\n1\n#\n")
    with pytest.raises(ValueError):
        read_cif(p)


# ---------------------------------------------------------------------------
# multi-model PDB
# ---------------------------------------------------------------------------


def _atom(serial: int, name: str, res: str, chain: str, resid: int, xyz, elem: str, rec: str = "ATOM  ") -> str:
    an = f" {name:<3}" if len(name) < 4 else name
    return (f"{rec}{serial:5d} {an} {res:>3} {chain}{resid:4d}    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{1.0:6.2f}{50.0:6.2f}          {elem:>2}")


def test_multimodel_pdb_keeps_first_model(tmp_path):
    lines = ["HEADER    TOY NMR ENSEMBLE"]
    for model, dx in ((1, 0.0), (2, 100.0), (3, 200.0)):
        lines.append(f"MODEL     {model:4d}")
        lines.append(_atom(1, "N", "ALA", "A", 1, (0.0 + dx, 0.0, 0.0), "N"))
        lines.append(_atom(2, "CA", "ALA", "A", 1, (1.458 + dx, 0.0, 0.0), "C"))
        lines.append(_atom(3, "CA", "GLY", "A", 2, (5.0 + dx, 1.0, 0.0), "C"))
        lines.append(_atom(4, "H", "GLY", "A", 2, (5.5 + dx, 1.0, 0.0), "H"))
        lines.append("ENDMDL")
    lines.append("END")
    p = tmp_path / "nmr.pdb"
    p.write_text("\n".join(lines) + "\n")
    s = read_pdb(p)
    assert len(s) == 4
    assert s.coords[:, 0].max() < 10.0
    assert s.sequence("A") == "AG"
    assert s.heavy_mask().sum() == 3


# ---------------------------------------------------------------------------
# experimental vs predicted
# ---------------------------------------------------------------------------

BOLTZ_DIR = Path(__file__).resolve().parent.parent / "validation" / "boltz_barnase_barstar" / "barnase_barstar"
_ATOM_LINE = "ATOM      1  CA  ALA A   1      11.104   6.134  -6.504  1.00 85.00           C\n"


@pytest.mark.parametrize("entry", ["1BRS", "2PTC", "4HHB"])
def test_crystal_structures_are_experimental(entry, pdb_path, cif_path):
    for p in (pdb_path(entry), cif_path(entry)):
        st = read_structure(p)
        assert st.experimental is True, p
        assert st.subset(st.chain_ids == st.chains[0]).experimental is True  # subset keeps the flag


def test_predicted_models_are_not_experimental(tmp_path):
    boltz = sorted(BOLTZ_DIR.glob("*_model_*.cif"))
    if boltz:  # Boltz-2 ModelCIF output
        assert read_structure(boltz[0]).experimental is False
    cases = {
        "plain.pdb": _ATOM_LINE,                                            # ColabFold / Boltz PDB
        "afdb.pdb": "EXPDTA    THEORETICAL MODEL\n" + _ATOM_LINE,           # AF-DB PDB
        "pred.pdb": "EXPDTA    PREDICTED MODEL\n" + _ATOM_LINE,
        "xray.pdb": "EXPDTA    X-RAY DIFFRACTION\n" + _ATOM_LINE,
        "nmr.pdb": "EXPDTA    SOLUTION NMR\n" + _ATOM_LINE,
    }
    expected = {"plain.pdb": False, "afdb.pdb": False, "pred.pdb": False, "xray.pdb": True, "nmr.pdb": True}
    for name, text in cases.items():
        (tmp_path / name).write_text(text)
        assert read_structure(tmp_path / name).experimental is expected[name], name

    atom_site = ("loop_\n_atom_site.group_PDB\n_atom_site.id\n_atom_site.type_symbol\n_atom_site.label_atom_id\n"
                 "_atom_site.label_comp_id\n_atom_site.auth_asym_id\n_atom_site.auth_seq_id\n"
                 "_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n_atom_site.B_iso_or_equiv\n"
                 "ATOM 1 C CA ALA A 1 1.0 2.0 3.0 85.0\n#\n")
    cifs = {
        "em.cif": ("data_x\n_exptl.method 'ELECTRON MICROSCOPY'\n#\n", True),
        "afdb.cif": ("data_x\n_exptl.method 'PREDICTED MODEL'\n#\n", False),
        "af3.cif": ("data_x\n#\n", False),
        "modelcif.cif": ("data_x\n_exptl.method 'X-RAY DIFFRACTION'\n_ma_data.id 1\n#\n", False),
        "computed.cif": ("data_x\n_struct.pdbx_structure_determination_methodology computational\n"
                         "_exptl.method 'X-RAY DIFFRACTION'\n#\n", False),
        "looped.cif": ("data_x\nloop_\n_exptl.entry_id\n_exptl.method\nX 'X-RAY DIFFRACTION'\n"
                       "X 'NEUTRON DIFFRACTION'\n#\n", True),
    }
    for name, (head, exp) in cifs.items():
        (tmp_path / name).write_text(head + atom_site)
        st = read_structure(tmp_path / name)
        assert len(st) == 1 and st.experimental is exp, name
    assert _toy(["A"]).experimental is False  # constructed in memory: default
