"""Minimal, dependency-free PDB / mmCIF reader + writer backed by numpy.

Only what scoring and MD-prep need: atom coords, names, elements, residue ids,
chain ids, B-factor (pLDDT in predicted models), polymer/hetero flag.
"""
from __future__ import annotations

import string
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PROTEIN_RES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE",
    "PRO", "SER", "THR", "TRP", "TYR", "VAL", "MSE", "SEC", "PYL", "UNK",
}
NA_RES = {"DA", "DC", "DG", "DT", "DU", "A", "C", "G", "U", "N", "DN"}
WATER = {"HOH", "WAT", "DOD"}
# PDB chain-id alphabet used when remapping multi-character chain ids on write
PDB_CHAIN_ALPHABET = string.ascii_uppercase + string.ascii_lowercase + string.digits

THREE2ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O", "UNK": "X",
}


@dataclass
class Structure:
    coords: np.ndarray          # (N,3) float32
    atom_names: np.ndarray      # (N,) str
    elements: np.ndarray        # (N,) str, upper-case
    res_names: np.ndarray       # (N,) str
    res_ids: np.ndarray         # (N,) int
    chain_ids: np.ndarray       # (N,) str
    bfactor: np.ndarray         # (N,) float32
    hetero: np.ndarray          # (N,) bool
    source: str = ""
    ins_codes: np.ndarray | None = field(default=None)  # (N,) str, "" when no insertion code
    # True for experimentally determined entries (PDB EXPDTA / mmCIF _exptl.method that is not
    # a prediction); their B-factors are crystallographic, not pLDDT
    experimental: bool = False

    def __post_init__(self) -> None:
        n = len(self.coords)
        if self.ins_codes is None:
            self.ins_codes = np.full(n, "", dtype=object)
        else:
            self.ins_codes = np.asarray(
                ["" if c is None else str(c).strip() for c in self.ins_codes], dtype=object
            ).reshape(-1)
            if len(self.ins_codes) != n:
                raise ValueError(f"ins_codes has length {len(self.ins_codes)}, expected {n}")

    # ---- derived -------------------------------------------------------
    def __len__(self) -> int:
        return len(self.coords)

    @property
    def chains(self) -> list[str]:
        seen: list[str] = []
        for c in self.chain_ids:
            if c not in seen:
                seen.append(c)
        return seen

    def polymer_mask(self) -> np.ndarray:
        return np.isin(self.res_names, list(PROTEIN_RES | NA_RES)) & ~self.hetero

    def protein_mask(self) -> np.ndarray:
        return np.isin(self.res_names, list(PROTEIN_RES))

    def ligand_mask(self) -> np.ndarray:
        return ~np.isin(self.res_names, list(PROTEIN_RES | NA_RES | WATER))

    def water_mask(self) -> np.ndarray:
        return np.isin(self.res_names, list(WATER))

    def heavy_mask(self) -> np.ndarray:
        return self.elements != "H"

    def chain_mask(self, chain: str) -> np.ndarray:
        return self.chain_ids == chain

    def ca_mask(self) -> np.ndarray:
        return (self.atom_names == "CA") & self.protein_mask()

    def rep_mask(self) -> np.ndarray:
        """One representative atom per residue: CA for protein, C1' for NA, all heavy atoms for ligands."""
        m = self.ca_mask() | ((self.atom_names == "C1'") & np.isin(self.res_names, list(NA_RES)))
        return m | (self.ligand_mask() & self.heavy_mask() & ~self.water_mask())

    def residue_keys(self) -> list[tuple[str, int, str]]:
        """Residue keys (chain, resid, insertion code) in file order, one per residue."""
        keys: list[tuple[str, int, str]] = []
        last = None
        for c, r, ic in zip(self.chain_ids, self.res_ids, self.ins_codes):
            k = (str(c), int(r), str(ic))
            if k != last:
                keys.append(k)
                last = k
        return keys

    def sequence(self, chain: str) -> str:
        m = self.chain_mask(chain) & self.ca_mask()
        return "".join(THREE2ONE.get(r, "X") for r in self.res_names[m])

    def subset(self, mask: np.ndarray) -> Structure:
        return Structure(
            self.coords[mask], self.atom_names[mask], self.elements[mask], self.res_names[mask],
            self.res_ids[mask], self.chain_ids[mask], self.bfactor[mask], self.hetero[mask], self.source,
            self.ins_codes[mask], self.experimental,
        )

    def per_residue_bfactor(self) -> ResidueDict:
        """Mean B-factor per residue keyed by (chain, resid, icode).

        Lookups with a legacy 2-tuple (chain, resid) resolve to the residue without an
        insertion code, so older callers keep working.
        """
        out: dict[tuple[str, int, str], list[float]] = {}
        for c, r, ic, b in zip(self.chain_ids, self.res_ids, self.ins_codes, self.bfactor):
            out.setdefault((str(c), int(r), str(ic)), []).append(float(b))
        return ResidueDict((k, float(np.mean(v))) for k, v in out.items())

    def pdb_chain_map(self) -> dict[str, str]:
        """Deterministic map from chain id to a single-character PDB chain id.

        Single-character ids are kept; longer ids get the first unused character of
        A-Z, a-z, 0-9 in order of first appearance. Raises ValueError for > 62 chains.
        """
        chains = [str(c) for c in self.chains]
        if len(chains) > len(PDB_CHAIN_ALPHABET):
            raise ValueError(f"{len(chains)} chains cannot be written to PDB format (max {len(PDB_CHAIN_ALPHABET)})")
        used = {c for c in chains if len(c) <= 1}
        free = (ch for ch in PDB_CHAIN_ALPHABET if ch not in used)
        mapping: dict[str, str] = {}
        for c in chains:
            if len(c) <= 1:
                mapping[c] = c
            else:
                try:
                    mapping[c] = next(free)
                except StopIteration:  # only reachable with exotic single-char ids
                    raise ValueError("ran out of single-character chain ids for PDB output") from None
        return mapping

    # ---- io ------------------------------------------------------------
    def write_pdb(self, path: str | Path) -> dict[str, str]:
        """Write PDB; returns {original chain id: written chain id} (identity for 1-char ids)."""
        cmap = self.pdb_chain_map()
        lines = []
        serial = 1
        for i in range(len(self)):
            rec = "HETATM" if self.hetero[i] else "ATOM  "
            an = self.atom_names[i]
            an_fmt = f" {an:<3}" if len(an) < 4 and not an[0].isdigit() else f"{an:<4}"
            x, y, z = self.coords[i]
            lines.append(
                f"{rec}{serial:5d} {an_fmt} {self.res_names[i]:>3} {cmap[str(self.chain_ids[i])]:1}"
                f"{int(self.res_ids[i]):4d}{str(self.ins_codes[i])[:1]:1}   "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{float(self.bfactor[i]):6.2f}          {self.elements[i]:>2}"
            )
            serial += 1
        lines.append("END")
        Path(path).write_text("\n".join(lines) + "\n")
        return cmap


class ResidueDict(dict):
    """dict keyed by (chain, resid, icode) that also accepts legacy (chain, resid) keys."""

    @staticmethod
    def _norm(key):
        if isinstance(key, tuple) and len(key) == 2:
            return (key[0], key[1], "")
        return key

    def __missing__(self, key):
        k = self._norm(key)
        if k is not key and dict.__contains__(self, k):
            return dict.__getitem__(self, k)
        raise KeyError(key)

    def __contains__(self, key) -> bool:
        return dict.__contains__(self, key) or dict.__contains__(self, self._norm(key))

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


# ---- readers -------------------------------------------------------------

# method strings that mark a computed model rather than an experiment
_PREDICTED_METHODS = ("PREDICTED", "THEORETICAL MODEL", "COMPUTED", "COMPUTATIONAL", "IN SILICO")


def _is_experimental_method(method: str) -> bool:
    m = method.strip().strip("'\"").upper()
    if not m or m in {".", "?"}:
        return False
    return not any(k in m for k in _PREDICTED_METHODS)


def _cif_is_experimental(lines: list[str]) -> bool:
    """mmCIF header check: ``_exptl.method`` present and not a prediction, and the file is not
    ModelCIF (``_ma_*`` categories) / declared computational. AF3, Boltz, Chai-1 and AF-DB
    models therefore stay non-experimental."""
    method_seen, methods = False, []
    pending = False
    for raw in lines:
        line = raw.strip()
        if line.startswith("_ma_"):
            return False
        if line.startswith("_struct.pdbx_structure_determination_methodology"):
            toks = _cif_tokens(line)
            if len(toks) > 1 and toks[1].lower() in {"computational", "integrative"}:
                return False
        if pending:  # value on the next line (long values / semicolon text)
            if line and not line.startswith("_"):
                methods.append(line.lstrip(";"))
            pending = False
        if line.startswith("_exptl.method"):
            method_seen = True
            toks = _cif_tokens(line)
            if len(toks) > 1:
                methods.append(toks[1])
            elif line.split()[0] == "_exptl.method":
                pending = True  # loop column or value on next line
        if line.startswith("_atom_site.") and method_seen:
            break
    if not method_seen:
        return False
    methods = [m for m in methods if m.strip()]
    return all(_is_experimental_method(m) for m in methods) if methods else True

def _guess_element(atom_name: str, res_name: str) -> str:
    an = atom_name.strip()
    if not an:
        return "X"
    if len(an) >= 2 and an[:2].upper() in {"FE", "ZN", "MG", "MN", "CA", "NA", "CL", "BR", "CU", "NI", "CO", "SE"} and res_name not in PROTEIN_RES:
        return an[:2].upper()
    for ch in an:
        if ch.isalpha():
            return ch.upper()
    return "X"


def read_pdb(path: str | Path) -> Structure:
    coords, names, elems, resn, resi, chains, bf, het, icode = [], [], [], [], [], [], [], [], []
    methods: list[str] = []
    for line in Path(path).read_text().splitlines():
        rec = line[:6]
        if rec not in ("ATOM  ", "HETATM"):
            if rec.startswith("ENDMDL"):
                break
            if rec == "EXPDTA":
                methods.extend(m for m in line[10:].replace(";", ",").split(",") if m.strip())
            continue
        altloc = line[16]
        if altloc not in (" ", "A"):
            continue
        an = line[12:16].strip()
        rn = line[17:20].strip()
        el = line[76:78].strip().upper() if len(line) >= 78 else ""
        coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        names.append(an)
        elems.append(el or _guess_element(an, rn))
        resn.append(rn)
        resi.append(int(line[22:26]))
        icode.append(line[26].strip() if len(line) > 26 else "")
        chains.append(line[21].strip() or "A")
        bf.append(float(line[60:66]) if line[60:66].strip() else 0.0)
        het.append(rec == "HETATM")
    st = _build(coords, names, elems, resn, resi, chains, bf, het, str(path), icode)
    st.experimental = bool(methods) and all(_is_experimental_method(m) for m in methods)
    return st


def _cif_tokens(line: str) -> list[str]:
    """Tokenize an mmCIF data line respecting quotes."""
    out, i, n = [], 0, len(line)
    while i < n:
        c = line[i]
        if c.isspace():
            i += 1
            continue
        if c in ("'", '"'):
            j = line.find(c, i + 1)
            while j != -1 and j + 1 < n and not line[j + 1].isspace():
                j = line.find(c, j + 1)
            if j == -1:
                j = n
            out.append(line[i + 1:j])
            i = j + 1
        else:
            j = i
            while j < n and not line[j].isspace():
                j += 1
            out.append(line[i:j])
            i = j
    return out


def read_cif(path: str | Path) -> Structure:
    lines = Path(path).read_text().splitlines()
    cols: list[str] = []
    rows: list[list[str]] = []
    in_loop = False
    in_atom_site = False
    for raw in lines:
        line = raw.strip()
        if line == "loop_":
            in_loop, in_atom_site, cols = True, False, []
            continue
        if in_loop and line.startswith("_atom_site."):
            in_atom_site = True
            cols.append(line.split(".", 1)[1].split()[0])
            continue
        if in_atom_site:
            if not line or line.startswith("_") or line.startswith("#") or line == "loop_":
                if rows:
                    break
                in_loop = line == "loop_"
                in_atom_site = False
                cols = []
                continue
            toks = _cif_tokens(line)
            if len(toks) == len(cols):
                rows.append(toks)
    if not rows:
        raise ValueError(f"no _atom_site loop found in {path}")
    ci = {c: i for i, c in enumerate(cols)}

    def col(*names: str, default: str | None = None) -> int | None:
        for nme in names:
            if nme in ci:
                return ci[nme]
        return default

    ix = col("Cartn_x")
    iy = col("Cartn_y")
    iz = col("Cartn_z")
    i_an = col("label_atom_id", "auth_atom_id")
    i_el = col("type_symbol")
    i_rn = col("label_comp_id", "auth_comp_id")
    i_ri = col("auth_seq_id", "label_seq_id")
    i_ch = col("auth_asym_id", "label_asym_id")
    i_bf = col("B_iso_or_equiv")
    i_grp = col("group_PDB")
    i_alt = col("label_alt_id")
    i_model = col("pdbx_PDB_model_num")
    i_ic = col("pdbx_PDB_ins_code")
    coords, names, elems, resn, resi, chains, bf, het, icode = [], [], [], [], [], [], [], [], []
    first_model = None
    for r in rows:
        if i_model is not None:
            if first_model is None:
                first_model = r[i_model]
            elif r[i_model] != first_model:
                break
        if i_alt is not None and r[i_alt] not in (".", "?", "A"):
            continue
        an = r[i_an]
        rn = r[i_rn]
        coords.append((float(r[ix]), float(r[iy]), float(r[iz])))
        names.append(an)
        el = r[i_el].upper() if i_el is not None and r[i_el] not in (".", "?") else ""
        elems.append(el or _guess_element(an, rn))
        resn.append(rn)
        ri = r[i_ri]
        resi.append(int(ri) if ri not in (".", "?") else 0)
        ic = r[i_ic] if i_ic is not None else ""
        icode.append("" if ic in (".", "?") else ic)
        chains.append(r[i_ch])
        bf.append(float(r[i_bf]) if i_bf is not None and r[i_bf] not in (".", "?") else 0.0)
        het.append((r[i_grp] == "HETATM") if i_grp is not None else rn not in PROTEIN_RES | NA_RES)
    st = _build(coords, names, elems, resn, resi, chains, bf, het, str(path), icode)
    st.experimental = _cif_is_experimental(lines)
    return st


def _build(coords, names, elems, resn, resi, chains, bf, het, source, icode=None) -> Structure:
    return Structure(
        coords=np.asarray(coords, dtype=np.float32).reshape(-1, 3),
        atom_names=np.asarray(names, dtype=object),
        elements=np.asarray(elems, dtype=object),
        res_names=np.asarray(resn, dtype=object),
        res_ids=np.asarray(resi, dtype=np.int64),
        chain_ids=np.asarray(chains, dtype=object),
        bfactor=np.asarray(bf, dtype=np.float32),
        hetero=np.asarray(het, dtype=bool),
        source=source,
        ins_codes=None if icode is None else np.asarray(icode, dtype=object),
    )


def read_structure(path: str | Path) -> Structure:
    p = Path(path)
    suf = p.suffix.lower()
    if suf in {".cif", ".mmcif"}:
        return read_cif(p)
    if suf in {".pdb", ".ent"}:
        return read_pdb(p)
    # sniff the whole file: real mmCIF headers easily exceed a few kB before _atom_site
    text = p.read_text()
    is_cif = text.lstrip().startswith("data_") or "_atom_site." in text
    return read_cif(p) if is_cif else read_pdb(p)
