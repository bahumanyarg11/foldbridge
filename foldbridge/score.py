"""Backend-agnostic, numpy-only scoring of predicted complexes.

Everything here is vectorised numpy; ``scipy.spatial.cKDTree`` is used for
neighbour search when scipy happens to be installed, otherwise a numpy
cell-list (uniform grid) search is used. No other dependencies.

Metrics (keys of the dict returned by :func:`score_structure`)
---------------------------------------------------------------
Bookkeeping   n_atoms, n_residues, n_tokens, n_chains
pLDDT         plddt, plddt_<chain>, plddt_interface        (0-100, from B-factors;
              0-1 inputs are auto-detected and rescaled; skipped if all B == 0)
Interface     n_interface_residues, n_contacts             (residue pairs in different
              chains with any heavy-atom pair < 5 A)
Clashes       clash_count, clash_fraction, n_clash_residues (heavy-atom pairs < 2.2 A in
              different residues, excluding sequence neighbours i,i+1 in one chain,
              CYS SG-SG disulphides and metal coordination; clash_fraction = clashing
              atoms / heavy atoms)
Buried SASA   bsa, bsa_<chain>                              (Shrake-Rupley, 100 points,
              Bondi radii + 1.4 A probe; bsa = sum_c SASA(c) - SASA(complex), equal to
              SASA(A)+SASA(B)-SASA(AB) for two chains; bsa_<chain> is the area of that
              chain's own atoms buried on complex formation)
PAE           pae_interchain (mean over all inter-chain token pairs), ipae (mean over
              contacting inter-chain token pairs; falls back to pae_interchain when there
              is no contact), ipsae and ipsae_<A>_<B> (Dunbrack 2025, d0res variant,
              max over both directions)
pDockQ        pdockq (Bryant 2022; CB-CB < 8 A, CA for GLY)
Reference     rmsd, n_aligned (Kabsch CA/C1' RMSD over (chain, resid) intersection
              with identical residue names); for two-chain complexes also
              fnat, irmsd, lrmsd, dockq (DockQ-lite, see :func:`dockq_lite`)
"""
from __future__ import annotations

import json
import math
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .structure import NA_RES, Structure, read_structure

try:  # optional accelerator
    from scipy.spatial import cKDTree as _cKDTree  # type: ignore
except Exception:  # pragma: no cover - depends on environment
    _cKDTree = None

__all__ = [
    "score_structure", "score_result", "shrake_rupley", "neighbor_pairs", "kabsch",
    "kabsch_rmsd", "dockq_lite", "ipsae", "pdockq", "load_pae",
]

INTERFACE_CUTOFF = 5.0      # heavy-atom contact definition (interface + Fnat)
CLASH_CUTOFF = 2.2
PAE_CUTOFF = 10.0           # ipSAE
DOCKQ_IRMS_CUTOFF = 10.0    # interface residues for iRMSD (DockQ v1/v2 uses 10 A)
PROBE = 1.4
N_SPHERE = 100
MIN_REFERENCE_COVERAGE = 0.5  # warn when fewer model residues than this are matched in the reference
EXPERIMENTAL_NOTE = "experimental structure: B-factors are not pLDDT"

# Bondi-style vdW radii (A); unknown elements default to 1.80
VDW_RADII = {
    "H": 1.10, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "SE": 1.90,
    "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98, "B": 1.92, "SI": 2.10,
    "NA": 2.27, "K": 2.75, "MG": 1.73, "CA": 2.31, "ZN": 1.39, "FE": 1.94, "MN": 1.97,
    "CU": 1.40, "NI": 1.63, "CO": 1.92,
}
DEFAULT_RADIUS = 1.80

METALS = {"LI", "NA", "K", "RB", "CS", "MG", "CA", "SR", "BA", "MN", "FE", "CO", "NI", "CU", "ZN",
          "CD", "HG", "PT", "AU", "AG", "MO", "W", "V", "CR", "AL", "GA"}

BACKBONE = {"N", "CA", "C", "O", "P", "O5'", "C5'", "C4'", "C3'", "O3'", "C1'"}


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------

def _as_structure(s: str | Path | Structure) -> Structure:
    return s if isinstance(s, Structure) else read_structure(s)


def _clean(st: Structure) -> Structure:
    """Drop hydrogens and waters; PAE token order is unaffected (rep atoms are heavy)."""
    keep = (st.elements != "H") & (st.elements != "D") & ~st.water_mask()
    return st if keep.all() else st.subset(keep)


def _residue_index(st: Structure) -> np.ndarray:
    """(N,) int residue index; a new residue starts when chain, resid, insertion code or resname changes."""
    n = len(st)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    ch = st.chain_ids.astype(str)
    ri = st.res_ids
    ic = st.ins_codes.astype(str)
    rn = st.res_names.astype(str)
    new = np.ones(n, dtype=bool)
    new[1:] = (ch[1:] != ch[:-1]) | (ri[1:] != ri[:-1]) | (ic[1:] != ic[:-1]) | (rn[1:] != rn[:-1])
    return np.cumsum(new) - 1


def _first_per_group(idx: np.ndarray, n_groups: int) -> np.ndarray:
    first = np.full(n_groups, -1, dtype=np.int64)
    # reversed so that the first occurrence wins
    rev = np.arange(len(idx))[::-1]
    first[idx[rev]] = rev
    return first


def _group_mean(values: np.ndarray, groups: np.ndarray, n_groups: int) -> np.ndarray:
    s = np.bincount(groups, weights=values, minlength=n_groups)
    c = np.bincount(groups, minlength=n_groups)
    with np.errstate(invalid="ignore", divide="ignore"):
        return s / c


def _to_plddt100(b: np.ndarray) -> np.ndarray | None:
    """Return B-factors as pLDDT on a 0-100 scale, or None if they carry no pLDDT."""
    b = np.asarray(b, dtype=np.float64)
    if b.size == 0 or not np.isfinite(b).any() or np.allclose(b, 0.0):
        return None
    mx = np.nanmax(b)
    if mx <= 1.0 + 1e-6:
        return b * 100.0
    return b


# ---------------------------------------------------------------------------
# neighbour search
# ---------------------------------------------------------------------------

_OFFSETS = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
                    dtype=np.int64)


def _grid_pairs(X: np.ndarray, Y: np.ndarray, cutoff: float, block: int = 8192):
    """All (i, j) with |X_i - Y_j| < cutoff using a uniform cell list (numpy only)."""
    allp = np.concatenate([X, Y], axis=0)
    mn = allp.min(axis=0)
    cx = np.floor((X - mn) / cutoff).astype(np.int64) + 1
    cy = np.floor((Y - mn) / cutoff).astype(np.int64) + 1
    dims = np.maximum(cx.max(axis=0), cy.max(axis=0)) + 2
    mult = np.array([dims[1] * dims[2], dims[2], 1], dtype=np.int64)
    ky = cy @ mult
    order = np.argsort(ky, kind="stable")
    ks = ky[order]
    off_keys = _OFFSETS @ mult
    c2 = cutoff * cutoff
    out_i, out_j = [], []
    for b0 in range(0, len(X), block):
        xb = X[b0:b0 + block]
        kx = cx[b0:b0 + block] @ mult
        for ok in off_keys:
            nk = kx + ok
            lo = np.searchsorted(ks, nk, side="left")
            hi = np.searchsorted(ks, nk, side="right")
            cnt = hi - lo
            tot = int(cnt.sum())
            if tot == 0:
                continue
            ii = np.repeat(np.arange(len(xb)), cnt)
            start = np.repeat(lo - (np.cumsum(cnt) - cnt), cnt)
            pos = start + np.arange(tot)
            jj = order[pos]
            d = xb[ii] - Y[jj]
            m = np.einsum("ij,ij->i", d, d) < c2
            out_i.append(ii[m] + b0)
            out_j.append(jj[m])
    if not out_i:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(out_i), np.concatenate(out_j)


def neighbor_pairs(X: np.ndarray, cutoff: float, Y: np.ndarray | None = None,
                   use_scipy: bool | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Index pairs closer than ``cutoff``.

    With ``Y is None`` returns unique pairs i < j within X; otherwise all (i in X, j in Y).
    Uses scipy's cKDTree when available (``use_scipy=None``), else a numpy cell list.
    """
    X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
    Yv = None if Y is None else np.asarray(Y, dtype=np.float64).reshape(-1, 3)
    empty = (np.zeros(0, np.int64), np.zeros(0, np.int64))
    if len(X) == 0 or (Yv is not None and len(Yv) == 0):
        return empty
    if use_scipy is None:
        use_scipy = _cKDTree is not None
    if use_scipy and _cKDTree is not None:
        tx = _cKDTree(X)
        if Yv is None:
            p = tx.query_pairs(cutoff, output_type="ndarray")
            i, j = (p[:, 0].astype(np.int64), p[:, 1].astype(np.int64)) if len(p) else empty
        else:
            sm = tx.sparse_distance_matrix(_cKDTree(Yv), cutoff, output_type="ndarray")
            i, j = sm["i"].astype(np.int64), sm["j"].astype(np.int64)
        # cKDTree is inclusive (<=); enforce strict inequality
        Yr = X if Yv is None else Yv
        d = X[i] - Yr[j]
        m = np.einsum("ij,ij->i", d, d) < cutoff * cutoff
        i, j = i[m], j[m]
    else:
        i, j = _grid_pairs(X, X if Yv is None else Yv, cutoff)
        if Yv is None:
            m = i < j
            i, j = i[m], j[m]
    o = np.lexsort((j, i))
    return i[o], j[o]


# ---------------------------------------------------------------------------
# geometry: Kabsch, SASA
# ---------------------------------------------------------------------------

def kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation R and translation t minimising |(P @ R.T + t) - Q|."""
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    pc, qc = P.mean(axis=0), Q.mean(axis=0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d if d != 0 else 1.0])
    R = Vt.T @ D @ U.T
    return R, qc - pc @ R.T


def _rmsd(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((np.asarray(A) - np.asarray(B)) ** 2, axis=1))))


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """RMSD after optimal superposition of P onto Q."""
    R, t = kabsch(P, Q)
    return _rmsd(np.asarray(P, dtype=np.float64) @ R.T + t, Q)


def _sphere_points(n: int = N_SPHERE) -> np.ndarray:
    """Golden-spiral quasi-uniform unit sphere points (deterministic)."""
    k = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * k / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * k
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def _radii(elements: np.ndarray) -> np.ndarray:
    return np.array([VDW_RADII.get(str(e).upper(), DEFAULT_RADIUS) for e in elements], dtype=np.float64)


def _burial(coords, rad, pi, pj, n_points, chunk=20000):
    """For directed pairs (pi -> pj) return a (len(pairs), n_points) bool: point of pi inside pj."""
    S = _sphere_points(n_points)
    out = np.zeros((len(pi), n_points), dtype=bool)
    for s in range(0, len(pi), chunk):
        a, b = pi[s:s + chunk], pj[s:s + chunk]
        pts = coords[a][:, None, :] + rad[a][:, None, None] * S[None, :, :]
        d = pts - coords[b][:, None, :]
        out[s:s + chunk] = np.einsum("pkx,pkx->pk", d, d) < (rad[b] ** 2)[:, None]
    return out


def _or_by_group(mask: np.ndarray, groups: np.ndarray, n: int, n_points: int) -> np.ndarray:
    """OR rows of ``mask`` into an (n, n_points) array by group id (groups need not be sorted)."""
    res = np.zeros((n, n_points), dtype=bool)
    if len(groups) == 0:
        return res
    o = np.argsort(groups, kind="stable")
    g = groups[o]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    res[g[starts]] = np.logical_or.reduceat(mask[o], starts, axis=0)
    return res


def shrake_rupley(coords: np.ndarray, radii: np.ndarray, n_points: int = N_SPHERE,
                  probe: float = PROBE) -> np.ndarray:
    """Per-atom solvent-accessible surface area (A^2), Shrake-Rupley, numpy only."""
    X = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    rad = np.asarray(radii, dtype=np.float64) + probe
    n = len(X)
    if n == 0:
        return np.zeros(0)
    i, j = neighbor_pairs(X, 2.0 * rad.max())
    m = np.linalg.norm(X[i] - X[j], axis=1) < rad[i] + rad[j]
    i, j = i[m], j[m]
    pi, pj = np.r_[i, j], np.r_[j, i]
    buried = _or_by_group(_burial(X, rad, pi, pj, n_points), pi, n, n_points)
    return 4.0 * np.pi * rad ** 2 * (1.0 - buried.mean(axis=1))


def _buried_sasa(X, elements, chain_idx, n_points=N_SPHERE, probe=PROBE):
    """Per-atom SASA lost on complexation (A^2); only atoms near another chain are computed."""
    n = len(X)
    rad = _radii(elements) + probe
    if n == 0:
        return np.zeros(0)
    i, j = neighbor_pairs(X, 2.0 * rad.max())
    m = np.linalg.norm(X[i] - X[j], axis=1) < rad[i] + rad[j]
    i, j = i[m], j[m]
    pi, pj = np.r_[i, j], np.r_[j, i]
    inter = chain_idx[pi] != chain_idx[pj]
    shell = np.zeros(n, dtype=bool)
    shell[pi[inter]] = True
    keep = shell[pi]
    pi, pj, inter = pi[keep], pj[keep], inter[keep]
    if len(pi) == 0:
        return np.zeros(n)
    bur = _burial(X, rad, pi, pj, n_points)
    b_cplx = _or_by_group(bur, pi, n, n_points)
    b_alone = _or_by_group(bur[~inter], pi[~inter], n, n_points)
    area = 4.0 * np.pi * rad ** 2 / n_points
    return area * (b_cplx.sum(axis=1) - b_alone.sum(axis=1))


# ---------------------------------------------------------------------------
# PAE
# ---------------------------------------------------------------------------

_PAE_KEYS = ("pae", "predicted_aligned_error", "pae_matrix", "paes")


def load_pae(pae: Any) -> np.ndarray | None:
    """Load a square PAE matrix from an array, .npy, .npz or AF2/ColabFold/AF3 .json."""
    if pae is None:
        return None
    if isinstance(pae, np.ndarray):
        arr = pae
    else:
        p = Path(pae)
        suf = p.suffix.lower()
        if suf == ".npy":
            arr = np.load(p, allow_pickle=False)
        elif suf == ".npz":
            with np.load(p, allow_pickle=False) as z:
                key = next((k for k in _PAE_KEYS if k in z.files), z.files[0] if z.files else None)
                if key is None:
                    return None
                arr = z[key]
        elif suf == ".json":
            obj = json.loads(p.read_text())
            if isinstance(obj, list) and obj:
                obj = obj[0]
            arr = None
            if isinstance(obj, dict):
                for k in _PAE_KEYS:
                    if k in obj:
                        arr = np.asarray(obj[k], dtype=np.float64)
                        break
                if arr is None and "distance" in obj and "residue1" in obj:  # legacy AF-DB
                    n = int(max(obj["residue1"]))
                    arr = np.zeros((n, n))
                    arr[np.asarray(obj["residue1"]) - 1, np.asarray(obj["residue2"]) - 1] = obj["distance"]
            if arr is None:
                return None
        else:
            raise ValueError(f"unsupported PAE file: {p}")
    arr = np.asarray(arr, dtype=np.float64)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return None
    return arr


def _calc_d0(L: np.ndarray, nucleic: bool = False) -> np.ndarray:
    L = np.maximum(np.asarray(L, dtype=np.float64), 27.0)
    d0 = 1.24 * np.cbrt(L - 15.0) - 1.8
    return np.maximum(2.0 if nucleic else 1.0, d0)


def ipsae(pae: np.ndarray, tok_chain: np.ndarray, a: str, b: str, cutoff: float = PAE_CUTOFF,
          nucleic: bool = False) -> float:
    """ipSAE (Dunbrack 2025), d0res variant, max over the A->B and B->A directions.

    For each aligned residue i in A: valid j in B have PAE(i, j) < cutoff; d0 from
    N0 = #valid j (floored at 27); score_i = mean_j 1 / (1 + (PAE_ij / d0)^2). The
    directional score is max_i score_i; ipSAE = max over both directions.
    """
    best = 0.0
    for x, y in ((a, b), (b, a)):
        rows = np.flatnonzero(tok_chain == x)
        cols = np.flatnonzero(tok_chain == y)
        if len(rows) == 0 or len(cols) == 0:
            continue
        sub = pae[np.ix_(rows, cols)]
        valid = sub < cutoff
        n0 = valid.sum(axis=1)
        d0 = _calc_d0(n0, nucleic)
        ptm = 1.0 / (1.0 + (sub / d0[:, None]) ** 2)
        with np.errstate(invalid="ignore", divide="ignore"):
            per_res = np.where(n0 > 0, (ptm * valid).sum(axis=1) / np.maximum(n0, 1), 0.0)
        best = max(best, float(per_res.max()))
    return best


def pdockq(mean_if_plddt: float, n_contacts: int) -> float:
    """pDockQ (Bryant et al. 2022) sigmoid; 0 when there are no contacts."""
    if n_contacts <= 0 or not np.isfinite(mean_if_plddt):
        return 0.0
    x = mean_if_plddt * math.log10(n_contacts)
    return 0.724 / (1.0 + math.exp(-0.052 * (x - 152.611))) + 0.018


# ---------------------------------------------------------------------------
# reference comparison
# ---------------------------------------------------------------------------

def _atom_keys(st: Structure, mask: np.ndarray) -> dict[tuple, int]:
    """(chain, resid, ins_code, atom_name) -> atom index; insertion codes keep 184/184A apart."""
    idx = np.flatnonzero(mask)
    return {(str(st.chain_ids[k]), int(st.res_ids[k]), str(st.ins_codes[k]), str(st.atom_names[k])): int(k)
            for k in idx}


def _matched_atoms(model: Structure, ref: Structure, mask_m: np.ndarray, mask_r: np.ndarray,
                   check_names: bool = True) -> tuple[np.ndarray, np.ndarray]:
    km, kr = _atom_keys(model, mask_m), _atom_keys(ref, mask_r)
    common = [k for k in km if k in kr]
    if check_names:
        common = [k for k in common if model.res_names[km[k]] == ref.res_names[kr[k]]]
    return (np.array([km[k] for k in common], dtype=np.int64),
            np.array([kr[k] for k in common], dtype=np.int64))


def _polymer_rep(st: Structure) -> np.ndarray:
    return st.ca_mask() | ((st.atom_names == "C1'") & np.isin(st.res_names, list(NA_RES)))


def _residue_contacts(st: Structure, a: str, b: str, cutoff: float) -> set[tuple]:
    """Set of ((resid, icode)_a, (resid, icode)_b) with any heavy-atom pair < cutoff."""
    ma, mb = np.flatnonzero(st.chain_ids == a), np.flatnonzero(st.chain_ids == b)
    i, j = neighbor_pairs(st.coords[ma], cutoff, st.coords[mb])
    ic = st.ins_codes.astype(str)
    ra = zip(st.res_ids[ma][i].tolist(), ic[ma][i].tolist())
    rb = zip(st.res_ids[mb][j].tolist(), ic[mb][j].tolist())
    return set(zip(ra, rb))


def dockq_lite(model: Structure, ref: Structure, receptor: str, ligand: str) -> dict[str, float]:
    """Approximate DockQ (v1 definitions; v2 is identical for a single protein interface).

    Approximations vs the DockQ v2 package: chains are matched by chain id (no chain
    permutation search), residues by (chain, resid) with identical names (no sequence
    alignment; insertion codes are part of the residue key), single interface only, no small-molecule/NA special handling.

    * Fnat  - fraction of reference residue contacts (any heavy atom < 5 A) reproduced.
    * iRMSD - backbone RMSD over reference interface residues (any heavy atom < 10 A).
    * LRMSD - backbone RMSD of the ligand (smaller) chain after superposing receptor backbone.
    * DockQ = (Fnat + 1/(1+(iRMS/1.5)^2) + 1/(1+(LRMS/8.5)^2)) / 3.
    """
    out: dict[str, float] = {}
    keys_m = set(zip(model.chain_ids.astype(str).tolist(),
                     zip(model.res_ids.tolist(), model.ins_codes.astype(str).tolist())))
    keys_r = set(zip(ref.chain_ids.astype(str).tolist(),
                     zip(ref.res_ids.tolist(), ref.ins_codes.astype(str).tolist())))
    shared = keys_m & keys_r
    nat = {c for c in _residue_contacts(ref, receptor, ligand, INTERFACE_CUTOFF)
           if (receptor, c[0]) in shared and (ligand, c[1]) in shared}
    mod = _residue_contacts(model, receptor, ligand, INTERFACE_CUTOFF)
    out["fnat"] = len(nat & mod) / len(nat) if nat else 0.0

    bb_m = np.isin(model.atom_names, list(BACKBONE)) & model.polymer_mask()
    bb_r = np.isin(ref.atom_names, list(BACKBONE)) & ref.polymer_mask()

    # interface residues at 10 A in the reference
    if_res = _residue_contacts(ref, receptor, ligand, DOCKQ_IRMS_CUTOFF)
    if_keys = {(receptor, a) for a, _ in if_res} | {(ligand, b) for _, b in if_res}
    in_if_r = np.array([(str(c), (int(r), str(ic))) in if_keys
                        for c, r, ic in zip(ref.chain_ids, ref.res_ids, ref.ins_codes)], dtype=bool)
    im, ir = _matched_atoms(model, ref, bb_m & np.isin(model.chain_ids, [receptor, ligand]), bb_r & in_if_r)
    out["irmsd"] = kabsch_rmsd(model.coords[im], ref.coords[ir]) if len(im) >= 3 else float("nan")

    rm, rr = _matched_atoms(model, ref, bb_m & (model.chain_ids == receptor), bb_r & (ref.chain_ids == receptor))
    lm, lr = _matched_atoms(model, ref, bb_m & (model.chain_ids == ligand), bb_r & (ref.chain_ids == ligand))
    if len(rm) >= 3 and len(lm) >= 1:
        R, t = kabsch(model.coords[rm], ref.coords[rr])
        out["lrmsd"] = _rmsd(model.coords[lm].astype(np.float64) @ R.T + t, ref.coords[lr])
    else:
        out["lrmsd"] = float("nan")

    if np.isfinite(out["irmsd"]) and np.isfinite(out["lrmsd"]):
        out["dockq"] = (out["fnat"] + 1.0 / (1.0 + (out["irmsd"] / 1.5) ** 2)
                        + 1.0 / (1.0 + (out["lrmsd"] / 8.5) ** 2)) / 3.0
    return out


def _reference_metrics(model: Structure, ref: Structure, chains: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    rep_m, rep_r = _polymer_rep(model), _polymer_rep(ref)
    im, ir = _matched_atoms(model, ref, rep_m, rep_r, check_names=True)
    if len(im) < 3:  # residue names disagree (e.g. mutants) -> match on position only
        im, ir = _matched_atoms(model, ref, rep_m, rep_r, check_names=False)
    if len(im) >= 3:
        out["rmsd"] = kabsch_rmsd(model.coords[im], ref.coords[ir])
        out["n_aligned"] = int(len(im))
    n_model = int(rep_m.sum())
    if n_model and len(im) < MIN_REFERENCE_COVERAGE * n_model:
        warnings.warn(f"reference matched only {len(im)} of {n_model} residues; check chain ids/numbering",
                      stacklevel=3)
    poly_chains = [c for c in chains
                   if (model.polymer_mask() & (model.chain_ids == c)).any()
                   and (ref.polymer_mask() & (ref.chain_ids == c)).any()]
    if len(poly_chains) == 2:
        a, b = poly_chains
        na = int((rep_r & (ref.chain_ids == a)).sum())
        nb = int((rep_r & (ref.chain_ids == b)).sum())
        receptor, ligand = (a, b) if na >= nb else (b, a)
        out.update(dockq_lite(model, ref, receptor, ligand))
    return out


# ---------------------------------------------------------------------------
# main entry points
# ---------------------------------------------------------------------------

def _f(x) -> float:
    return float(x)


def _missing_chains_error(want: list[str], st: Structure) -> ValueError | None:
    have = [str(c) for c in st.chains]
    missing = [c for c in want if c not in have]
    if not missing:
        return None
    name = Path(st.source).name if st.source else "structure"
    what = "chain" if len(missing) == 1 else "chains"
    return ValueError(f"{what} {','.join(missing)} not found in {name} (available: {', '.join(have) or 'none'})")


def score_structure(structure: str | Path | Structure, pae: Any = None, reference: Any = None,
                    chains: Iterable[str] | None = None, contact_cutoff: float = 8.0) -> dict[str, Any]:
    """Compute backend-independent quality metrics for one predicted complex.

    Parameters
    ----------
    structure : path or Structure
    pae : NxN array or path (.npz/.npy/.json); N must equal the number of tokens
        (``Structure.rep_mask()`` atoms: CA / C1' per polymer residue, one per ligand heavy
        atom) or, failing that, the number of polymer residues. Mismatches are skipped
        with a warning.
    reference : path or Structure for RMSD / DockQ-lite (chains matched by id). A warning is
        issued when fewer than half of the model residues are matched in it.
    chains : restrict all metrics to these chain ids (e.g. a pair out of a multimer). Any id
        that is not in the structure raises ValueError.
    contact_cutoff : CB-CB distance used for pDockQ contacts (Bryant 2022 uses 8 A).

    For experimentally determined entries (``Structure.experimental``: PDB EXPDTA / mmCIF
    ``_exptl.method``) B-factors are not pLDDT: plddt*, plddt_interface and pdockq are left
    out and ``note`` says so. Counts (n_*, clash_count, pdockq_contacts) are ints.
    """
    full = _clean(_as_structure(structure))
    out: dict[str, Any] = {}
    experimental = bool(getattr(full, "experimental", False))
    if chains is not None:
        want = list(dict.fromkeys(str(c) for c in chains))
        err = _missing_chains_error(want, full)
        if err is not None:
            raise err

    # ---- tokens + PAE on the full structure, then restrict to requested chains
    tok_mask_full = full.rep_mask()
    tok_chain_full = full.chain_ids[tok_mask_full].astype(str)
    pae_arr = None
    if pae is not None:
        try:
            pae_arr = load_pae(pae)
        except Exception as e:  # unreadable PAE should not kill scoring
            warnings.warn(f"could not read PAE {pae!r}: {e}", stacklevel=2)
        if pae_arr is not None and pae_arr.shape[0] != len(tok_chain_full):
            poly_tok = (tok_mask_full & full.polymer_mask())[tok_mask_full]
            if pae_arr.shape[0] == int(poly_tok.sum()):
                tok_chain_full = tok_chain_full[poly_tok]
                tok_mask_full = tok_mask_full & full.polymer_mask()
            else:
                warnings.warn(f"PAE size {pae_arr.shape[0]} != {len(tok_chain_full)} tokens; skipping PAE metrics", stacklevel=2)
                pae_arr = None

    chain_list = [c for c in full.chains]
    if chains is not None:
        want = [str(c) for c in chains]
        chain_list = [c for c in chain_list if c in want]
        sel = np.isin(full.chain_ids.astype(str), chain_list)
        if pae_arr is not None:
            ts = np.isin(tok_chain_full, chain_list)
            pae_arr = pae_arr[np.ix_(ts, ts)]
            tok_chain_full = tok_chain_full[ts]
        tok_mask_full = tok_mask_full & sel
        st = full.subset(sel)
        tok_mask = tok_mask_full[sel]
    else:
        st, tok_mask = full, tok_mask_full

    n = len(st)
    ridx = _residue_index(st)
    n_res = int(ridx.max()) + 1 if n else 0
    chain_ids = st.chain_ids.astype(str)
    cmap = {c: k for k, c in enumerate(chain_list)}
    cidx = np.array([cmap[c] for c in chain_ids], dtype=np.int64)
    res_chain = cidx[_first_per_group(ridx, n_res)] if n else np.zeros(0, np.int64)
    out.update(n_atoms=int(n), n_residues=int(n_res), n_tokens=int(tok_mask.sum()), n_chains=len(chain_list))
    if experimental:
        out["note"] = EXPERIMENTAL_NOTE
    if n == 0:
        return out

    # ---- interface (heavy-atom < 5 A across chains)
    X = st.coords.astype(np.float64)
    i, j = neighbor_pairs(X, max(INTERFACE_CUTOFF, CLASH_CUTOFF))
    d = np.linalg.norm(X[i] - X[j], axis=1)
    inter = cidx[i] != cidx[j]
    ii, jj = ridx[i[inter & (d < INTERFACE_CUTOFF)]], ridx[j[inter & (d < INTERFACE_CUTOFF)]]
    lo, hi = np.minimum(ii, jj), np.maximum(ii, jj)
    res_pairs = np.unique(lo * n_res + hi) if len(lo) else np.zeros(0, np.int64)
    rp_a, rp_b = res_pairs // max(n_res, 1), res_pairs % max(n_res, 1)
    if_res = np.zeros(n_res, dtype=bool)
    if_res[rp_a] = True
    if_res[rp_b] = True
    out["n_contacts"] = int(len(res_pairs))
    out["n_interface_residues"] = int(if_res.sum())

    # ---- clashes
    close = d < CLASH_CUTOFF
    ci, cj = i[close], j[close]
    ri, rj = ridx[ci], ridx[cj]
    same_chain = cidx[ci] == cidx[cj]
    bonded = (ri == rj) | (same_chain & (np.abs(ri - rj) == 1))
    disulf = (st.atom_names[ci] == "SG") & (st.atom_names[cj] == "SG")
    metal = np.isin(st.elements[ci], list(METALS)) | np.isin(st.elements[cj], list(METALS))
    clash = ~bonded & ~disulf & ~metal
    ci, cj = ci[clash], cj[clash]
    out["clash_count"] = int(len(ci))
    clashing_atoms = np.unique(np.r_[ci, cj])
    out["clash_fraction"] = _f(len(clashing_atoms) / n)
    out["n_clash_residues"] = int(len(np.unique(ridx[clashing_atoms])))

    # ---- pLDDT (0-100)
    plddt_atom = None if experimental else _to_plddt100(st.bfactor)
    tok_plddt = None
    res_plddt = None
    if plddt_atom is not None:
        tok_plddt = plddt_atom[tok_mask]
        res_plddt = _group_mean(plddt_atom, ridx, n_res)
        src = tok_plddt if len(tok_plddt) else res_plddt
        out["plddt"] = _f(np.mean(src))
        tok_c = cidx[tok_mask]
        for k, c in enumerate(chain_list):
            vals = tok_plddt[tok_c == k] if len(tok_plddt) else res_plddt[res_chain == k]
            if len(vals):
                out[f"plddt_{c}"] = _f(vals.mean())
        if if_res.any():
            out["plddt_interface"] = _f(res_plddt[if_res].mean())

    # ---- buried SASA
    if len(chain_list) >= 2:
        dsasa = _buried_sasa(X, st.elements, cidx)
        out["bsa"] = _f(dsasa.sum())
        per_chain = np.bincount(cidx, weights=dsasa, minlength=len(chain_list))
        for k, c in enumerate(chain_list):
            out[f"bsa_{c}"] = _f(per_chain[k])

    # ---- pDockQ (protein CB, CA for GLY / missing CB)
    prot = st.protein_mask()
    cb = prot & (st.atom_names == "CB")
    ca = prot & (st.atom_names == "CA")
    has_cb = np.zeros(n_res, dtype=bool)
    has_cb[ridx[cb]] = True
    beta = np.flatnonzero(cb | (ca & ~has_cb[ridx]))
    if len(chain_list) >= 2 and len(beta):
        bi, bj = neighbor_pairs(X[beta], contact_cutoff)
        m = cidx[beta][bi] != cidx[beta][bj]
        bi, bj = beta[bi[m]], beta[bj[m]]
        n_ct = len(bi)
        out["pdockq_contacts"] = int(n_ct)
        if plddt_atom is not None:
            if n_ct:
                if_atoms = np.unique(np.r_[bi, bj])
                if_pl = float(plddt_atom[if_atoms].mean())
            else:
                if_pl = float("nan")
            out["pdockq"] = pdockq(if_pl, n_ct)

    # ---- PAE-derived
    if pae_arr is not None and len(chain_list) >= 2:
        tc = tok_chain_full
        diff = tc[:, None] != tc[None, :]
        out["pae_interchain"] = _f(pae_arr[diff].mean()) if diff.any() else float("nan")
        # contacting token pairs: tokens whose residues form a 5 A residue contact
        tok_res = ridx[tok_mask]
        if len(res_pairs) and len(tok_res) == len(tc):
            key = (np.minimum(tok_res[:, None], tok_res[None, :]) * n_res
                   + np.maximum(tok_res[:, None], tok_res[None, :]))
            pos = np.clip(np.searchsorted(res_pairs, key), 0, len(res_pairs) - 1)
            cm = (res_pairs[pos] == key) & diff
            out["ipae"] = _f(pae_arr[cm].mean()) if cm.any() else out["pae_interchain"]
        out.setdefault("ipae", out["pae_interchain"])
        best = 0.0
        is_na = {c: bool(np.isin(st.res_names[chain_ids == c], list(NA_RES)).any()) for c in chain_list}
        for x in range(len(chain_list)):
            for y in range(x + 1, len(chain_list)):
                a, b = chain_list[x], chain_list[y]
                v = ipsae(pae_arr, tc, a, b, nucleic=is_na[a] or is_na[b])
                out[f"ipsae_{a}_{b}"] = v
                best = max(best, v)
        out["ipsae"] = best

    # ---- reference
    if reference is not None:
        ref = _clean(_as_structure(reference))
        if chains is not None:
            ref = ref.subset(np.isin(ref.chain_ids.astype(str), chain_list))
        out.update(_reference_metrics(st, ref, chain_list))

    return out


def score_result(result, reference: Any = None, **kwargs) -> dict[str, Any]:
    """Score a backends.base.ModelResult; merges backend-native scores with computed metrics.

    Computed metrics win on key collisions; the backend value is kept as ``<key>_backend``.
    A backend-only ``plddt`` on a 0-1 scale is rescaled to 0-100.
    """
    row: dict[str, Any] = {
        "id": f"{result.backend}_s{result.seed}_m{result.sample}",
        "backend": result.backend,
        "seed": int(result.seed),
        "sample": int(result.sample),
        "path": str(result.path),
    }
    for k, v in (result.scores or {}).items():
        try:
            row[k] = float(v)
        except (TypeError, ValueError):
            row[k] = v
    pae = result.pae_path if result.pae_path is not None and Path(result.pae_path).exists() else None
    computed = score_structure(result.path, pae=pae, reference=reference, **kwargs)
    for k, v in computed.items():
        if k in row and k not in ("id", "backend", "seed", "sample", "path") and row[k] != v:
            row[f"{k}_backend"] = row[k]
        row[k] = v
    if "plddt" in row and "plddt" not in computed and isinstance(row["plddt"], float) and row["plddt"] <= 1.0:
        row["plddt"] = row["plddt"] * 100.0
    return row
