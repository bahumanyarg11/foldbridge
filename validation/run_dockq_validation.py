"""Validate foldbridge DockQ-lite against the official DockQ v2 package.

Steps (all deterministic, seed 20261009):
  1. Extract two-chain natives (protein heavy atoms only) from tests/data:
     1BRS A+D (barnase-barstar), 2PTC E+I (trypsin-BPTI, has insertion codes 184A/188A/221A),
     1A2K A+D (NTF2-Ran).
  2. Generate ~60 decoys per native: rigid-body perturbation of the smaller (ligand) chain
     (rotation 0-60 deg about its centroid, translation 0-15 A) + Gaussian coordinate noise on
     all atoms; plus noise-only, "interface-swapped" (ligand moved to the opposite face of the
     receptor) and "far" (ligand pulled 25-40 A away) decoys.
  3. Score every decoy with the official DockQ v2 python API (explicit identity chain map) and
     with foldbridge.score.score_structure(decoy, reference=native).
  4. Write validation/dockq_agreement.csv, validation/dockq_summary.json,
     validation/dockq_scatter.png and the small regression fixtures in validation/fixtures/.

Decoy PDBs are written to validation/_work/ (git-ignored, regenerable).

Official DockQ runs in its own venv (DockQ 2.1.x pins numpy<2; foldbridge uses numpy 2):
    python3 -m venv validation/.dockq-venv && validation/.dockq-venv/bin/pip install DockQ==2.1.3
    .venv/bin/pip install matplotlib
    .venv/bin/python validation/run_dockq_validation.py      # override with DOCKQ_PYTHON=...
The regression test (tests/test_validation_dockq.py) needs neither.
"""
from __future__ import annotations

import csv
import dataclasses
import gzip
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from foldbridge.score import score_structure  # noqa: E402
from foldbridge.structure import read_structure  # noqa: E402

VAL = ROOT / "validation"
WORK = VAL / "_work"
FIX = VAL / "fixtures"
DOCKQ_PYTHON = os.environ.get("DOCKQ_PYTHON", str(VAL / ".dockq-venv" / "bin" / "python"))
SEED = 20261009

NATIVES = [  # name, source file, receptor (larger), ligand (smaller)
    ("1BRS", "1BRS.pdb", "A", "D"),
    ("2PTC", "2PTC.pdb", "E", "I"),
    ("1A2K", "1A2K.pdb", "D", "A"),
]
N_GRADED = 50
CAPRI = [(0.23, "incorrect"), (0.49, "acceptable"), (0.80, "medium"), (math.inf, "high")]


def capri_class(d: float) -> str:
    if d is None or not np.isfinite(d):
        return "nan"
    for thr, name in CAPRI:
        if d < thr:
            return name
    return "high"


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def rotmat(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    t = math.radians(angle_deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K


def unit(rng) -> np.ndarray:
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def make_decoy(nat, lig_mask, rng, angle, trans, noise, about=None):
    """Rotate ligand by `angle` about `about` (default: ligand centroid), translate, add noise."""
    X = nat.coords.astype(np.float64).copy()
    L = X[lig_mask]
    c = L.mean(axis=0) if about is None else np.asarray(about, float)
    R = rotmat(unit(rng), angle)
    X[lig_mask] = (L - c) @ R.T + c + unit(rng) * trans
    if noise > 0:
        X += rng.normal(scale=noise, size=X.shape)
    return dataclasses.replace(nat, coords=X.astype(np.float32))


# ---------------------------------------------------------------------------
# official DockQ (subprocess in its own venv)
# ---------------------------------------------------------------------------

def run_official(jobs: list[dict]) -> dict:
    jp, op = WORK / "official_jobs.json", WORK / "official_out.json"
    jp.write_text(json.dumps(jobs))
    subprocess.run([DOCKQ_PYTHON, str(VAL / "official_dockq_batch.py"), str(jp), str(op)], check=True)
    return json.loads(op.read_text())


def renumbered(st):
    """Copy with residues renumbered 1..n per chain and no insertion codes."""
    new = np.zeros(len(st), dtype=np.int64)
    for c in st.chains:
        m = np.flatnonzero(st.chain_ids == c)
        keys = list(zip(st.res_ids[m].tolist(), st.ins_codes[m].tolist()))
        first = {}
        for k in keys:
            first.setdefault(k, len(first) + 1)
        new[m] = [first[k] for k in keys]
    return dataclasses.replace(st, res_ids=new, ins_codes=np.full(len(st), "", dtype=object))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(SEED)
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    FIX.mkdir(parents=True, exist_ok=True)
    meta = []
    native_paths = {}
    for name, src, rec, lig in NATIVES:
        full = read_structure(ROOT / "tests" / "data" / src)
        keep = (full.protein_mask() & ~full.hetero & (full.elements != "H")
                & np.isin(full.chain_ids.astype(str), [rec, lig]))
        nat = full.subset(keep)
        npath = WORK / f"{name}_native.pdb"
        cmap = nat.write_pdb(npath)
        assert cmap is None or all(k == v for k, v in cmap.items()), cmap
        nat = read_structure(npath)  # exactly what both tools see
        native_paths[name] = npath
        lig_mask = nat.chain_ids == lig

        rec_c = nat.coords[nat.chain_ids == rec].astype(float).mean(axis=0)
        specs = []
        # graded rigid-body perturbations; severity s spread over [0,1] so DockQ spans 0-1
        for k in range(N_GRADED):
            s = (k + rng.uniform()) / N_GRADED
            angle = 60.0 * s * rng.uniform(0.6, 1.0)
            trans = 15.0 * s * rng.uniform(0.6, 1.0)
            noise = rng.uniform(0.0, 0.6)
            specs.append(("graded", dict(angle=angle, trans=trans, noise=noise)))
        for noise in (0.25, 0.5, 1.0):  # noise only (near-native, high quality)
            specs.append(("noise", dict(angle=0.0, trans=0.0, noise=noise)))
        for _ in range(4):  # ligand rotated 90-180 deg about the receptor centroid -> other face
            specs.append(("swapped", dict(angle=rng.uniform(90, 180), trans=0.0, noise=0.3, about=rec_c)))
        for _ in range(3):  # far: no contacts
            specs.append(("far", dict(angle=rng.uniform(0, 180), trans=rng.uniform(25, 40), noise=0.3)))

        for k, (kind, kw) in enumerate(specs):
            dec = make_decoy(nat, lig_mask, rng, **kw)
            did = f"{name}_{k:02d}_{kind}"
            dpath = WORK / f"{did}.pdb"
            dec.write_pdb(dpath)
            meta.append({"id": did, "native": name, "kind": kind, "receptor": rec, "ligand": lig,
                         "rot_deg": round(kw["angle"], 3), "trans_A": round(kw["trans"], 3),
                         "noise_A": round(kw["noise"], 3)})
            if name == "2PTC":  # control: same coordinates, sequential numbering, no insertion codes
                renumbered(dec).write_pdb(WORK / f"{did}_renum.pdb")
        if name == "2PTC":
            renumbered(nat).write_pdb(WORK / f"{name}_native_renum.pdb")

    # ---- official DockQ v2 (one subprocess for all decoys; CLI cross-check on every 37th)
    jobs = [{"id": m["id"], "model": str(WORK / f"{m['id']}.pdb"), "native": str(native_paths[m["native"]]),
             "receptor": m["receptor"], "ligand": m["ligand"], "cli": i % 37 == 0} for i, m in enumerate(meta)]
    # secondary check: every 6th decoy scored against the *raw* PDB entry (all chains, waters,
    # HETATM ligands such as GDP/Mg on Ran or Ca2+ on trypsin, altlocs) -- what a user would pass
    src_of = {n: ROOT / "tests" / "data" / f for n, f, _, _ in NATIVES}
    raw_ids = [m["id"] for m in meta[::6]]
    jobs += [{"id": f"{m['id']}@raw", "model": str(WORK / f"{m['id']}.pdb"), "native": str(src_of[m["native"]]),
              "receptor": m["receptor"], "ligand": m["ligand"], "cli": False} for m in meta[::6]]
    official = run_official(jobs)
    dockq_version = official["dockq_version"]
    official = official["results"]

    rows, control = [], []
    for m in meta:
        did, name = m["id"], m["native"]
        off = official[did]
        fb = score_structure(WORK / f"{did}.pdb", reference=native_paths[name])
        row = dict(m)
        for k in ("fnat", "irmsd", "lrmsd", "dockq"):
            row[f"{k}_official"] = float("nan") if off[k] is None else off[k]
            row[f"{k}_foldbridge"] = float(fb.get(k, float("nan")))
        row["capri_official"] = capri_class(row["dockq_official"])
        row["capri_foldbridge"] = capri_class(row["dockq_foldbridge"])
        rows.append(row)
        if name == "2PTC":
            fr = score_structure(WORK / f"{did}_renum.pdb", reference=WORK / f"{name}_native_renum.pdb")
            control.append(abs(fr["dockq"] - row["dockq_official"]))
        print(f"{did:24s} DockQ off={row['dockq_official']:.4f} fb={row['dockq_foldbridge']:.4f}", flush=True)
    raw_check = []
    by_id = {r["id"]: r for r in rows}
    for did in raw_ids:
        r = by_id[did]
        off = official[f"{did}@raw"]
        fb = score_structure(WORK / f"{did}.pdb", reference=src_of[r["native"]])
        raw_check.append({"id": did, **{f"{k}_official_raw": off[k] for k in ("fnat", "irmsd", "lrmsd", "dockq")},
                          **{f"{k}_foldbridge_raw": float(fb.get(k, float("nan"))) for k in ("fnat", "irmsd", "lrmsd", "dockq")}})
    cli_checks = [{"id": j["id"], "dockq_cli_3dp": official[j["id"]]["dockq_cli"],
                   "dockq_api": official[j["id"]]["dockq"]} for j in jobs if j["cli"]]

    # ---- csv
    fields = list(rows[0].keys())
    with open(VAL / "dockq_agreement.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})

    summary = summarize(rows)
    summary["cli_vs_api"] = cli_checks
    summary["n_decoys"] = len(rows)
    summary["dockq_version"] = dockq_version
    summary["raw_native_check"] = {
        "n": len(raw_check),
        **{f"{k}_max_abs_diff": float(max(abs(c[f"{k}_official_raw"] - c[f"{k}_foldbridge_raw"]) for c in raw_check))
           for k in ("fnat", "irmsd", "lrmsd", "dockq")},
        "official_raw_vs_clean_dockq_max_abs_diff": float(max(abs(c["dockq_official_raw"] - by_id[c["id"]]["dockq_official"])
                                                              for c in raw_check)),
        "rows": raw_check}
    summary["control_2PTC_renumbered_dockq_max_abs_diff"] = float(max(control))
    summary["control_2PTC_renumbered_dockq_mean_abs_diff"] = float(np.mean(control))
    (VAL / "dockq_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    plot(rows, summary)
    write_fixtures(rows, native_paths)


def summarize(rows) -> dict:
    def pearson(x, y):
        return float(np.corrcoef(x, y)[0, 1])

    def ranks(x):  # average ranks for ties
        o = np.argsort(x, kind="mergesort")
        r = np.empty(len(x))
        r[o] = np.arange(len(x))
        for v in np.unique(x):
            m = x == v
            r[m] = r[m].mean()
        return r

    out: dict = {"per_metric": {}, "per_native": {}}
    for m in ("fnat", "irmsd", "lrmsd", "dockq"):
        a = np.array([r[f"{m}_official"] for r in rows])
        b = np.array([r[f"{m}_foldbridge"] for r in rows])
        ok = np.isfinite(a) & np.isfinite(b)
        d = np.abs(a[ok] - b[ok])
        imax = int(np.argmax(d))
        out["per_metric"][m] = {
            "n": int(ok.sum()),
            "pearson": pearson(a[ok], b[ok]),
            "spearman": pearson(ranks(a[ok]), ranks(b[ok])),
            "mean_abs_diff": float(d.mean()),
            "max_abs_diff": float(d.max()),
            "max_diff_id": [r["id"] for r, o in zip(rows, ok) if o][imax],
        }
    for name in sorted({r["native"] for r in rows}):
        sub = [r for r in rows if r["native"] == name]
        d = np.array([abs(r["dockq_official"] - r["dockq_foldbridge"]) for r in sub])
        out["per_native"][name] = {"n": len(sub), "dockq_mean_abs_diff": float(d.mean()),
                                   "dockq_max_abs_diff": float(d.max())}
    agree = sum(r["capri_official"] == r["capri_foldbridge"] for r in rows)
    out["capri_agreement"] = {"agree": agree, "n": len(rows), "fraction": agree / len(rows)}
    names = [n for _, n in CAPRI]
    conf = {a: {b: 0 for b in names} for a in names}
    for r in rows:
        conf[r["capri_official"]][r["capri_foldbridge"]] += 1
    out["capri_confusion_official_rows"] = conf
    out["capri_counts_official"] = {n: sum(r["capri_official"] == n for r in rows) for n in names}
    return out


def plot(rows, summary) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"1BRS": "#2a6fdb", "2PTC": "#e07b00", "1A2K": "#1a9e5c"}
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    for ax, m, lab in zip(axes, ("dockq", "fnat", "irmsd", "lrmsd"),
                          ("DockQ", "Fnat", "iRMSD (Å)", "LRMSD (Å)")):
        for name, col in colors.items():
            sub = [r for r in rows if r["native"] == name]
            ax.scatter([r[f"{m}_official"] for r in sub], [r[f"{m}_foldbridge"] for r in sub],
                       s=14, alpha=0.75, color=col, label=name, edgecolor="none")
        vals = [v for r in rows for v in (r[f"{m}_official"], r[f"{m}_foldbridge"]) if np.isfinite(v)]
        lo, hi = min(vals), max(vals)
        ax.plot([lo, hi], [lo, hi], color="0.5", lw=0.8, ls="--", zorder=0)
        if m == "dockq":
            for t in (0.23, 0.49, 0.80):
                ax.axvline(t, color="0.85", lw=0.7, zorder=0)
                ax.axhline(t, color="0.85", lw=0.7, zorder=0)
        s = summary["per_metric"][m]
        ax.set_title(f"{lab}\nr={s['pearson']:.4f}  MAD={s['mean_abs_diff']:.2g}  max={s['max_abs_diff']:.2g}",
                     fontsize=9.5)
        ax.set_xlabel("official DockQ v2")
        ax.set_ylabel("foldbridge DockQ-lite")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"foldbridge DockQ-lite vs official DockQ v2 ({len(rows)} decoys, 3 natives)", fontsize=11)
    fig.tight_layout()
    fig.savefig(VAL / "dockq_scatter.png", dpi=130)


def write_fixtures(rows, native_paths) -> None:
    """10 decoys spanning DockQ 0-1 over all three natives, gzipped, + official values."""
    for p in FIX.glob("*"):
        p.unlink()
    picks = []
    for name in ("1BRS", "2PTC", "1A2K"):
        sub = sorted([r for r in rows if r["native"] == name], key=lambda r: r["dockq_official"])
        n = 4 if name == "1BRS" else 3
        idx = np.linspace(0, len(sub) - 1, n).round().astype(int)
        picks += [sub[i] for i in idx]
    for name, npath in native_paths.items():
        with open(npath, "rb") as fi, gzip.open(FIX / f"{name}_native.pdb.gz", "wb", compresslevel=9) as fo:
            fo.write(fi.read())
    out = []
    for r in picks:
        with open(WORK / f"{r['id']}.pdb", "rb") as fi, gzip.open(FIX / f"{r['id']}.pdb.gz", "wb", compresslevel=9) as fo:
            fo.write(fi.read())
        out.append({k: r[k] for k in ("id", "native", "receptor", "ligand", "fnat_official", "irmsd_official",
                                       "lrmsd_official", "dockq_official", "capri_official")})
    (FIX / "official_dockq_v2.json").write_text(json.dumps(
        {"source": "DockQ v2 python API (run_on_all_native_interfaces, identity chain map)",
         "decoys": out}, indent=2) + "\n")


if __name__ == "__main__":
    main()
