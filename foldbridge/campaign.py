"""Binder-design screening: fold many binder sequences against one target, score, rank.

    foldbridge campaign --target target.yaml --designs designs.fasta -b boltz -o screen/

For every FASTA record a job spec = target chains/ligands + one binder chain (id ``B``, or
the next free single-letter id) is written to ``<out>/designs/<design>/`` and run through
the same machinery as ``foldbridge run`` (result cache included, so identical designs are
never folded twice). Boltz-2 and Chai-1 take one seed per invocation, so each seed is its
own job (``<design>/s<seed>/``); ColabFold runs all seeds in one job.

A failing design is recorded and skipped; the campaign carries on. Progress lives in
``<out>/campaign_state.json`` (written atomically after every design): rerunning the same
command skips designs that are already done, retries failed / interrupted ones and picks
up new FASTA records. ``max_jobs`` caps how many not-yet-done designs one invocation
processes, so a large screen can be fed through in slices.

When everything has run, all models are scored in parallel (:func:`foldbridge.batch.score_many`,
checkpointed to ``<out>/scores.jsonl``), ranked together with the length-normalised
composite, and each design is represented by its best model:

    campaign.csv / campaign.json   one row per design, best first
    campaign_models.csv            every model, ranked
    report.html                    sortable table, best design first
    top_<N>.fasta                  the N best binder sequences
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .spec import Chain, JobSpec

__all__ = ["run_campaign", "read_designs", "sanitize_name", "free_chain_id", "design_spec"]

STATE_FILE = "campaign_state.json"
STATE_VERSION = 1
PER_SEED_BACKENDS = {"boltz", "chai"}  # these take a single --seed per run
# every campaign.csv / campaign.json row has exactly these keys (None when not applicable)
DESIGN_COLS = ("rank", "id", "design_id", "composite", "ipsae", "ipsae_complex", "iptm", "ptm", "pdockq",
               "ipae", "plddt", "plddt_interface", "plddt_binder", "clash_fraction", "binder_length",
               "n_residues", "n_models", "best_model", "status", "error", "sequence", "path")
PER_CHAIN_PREFIXES = ("ipsae_", "plddt_", "bsa_")
CHAIN_LETTERS = "BCDEFGHIJKLMNOPQRSTUVWXYZAabcdefghijklmnopqrstuvwxyz0123456789"


# ---- inputs -------------------------------------------------------------------------

def read_designs(path: str | Path) -> list[tuple[str, str]]:
    """``(id, sequence)`` per FASTA record; id = first header token. Trailing ``*`` dropped."""
    records: list[tuple[str, str]] = []
    cur, buf = None, []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith(">"):
            if cur is not None:
                records.append((cur, "".join(buf)))
            toks = line[1:].split()
            cur, buf = (toks[0] if toks else f"design{len(records) + 1}"), []
        elif cur is None:
            raise ValueError(f"{path}: sequence data before the first '>' header")
        else:
            buf.append(line.replace(" ", ""))
    if cur is not None:
        records.append((cur, "".join(buf)))
    return [(i, s.rstrip("*").upper()) for i, s in records]


def sanitize_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._-")
    return s[:80] or "design"


def _warn_duplicates(ids: list[str], names: list[str], path: Path) -> None:
    """stderr warning for repeated FASTA ids (or ids that collide once sanitised)."""
    first: dict[str, int] = {}
    for k, raw in enumerate(ids, start=1):
        key = sanitize_name(raw).lower()
        if key in first:
            print(f"warning: {path}: design id {raw!r} (record {k}) duplicates record {first[key]} "
                  f"({ids[first[key] - 1]!r}); renamed to {names[k - 1]!r}", file=sys.stderr)
        else:
            first[key] = k


def _unique_names(ids: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for raw in ids:
        base = sanitize_name(raw)
        name, k = base, seen.get(base.lower(), 0)
        while name.lower() in seen:
            k += 1
            name = f"{base}_{k}"
        seen[base.lower()] = k
        seen[name.lower()] = 0
        out.append(name)
    return out


def _used_ids(spec: JobSpec) -> set[str]:
    used: set[str] = set()
    for c in spec.chains:
        used.update(c.chain_ids())
    for lig in spec.ligands:
        used.update([lig.id] + [f"{lig.id}{i}" for i in range(2, lig.copies + 1)])
    return used


def free_chain_id(spec: JobSpec, preferred: str = "B") -> str:
    """``preferred`` if unused by any chain copy / ligand of ``spec``, else the next free letter."""
    used = _used_ids(spec)
    for c in [preferred] + list(CHAIN_LETTERS):
        if c not in used:
            return c
    raise ValueError("no free single-character chain id left for the binder")


def design_spec(target: JobSpec, name: str, sequence: str, binder_id: str,
                samples: int | None = None, seeds: list[int] | None = None,
                binder_msa: str | None = None) -> JobSpec:
    """Target + binder chain as a new, validated JobSpec."""
    if not sequence:
        raise ValueError("empty binder sequence")
    spec = copy.deepcopy(target)
    spec.name = name
    spec.reference = None
    spec.chains.append(Chain(id=binder_id, sequence=sequence, type="protein", msa=binder_msa))
    if samples is not None:
        spec.samples = int(samples)
    if seeds:
        spec.seeds = [int(s) for s in seeds]
    spec.validate()
    return spec


def _single_sequence_a3m(cache_root: Path, seq: str) -> str:
    """Stable single-sequence a3m for a designed binder (designs have no homologues)."""
    d = cache_root / "campaign_msa"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{hashlib.sha256(seq.encode()).hexdigest()[:24]}.a3m"
    if not p.is_file():
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-", suffix=".a3m")
        with os.fdopen(fd, "w") as fh:
            fh.write(f">binder\n{seq}\n")
        os.replace(tmp, p)
    return str(p.resolve())


# ---- state --------------------------------------------------------------------------

def _load_state(out: Path) -> dict[str, Any]:
    p = out / STATE_FILE
    if p.is_file():
        try:
            st = json.loads(p.read_text())
            if isinstance(st, dict) and isinstance(st.get("designs"), dict):
                return st
        except (OSError, json.JSONDecodeError):
            pass
    return {"version": STATE_VERSION, "designs": {}}


def _save_state(out: Path, state: dict[str, Any]) -> None:
    from .cli import _clean_json

    state["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    fd, tmp = tempfile.mkstemp(dir=out, prefix=".state-", suffix=".json")
    with os.fdopen(fd, "w") as fh:
        json.dump(_clean_json(state), fh, indent=2)
    os.replace(tmp, out / STATE_FILE)


def _is_done(entry: dict[str, Any] | None, spec_hash: str) -> bool:
    if not entry or entry.get("status") != "done" or entry.get("spec_hash") != spec_hash:
        return False
    models = entry.get("models") or []
    return bool(models) and all(Path(m["path"]).is_file() for m in models)


# ---- scoring helpers ----------------------------------------------------------------

_META = {"id", "backend", "seed", "sample", "path", "pae_path", "rank", "composite", "source_file"}


def translate_chain_keys(scored: dict[str, Any], chain_map: dict[str, str] | None) -> dict[str, Any]:
    """Rename per-chain metric keys (``ipsae_X_Y``, ``plddt_X``, ``bsa_X``) from the model's output
    chain ids to spec chain ids. Chai-1 and ColabFold relabel chains A, B, C... in spec order, so
    output chain ``B`` may be a target copy (``A2``) rather than the binder. No map = identity."""
    if not chain_map:
        return dict(scored)
    cmap = {str(k): str(v) for k, v in chain_map.items()}
    out: dict[str, Any] = {}
    for k, v in scored.items():
        for pre in PER_CHAIN_PREFIXES:
            if k.startswith(pre):
                parts = k[len(pre):].split("_")
                if parts and all(p in cmap for p in parts):
                    k = pre + "_".join(cmap[p] for p in parts)
                break
        out[k] = v
    return out


def _backend_chain_map(backend: str, spec: JobSpec) -> dict[str, str]:
    """``{output chain id: spec chain id}`` from the backend's ``chain_map(spec)`` if it has one,
    else ``{}`` (identity)."""
    try:
        from .backends.base import get_backend

        fn = getattr(get_backend(backend), "chain_map", None)
        return {str(k): str(v) for k, v in (fn(spec) if callable(fn) else {}).items()}
    except Exception:  # noqa: BLE001 - a missing map only means "assume identity"
        return {}


def _merge(model: dict[str, Any], scored: dict[str, Any], binder_id: str) -> dict[str, Any]:
    """Backend-native scores + computed metrics (computed wins; backend kept as ``<k>_backend``).

    ``binder_id`` is the binder's spec chain id; ``model["chain_map"]`` (output -> spec chain id,
    when the backend renamed chains) is applied to per-chain keys first.
    """
    scored = translate_chain_keys(scored, model.get("chain_map"))
    row = {k: model.get(k) for k in ("design", "id", "backend", "seed", "sample", "path", "pae_path")}
    for k, v in (model.get("scores") or {}).items():
        row[k] = v
    for k, v in scored.items():
        if k in ("id", "path", "pae_path"):
            continue
        if k in row and k not in _META and row[k] != v:
            row[f"{k}_backend"] = row[k]
        row[k] = v
    if "plddt" not in scored and isinstance(row.get("plddt"), (int, float)) and row["plddt"] <= 1.0:
        row["plddt"] = row["plddt"] * 100.0
    # binder-specific ipSAE: best interface that involves the binder chain
    pairs = {k: v for k, v in scored.items() if k.startswith("ipsae_") and isinstance(v, (int, float))}
    mine = [v for k, v in pairs.items() if binder_id in k[len("ipsae_"):].split("_")]
    if mine and len(mine) < len(pairs):
        row["ipsae_complex"] = row.get("ipsae")
        row["ipsae"] = max(mine)
    if f"plddt_{binder_id}" in scored:
        row["plddt_binder"] = scored[f"plddt_{binder_id}"]
    return row


def _design_row(name: str, entry: dict[str, Any], best: dict[str, Any] | None) -> dict[str, Any]:
    row: dict[str, Any] = {k: None for k in DESIGN_COLS}  # same columns for every design
    row["id"] = name
    if best is not None:
        for k in DESIGN_COLS:
            if k in best and k not in ("id", "rank", "status", "sequence", "design_id", "error"):
                row[k] = best[k]
        row["best_model"] = best.get("id")
    row["binder_length"] = len(entry.get("sequence", ""))
    row["n_models"] = len(entry.get("models") or [])
    row["status"] = entry.get("status", "pending")
    if entry.get("error"):
        row["error"] = entry["error"]
    row["design_id"] = entry.get("design_id", name)
    row["sequence"] = entry.get("sequence", "")
    return {k: row[k] for k in DESIGN_COLS}


# ---- main ---------------------------------------------------------------------------

def run_campaign(target: str | Path, designs: str | Path, backend: str, out: str | Path,
                 samples: int | None = None, seeds: list[int] | None = None, dry_run: bool = False,
                 max_jobs: int | None = None, cpu: bool = False, use_cache: bool = True,
                 cache_root: str | Path | None = None, workers: int | None = None, top: int = 10,
                 binder_chain: str | None = None, binder_msa: str = "auto", progress: bool = True,
                 log: Callable[[str], None] = print) -> dict[str, Any]:
    """Run / resume a binder screen. Returns a summary dict (paths, counts, ranked rows).

    Raises ``cli.CLIError`` for bad inputs and when the backend is not installed; every
    per-design problem is recorded in the state file instead.
    """
    from . import report
    from .batch import score_many
    from .cache import Cache
    from .cli import EXIT_MISSING, CLIError, _clean_json, run_job
    from .rank import rank
    from .spec import load_spec

    tp, dp, out = Path(target), Path(designs), Path(out)
    if not tp.is_file():
        raise CLIError(f"target spec not found: {tp}")
    if not dp.is_file():
        raise CLIError(f"designs FASTA not found: {dp}")
    try:
        target_spec = load_spec(tp)
    except Exception as e:
        raise CLIError(f"invalid target spec {tp}: {e}") from e
    try:
        records = read_designs(dp)
    except (ValueError, UnicodeDecodeError) as e:
        raise CLIError(str(e)) from e
    if not records:
        raise CLIError(f"no sequences in {dp}")
    if binder_msa not in ("auto", "single", "inherit"):
        raise CLIError("binder_msa must be auto, single or inherit")
    if max_jobs is not None and max_jobs < 0:
        raise CLIError("--max-jobs must be >= 0")
    if binder_chain is not None:
        if len(binder_chain) != 1 or binder_chain in _used_ids(target_spec):
            raise CLIError(f"binder chain id {binder_chain!r} must be one free character")
        bid = binder_chain
    else:
        bid = free_chain_id(target_spec)
    single = binder_msa == "single" or (binder_msa == "auto" and target_spec.msa not in ("server", "none"))
    croot = Cache(cache_root).root

    out.mkdir(parents=True, exist_ok=True)
    state = _load_state(out)
    state.update(version=STATE_VERSION, target=str(tp.resolve()), designs_file=str(dp.resolve()),
                 backend=backend, binder_chain=bid, samples=samples, seeds=seeds)
    names = _unique_names([r[0] for r in records])
    _warn_duplicates([r[0] for r in records], names, dp)
    started = 0
    commands: list[str] = []
    n_skipped = n_run = n_failed = 0

    for (raw_id, seq), name in zip(records, names):
        entry = state["designs"].get(name) or {}
        try:
            msa = _single_sequence_a3m(croot, seq) if single and seq else None
            spec = design_spec(target_spec, name, seq, bid, samples=samples, seeds=seeds, binder_msa=msa)
        except (ValueError, TypeError) as e:
            state["designs"][name] = {"design_id": raw_id, "sequence": seq, "status": "failed",
                                      "error": f"invalid design: {e}", "models": []}
            n_failed += 1
            log(f"[{name}] invalid design: {e}")
            _save_state(out, state)
            continue
        spec_hash = spec.content_hash(backend)
        if _is_done(entry, spec_hash):
            n_skipped += 1
            continue
        if max_jobs is not None and started >= max_jobs:
            keep = entry.get("spec_hash") == spec_hash and entry.get("status") in ("failed", "dry_run")
            state["designs"][name] = {**(entry if keep else {}), "design_id": raw_id, "sequence": seq,
                                      "spec_hash": spec_hash, "status": entry["status"] if keep else "pending",
                                      "models": []}
            continue
        started += 1

        ddir = out / "designs" / name
        seed_groups = ([[s] for s in spec.seeds] if backend in PER_SEED_BACKENDS and len(spec.seeds) > 1
                       else [list(spec.seeds)])
        entry = {"design_id": raw_id, "sequence": seq, "spec_hash": spec_hash, "status": "running",
                 "binder_chain": bid, "jobs": [], "models": []}
        state["designs"][name] = entry
        _save_state(out, state)
        try:
            for group in seed_groups:
                job = copy.deepcopy(spec)
                job.seeds = group
                jdir = ddir / (f"s{group[0]}" if len(seed_groups) > 1 else "")
                jdir.mkdir(parents=True, exist_ok=True)
                spec_path = jdir / "spec.yaml"
                job.save(spec_path)
                summ = run_job(spec_path, backend, out=jdir, cpu=cpu, dry_run=dry_run, use_cache=use_cache,
                               score=False, cache_root=cache_root, log=lambda m, n=name: log(f"[{n}] {m}"))
                entry["jobs"].append({k: summ.get(k) for k in ("cache_key", "cache_hit", "results_json",
                                                                "command", "runtime_s", "out_dir")})
                if summ.get("command"):
                    commands.append(summ["command"])
                job_map = None
                for r in summ.get("results", []):
                    cmap = r.get("chain_map") or (r.get("extra") or {}).get("chain_map")
                    if not cmap:  # rows without the backend's extra: ask the backend directly
                        job_map = job_map if job_map is not None else _backend_chain_map(backend, job)
                        cmap = job_map
                    entry["models"].append({
                        "design": name, "id": f"{name}_{r.get('backend', backend)}_s{r.get('seed')}_m{r.get('sample')}",
                        "backend": r.get("backend", backend), "seed": r.get("seed"), "sample": r.get("sample"),
                        "path": str(Path(r["path"]).resolve()),
                        "pae_path": str(Path(r["pae_path"]).resolve()) if r.get("pae_path") else None,
                        "scores": {k: v for k, v in r.items()
                                   if k not in _META and k not in ("chain_map", "extra")},
                        "chain_map": cmap or None,
                    })
            entry["status"] = "dry_run" if dry_run else "done"
            n_run += 1
        except CLIError as e:
            if e.code == EXIT_MISSING:
                entry["status"] = "pending"
                _save_state(out, state)
                raise
            entry.update(status="failed", error=str(e))
            n_failed += 1
            log(f"[{name}] failed: {e}")
        except Exception as e:  # noqa: BLE001 - one bad design must not stop the screen
            entry.update(status="failed", error=f"{type(e).__name__}: {e}")
            n_failed += 1
            log(f"[{name}] failed: {type(e).__name__}: {e}")
        _save_state(out, state)

    if commands:
        (out / "commands.txt").write_text("\n".join(commands) + "\n")

    # ---- score + rank everything that is done
    current = [(name, state["designs"].get(name, {})) for name in names]
    models = [m for _, e in current if e.get("status") == "done" for m in e.get("models", [])]
    model_rows: list[dict[str, Any]] = []
    if models:
        log(f"scoring {len(models)} models from {sum(1 for _, e in current if e.get('status') == 'done')} designs")
        scored = score_many([m["path"] for m in models], workers=workers, progress=progress,
                            checkpoint=out / "scores.jsonl",
                            pae={m["path"]: m.get("pae_path") for m in models})
        by_path = {str(Path(r["path"]).resolve()): r for r in scored}
        score_err: dict[str, str] = {}
        for m in models:
            s = by_path.get(str(Path(m["path"]).resolve()), {"error": "not scored"})
            if s.get("error"):
                score_err.setdefault(m["design"], f"scoring {Path(m['path']).name}: {s['error']}")
                continue
            model_rows.append(_merge(m, s, bid))
        model_rows = rank(model_rows, length_normalize=True)
        for name, err in score_err.items():
            if not any(r["design"] == name for r in model_rows):
                state["designs"][name]["error"] = err
        _save_state(out, state)

    # a design is ranked by its best model; models without any metric have rank None (never "best")
    best: dict[str, dict[str, Any]] = {}
    for r in model_rows:
        if r.get("rank") is not None:
            best.setdefault(r["design"], r)
    design_rows = [_design_row(n, state["designs"].get(n, {}), best[n]) for n in best]
    for k, r in enumerate(design_rows, start=1):
        r["rank"] = k
    ranked_names = set(best)
    # failed / pending / dry-run designs: no metrics, no rank, listed after the ranked ones
    design_rows += [_design_row(n, e, None) for n, e in current if n not in ranked_names]

    res_csv = report.write_csv(design_rows, out / "campaign.csv")
    meta = {"target": str(tp.resolve()), "designs": str(dp.resolve()), "backend": backend,
            "binder_chain": bid, "n_designs": len(names), "n_ranked": len(best), "dry_run": dry_run}
    (out / "campaign.json").write_text(json.dumps(_clean_json({"campaign": meta, "rows": design_rows}),
                                                  indent=2) + "\n")
    if model_rows:
        report.write_csv(model_rows, out / "campaign_models.csv")
    html = report.write_html(design_rows, out / "report.html",
                             title=f"foldbridge campaign: {target_spec.name} x {len(names)} designs ({backend})")
    fasta = None
    for old in out.glob("top_*.fasta"):  # a previous run may have had fewer designs / another --top
        if re.fullmatch(r"top_\d+\.fasta", old.name):
            old.unlink()
    if best:
        n_top = max(1, min(int(top), len(best)))
        fasta = out / f"top_{n_top}.fasta"
        lines = []
        for r in design_rows[:n_top]:
            tags = " ".join(f"{k}={r[k]:.3f}" for k in ("composite", "ipsae", "iptm", "pdockq")
                            if isinstance(r.get(k), (int, float)))
            lines += [f">{r['id']} rank={r['rank']} {tags}".rstrip(), r["sequence"]]
        fasta.write_text("\n".join(lines) + "\n")

    status_counts: dict[str, int] = {}
    for name in names:
        s = state["designs"].get(name, {}).get("status", "pending")
        status_counts[s] = status_counts.get(s, 0) + 1
    return {
        "out_dir": str(out.resolve()), "state": str(out / STATE_FILE), "csv": str(res_csv),
        "json": str(out / "campaign.json"), "report_html": str(html), "top_fasta": str(fasta) if fasta else None,
        "commands": str(out / "commands.txt") if commands else None,
        "n_designs": len(names), "n_run": n_run, "n_skipped": n_skipped, "n_failed": n_failed,
        "status": status_counts, "binder_chain": bid, "rows": design_rows,
    }
