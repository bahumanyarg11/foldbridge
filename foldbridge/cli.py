"""foldbridge command line.

    foldbridge backends
    foldbridge run SPEC --backend boltz [--out DIR] [--cpu] [--dry-run] [--no-cache] [--score] [--reference PDB]
    foldbridge score PATH... [--pae FILE] [--reference REF] [--out results.csv] [--workers N] [--checkpoint F]
    foldbridge rank RESULTS.csv|json... [--by composite] [--no-length-norm] [--top N] [--html report.html]
    foldbridge prep-md STRUCTURE [--out DIR] [--ph 7.0] [--keep-ligands] [--solvate]
    foldbridge campaign --target T.yaml --designs D.fasta --backend boltz --out DIR [--samples N]
                        [--seeds 1,2] [--dry-run] [--max-jobs N] [--top 10]
    foldbridge mcp

User errors print one line to stderr (no traceback; set FOLDBRIDGE_DEBUG=1 or pass
--debug for one). Exit codes: 0 ok, 1 runtime failure, 2 bad input/usage,
3 backend not installed, 130 interrupted.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import re
import shutil
import sys
import time
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from . import __version__
from .batch import (  # noqa: F401 (re-exported)
    PAE_SUFFIXES,
    STRUCT_SUFFIXES,
    WARNING_SEP,
    _pae_candidates,
    find_pae,
)

EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_MISSING = 0, 1, 2, 3


class CLIError(Exception):
    """An error caused by user input: printed without traceback."""

    def __init__(self, msg: str, code: int = EXIT_USAGE):
        super().__init__(msg)
        self.code = code


# ---- small helpers ------------------------------------------------------------------

def _clean_json(x: Any) -> Any:
    """NaN/inf -> None, Paths -> str, numpy scalars -> python, recursively."""
    if isinstance(x, dict):
        return {str(k): _clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean_json(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, float):
        if not math.isfinite(x):
            return None
        return 0.0 if abs(x) < 1e-9 else x
    if isinstance(x, (str, int, bool)) or x is None:
        return x
    try:
        return _clean_json(x.item())
    except Exception:
        return str(x)


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if not math.isfinite(v):
            return "-"
        return f"{v:.3f}" if abs(v) < 1000 else f"{v:.1f}"
    return "-" if v is None or v == "" else str(v)


def print_table(rows: Sequence[dict[str, Any]], cols: Sequence[str], file=None) -> None:
    file = file or sys.stdout
    cols = [c for c in cols if any(c in r for r in rows)] or list(cols)
    cells = [[_fmt(r.get(c)) for c in cols] for r in rows]
    widths = [max([len(c)] + [len(row[i]) for row in cells]) for i, c in enumerate(cols)]
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)), file=file)
    print("  ".join("-" * w for w in widths), file=file)
    for row in cells:
        print("  ".join(v.ljust(w) for v, w in zip(row, widths)), file=file)


TABLE_COLS = ["rank", "id", "backend", "composite", "iptm", "ptm", "plddt", "ipsae", "pdockq",
              "ipae", "clash_fraction", "rmsd", "dockq"]


def _row_id(path: Path) -> str:
    return path.stem


def _row_ids(files: Sequence[Path]) -> list[str]:
    """File stems, except that files sharing a stem (1A2K.pdb + 1A2K.cif) get their file name
    and files sharing a name get the path as given."""
    stems = [f.stem for f in files]
    names = [f.name for f in files]
    return [st if stems.count(st) < 2 else (nm if names.count(nm) < 2 else str(f))
            for f, st, nm in zip(files, stems, names)]


OUTPUT_SUFFIXES = {".csv", ".json", ".html", ".htm"}


def _check_output(path: str | Path | None, flag: str, allowed: set[str] = OUTPUT_SUFFIXES) -> None:
    """Reject an unsupported output extension before any work is done."""
    if path is None:
        return
    p = Path(path)
    if p.suffix.lower() not in allowed:
        kinds = ", ".join(sorted(allowed))
        raise CLIError(f"{flag} {p}: unsupported output type {p.suffix or '(none)'!r} (use {kinds})")
    if p.is_dir():
        raise CLIError(f"{flag} {p} is a directory; give a file name such as {p / ('results' + sorted(allowed)[0])}")


def _print_row_messages(rows: Iterable[dict[str, Any]], file=None) -> None:
    """Per-file warnings / notes on stderr as ``warning: <file>: <msg>``."""
    file = file or sys.stderr
    for r in rows:
        name = r.get("path") or r.get("id")
        if r.get("note"):
            print(f"note: {name}: {r['note']}", file=file)
        w = r.get("warnings")
        msgs = w if isinstance(w, list) else ([m for m in str(w).split(WARNING_SEP) if m] if w else [])
        for m in msgs:
            print(f"warning: {name}: {m}", file=file)


# ---- structure / PAE discovery --------------------------------------------------------

def collect_structures(paths: Iterable[str | Path]) -> list[Path]:
    """Files as given; directories searched recursively for *.cif / *.pdb (sorted, de-duplicated)."""
    out: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            found = sorted(q for q in p.rglob("*") if q.is_file() and q.suffix.lower() in STRUCT_SUFFIXES)
            out.extend(found)
        elif p.is_file():
            out.append(p)
        else:
            raise CLIError(f"no such file or directory: {p}")
    seen, uniq = set(), []
    for p in out:
        k = p.resolve()
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    if not uniq:
        raise CLIError("no structure files (*.cif / *.pdb) found in: " + " ".join(map(str, paths)))
    return uniq


def score_paths(paths: Iterable[str | Path], pae: str | Path | None = None,
                reference: str | Path | None = None, chains: Sequence[str] | None = None,
                on_error=None) -> list[dict[str, Any]]:
    """Score every structure under ``paths``. ``pae`` only allowed with a single structure."""
    from .score import score_structure

    files = collect_structures(paths)
    if pae is not None:
        if len(files) != 1:
            raise CLIError(f"--pae given but {len(files)} structures found; pass exactly one structure")
        if not Path(pae).is_file():
            raise CLIError(f"PAE file not found: {pae}")
    if reference is not None and not Path(reference).is_file():
        raise CLIError(f"reference not found: {reference}")
    rows = []
    for f, rid in zip(files, _row_ids(files)):
        pf = Path(pae) if pae is not None else find_pae(f)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                s = score_structure(f, pae=pf, reference=reference, chains=chains)
        except Exception as e:
            if on_error is None:
                raise CLIError(f"failed to score {f}: {e}", EXIT_FAIL) from e
            on_error(f, e)
            continue
        row = {"id": rid, "path": str(f), "pae_path": str(pf) if pf else None}
        row.update(s)
        msgs = list(dict.fromkeys(str(w.message) for w in caught))
        if msgs:
            row["warnings"] = WARNING_SEP.join(msgs)
        rows.append(row)
    return rows


# ---- results files --------------------------------------------------------------------

# columns that hold text even when a value looks numeric (design "007", sequence "1", ...)
STRING_COLUMNS = {"id", "design_id", "design", "sequence", "path", "pae_path", "backend", "status", "error",
                  "best_model", "note", "warnings", "source_file", "name", "spec", "command"}


def _parse_cell(v: str) -> Any:
    if v is None:
        return None
    s = v.strip()
    if s == "":
        return None
    if s.lower() in {"nan", "none", "null"}:
        return float("nan")
    try:
        if s.lstrip("-").isdigit():
            return int(s)
        return float(s)
    except ValueError:
        return s


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    """Rows from a results .csv or .json (a list, or a dict holding results/rows/models)."""
    p = Path(path)
    if p.is_dir():
        raise CLIError(f"{p} is a directory; rank takes results .csv/.json (use foldbridge score DIR first)")
    if not p.is_file():
        raise CLIError(f"results file not found: {p}")
    suf = p.suffix.lower()
    try:
        if suf == ".csv":
            with open(p, newline="") as fh:
                rows = [{k: ((v if v is None or v.strip() else None) if k in STRING_COLUMNS else _parse_cell(v))
                         for k, v in r.items() if k is not None} for r in csv.DictReader(fh)]
        elif suf == ".json":
            data = json.loads(p.read_text())
            if isinstance(data, dict):
                if data.get("dry_run") and not data.get("results"):
                    raise CLIError(f"{p} is from a --dry-run (0 models); run without --dry-run first")
                for key in ("results", "rows", "models"):
                    if isinstance(data.get(key), list):
                        data = data[key]
                        break
            if not isinstance(data, list) or not all(isinstance(r, dict) for r in data):
                raise CLIError(f"{p}: expected a list of result rows")
            rows = [dict(r) for r in data]
        else:
            raise CLIError(f"{p}: results must be .csv or .json")
    except (json.JSONDecodeError, csv.Error, UnicodeDecodeError) as e:
        raise CLIError(f"cannot parse {p}: {e}") from e
    for r in rows:
        r.setdefault("source_file", str(p))
    return rows


def write_rows(rows: list[dict[str, Any]], path: str | Path, title: str = "foldbridge") -> Path:
    from . import report

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    suf = p.suffix.lower()
    if suf == ".csv":
        report.write_csv(rows, p)
    elif suf == ".json":
        report.write_json(rows, p)
    elif suf in {".html", ".htm"}:
        report.write_html(rows, p, title)
    else:
        raise CLIError(f"unsupported output type {p.suffix!r} (use .csv, .json or .html)")
    return p


# ---- run --------------------------------------------------------------------------------

def _stage_cached_msas(spec, cache, msa_root: Path) -> Path | None:
    """Copy cached a3m files to <msa_root>/<chain>.a3m for each protein chain that has one."""
    found = 0
    for cid, seq in spec.protein_sequences():
        p = cache.msa_path(seq)
        if p is not None:
            msa_root.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, msa_root / f"{cid}.a3m")
            found += 1
    return msa_root if found else None


def _log(msg: str) -> None:
    """Progress on stdout; ``warning:`` / ``note:`` lines on stderr."""
    print(msg, file=sys.stderr if str(msg).startswith(("warning:", "note:")) else sys.stdout)


PER_SEED_BACKENDS = {"boltz", "chai"}  # take a single --seed per invocation


def safe_name(name: str) -> str:
    """``name`` reduced to [A-Za-z0-9._-] for use as a directory name; rejects traversal."""
    raw = str(name)
    if any(part == ".." for part in re.split(r"[\\/]", raw)):
        raise CLIError(f"invalid job name {raw!r}: path traversal is not allowed")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._")
    if not safe:
        raise CLIError(f"invalid job name {raw!r}: needs at least one letter or digit")
    return safe


def run_job(spec_path: str | Path, backend: str, out: str | Path | None = None, cpu: bool = False,
            dry_run: bool = False, use_cache: bool = True, score: bool = False,
            reference: str | Path | None = None, cache_root: str | Path | None = None,
            log=None) -> dict[str, Any]:
    """Shared implementation of ``foldbridge run`` (also used by the MCP server).

    Backend inputs/outputs go to ``<out>/work/<cache key>/`` (wiped before a real run, so a
    rerun never collects stale models). Boltz-2 and Chai-1 take one seed per invocation:
    a multi-seed spec runs once per seed in ``.../<cache key>/s<seed>/`` and the results
    are merged; ColabFold gets all seeds in one call.
    """
    from .backends.base import get_backend
    from .cache import Cache, harvest_msas, result_to_dict
    from .spec import load_spec

    log = log or _log
    sp = Path(spec_path)
    if not sp.is_file():
        raise CLIError(f"spec not found: {sp}")
    try:
        spec = load_spec(sp)
    except (ValueError, KeyError, TypeError) as e:
        msg = e.args[0] if isinstance(e, KeyError) and e.args else e
        raise CLIError(f"invalid spec {sp}: {msg}") from e
    except Exception as e:  # yaml / json syntax
        raise CLIError(f"cannot read spec {sp}: {e}") from e
    try:
        be = get_backend(backend)
    except KeyError as e:
        raise CLIError(str(e.args[0] if e.args else e)) from e

    available = be.available()
    if not available and not dry_run:
        raise CLIError(f"backend {backend!r} is not installed or not on PATH "
                       f"(try `foldbridge backends`; use --dry-run to only write inputs)", EXIT_MISSING)
    version = be.version() if available else "unknown"
    cache = Cache(cache_root)
    key = spec.content_hash(backend, version)
    # absolute: backends report model paths under the workdir, and the cache must find them
    out_dir = (Path(out) if out else Path("foldbridge_out") / f"{safe_name(spec.name)}-{backend}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    workdir = out_dir / "work" / key
    ref = reference or spec.reference
    if ref is not None and not Path(ref).is_file():
        raise CLIError(f"reference not found: {ref}")

    summary: dict[str, Any] = {
        "foldbridge_version": __version__, "spec": str(sp.resolve()), "name": spec.name,
        "backend": backend, "backend_version": version, "cache_key": key, "cache_root": str(cache.root),
        "dry_run": dry_run, "cache_hit": False, "out_dir": str(out_dir),
        "workdir": str(workdir), "reference": str(ref) if ref else None,
    }

    results = None
    if use_cache and not dry_run:
        results = cache.load(key)
        if results is not None:
            summary["cache_hit"] = True
            log(f"cache hit {key} ({len(results)} models) from {cache.path_for(key)}")
            # copy into the out dir so the report outlives the cache
            mdir = out_dir / "models"
            mdir.mkdir(exist_ok=True)
            for r in results:
                dst = mdir / r.path.name
                shutil.copy2(r.path, dst)
                r.path = dst
                if r.pae_path is not None:
                    pdst = mdir / r.pae_path.name
                    shutil.copy2(r.pae_path, pdst)
                    r.pae_path = pdst

    if results is None:
        if not dry_run and workdir.exists():
            shutil.rmtree(workdir)  # stale outputs would be skipped over / collected again
        msa_dir = None
        if spec.msa == "cached":
            msa_dir = _stage_cached_msas(spec, cache, workdir / "msa_cache")
        seeds = list(spec.seeds or [])
        if backend in PER_SEED_BACKENDS and len(seeds) > 1:
            jobs = []
            for seed in seeds:
                job = copy.deepcopy(spec)
                job.seeds = [seed]
                jobs.append((job, workdir / f"s{seed}", seed))
        else:
            jobs = [(spec, workdir, None)]
        t0 = time.time()
        results = []
        commands: list[str] = []
        for job, jdir, seed in jobs:
            try:
                res = be.run(job, jdir, msa_dir=msa_dir, gpu=not cpu, dry_run=dry_run)
            except ValueError as e:  # bad spec for this backend (e.g. missing MSA)
                raise CLIError(str(e)) from e
            except (RuntimeError, OSError) as e:
                raise CLIError(str(e), EXIT_FAIL) from e
            if seed is not None:
                for r in res:
                    r.seed = int(seed)
            results.extend(res)
            cmd_file = jdir / "command.txt"
            if cmd_file.exists():
                commands.append(cmd_file.read_text().strip())
        summary["runtime_s"] = round(time.time() - t0, 2)
        if commands:
            summary["command"] = "\n".join(commands)
            cmd = workdir / "command.txt"
            if len(jobs) > 1:
                cmd.write_text(summary["command"] + "\n")
            summary["command_file"] = str(cmd)
        if dry_run:
            log(f"dry run: inputs in {workdir}")
            if "command" in summary:
                log(f"command: {summary['command']}")
        else:
            if not results:
                raise CLIError(f"{backend} finished but produced no models; see {jobs[0][1] / 'run.log'}",
                               EXIT_FAIL)
            if use_cache:
                try:
                    cache.store(key, results, workdir, meta={k: summary[k] for k in
                                ("name", "backend", "backend_version", "spec")})
                    harvest_msas(cache, workdir, [s for _, s in spec.protein_sequences()])
                    log(f"cached as {key}")
                except OSError as e:
                    log(f"warning: could not write cache: {e}")

    rows: list[dict[str, Any]] = []
    if results:
        if score:
            from .score import score_result

            for r in results:
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        row = score_result(r, reference=ref)
                    for m in dict.fromkeys(str(w.message) for w in caught):
                        log(f"warning: {r.path.name}: {m}")
                    if row.get("note"):
                        log(f"note: {r.path.name}: {row['note']}")
                    rows.append(row)
                except Exception as e:
                    log(f"warning: scoring {r.path.name} failed: {e}")
                    rows.append({"id": r.path.stem, **result_to_dict(r)})
        else:
            for r in results:
                d = result_to_dict(r)
                rows.append({"id": f"{r.backend}_s{r.seed}_m{r.sample}", "backend": r.backend,
                             "seed": r.seed, "sample": r.sample, "path": d["path"],
                             "pae_path": d["pae_path"], **d["scores"]})
        try:
            from .rank import rank

            rows = rank(rows)
        except Exception as e:  # ranking is a convenience here
            log(f"warning: ranking failed: {e}")

    summary["n_models"] = len(rows)
    summary["results"] = rows
    res_json = out_dir / "results.json"
    res_json.write_text(json.dumps(_clean_json(summary), indent=2))
    summary["results_json"] = str(res_json)
    if rows:
        try:
            html = write_rows(rows, out_dir / "report.html", title=f"foldbridge: {spec.name} ({backend})")
            summary["report_html"] = str(html)
        except Exception as e:
            log(f"warning: could not write report.html: {e}")
    return summary


# ---- subcommands -----------------------------------------------------------------------------

def cmd_backends(args) -> int:
    from .backends.base import list_backends

    info = list_backends()
    if args.json:
        print(json.dumps(info, indent=2))
        return EXIT_OK
    rows = [{"backend": n, "available": v["available"], "version": v["version"] or "-"}
            for n, v in sorted(info.items())]
    print_table(rows, ["backend", "available", "version"])
    return EXIT_OK


def cmd_run(args) -> int:
    summary = run_job(args.spec, args.backend, out=args.out, cpu=args.cpu, dry_run=args.dry_run,
                      use_cache=not args.no_cache, score=args.score, reference=args.reference,
                      cache_root=args.cache_dir)
    rows = summary["results"]
    if rows:
        print_table(rows, TABLE_COLS)
    print(f"results: {summary['results_json']}")
    if summary.get("report_html"):
        print(f"report:  {summary['report_html']}")
    return EXIT_OK


def cmd_score(args) -> int:
    from .batch import score_many

    _check_output(args.out, "--out")
    files = collect_structures(args.paths)
    if args.pae is not None:
        if len(files) != 1:
            raise CLIError(f"--pae given but {len(files)} structures found; pass exactly one structure")
        if not Path(args.pae).is_file():
            raise CLIError(f"PAE file not found: {args.pae}")
    if args.reference is not None and not Path(args.reference).is_file():
        raise CLIError(f"reference not found: {args.reference}")
    if args.workers is not None and args.workers < 1:
        raise CLIError("--workers must be >= 1")
    chains = [c.strip() for c in args.chains.split(",") if c.strip()] if args.chains else None
    if args.chains is not None and not chains:
        raise CLIError("--chains: expected comma-separated chain ids, e.g. A,B")
    rows = score_many(files, reference=args.reference, workers=args.workers, chains=chains,
                      progress=not args.no_progress, checkpoint=args.checkpoint, pae=args.pae)
    failed = [r for r in rows if r.get("error")]
    for r in failed:
        print(f"warning: failed to score {r['path']}: {r['error']}", file=sys.stderr)
    rows = [r for r in rows if not r.get("error")]
    _print_row_messages(rows)
    if not rows:
        chain_errs = [r["error"] for r in failed if " not found in " in r["error"] and "(available:" in r["error"]]
        if chain_errs and len(chain_errs) == len(failed):
            raise CLIError(chain_errs[0].split(": ", 1)[-1])
        raise CLIError("no structures could be scored", EXIT_FAIL)
    out = write_rows(rows, args.out)
    if len(rows) <= 50:
        print_table(rows, ["id"] + TABLE_COLS[3:])
    print(f"wrote {out} ({len(rows)} structures" + (f", {len(failed)} failed)" if failed else ")"))
    return EXIT_OK


_NON_METRIC = {"rank", "seed", "sample", "composite"}


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _available_metrics(rows: Sequence[dict[str, Any]]) -> list[str]:
    keys = {k for r in rows for k, v in r.items() if _is_num(v)}
    return sorted(keys - _NON_METRIC - STRING_COLUMNS)


def _has_metrics(row: dict[str, Any], by: str) -> bool:
    from .rank import DEFAULT_WEIGHTS, HIGHER_BETTER, LOWER_BETTER

    known = set(HIGHER_BETTER) | set(LOWER_BETTER) | set(DEFAULT_WEIGHTS)
    if by != "composite":
        known.add(by)
    return any(_is_num(row.get(k)) for k in known)


def cmd_rank(args) -> int:
    from .rank import rank

    _check_output(args.out, "--out", {".csv", ".json"})
    _check_output(args.html, "--html", {".html", ".htm"})
    if args.top is not None and args.top < 1:
        raise CLIError("--top must be >= 1")
    rows: list[dict[str, Any]] = []
    for p in args.results:
        rows.extend(load_rows(p))
    if not rows:
        raise CLIError("no rows found in results files")
    # the same model listed twice (overlapping results files): keep the first row
    seen: set[str] = set()
    uniq = []
    for r in rows:
        key = r.get("path")
        if key:
            key = str(Path(str(key)).resolve())
            if key in seen:
                continue
            seen.add(key)
        uniq.append(r)
    rows = uniq
    if args.by != "composite" and not any(_is_num(r.get(args.by)) for r in rows):
        avail = _available_metrics(rows)
        raise CLIError(f"metric {args.by!r} not present in results (available: "
                       f"{', '.join(['composite'] + avail)})")
    scored = [r for r in rows if _has_metrics(r, args.by)]
    unscored = [dict(r, rank=None) for r in rows if not _has_metrics(r, args.by)]
    if not scored:
        raise CLIError("no rows with metrics to rank (are these results from a --dry-run or failed runs?)")
    ranked = rank(scored, by=args.by, length_normalize=not args.no_length_norm)
    if args.top:
        ranked = ranked[: args.top]
    elif unscored:
        print(f"warning: {len(unscored)} row(s) without metrics left unranked (listed last)", file=sys.stderr)
        ranked = ranked + unscored
    cols = TABLE_COLS if args.by in TABLE_COLS else TABLE_COLS[:4] + [args.by] + TABLE_COLS[4:]
    print_table(ranked, cols)
    if args.out:
        print(f"wrote {write_rows(ranked, args.out)}")
    if args.html:
        print(f"wrote {write_rows(ranked, args.html, title='foldbridge ranking')}")
    return EXIT_OK


def cmd_prep_md(args) -> int:
    from . import mdprep
    from .mdprep import prepare_for_md

    src = Path(args.structure)
    if not src.is_file():
        raise CLIError(f"structure not found: {src}")
    if args.solvate and not mdprep.openmm_available():
        raise CLIError("--solvate needs openmm + pdbfixer, which are not installed; install them with "
                       "`pip install foldbridge[md]` (or `conda install -c conda-forge openmm pdbfixer`), "
                       "or drop --solvate for a numpy-only cleanup")
    out = (Path(args.out) if args.out else Path.cwd() / f"{src.stem}_md").resolve()
    try:
        rep = prepare_for_md(src, out, ph=args.ph, keep_ligands=args.keep_ligands, solvate=args.solvate,
                             padding_nm=args.padding, plddt_cutoff=args.plddt_cutoff)
    except ValueError as e:
        raise CLIError(str(e)) from e
    def full(x: Any) -> str:
        return str(Path(str(x)).resolve())

    print(f"output dir:    {out}")
    print(f"openmm:        {'yes' if rep['openmm'] else 'no (numpy fallback)'}")
    print(f"prepared:      {full(rep['prepared_pdb'])}  ({rep['atoms']} atoms, "
          f"{len(rep['chains'])} chain(s), charge {rep.get('charge_estimate')})")
    print(f"low pLDDT<50:  {rep['n_low_plddt_residues']} residue(s)")
    if rep.get("restraint_files"):
        print(f"restraints:    {', '.join(full(f) for f in rep['restraint_files'])}")
    if rep.get("ligands"):
        print(f"ligand folder: {out / 'ligands'}  ({len(rep['ligands'])} file(s))")
    if rep.get("ligand_note"):
        print(f"ligands:       {rep['ligand_note']}")
    for w in rep.get("warnings", []):
        print(f"warning: {w}", file=sys.stderr)
    print(f"report:        {full(rep['report'])}")
    return EXIT_OK


def _int_list(text: str | None, flag: str) -> list[int] | None:
    if text is None:
        return None
    try:
        vals = [int(t) for t in text.replace(",", " ").split()]
    except ValueError as e:
        raise CLIError(f"{flag}: expected comma-separated integers, got {text!r}") from e
    if not vals:
        raise CLIError(f"{flag}: empty list")
    return vals


def cmd_campaign(args) -> int:
    from .campaign import run_campaign

    if args.samples is not None and args.samples < 1:
        raise CLIError("--samples must be >= 1")
    summary = run_campaign(args.target, args.designs, args.backend, args.out, samples=args.samples,
                           seeds=_int_list(args.seeds, "--seeds"), dry_run=args.dry_run,
                           max_jobs=args.max_jobs, cpu=args.cpu, use_cache=not args.no_cache,
                           cache_root=args.cache_dir, workers=args.workers, top=args.top,
                           binder_chain=args.binder_chain, binder_msa=args.binder_msa,
                           progress=not args.no_progress)
    ranked = [r for r in summary["rows"] if r.get("rank")]
    if ranked:
        print_table(ranked[: max(args.top, 1)], ["rank", "id", "composite", "ipsae", "iptm", "pdockq", "ipae",
                                                  "plddt_interface", "clash_fraction", "binder_length"])
    counts = ", ".join(f"{v} {k}" for k, v in sorted(summary["status"].items()))
    print(f"designs: {summary['n_designs']} ({counts}); this run: {summary['n_run']} run, "
          f"{summary['n_skipped']} already done, {summary['n_failed']} failed")
    print(f"results: {summary['csv']}")
    print(f"report:  {summary['report_html']}")
    if summary.get("top_fasta"):
        print(f"top:     {summary['top_fasta']}")
    if summary.get("commands"):
        print(f"commands: {summary['commands']}")
    print(f"state:   {summary['state']}")
    return EXIT_OK


def cmd_mcp(args) -> int:
    from .mcp_server import main as mcp_main

    return mcp_main()


# ---- parser ------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="foldbridge",
                                description="One spec, many folders: Boltz-2 / Chai-1 / ColabFold -> "
                                            "uniform scores -> MD / agents.")
    p.add_argument("--version", action="version", version=f"foldbridge {__version__}")
    p.add_argument("--debug", action="store_true", help="show tracebacks on errors")
    # --debug is also accepted after the subcommand; SUPPRESS keeps the subparser from
    # resetting a top-level --debug to False
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--debug", action="store_true", default=argparse.SUPPRESS,
                        help="show tracebacks on errors")
    sub = p.add_subparsers(dest="command", metavar="COMMAND")

    s = sub.add_parser("backends", parents=[common], help="list backends and whether they are installed")
    s.add_argument("--json", action="store_true", help="machine-readable output")
    s.set_defaults(func=cmd_backends)

    s = sub.add_parser("run", parents=[common], help="run a prediction from a job spec")
    s.add_argument("spec", help="job spec (.yaml/.json/.fasta)")
    s.add_argument("--backend", "-b", required=True, choices=["boltz", "chai", "colabfold"])
    s.add_argument("--out", "-o", help="output directory (default foldbridge_out/<name>-<backend>)")
    s.add_argument("--cpu", action="store_true", help="run on CPU")
    s.add_argument("--dry-run", action="store_true", help="write backend inputs + command.txt, don't run")
    s.add_argument("--no-cache", action="store_true", help="ignore and don't write the result cache")
    s.add_argument("--cache-dir", help="cache root (default $FOLDBRIDGE_CACHE or ~/.cache/foldbridge)")
    s.add_argument("--score", action="store_true", help="compute uniform metrics for each model")
    s.add_argument("--reference", help="reference structure for RMSD/DockQ (overrides spec)")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("score", parents=[common], help="score structure files / directories")
    s.add_argument("paths", nargs="+", metavar="PATH")
    s.add_argument("--pae", help="PAE file (.npz/.npy/.json) for a single structure")
    s.add_argument("--reference", help="reference structure for RMSD/DockQ")
    s.add_argument("--out", "-o", default="results.csv", help="output .csv/.json/.html (default results.csv)")
    s.add_argument("--chains", help="comma-separated chain ids to restrict metrics to (e.g. A,B)")
    s.add_argument("--workers", "-j", type=int,
                   help="parallel processes (default: CPUs - 1; in-process for < 4 files)")
    s.add_argument("--checkpoint", metavar="FILE",
                   help="append each finished row to this JSONL file; rerun to resume (skips done files)")
    s.add_argument("--no-progress", action="store_true", help="no progress line on stderr")
    s.set_defaults(func=cmd_score)

    s = sub.add_parser("rank", parents=[common], help="rank rows from results files")
    s.add_argument("results", nargs="+", metavar="RESULTS")
    s.add_argument("--by", default="composite", help="metric to sort by (default composite)")
    s.add_argument("--no-length-norm", action="store_true", help="don't length-normalise the composite")
    s.add_argument("--top", type=int, help="keep only the top N")
    s.add_argument("--html", help="write an HTML report")
    s.add_argument("--out", "-o", help="write ranked rows (.csv/.json)")
    s.set_defaults(func=cmd_rank)

    s = sub.add_parser("prep-md", parents=[common], help="clean/protonate/solvate a structure for MD")
    s.add_argument("structure")
    s.add_argument("--out", "-o", help="output directory (default ./<stem>_md in the current directory)")
    s.add_argument("--ph", type=float, default=7.0,
                   help="pH for protonation states (default 7.0; used by the openmm/pdbfixer path)")
    s.add_argument("--keep-ligands", action="store_true", help="export ligands for parameterisation")
    s.add_argument("--solvate", action="store_true", help="add TIP3P water + 0.15 M NaCl (needs openmm)")
    s.add_argument("--padding", type=float, default=1.0, help="solvent padding in nm (default 1.0)")
    s.add_argument("--plddt-cutoff", type=float, default=70.0,
                   help="restrain CA atoms with pLDDT >= this (default 70)")
    s.set_defaults(func=cmd_prep_md)

    s = sub.add_parser("campaign", parents=[common], help="binder-design screen: fold many binders against one target, rank")
    s.add_argument("--target", "-t", required=True, help="job spec with the target chain(s) / ligands")
    s.add_argument("--designs", "-d", required=True, help="FASTA of binder sequences (one record per design)")
    s.add_argument("--backend", "-b", required=True, choices=["boltz", "chai", "colabfold"])
    s.add_argument("--out", "-o", required=True, help="campaign directory (rerun with the same one to resume)")
    s.add_argument("--samples", type=int, help="samples per seed (default: from the target spec)")
    s.add_argument("--seeds", help="comma-separated seeds (default: from the target spec)")
    s.add_argument("--dry-run", action="store_true", help="write every design's inputs + commands.txt only")
    s.add_argument("--max-jobs", type=int, metavar="N",
                   help="process at most N not-yet-done designs in this invocation (rerun to continue)")
    s.add_argument("--top", type=int, default=10, help="write the N best binders to top_N.fasta (default 10)")
    s.add_argument("--binder-chain", help="chain id for the binder (default B, or the next free id)")
    s.add_argument("--binder-msa", choices=["auto", "single", "inherit"], default="auto",
                   help="binder MSA: single-sequence a3m, or the target spec's msa mode "
                        "(auto: single unless the spec uses server/none)")
    s.add_argument("--cpu", action="store_true", help="run predictions on CPU")
    s.add_argument("--no-cache", action="store_true", help="ignore and don't write the result cache")
    s.add_argument("--cache-dir", help="cache root (default $FOLDBRIDGE_CACHE or ~/.cache/foldbridge)")
    s.add_argument("--workers", "-j", type=int, help="scoring processes (default: CPUs - 1)")
    s.add_argument("--no-progress", action="store_true", help="no scoring progress line on stderr")
    s.set_defaults(func=cmd_campaign)

    s = sub.add_parser("mcp", parents=[common], help="start the MCP server (stdio)")
    s.set_defaults(func=cmd_mcp)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    debug = args.debug or os.environ.get("FOLDBRIDGE_DEBUG") not in (None, "", "0")
    if not getattr(args, "func", None):
        parser.print_help(sys.stderr)
        return EXIT_USAGE
    try:
        return int(args.func(args) or 0)
    except CLIError as e:
        if debug:
            raise
        print(f"foldbridge: error: {e}", file=sys.stderr)
        return e.code
    except KeyboardInterrupt:
        print("foldbridge: interrupted", file=sys.stderr)
        return 130
    except (FileNotFoundError, IsADirectoryError, PermissionError) as e:
        if debug:
            raise
        print(f"foldbridge: error: {e}", file=sys.stderr)
        return EXIT_USAGE
    except ImportError as e:
        if debug:
            raise
        print(f"foldbridge: error: missing dependency: {e}", file=sys.stderr)
        return EXIT_FAIL
    except Exception as e:  # unexpected: short message, traceback only with --debug
        if debug:
            raise
        print(f"foldbridge: error: {type(e).__name__}: {e} (rerun with --debug for a traceback)",
              file=sys.stderr)
        return EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
