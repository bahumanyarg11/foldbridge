"""Score many structures in parallel, with PAE auto-pairing, per-file error capture,
a dependency-free progress line and resumable JSONL checkpoints.

    from foldbridge.batch import score_many
    rows = score_many(["models/"], workers=8, checkpoint="scores.jsonl")

Every input yields exactly one row, in input order. A file that cannot be scored gets a
row with an ``error`` field instead of stopping the batch. With ``checkpoint`` each finished
row is appended to a JSONL file as soon as it is done; a restarted call skips files whose
row is already there (same path, size, mtime, PAE, reference and chains), so an interrupted
1000-file batch resumes where it stopped.

This module must not import :mod:`foldbridge.cli` (the CLI imports it).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from functools import lru_cache
from pathlib import Path
from typing import Any

__all__ = ["score_many", "expand_paths", "find_pae", "STRUCT_SUFFIXES", "PAE_SUFFIXES",
           "default_workers", "Progress"]

STRUCT_SUFFIXES = {".cif", ".mmcif", ".pdb", ".ent"}
PAE_SUFFIXES = (".npz", ".npy", ".json")
CHECKPOINT_VERSION = 1
SMALL_BATCH = 4  # with workers=None, batches smaller than this run in-process
WARNING_SEP = " | "  # joins several per-file warnings in the ``warnings`` column


# ---- discovery ----------------------------------------------------------------------

def _pae_candidates(model: Path) -> list[Path]:
    d, stem = model.parent, model.stem
    names: list[str] = []
    for suf in PAE_SUFFIXES:
        names += [f"pae_{stem}{suf}", f"{stem}_pae{suf}", f"{stem}.pae{suf}", f"{stem}_full_pae{suf}"]
    # Chai-1: pred.model_idx_0.cif -> pae.model_idx_0.npy
    if stem.startswith("pred."):
        names += [f"pae.{stem[5:]}{suf}" for suf in PAE_SUFFIXES]
    # ColabFold: X_unrelaxed_rank_001_..._seed_000.pdb -> X_scores_rank_001_..._seed_000.json
    for tag in ("_unrelaxed_", "_relaxed_"):
        if tag in stem:
            names.append(stem.replace(tag, "_scores_", 1) + ".json")
            names.append(stem.split(tag, 1)[0] + "_predicted_aligned_error_v1.json")
    # AF3 style: X_model.cif -> X_confidences.json
    if stem.endswith("_model"):
        names.append(stem[: -len("_model")] + "_confidences.json")
    return [d / n for n in names]


def find_pae(model: str | Path) -> Path | None:
    """PAE file that sits next to ``model`` under a known naming convention, if any."""
    for c in _pae_candidates(Path(model)):
        if c.is_file():
            return c
    return None


def expand_paths(paths: Iterable[str | Path]) -> list[Path]:
    """Files as given; directories searched recursively for structure files (sorted).

    Duplicates (same resolved path) are dropped. Missing paths are kept so that the caller
    reports them as per-file errors rather than failing the whole batch.
    """
    out: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            out.extend(sorted(q for q in p.rglob("*") if q.is_file() and q.suffix.lower() in STRUCT_SUFFIXES))
        else:
            out.append(p)
    seen, uniq = set(), []
    for p in out:
        k = p.resolve()
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


# ---- worker -------------------------------------------------------------------------

def _plain(x: Any) -> Any:
    """JSON-friendly copy that keeps NaN (checkpoint rows must round-trip exactly)."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    try:
        return _plain(x.item())
    except Exception:
        return str(x)


def _stat(path: str | Path | None) -> tuple[int | None, int | None]:
    """(size, mtime_ns) of a file, (None, None) when missing / not given."""
    if path is None:
        return None, None
    try:
        st = Path(path).stat()
        return st.st_size, st.st_mtime_ns
    except OSError:
        return None, None


@lru_cache(maxsize=4)
def _reference_cached(path: str, size: int | None, mtime_ns: int | None):
    from .structure import read_structure

    return read_structure(path)


def _reference(path: str):
    """Parse a reference once per process instead of once per scored file (re-read if it changed)."""
    return _reference_cached(path, *_stat(path))


def _score_one(path: str, pae: str | None, reference: str | None,
               chains: tuple[str, ...] | None) -> dict[str, Any]:
    """Score one file; never raises (errors become an ``error`` field)."""
    row: dict[str, Any] = {"id": Path(path).stem, "path": path, "pae_path": pae}
    try:
        from .score import score_structure

        if not Path(path).is_file():
            raise FileNotFoundError(f"no such file: {path}")
        ref = _reference(reference) if reference else None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            s = score_structure(path, pae=pae, reference=ref, chains=list(chains) if chains else None)
        if not s.get("n_atoms"):
            raise ValueError("no atoms could be read (empty or not a PDB/mmCIF file)")
        row.update(s)
        msgs = list(dict.fromkeys(str(w.message) for w in caught))
        if msgs:
            row["warnings"] = WARNING_SEP.join(msgs)
    except Exception as e:  # noqa: BLE001 - per-file capture is the point
        row["error"] = f"{type(e).__name__}: {e}"
    return _plain(row)


# ---- progress -----------------------------------------------------------------------

def _hms(sec: float) -> str:
    if not math.isfinite(sec):
        return "--:--"
    sec = int(round(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class Progress:
    """``label n/N  rate/s  ETA mm:ss`` on stderr: rewritten in place on a TTY, otherwise
    one line roughly every 10 % (so logs stay readable)."""

    def __init__(self, total: int, label: str = "score", stream=None, enabled: bool = True,
                 min_interval: float = 0.2):
        self.total, self.label, self.enabled = total, label, enabled and total > 0
        self.stream = stream or sys.stderr
        self.tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.n = self.errors = self.cached = 0
        self.t0 = time.monotonic()
        self._last = 0.0
        self._last_pct = -1
        self.min_interval = min_interval

    def line(self) -> str:
        dt = max(time.monotonic() - self.t0, 1e-9)
        fresh = self.n - self.cached
        rate = fresh / dt  # only newly scored files count towards the rate
        left = self.total - self.n
        eta = 0.0 if left <= 0 else (left / rate if rate > 0 else float("inf"))
        cached = f" ({self.cached} cached)" if self.cached else ""
        err = f"  {self.errors} failed" if self.errors else ""
        return f"{self.label} {self.n}/{self.total}{cached}  {rate:.1f}/s  ETA {_hms(eta)}{err}"

    def skip(self, k: int) -> None:
        """Count ``k`` rows taken from a checkpoint (shown as cached, excluded from the rate)."""
        self.cached += k
        self.update(k)

    def update(self, k: int = 1, error: bool = False) -> None:
        self.n += k
        self.errors += int(error)
        if not self.enabled:
            return
        now = time.monotonic()
        done = self.n >= self.total
        if self.tty:
            if done or now - self._last >= self.min_interval:
                self._last = now
                print("\r" + self.line() + "\033[K", end="", file=self.stream, flush=True)
        else:
            pct = int(10 * self.n / self.total)
            if done or pct > self._last_pct:
                self._last_pct = pct
                print(self.line(), file=self.stream, flush=True)

    def close(self) -> None:
        if self.enabled and self.tty:
            print(file=self.stream, flush=True)


# ---- checkpoint ---------------------------------------------------------------------

def _signature(path: Path, pae: str | None, reference: str | None,
               chains: tuple[str, ...] | None) -> dict[str, Any]:
    """What a checkpointed row depends on: the model, PAE and reference files (path, size,
    mtime) and the chain selection."""
    size, mtime = _stat(path)
    return {"size": size, "mtime_ns": mtime, "pae": pae, "pae_stat": list(_stat(pae)) if pae else None,
            "reference": reference, "reference_stat": list(_stat(reference)) if reference else None,
            "chains": list(chains) if chains else None}


def _read_checkpoint(path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    """(resolved path -> latest record, number of lines read). Later lines win (dedupe by
    path); torn / foreign lines are ignored."""
    done: dict[str, dict[str, Any]] = {}
    n_lines = 0
    if not path.is_file():
        return done, 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # half-written last line after a crash
            if isinstance(rec, dict) and isinstance(rec.get("row"), dict) and "key" in rec:
                done.pop(rec["key"], None)  # keep file order = order of last write
                done[rec["key"]] = rec
    return done, n_lines


def _load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    """resolved path -> {"sig": ..., "row": ...}; later lines win; torn lines ignored."""
    return _read_checkpoint(path)[0]


def compact_checkpoint(path: str | Path) -> bool:
    """Rewrite a checkpoint with one line per file (the last one written). Atomic; returns
    True when the file changed."""
    path = Path(path)
    recs, n_lines = _read_checkpoint(path)
    if n_lines == len(recs):
        return False
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for rec in recs.values():
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
    os.replace(tmp, path)
    return True


class _CheckpointWriter:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        # a crash may have left a torn last line without "\n"; start on a fresh line
        needs_nl = path.is_file() and path.stat().st_size > 0 and not path.read_bytes().endswith(b"\n")
        self.fh = open(path, "a", encoding="utf-8")
        if needs_nl:
            self.fh.write("\n")

    def write(self, key: str, sig: dict[str, Any], row: dict[str, Any]) -> None:
        rec = {"v": CHECKPOINT_VERSION, "key": key, "sig": sig, "row": row}
        self.fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


# ---- main entry ---------------------------------------------------------------------

def _resolve_pae(files: Sequence[Path], pae: Any) -> list[str | None]:
    if pae is None:
        return [str(p) if (p := find_pae(f)) else None for f in files]
    if isinstance(pae, Mapping):
        table = {str(Path(k).resolve()): v for k, v in pae.items()}
        out = []
        for f in files:
            v = table.get(str(f.resolve()), "auto")
            if v == "auto":
                v = find_pae(f)
            out.append(str(v) if v else None)
        return out
    if len(files) != 1:
        raise ValueError(f"a single PAE file was given but {len(files)} structures; pass a mapping")
    return [str(pae)]


def score_many(paths: Iterable[str | Path], reference: str | Path | None = None,
               workers: int | None = None, chains: Sequence[str] | None = None,
               progress: bool = True, checkpoint: str | Path | None = None,
               pae: str | Path | Mapping[str | Path, str | Path | None] | None = None,
               retry_errors: bool = True,
               on_row: Callable[[dict[str, Any]], None] | None = None) -> list[dict[str, Any]]:
    """Score structure files (directories are searched recursively), one row per file.

    Parameters
    ----------
    paths : files and/or directories.
    reference : structure for RMSD / DockQ-lite (parsed once per worker process).
    workers : process count; default ``os.cpu_count() - 1`` (in-process when the batch has
        fewer than 4 files). ``workers <= 1`` always runs in-process.
    chains : restrict metrics to these chain ids.
    progress : print ``n/N, rate, ETA`` to stderr.
    checkpoint : JSONL file; finished rows are appended as they complete and files already
        recorded there (unchanged size/mtime and same PAE/reference/chains) are skipped.
    pae : PAE for a single structure, or a mapping ``{structure: pae}`` (missing entries
        fall back to auto-pairing; a ``None`` value means "no PAE"). Default: auto-pair each
        file with a PAE next to it (Boltz ``pae_<stem>.npz``, Chai ``pae.<...>.npy``,
        ColabFold ``*_scores_*.json``, AF3 ``*_confidences.json``, ...).
    retry_errors : re-score files whose checkpointed row has an ``error`` (default True).
    on_row : called in the parent process with each freshly computed row.

    Returns rows in input order: ``id``, ``path``, ``pae_path`` + metrics, or ``error``.
    """
    files = expand_paths(paths)
    if not files:
        return []
    ref = str(Path(reference).resolve()) if reference is not None else None
    ch = tuple(str(c) for c in chains) if chains else None
    paes = _resolve_pae(files, pae)
    keys = [str(f.resolve()) for f in files]
    sigs = [_signature(f, p, ref, ch) for f, p in zip(files, paes)]

    rows: list[dict[str, Any] | None] = [None] * len(files)
    prev: dict[str, dict[str, Any]] = {}
    if checkpoint is not None:
        ck_path = Path(checkpoint)
        prev = _load_checkpoint(ck_path)
        for i, (k, s) in enumerate(zip(keys, sigs)):
            rec = prev.get(k)
            if rec is None or rec.get("sig") != s:
                continue
            if retry_errors and "error" in rec["row"]:
                continue
            row = dict(rec["row"])
            row["path"] = str(files[i])  # report the path as the caller spelled it now
            rows[i] = row
    todo = [i for i, r in enumerate(rows) if r is None]

    n_workers = default_workers() if workers is None else int(workers)
    if workers is None and len(todo) < SMALL_BATCH:
        n_workers = 1
    n_workers = max(1, min(n_workers, len(todo) or 1))

    bar = Progress(len(files), enabled=progress)
    if len(files) - len(todo):
        bar.skip(len(files) - len(todo))
    writer = _CheckpointWriter(Path(checkpoint)) if checkpoint is not None and todo else None

    def finish(i: int, row: dict[str, Any]) -> None:
        row["path"] = str(files[i])
        rows[i] = row
        if writer is not None:
            old = prev.get(keys[i])
            # an unchanged failure is not appended again on every retry
            if not (old is not None and old.get("sig") == sigs[i] and old.get("row") == row):
                writer.write(keys[i], sigs[i], row)
        if on_row is not None:
            on_row(row)
        bar.update(error="error" in row)

    try:
        if n_workers > 1:
            try:
                _score_pool(n_workers, todo, keys, paes, ref, ch, finish)
            except BrokenProcessPool:
                print("warning: worker processes could not start or died (BrokenProcessPool); scoring the "
                      "remaining files serially in-process. When calling score_many from a script, put the "
                      "call under `if __name__ == \"__main__\":` (needed for multiprocessing with the "
                      "spawn start method on macOS/Windows), or pass workers=1.", file=sys.stderr, flush=True)
        for i in todo:
            if rows[i] is None:
                finish(i, _score_one(keys[i], paes[i], ref, ch))
    finally:
        bar.close()
        if writer is not None:
            writer.close()
            try:
                compact_checkpoint(Path(checkpoint))  # type: ignore[arg-type]
            except OSError:
                pass
    out = [r for r in rows if r is not None]
    _disambiguate_ids(out, files)
    return out


def _score_pool(n_workers: int, todo: list[int], keys: list[str], paes: list[str | None],
                ref: str | None, ch: tuple[str, ...] | None, finish: Callable[[int, dict[str, Any]], None]) -> None:
    """Score ``todo`` in a process pool. Raises BrokenProcessPool if the pool breaks (the
    caller scores whatever is left in-process)."""
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_score_one, keys[i], paes[i], ref, ch): i for i in todo}
        try:
            for fut in as_completed(futs):
                finish(futs[fut], fut.result())
        except BaseException:
            for f in futs:
                f.cancel()
            raise


def _disambiguate_ids(rows: list[dict[str, Any]], files: Sequence[Path]) -> None:
    """ids default to the file stem; files sharing a stem (1A2K.pdb + 1A2K.cif) get their
    file name instead, and files sharing a name get the path as given."""
    if len(rows) != len(files):
        return
    stems: dict[str, int] = {}
    names: dict[str, int] = {}
    for f in files:
        stems[f.stem] = stems.get(f.stem, 0) + 1
        names[f.name] = names.get(f.name, 0) + 1
    for r, f in zip(rows, files):
        if r.get("id") != f.stem or stems[f.stem] < 2:
            continue
        r["id"] = f.name if names[f.name] < 2 else str(f)
