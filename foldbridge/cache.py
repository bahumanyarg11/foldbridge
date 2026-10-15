"""Content-addressed cache for prediction results and MSAs.

Layout (root defaults to ``$FOLDBRIDGE_CACHE`` or ``~/.cache/foldbridge``)::

    <root>/results/<key>/manifest.json      # list of serialised ModelResults
    <root>/results/<key>/files/00_<name>    # copied model / PAE files
    <root>/msa/<sha256(seq)>.a3m            # one a3m per unique protein sequence

``key`` is ``JobSpec.content_hash(backend, backend_version)``, so a changed
spec, backend or backend version gives a fresh entry. Writes are atomic: files
are staged in a temp dir next to the target and moved in with ``os.replace``.
Temp dirs/files get the umask-derived mode (``mkdtemp``/``mkstemp`` default to
owner-only), so a shared ``FOLDBRIDGE_CACHE`` stays group/other-readable when
the umask allows it.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .backends.base import ModelResult

MANIFEST = "manifest.json"
MANIFEST_VERSION = 1


def _umask() -> int:
    mask = os.umask(0o022)
    os.umask(mask)
    return mask


def _chmod_default(path: str | Path, is_dir: bool) -> None:
    """Give a mkdtemp/mkstemp path the mode a plain mkdir/open would have."""
    os.chmod(path, (0o777 if is_dir else 0o666) & ~_umask())


# ---- ModelResult (de)serialisation -------------------------------------------

def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value
    try:  # numpy scalars and friends
        return value.item()
    except Exception:
        return str(value)


def result_to_dict(r: ModelResult, base: str | Path | None = None) -> dict[str, Any]:
    """Serialise a ModelResult. Paths are stored relative to ``base`` when given."""
    def rel(p: Path | None) -> str | None:
        if p is None:
            return None
        p = Path(p)
        if base is not None:
            try:
                return str(p.resolve().relative_to(Path(base).resolve()))
            except ValueError:
                pass
        return str(p)

    return {
        "backend": r.backend,
        "path": rel(r.path),
        "seed": int(r.seed),
        "sample": int(r.sample),
        "scores": {str(k): _jsonable(v) for k, v in (r.scores or {}).items()},
        "pae_path": rel(r.pae_path),
        "extra": _jsonable(r.extra or {}),
    }


def result_from_dict(d: dict[str, Any], base: str | Path | None = None) -> ModelResult:
    """Inverse of :func:`result_to_dict`. Relative paths are resolved against ``base``."""
    def absp(p: str | None) -> Path | None:
        if p is None:
            return None
        q = Path(p)
        if not q.is_absolute() and base is not None:
            q = Path(base) / q
        return q

    path = absp(d["path"])
    assert path is not None
    return ModelResult(
        backend=d.get("backend", "unknown"),
        path=path,
        seed=int(d.get("seed", 0)),
        sample=int(d.get("sample", 0)),
        scores=dict(d.get("scores") or {}),
        pae_path=absp(d.get("pae_path")),
        extra=dict(d.get("extra") or {}),
    )


def sequence_hash(seq: str) -> str:
    norm = "".join(seq.split()).upper()
    return hashlib.sha256(norm.encode()).hexdigest()


def default_root() -> Path:
    env = os.environ.get("FOLDBRIDGE_CACHE")
    return Path(env).expanduser() if env else Path("~/.cache/foldbridge").expanduser()


# ---- cache ---------------------------------------------------------------------

class Cache:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root).expanduser() if root is not None else default_root()

    def __repr__(self) -> str:
        return f"Cache({str(self.root)!r})"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    @property
    def msa_dir(self) -> Path:
        return self.root / "msa"

    @staticmethod
    def key_for(spec, backend: str, backend_version: str | None = None) -> str:
        return spec.content_hash(backend, backend_version)

    # ---- results -------------------------------------------------------------
    def path_for(self, key: str) -> Path:
        if not key or "/" in key or key.startswith("."):
            raise ValueError(f"invalid cache key {key!r}")
        return self.results_dir / key

    def has(self, key: str) -> bool:
        return (self.path_for(key) / MANIFEST).is_file()

    def manifest(self, key: str) -> dict[str, Any] | None:
        mf = self.path_for(key) / MANIFEST
        if not mf.is_file():
            return None
        try:
            return json.loads(mf.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def load(self, key: str) -> list[ModelResult] | None:
        """Cached results with absolute paths, or None on miss / corrupt entry."""
        man = self.manifest(key)
        if man is None:
            return None
        base = self.path_for(key)
        try:
            results = [result_from_dict(d, base) for d in man.get("results", [])]
        except (KeyError, TypeError, ValueError):
            return None
        for r in results:
            if not r.path.exists() or (r.pae_path is not None and not r.pae_path.exists()):
                return None  # files went missing -> treat as a miss
        return results

    def store(self, key: str, results: Iterable[ModelResult], workdir: str | Path | None = None,
              meta: dict[str, Any] | None = None) -> list[ModelResult]:
        """Copy model/PAE files into the cache and write the manifest atomically.

        Returns the results re-pointed at the cached copies. ``workdir`` is only
        used to resolve relative result paths.
        """
        target = self.path_for(key)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=f".tmp-{key}-", dir=self.results_dir))
        try:
            _chmod_default(tmp, is_dir=True)
            (tmp / "files").mkdir()
            stored: list[ModelResult] = []
            for i, r in enumerate(results):
                src = self._resolve(r.path, workdir)
                if not src.is_file():
                    raise FileNotFoundError(f"cannot cache missing model file {src}")
                dst = tmp / "files" / f"{i:02d}_{src.name}"
                shutil.copy2(src, dst)
                pae_rel = None
                if r.pae_path is not None:
                    psrc = self._resolve(r.pae_path, workdir)
                    if psrc.is_file():
                        pdst = tmp / "files" / f"{i:02d}_{psrc.name}"
                        shutil.copy2(psrc, pdst)
                        pae_rel = pdst
                stored.append(ModelResult(r.backend, dst, r.seed, r.sample, dict(r.scores),
                                          pae_rel, dict(r.extra)))
            manifest = {
                "manifest_version": MANIFEST_VERSION,
                "key": key,
                "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "meta": _jsonable(meta or {}),
                "results": [result_to_dict(r, tmp) for r in stored],
            }
            (tmp / MANIFEST).write_text(json.dumps(manifest, indent=2))
            # swap in: move any old entry aside first so the final rename is atomic
            old = None
            if target.exists():
                old = target.with_name(f".old-{key}-{os.getpid()}-{time.time_ns()}")
                os.replace(target, old)
            os.replace(tmp, target)
            if old is not None:
                shutil.rmtree(old, ignore_errors=True)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return self.load(key) or []

    def remove(self, key: str) -> bool:
        p = self.path_for(key)
        if p.exists():
            shutil.rmtree(p)
            return True
        return False

    def keys(self) -> list[str]:
        if not self.results_dir.is_dir():
            return []
        return sorted(p.name for p in self.results_dir.iterdir()
                      if not p.name.startswith(".") and (p / MANIFEST).is_file())

    @staticmethod
    def _resolve(p: str | Path, workdir: str | Path | None) -> Path:
        p = Path(p)
        if not p.is_absolute() and workdir is not None:
            p = Path(workdir) / p
        return p

    # ---- MSAs ----------------------------------------------------------------
    def msa_file(self, seq: str) -> Path:
        """Canonical cache location for ``seq``'s a3m (may not exist)."""
        return self.msa_dir / f"{sequence_hash(seq)}.a3m"

    def msa_path(self, seq: str) -> Path | None:
        """Cached a3m for this protein sequence, or None."""
        p = self.msa_file(seq)
        return p if p.is_file() else None

    def has_msa(self, seq: str) -> bool:
        return self.msa_path(seq) is not None

    def store_msa(self, seq: str, a3m_path: str | Path) -> Path:
        src = Path(a3m_path)
        if not src.is_file():
            raise FileNotFoundError(f"a3m not found: {src}")
        return self._write_msa(seq, lambda tmp: shutil.copyfile(src, tmp))

    def store_msa_text(self, seq: str, text: str) -> Path:
        """Store a3m content given as a string (e.g. converted from Boltz's csv)."""
        return self._write_msa(seq, lambda tmp: Path(tmp).write_text(text))

    def _write_msa(self, seq: str, fill) -> Path:
        dst = self.msa_file(seq)
        dst.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".a3m", dir=dst.parent)
        os.close(fd)
        try:
            _chmod_default(tmp, is_dir=False)
            fill(tmp)
            os.replace(tmp, dst)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return dst


def a3m_query(path: str | Path) -> str | None:
    """First (query) sequence of an a3m file, gaps/insertions stripped, or None."""
    try:
        seq: list[str] = []
        started = False
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith(">"):
                    if started:
                        break
                    started = True
                    continue
                if started:
                    seq.append(line)
        q = "".join(ch for ch in "".join(seq) if ch.isupper())
        return q or None
    except (OSError, UnicodeDecodeError):
        return None


# Raw MMseqs2-server outputs (colabfold.colabfold.run_mmseqs2 writes them into
# <prefix>_<mode>/ dirs): env-only, uniref-only or paired-only, never a final MSA.
_RAW_A3M_NAMES = {"pair.a3m", "uniref.a3m", "bfd.mgnify30.metaeuk30.smag30.a3m"}
_RAW_A3M_WORDS = {"pair", "paired", "env"}            # whole words in the file stem
_RAW_A3M_PREFIXES = ("uniref", "bfd", "mgnify")        # database-named hit files
# Scratch dirs: Boltz passes run_mmseqs2 the prefixes <target>_paired_tmp /
# <target>_unpaired_tmp, which then appends _<mode> (env, all, pairgreedy, ...).
_RAW_DIR = re.compile(r"_tmp(_(env|all|nofilter|pair\w*))?$")


def _is_intermediate(path: Path, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    if any(_RAW_DIR.search(d) or d.startswith(".tmp") for d in parts[:-1]):
        return True
    name = path.name.lower()
    words = re.split(r"[^a-z0-9]+", Path(name).stem)
    return (name in _RAW_A3M_NAMES
            or any(w in _RAW_A3M_WORDS or w.startswith(_RAW_A3M_PREFIXES) for w in words))


def boltz_csv_to_a3m(path: str | Path) -> str | None:
    """Boltz's final per-entity MSA (``msa/<target>_<entity>.csv``: ``key,sequence``) as a3m.

    The first row is the query (Boltz puts the paired block, whose first row is
    the query, ahead of the unpaired rows). Pairing keys are dropped. None if
    the file is not a Boltz MSA csv.
    """
    try:
        lines = Path(path).read_text().splitlines()
    except (OSError, UnicodeDecodeError):
        return None
    if not lines or lines[0].strip().replace(" ", "") != "key,sequence":
        return None
    out = []
    for i, line in enumerate(lines[1:]):
        seq = line.split(",", 1)[1].strip() if "," in line else ""
        if seq:
            out += [f">{'query' if not out else f'row{i}'}", seq]
    return "\n".join(out) + "\n" if out else None


def harvest_msas(cache: Cache, workdir: str | Path, sequences: Iterable[str]) -> list[Path]:
    """Store final per-chain MSAs under ``workdir`` whose query matches ``sequences``.

    Picked up: plain ``*.a3m`` files (ColabFold's ``<job>.a3m``, staged inputs)
    and Boltz's per-entity ``msa/*.csv`` (converted to a3m). Skipped: MMseqs2
    server scratch (``*_tmp*`` / ``*_env`` / ``*_pair*`` dirs, ``pair.a3m``,
    ``uniref.a3m``, ``bfd...a3m``), which hold env-only, uniref-only or
    paired-only alignments, never the merged MSA a backend actually used.
    """
    root = Path(workdir)
    wanted = {"".join(s.split()).upper() for s in sequences}
    stored = []
    candidates = sorted(root.rglob("*.a3m")) + sorted(
        p for p in root.rglob("*.csv") if p.parent.name == "msa")
    for path in candidates:
        if _is_intermediate(path, root):
            continue
        if path.suffix == ".csv":
            text = boltz_csv_to_a3m(path)
            if text is None:
                continue
            q = "".join(ch for ch in text.splitlines()[1] if ch.isupper())
            if q in wanted and not cache.has_msa(q):
                stored.append(cache.store_msa_text(q, text))
            continue
        q = a3m_query(path)
        if q and q in wanted and not cache.has_msa(q):
            stored.append(cache.store_msa(q, path))
    return stored
