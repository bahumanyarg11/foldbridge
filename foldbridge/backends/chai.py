"""Chai backend (Chai-1 via the ``chai-lab`` CLI).

Native input is a typed FASTA, one record per chain copy / ligand copy::

    >protein|name=A
    MSKGEELFT...
    >ligand|name=L
    CC(=O)Oc1ccccc1C(=O)O          # SMILES as the "sequence"
    >dna|name=B
    ACGT...
    >rna|name=C
    ACGU...

Native output layout (relative to the output dir, ``<workdir>/chai_out``)::

    pred.model_idx_<i>.cif
    scores.model_idx_<i>.npz        # aggregate_score, ptm, iptm, per_chain_ptm,
                                    # per_chain_pair_iptm, has_inter_chain_clashes, (pae)

Chai refuses to write into a non-empty output directory, so outputs go into a
dedicated ``chai_out`` sub-directory rather than the workdir itself.

Chain naming: unless ``--fasta-names-as-cif-chains`` is passed, Chai names the
output chains A, B, C, ... (then a..z, AA, AB, ...) in FASTA record order, i.e.
spec chains expanded by copies, then ligands. ``collect()`` records the
mapping in ``ModelResult.extra["chain_map"]`` (output chain id -> spec copy id).

Explicit MSAs: each unique protein sequence's a3m is staged in its own
``<workdir>/msa_a3m/<seqhash>/`` dir and converted separately with
``chai-lab a3m-to-pqt`` (which merges *all* a3m files in a directory into one
query's MSA) into ``<workdir>/msas/<sha256(seq)>.aligned.pqt``, the directory
passed as ``--msa-directory``.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import shutil
import string
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..spec import Chain, JobSpec
from .base import Backend, ModelResult, register

OUT_SUBDIR = "chai_out"
MSA_SUBDIR = "msas"          # converted .aligned.pqt files (what Chai reads)
MSA_A3M_SUBDIR = "msa_a3m"   # raw a3m staging, one sub-dir per unique sequence
FASTA_NAMES_FLAG = "--fasta-names-as-cif-chains"
_MODEL_IDX = re.compile(r"model_idx_(\d+)$")
# chai_lab.data.io.cif_utils._CHAIN_VOCAB: A..Z, a..z, then two-letter combos
_CHAI_VOCAB = [*string.ascii_uppercase, *string.ascii_lowercase]
_CHAI_VOCAB = _CHAI_VOCAB + [x + y for x in _CHAI_VOCAB for y in _CHAI_VOCAB]


def _venv_bin_dir() -> Path:
    """Directory of the running interpreter (the venv ``bin``), unresolved."""
    return Path(sys.executable).parent


def find_executable(backend: Backend, exe: str) -> str | None:
    """``exe`` on PATH (via ``backend.which``) or next to ``sys.executable``."""
    try:
        hit = backend.which(exe)
        if hit:
            return hit
        for name in (exe, exe + ".exe") if os.name == "nt" else (exe,):
            cand = _venv_bin_dir() / name
            if cand.is_file() and os.access(cand, os.X_OK):
                return str(cand)
    except Exception:
        return None
    return None


def module_runnable(module: str) -> bool:
    """True if ``python -m <module>`` can be located (never raises)."""
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:
        return False


def resolve_cli(backend: Backend, exe: str, module: str | None) -> list[str] | None:
    """argv prefix for a tool CLI: the executable, else ``python -m module``, else None.

    ``module`` must be a module with an ``if __name__ == "__main__"`` entry point
    (checked upstream); pass None when only the console script is supported.
    """
    found = find_executable(backend, exe)
    if found:
        return [found]
    if module and module_runnable(module):
        return [sys.executable, "-m", module]
    return None


def spec_copy_ids(spec: JobSpec) -> list[tuple[str, str | None]]:
    """(copy id, sequence) in backend input order: chains x copies, then ligands.

    Ligands carry ``None`` as their sequence.
    """
    out: list[tuple[str, str | None]] = []
    for chain in spec.chains:
        out += [(cid, chain.sequence) for cid in chain.chain_ids()]
    for lig in spec.ligands:
        out += [(lid, None) for lid in [lig.id] + [f"{lig.id}{i}" for i in range(2, lig.copies + 1)]]
    return out


def _seq_hash(seq: str) -> str:
    return hashlib.sha256(seq.upper().encode()).hexdigest()


def _as_float(value: Any) -> float | None:
    """Best-effort scalar extraction from JSON / npz values."""
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.size == 0:
        return None
    val = float(arr.mean())
    return val if np.isfinite(val) else None


def _plddt_100(val: float | None) -> float | None:
    """Normalise a mean pLDDT to the 0-100 scale."""
    if val is None:
        return None
    return val * 100.0 if val <= 1.0 else val


def _a3m_query(path: Path) -> str | None:
    """First sequence of an a3m (the query), upper-cased with gaps removed."""
    try:
        buf: list[str] = []
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(">"):
                if buf:
                    break
                continue
            buf.append(line)
        seq = "".join(buf)
        return "".join(ch for ch in seq if ch.isupper()) or None
    except Exception:
        return None


def resolve_a3m(backend: str, spec: JobSpec, chain: Chain, msa_dir: Path | None) -> Path | None:
    """Explicit a3m for a protein chain, mirroring Boltz's lookup rules.

    None => no explicit MSA (job-level ``server`` / ``none`` mode applies).
    """
    if chain.type != "protein":
        return None
    if chain.msa:
        p = Path(chain.msa).expanduser().resolve()
        if not p.is_file():
            raise ValueError(f"{backend}: chain {chain.id} msa {chain.msa!r} does not exist")
        return p
    if spec.msa in ("none", "server"):
        return None
    search = Path(spec.msa).expanduser() if spec.msa != "cached" else None
    if search is not None and search.is_file():
        return search.resolve()
    looked: list[str] = []
    missing: list[str] = []
    for base in (search, msa_dir):
        if base is None:
            continue
        base = Path(base)
        if not base.is_dir():
            missing.append(str(base))
            continue
        looked.append(str(base))
        for cand in (base / f"{chain.id}.a3m", base / f"{spec.name}_{chain.id}.a3m"):
            if cand.exists():
                return cand.resolve()
        hits = sorted(base.glob("*.a3m"))
        # Single-file fallback only when the a3m's query really is this chain
        # (a heteromer must not silently reuse chain A's MSA for chain B).
        if len(hits) == 1 and _a3m_query(hits[0]) in (None, chain.sequence.upper()):
            return hits[0].resolve()
    raise ValueError(f"{backend}: msa={spec.msa!r} but no a3m found for chain {chain.id}"
                     f" ({_where(looked, missing)}); "
                     "set msa: server/none or point msa at an a3m file or directory")


def _where(looked: list[str], missing: list[str]) -> str:
    parts = []
    if looked:
        parts.append("looked for <chain>.a3m / <name>_<chain>.a3m in " + ", ".join(looked))
    if missing:
        parts.append("MSA directory " + ", ".join(missing) + " does not exist")
    if not parts:
        parts.append("no MSA directory was given to search")
    return "; ".join(parts)


@register
class ChaiBackend(Backend):
    """Run Chai-1 via its ``chai-lab fold`` CLI."""

    name = "chai"

    # Console script `chai-lab` = chai_lab.main:cli; chai_lab/main.py also has an
    # `if __name__ == "__main__": cli()` guard, so `python -m chai_lab.main` works.
    EXE = "chai-lab"
    MODULE = "chai_lab.main"

    # ---- discovery -------------------------------------------------------
    def _cli(self) -> list[str] | None:
        return resolve_cli(self, self.EXE, self.MODULE)

    def available(self) -> bool:
        """True if `chai-lab` is on PATH / in the venv bin, or `python -m chai_lab.main` resolves."""
        return self._cli() is not None

    def version(self) -> str:
        exe = find_executable(self, self.EXE)
        if exe:
            try:
                proc = subprocess.run(
                    [exe, "--version"], capture_output=True, text=True, timeout=20
                )
                out = (proc.stdout or proc.stderr or "").strip()
                if proc.returncode == 0 and out:  # usage errors must not become a "version"
                    return out.splitlines()[0].strip()
            except Exception:
                pass
        for dist in ("chai_lab", "chai-lab"):
            try:
                from importlib.metadata import version as _dist_version

                return _dist_version(dist)
            except Exception:
                continue
        return "unknown"

    # ---- lifecycle -------------------------------------------------------
    def prepare(self, spec: JobSpec, workdir: Path, msa_dir: Path | None = None) -> Path:
        """Write ``<workdir>/<name>.fasta`` (Chai typed FASTA). Returns that path.

        Explicit a3m files (chain.msa / cached / msa dir) are staged one unique
        sequence per ``<workdir>/msa_a3m/<seqhash>/`` dir and each converted to
        ``<workdir>/msas/<sha256>.aligned.pqt`` (passed via ``--msa-directory``).
        Mixing explicit MSAs with ``msa: server`` chains is rejected: Chai cannot
        combine ``--msa-directory`` with ``--use-msa-server``.
        """
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        spec.validate()

        records: list[str] = []
        staged: dict[str, tuple[str, Path]] = {}  # SEQ -> (chain id, a3m)
        no_msa: list[Chain] = []
        for chain in spec.chains:
            a3m = resolve_a3m(self.name, spec, chain, msa_dir)
            for cid in chain.chain_ids():
                records.append(f">{chain.type}|name={cid}\n{chain.sequence}")
            if a3m is not None:
                staged.setdefault(chain.sequence.upper(), (chain.id, a3m))
            elif chain.type == "protein":
                no_msa.append(chain)
        # Chai looks MSAs up by sequence, so an identical-sequence chain is covered.
        server_chains = [c.id for c in no_msa
                         if spec.msa == "server" and c.sequence.upper() not in staged]
        if staged and server_chains:
            raise ValueError(
                f"chai: chains {sorted(c for c, _ in staged.values())} have explicit MSAs but "
                f"chains {server_chains} rely on msa: server; Chai cannot use --msa-directory "
                "and --use-msa-server together. Give every protein chain an a3m, or drop the "
                "explicit ones and let the server build all MSAs"
            )

        for lig in spec.ligands:
            if not lig.smiles:
                raise ValueError(
                    f"chai: ligand {lig.id} has only a CCD code ({lig.ccd!r}); "
                    "Chai-1 FASTA input needs a SMILES string"
                )
            lig_ids = [lig.id] + [f"{lig.id}{i}" for i in range(2, lig.copies + 1)]
            for lid in lig_ids:
                records.append(f">ligand|name={lid}\n{lig.smiles}")

        a3m_root, msa_out = workdir / MSA_A3M_SUBDIR, workdir / MSA_SUBDIR
        for stale in (a3m_root, msa_out):  # foldbridge-owned; never mix runs
            shutil.rmtree(stale, ignore_errors=True)
        if staged:
            msa_out.mkdir(parents=True)
            for seq, (cid, src) in staged.items():
                sub = a3m_root / _seq_hash(seq)[:16]
                sub.mkdir(parents=True)
                # a3m-to-pqt infers the source database from the file name
                # (hits_<db>.a3m; unknown names default to uniref90).
                shutil.copyfile(src, sub / src.name)
                self._convert_a3m(sub, msa_out, cid, seq)

        path = workdir / f"{self._job_stem(spec)}.fasta"
        path.write_text("\n".join(records) + "\n")
        return path

    def command(
        self, spec: JobSpec, workdir: Path, input_path: Path, gpu: bool = True
    ) -> list[str]:
        """argv for ``chai-lab fold``. Results land under ``<workdir>/chai_out``."""
        workdir = Path(workdir)
        cli = self._cli() or [self.EXE]
        # Chai takes a single --seed; extra seeds need extra runs (one workdir each).
        seed = spec.seeds[0] if spec.seeds else 0
        argv = [
            *cli,
            "fold",
            str(Path(input_path).resolve()),
            str((workdir / OUT_SUBDIR).resolve()),
            "--num-trunk-recycles",
            str(max(0, spec.recycles)),
            "--num-diffn-samples",
            str(max(1, spec.samples)),
            "--seed",
            str(seed),
        ]
        msa_out = workdir / MSA_SUBDIR
        if msa_out.is_dir():
            # Chai reads <sha256(SEQ)>.aligned.pqt from here; see _convert_a3m.
            argv += ["--msa-directory", str(msa_out.resolve())]
        elif spec.msa == "server":
            argv.append("--use-msa-server")
        if not gpu:
            argv += ["--device", "cpu"]
        argv += [str(a) for a in spec.extra.get("chai_args", [])]
        return argv

    def collect(self, spec: JobSpec, workdir: Path) -> list[ModelResult]:
        """Parse ``scores.model_idx_<i>.npz`` next to each ``pred.model_idx_<i>.cif``, best-first.

        Common score keys: ptm, iptm, plddt (0-100), ranking (= Chai aggregate_score).

        Chai renames chains: the output CIF uses A, B, C, ... in FASTA order
        (not the spec ids) unless ``--fasta-names-as-cif-chains`` is in
        ``chai_args``. ``extra["chain_map"]`` maps output chain id -> spec copy
        id (A, A2, ..., L, L2); anything comparing chains to the spec or a
        reference must translate through it.
        """
        workdir = Path(workdir)
        results: list[ModelResult] = []
        seed = spec.seeds[0] if spec.seeds else 0
        chain_map = self.chain_map(spec)

        for npz in sorted(workdir.glob("**/scores.*model_idx_*.npz")):
            try:
                stem = npz.name[len("scores.") : -len(".npz")]  # model_idx_<i> (maybe trunk-prefixed)
                model = next(
                    (
                        p
                        for p in (npz.parent / f"pred.{stem}.cif", npz.parent / f"pred.{stem}.pdb")
                        if p.exists()
                    ),
                    None,
                )
                if model is None:
                    continue
                m = _MODEL_IDX.search(stem)
                idx = int(m.group(1)) if m else 0

                scores: dict[str, float] = {}
                extra: dict[str, Any] = {"record": stem, "scores_npz": str(npz),
                                         "chain_map": dict(chain_map)}
                pae_path: Path | None = None
                with np.load(npz, allow_pickle=False) as data:
                    for common, native in (
                        ("ranking", "aggregate_score"),
                        ("ptm", "ptm"),
                        ("iptm", "iptm"),
                        ("has_inter_chain_clashes", "has_inter_chain_clashes"),
                    ):
                        val = _as_float(data[native]) if native in data.files else None
                        if val is not None:
                            scores[common] = val
                    if "plddt" in data.files:
                        val = _plddt_100(_as_float(data["plddt"]))
                        if val is not None:
                            scores["plddt"] = val
                    for native in ("per_chain_ptm", "per_chain_pair_iptm"):
                        if native in data.files:
                            extra[native] = np.asarray(data[native], dtype=float).squeeze().tolist()
                    if "pae" in data.files:
                        pae = np.asarray(data["pae"], dtype=float)
                        while pae.ndim > 2 and pae.shape[0] == 1:
                            pae = pae[0]
                        if pae.ndim == 2:
                            pae_path = workdir / f"pae_{idx}.npy"
                            np.save(pae_path, pae)
                if "plddt" not in scores:
                    val = self._bfactor_plddt(model)
                    if val is not None:
                        scores["plddt"] = val

                results.append(
                    ModelResult(
                        backend=self.name,
                        path=model,
                        seed=seed,
                        sample=idx,
                        scores=scores,
                        pae_path=pae_path,
                        extra=extra,
                    )
                )
            except Exception:
                continue  # defensive: a single unreadable model must not kill the run

        results.sort(key=lambda r: r.scores.get("ranking", float("-inf")), reverse=True)
        return results

    # ---- helpers ---------------------------------------------------------
    @staticmethod
    def _job_stem(spec: JobSpec) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", spec.name) or "job"

    @staticmethod
    def _bfactor_plddt(model: Path) -> float | None:
        """Mean per-residue pLDDT from the model's B-factor column (0-100)."""
        try:
            from ..structure import read_structure

            s = read_structure(model)
            per_res = list(s.per_residue_bfactor().values())
            return _plddt_100(_as_float(per_res)) if per_res else None
        except Exception:
            return None

    @staticmethod
    def chain_map(spec: JobSpec) -> dict[str, str]:
        """Output CIF chain id -> spec copy id, mirroring Chai's naming rules."""
        ids = [cid for cid, _ in spec_copy_ids(spec)]
        if FASTA_NAMES_FLAG in [str(a) for a in spec.extra.get("chai_args", [])]:
            return {cid: cid for cid in ids}
        return {_CHAI_VOCAB[i]: cid for i, cid in enumerate(ids)}

    def _convert_a3m(self, a3m_dir: Path, msa_out: Path, chain_id: str, seq: str) -> None:
        """``chai-lab a3m-to-pqt <a3m_dir> --output-directory <msa_out>`` for ONE sequence.

        a3m-to-pqt merges every a3m in ``a3m_dir`` into a single query's MSA, so
        each unique sequence gets its own dir. Skipped (inputs stay staged) when
        no chai-lab CLI is available, e.g. for a dry run on a machine without
        Chai; a failed or mismatched conversion raises ValueError.
        """
        cli = self._cli()
        if cli is None:
            return
        argv = [*cli, "a3m-to-pqt", str(a3m_dir), "--output-directory", str(msa_out)]
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=600)
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError(f"chai: a3m-to-pqt failed for chain {chain_id}: {exc}") from exc
        if proc.returncode != 0:
            raise ValueError(
                f"chai: a3m-to-pqt failed for chain {chain_id} (exit {proc.returncode}): "
                f"{(proc.stderr or proc.stdout or '').strip()[-2000:]}"
            )
        expected = msa_out / f"{_seq_hash(seq)}.aligned.pqt"
        if not expected.is_file():
            raise ValueError(
                f"chai: a3m-to-pqt for chain {chain_id} did not produce {expected.name}; "
                "the a3m's query sequence must equal the chain sequence"
            )
