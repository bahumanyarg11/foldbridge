"""Boltz backend (Boltz-1 / Boltz-2).

Native input is the Boltz YAML schema v1::

    version: 1
    sequences:
      - protein:
          id: A
          sequence: MSK...
          msa: empty            # or /path/to/chain.a3m; omitted => --use_msa_server
      - ligand:
          id: L
          smiles: CC(=O)O

Native output layout (relative to ``--out_dir``)::

    boltz_results_<input_stem>/predictions/<record>/<record>_model_<i>.cif
                                                   /confidence_<record>_model_<i>.json
                                                   /pae_<record>_model_<i>.npz
                                                   /plddt_<record>_model_<i>.npz

Chain naming: Boltz writes the YAML entity ids verbatim as output chain ids
(A, A2, ..., L, L2 here), so ``extra["chain_map"]`` is the identity map.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..spec import Chain, JobSpec
from .base import Backend, ModelResult, register
from .chai import find_executable, resolve_a3m, resolve_cli, spec_copy_ids

_MODEL_IDX = re.compile(r"_model_(\d+)$")


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


@register
class BoltzBackend(Backend):
    """Run Boltz via its ``boltz predict`` CLI."""

    name = "boltz"

    # Console script `boltz` = boltz.main:cli; boltz/main.py ends with
    # `if __name__ == "__main__": cli()`, so `python -m boltz.main` works too.
    EXE = "boltz"
    MODULE = "boltz.main"

    # ---- discovery -------------------------------------------------------
    def _cli(self) -> list[str] | None:
        return resolve_cli(self, self.EXE, self.MODULE)

    def available(self) -> bool:
        """True if `boltz` is on PATH / in the venv bin, or `python -m boltz.main` resolves."""
        return self._cli() is not None

    def version(self) -> str:
        exe = find_executable(self, self.EXE)
        if exe:
            try:
                proc = subprocess.run(
                    [exe, "--version"], capture_output=True, text=True, timeout=20
                )
                out = (proc.stdout or proc.stderr or "").strip()
                if proc.returncode == 0 and out:
                    return out.splitlines()[0].strip()
            except Exception:
                pass
        try:
            from importlib.metadata import version as _dist_version

            return _dist_version("boltz")
        except Exception:
            return "unknown"

    # ---- lifecycle -------------------------------------------------------
    def prepare(self, spec: JobSpec, workdir: Path, msa_dir: Path | None = None) -> Path:
        """Write ``<workdir>/<name>.yaml`` in Boltz schema v1. Returns that path."""
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        spec.validate()

        sequences: list[dict[str, Any]] = []
        for chain in spec.chains:
            ids = chain.chain_ids()
            entity: dict[str, Any] = {
                "id": ids if len(ids) > 1 else ids[0],
                "sequence": chain.sequence,
            }
            msa = self._msa_value(spec, chain, msa_dir)
            if msa is not None:
                entity["msa"] = msa
            sequences.append({chain.type: entity})  # protein | dna | rna

        for lig in spec.ligands:
            lig_ids = [lig.id] + [f"{lig.id}{i}" for i in range(2, lig.copies + 1)]
            entity = {"id": lig_ids if len(lig_ids) > 1 else lig_ids[0]}
            if lig.smiles:
                entity["smiles"] = lig.smiles
            else:
                entity["ccd"] = lig.ccd
            sequences.append({"ligand": entity})

        doc: dict[str, Any] = {"version": 1, "sequences": sequences}
        # Optional schema-v1 sections passed straight through from spec.extra.
        for key in ("constraints", "templates", "properties"):
            if spec.extra.get(key):
                doc[key] = spec.extra[key]

        path = workdir / f"{self._job_stem(spec)}.yaml"
        path.write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=False))
        return path

    def command(
        self, spec: JobSpec, workdir: Path, input_path: Path, gpu: bool = True
    ) -> list[str]:
        """argv for ``boltz predict``. Results land under ``<workdir>/boltz_results_*``."""
        cli = self._cli() or [self.EXE]
        # Boltz takes a single --seed; extra seeds need extra runs (one workdir each).
        seed = spec.seeds[0] if spec.seeds else 0
        argv = [
            *cli,
            "predict",
            str(Path(input_path).resolve()),
            "--out_dir",
            str(Path(workdir).resolve()),
            "--diffusion_samples",
            str(max(1, spec.samples)),
            "--recycling_steps",
            str(max(0, spec.recycles)),
            "--seed",
            str(seed),
            "--output_format",
            "mmcif",
            "--write_full_pae",  # needed for pae_*.npz, which collect() picks up
        ]
        if spec.msa == "server":
            argv.append("--use_msa_server")
        if not gpu:
            argv += ["--accelerator", "cpu"]
        argv += [str(a) for a in spec.extra.get("boltz_args", [])]
        return argv

    def collect(self, spec: JobSpec, workdir: Path) -> list[ModelResult]:
        """Parse ``confidence_*.json`` next to each predicted mmCIF, best-first.

        Common score keys: ptm, iptm, plddt (0-100), ranking (= Boltz confidence_score).

        Boltz keeps the YAML ids as output chain ids, so ``extra["chain_map"]``
        (output chain id -> spec copy id) is the identity; it is set anyway so
        score/campaign code can translate chains uniformly across backends.
        """
        workdir = Path(workdir)
        results: list[ModelResult] = []
        seed = spec.seeds[0] if spec.seeds else 0
        chain_map = self.chain_map(spec)

        for conf in sorted(workdir.glob("**/predictions/*/confidence_*.json")):
            try:
                stem = conf.name[len("confidence_") : -len(".json")]
                pred_dir = conf.parent
                model = next(
                    (p for p in (pred_dir / f"{stem}.cif", pred_dir / f"{stem}.pdb") if p.exists()),
                    None,
                )
                if model is None:
                    continue
                raw = json.loads(conf.read_text())
                if not isinstance(raw, dict):
                    continue

                scores: dict[str, float] = {}
                for common, native in (
                    ("ranking", "confidence_score"),
                    ("ptm", "ptm"),
                    ("iptm", "iptm"),
                    ("plddt", "complex_plddt"),
                ):
                    val = _as_float(raw.get(native)) if native in raw else None
                    if val is not None:
                        scores[common] = val
                for native in ("ligand_iptm", "protein_iptm", "complex_iplddt", "complex_pde"):
                    val = _as_float(raw.get(native)) if native in raw else None
                    if val is not None:
                        scores[native] = val
                if "plddt" not in scores:
                    plddt_npz = pred_dir / f"plddt_{stem}.npz"
                    if plddt_npz.exists():
                        val = self._npz_mean(plddt_npz)
                        if val is not None:
                            scores["plddt"] = val
                if "plddt" in scores and scores["plddt"] <= 1.0:
                    scores["plddt"] *= 100.0  # common 0-100 scale across backends

                pae = pred_dir / f"pae_{stem}.npz"
                m = _MODEL_IDX.search(stem)
                results.append(
                    ModelResult(
                        backend=self.name,
                        path=model,
                        seed=seed,
                        sample=int(m.group(1)) if m else 0,
                        scores=scores,
                        pae_path=pae if pae.exists() else None,
                        extra={"record": stem, "confidence_json": str(conf),
                               "chain_map": dict(chain_map)},
                    )
                )
            except Exception:
                continue  # defensive: a single unreadable model must not kill the run

        results.sort(key=lambda r: r.scores.get("ranking", float("-inf")), reverse=True)
        return results

    # ---- helpers ---------------------------------------------------------
    @staticmethod
    def _job_stem(spec: JobSpec) -> str:
        """Boltz derives record + results-dir names from the input filename."""
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", spec.name) or "job"

    @staticmethod
    def _npz_mean(path: Path) -> float | None:
        try:
            with np.load(path, allow_pickle=False) as data:
                for key in ("plddt", "arr_0", *data.files):
                    if key in data.files:
                        return _as_float(data[key])
        except Exception:
            return None
        return None

    @staticmethod
    def chain_map(spec: JobSpec) -> dict[str, str]:
        """Output chain id -> spec copy id (identity: Boltz keeps YAML ids)."""
        return {cid: cid for cid, _ in spec_copy_ids(spec)}

    @staticmethod
    def _msa_value(spec: JobSpec, chain: Chain, msa_dir: Path | None) -> str | None:
        """Value for a protein entity's ``msa`` key (None => rely on --use_msa_server)."""
        if chain.type != "protein":
            return None  # Boltz only accepts msa on protein entities
        a3m = resolve_a3m("boltz", spec, chain, msa_dir)
        if a3m is not None:
            return str(a3m)
        return "empty" if spec.msa == "none" else None  # none => single-sequence mode
