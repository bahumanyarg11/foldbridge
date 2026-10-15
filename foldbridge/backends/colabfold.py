"""ColabFold backend (AlphaFold2 / AF2-multimer via ``colabfold_batch``).

Native input is a FASTA with all complex chains joined by ``:`` on one record::

    >job
    MSKGEELFT...:MSKGEELFT...:ACDEFG...

or, when explicit MSAs are supplied, a ColabFold-style a3m. For complexes that
a3m carries a ``#len1,len2<TAB>card1,card2`` header followed by the
concatenated query and each unique chain's (unpaired) rows padded with gaps.

Native output layout (relative to ``<workdir>/colabfold_out``)::

    <job>_unrelaxed_rank_00X_<model_type>_model_<m>_seed_<s>.pdb
    <job>_scores_rank_00X_<model_type>_model_<m>_seed_<s>.json   # plddt[], ptm, iptm, pae[][], max_pae

ColabFold (AF2) cannot model ligands or nucleic acids.

Chain naming: ColabFold groups identical sequences (first-appearance order,
``query_seqs_unique`` x ``query_seqs_cardinality`` in ``colabfold.batch``) and
names output chains A, B, C, ... in that grouped order, so spec ids are lost
and chain order can differ from the spec (A=s1, B=s2, C=s1 -> A, C, B).
``collect()`` records ``extra["chain_map"]`` (output chain id -> spec copy id).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from ..spec import JobSpec
from .base import Backend, ModelResult, register
from .chai import _as_float, _plddt_100, find_executable, resolve_a3m, resolve_cli, spec_copy_ids

OUT_SUBDIR = "colabfold_out"
_RANK = re.compile(r"_rank_(\d+)")
_MODEL = re.compile(r"_model_(\d+)")
_SEED = re.compile(r"_seed_(\d+)")
# alphafold.common.protein.PDB_CHAIN_IDS
PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


@register
class ColabFoldBackend(Backend):
    """Run ColabFold via its ``colabfold_batch`` CLI."""

    name = "colabfold"

    # Console script `colabfold_batch` = colabfold.batch:main; colabfold/batch.py
    # ends with `if __name__ == "__main__": main()`, so `python -m colabfold.batch` works.
    EXE = "colabfold_batch"
    MODULE = "colabfold.batch"

    # ---- discovery -------------------------------------------------------
    def _cli(self) -> list[str] | None:
        return resolve_cli(self, self.EXE, self.MODULE)

    def available(self) -> bool:
        """True if `colabfold_batch` is on PATH / in the venv bin, or `python -m colabfold.batch` resolves."""
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
        try:
            from importlib.metadata import version as _dist_version

            return _dist_version("colabfold")
        except Exception:
            return "unknown"

    # ---- lifecycle -------------------------------------------------------
    def prepare(self, spec: JobSpec, workdir: Path, msa_dir: Path | None = None) -> Path:
        """Write ``<workdir>/<name>.fasta`` (or ``.a3m`` with explicit MSAs). Returns that path."""
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        spec.validate()
        if spec.ligands:
            raise ValueError(
                f"colabfold: cannot model ligands ({[lig.id for lig in spec.ligands]}); "
                "drop them from the spec or use the boltz / chai backend"
            )
        non_protein = [c.id for c in spec.chains if c.type != "protein"]
        if non_protein:
            raise ValueError(
                f"colabfold: cannot model DNA/RNA chains {non_protein}; "
                "use the boltz / chai backend"
            )

        stem = self._job_stem(spec)
        a3ms = {c.id: resolve_a3m(self.name, spec, c, msa_dir) for c in spec.chains}

        if any(p is not None for p in a3ms.values()):
            path = workdir / f"{stem}.a3m"
            if len(spec.chains) == 1 and spec.chains[0].copies == 1:
                shutil.copyfile(a3ms[spec.chains[0].id], path)  # plain monomer a3m
            else:
                path.write_text(self._complex_a3m(spec, a3ms))
            return path

        seqs = [c.sequence for c in spec.chains for _ in range(c.copies)]
        path = workdir / f"{stem}.fasta"
        path.write_text(f">{stem}\n{':'.join(seqs)}\n")
        return path

    def command(
        self, spec: JobSpec, workdir: Path, input_path: Path, gpu: bool = True
    ) -> list[str]:
        """argv for ``colabfold_batch``. Results land under ``<workdir>/colabfold_out``."""
        workdir = Path(workdir)
        cli = self._cli() or [self.EXE]
        seeds = spec.seeds or [0]
        argv = [
            *cli,
            str(Path(input_path).resolve()),
            str((workdir / OUT_SUBDIR).resolve()),
            "--num-recycle",
            str(max(0, spec.recycles)),
            # ColabFold runs seeds random_seed, random_seed+1, ...; it cannot take an
            # arbitrary list, so only the count and the first seed are honoured.
            "--num-seeds",
            str(max(1, len(seeds))),
            "--random-seed",
            str(seeds[0]),
            "--num-models",
            str(min(5, max(1, spec.samples))),  # AF2 has 5 model weights
        ]
        if spec.msa == "none" and Path(input_path).suffix.lower() != ".a3m":
            argv += ["--msa-mode", "single_sequence"]
        # No device flag: JAX falls back to CPU on its own when no GPU is visible.
        argv += [str(a) for a in spec.extra.get("colabfold_args", [])]
        return argv

    def collect(self, spec: JobSpec, workdir: Path) -> list[ModelResult]:
        """Parse ``*_scores_rank_*.json`` next to each ``*_unrelaxed_rank_*.pdb``, best-first.

        Common score keys: ptm, iptm, plddt (0-100), ranking (0.8*iptm + 0.2*ptm, or ptm).

        ColabFold renames (and may reorder) chains: output chains are A, B, C, ...
        over identical-sequence groups. ``extra["chain_map"]`` maps output chain
        id -> spec copy id (A, A2, B, ...); anything comparing chains to the spec
        or a reference must translate through it.
        """
        workdir = Path(workdir)
        results: list[ModelResult] = []
        default_seed = spec.seeds[0] if spec.seeds else 0
        chain_map = self.chain_map(spec)

        for js in sorted(workdir.glob("**/*_scores_rank_*.json")):
            try:
                model = next(
                    (
                        p
                        for kind in ("_unrelaxed_", "_relaxed_")
                        for p in (js.parent / (js.stem.replace("_scores_", kind, 1) + ".pdb"),)
                        if p.exists()
                    ),
                    None,
                )
                if model is None:
                    continue
                raw = json.loads(js.read_text())
                if not isinstance(raw, dict):
                    continue

                scores: dict[str, float] = {}
                for key in ("ptm", "iptm", "max_pae"):
                    val = _as_float(raw.get(key)) if key in raw else None
                    if val is not None:
                        scores[key] = val
                plddt = _plddt_100(_as_float(raw.get("plddt"))) if "plddt" in raw else None
                if plddt is not None:
                    scores["plddt"] = plddt
                if "iptm" in scores and "ptm" in scores:
                    scores["ranking"] = 0.8 * scores["iptm"] + 0.2 * scores["ptm"]
                elif "ptm" in scores:
                    scores["ranking"] = scores["ptm"]

                rank_m = _RANK.search(js.name)
                rank = int(rank_m.group(1)) if rank_m else 0
                seed_m = _SEED.search(js.name)
                model_m = _MODEL.search(js.name)

                pae_path: Path | None = None
                if raw.get("pae") is not None:
                    pae = np.asarray(raw["pae"], dtype=float)
                    if pae.ndim == 2 and pae.size:
                        pae_path = workdir / f"pae_{max(0, rank - 1)}.npy"
                        np.save(pae_path, pae)

                results.append(
                    ModelResult(
                        backend=self.name,
                        path=model,
                        seed=int(seed_m.group(1)) if seed_m else default_seed,
                        sample=int(model_m.group(1)) if model_m else 0,
                        scores=scores,
                        pae_path=pae_path,
                        extra={"record": js.stem, "rank": rank, "scores_json": str(js),
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
        """ColabFold derives the output jobname from the record header / filename."""
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", spec.name) or "job"

    @staticmethod
    def chain_map(spec: JobSpec) -> dict[str, str]:
        """Output chain id -> spec copy id, mirroring ColabFold's sequence grouping."""
        groups: dict[str, list[str]] = {}
        for cid, seq in spec_copy_ids(spec):
            if seq is not None:  # ColabFold has no ligands
                groups.setdefault(seq.upper(), []).append(cid)
        ordered = [cid for ids in groups.values() for cid in ids]
        return {PDB_CHAIN_IDS[i]: cid for i, cid in enumerate(ordered) if i < len(PDB_CHAIN_IDS)}

    @staticmethod
    def _a3m_records(path: Path) -> list[tuple[str, str]]:
        recs: list[tuple[str, str]] = []
        header, buf = None, []
        for line in Path(path).read_text().splitlines():
            line = line.rstrip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(">"):
                if header is not None:
                    recs.append((header, "".join(buf)))
                header, buf = line[1:], []
            else:
                buf.append(line.strip())
        if header is not None:
            recs.append((header, "".join(buf)))
        return recs

    def _complex_a3m(self, spec: JobSpec, a3ms: dict[str, Path | None]) -> str:
        """Unpaired complex a3m in ColabFold's ``msa_to_str`` layout.

        Chains with identical sequences are merged (cardinality = total copies).
        Chains without an explicit a3m contribute only their query row.
        """
        # TODO verify format against colabfold.batch.msa_to_str for the installed release.
        unique: list[str] = []
        card: dict[str, int] = {}
        src: dict[str, Path | None] = {}
        for c in spec.chains:
            if c.sequence not in card:
                unique.append(c.sequence)
                card[c.sequence] = 0
                src[c.sequence] = a3ms.get(c.id)
            card[c.sequence] += c.copies
            if src[c.sequence] is None:
                src[c.sequence] = a3ms.get(c.id)

        lens = [len(s) for s in unique]
        names = [str(101 + i) for i in range(len(unique))]
        out = [
            "#" + ",".join(map(str, lens)) + "\t" + ",".join(str(card[s]) for s in unique),
            ">" + "\t".join(names),
            "".join(unique),
        ]
        for i, seq in enumerate(unique):
            left = "-" * sum(lens[:i])
            right = "-" * sum(lens[i + 1 :])
            recs = self._a3m_records(src[seq]) if src[seq] is not None else []
            if not recs:
                recs = [(names[i], seq)]
            for j, (hdr, row) in enumerate(recs):
                out.append(">" + (names[i] if j == 0 else hdr))
                out.append(left + row + right)
        return "\n".join(out) + "\n"
