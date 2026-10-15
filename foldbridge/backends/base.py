"""Backend contract. Every folder (Boltz-2, Chai-1, ColabFold, ...) implements this."""
from __future__ import annotations

import shlex
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..spec import JobSpec


@dataclass
class ModelResult:
    """One predicted model (one seed x one sample)."""
    backend: str
    path: Path                      # .cif or .pdb
    seed: int = 0
    sample: int = 0
    scores: dict[str, float] = field(default_factory=dict)  # ptm, iptm, plddt, ranking ... (backend-native names normalised)
    pae_path: Path | None = None    # .npz/.npy/.json holding NxN PAE if available
    extra: dict[str, Any] = field(default_factory=dict)


class Backend(ABC):
    name: str = "base"

    # ---- discovery -------------------------------------------------------
    @abstractmethod
    def available(self) -> bool: ...

    def version(self) -> str:
        return "unknown"

    # ---- lifecycle -------------------------------------------------------
    @abstractmethod
    def prepare(self, spec: JobSpec, workdir: Path, msa_dir: Path | None = None) -> Path:
        """Write backend-native inputs into workdir. Return the primary input path."""

    @abstractmethod
    def command(self, spec: JobSpec, workdir: Path, input_path: Path, gpu: bool = True) -> list[str]:
        """Shell argv to run the prediction."""

    @abstractmethod
    def collect(self, spec: JobSpec, workdir: Path) -> list[ModelResult]:
        """Parse backend outputs into ModelResult list (sorted best first)."""

    def run(self, spec: JobSpec, workdir: Path, msa_dir: Path | None = None, gpu: bool = True,
            dry_run: bool = False, log: Path | None = None) -> list[ModelResult]:
        # absolute, so collected model paths are absolute too (relative ones would be
        # resolved against the workdir a second time by the cache)
        workdir = Path(workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        inp = self.prepare(spec, workdir, msa_dir)
        argv = [str(a) for a in self.command(spec, workdir, inp, gpu=gpu)]
        (workdir / "command.txt").write_text(shlex.join(argv) + "\n")
        if dry_run:
            return []
        with open(log or workdir / "run.log", "a") as logf:
            try:
                proc = subprocess.run(argv, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                      cwd=workdir)
            except FileNotFoundError as e:
                raise RuntimeError(f"{self.name}: executable {argv[0]!r} not found; is {self.name} installed "
                                   f"and on PATH? (see `foldbridge backends`)") from e
            except PermissionError as e:
                raise RuntimeError(f"{self.name}: cannot execute {argv[0]!r}: {e}") from e
        if proc.returncode != 0:
            raise RuntimeError(f"{self.name} failed (exit {proc.returncode}); see {workdir/'run.log'}")
        return self.collect(spec, workdir)

    # ---- helpers ---------------------------------------------------------
    @staticmethod
    def which(exe: str) -> str | None:
        return shutil.which(exe)


BACKENDS: dict[str, type[Backend]] = {}


def register(cls: type[Backend]) -> type[Backend]:
    BACKENDS[cls.name] = cls
    return cls


def get_backend(name: str) -> Backend:
    _load_all()
    if name not in BACKENDS:
        raise KeyError(f"unknown backend {name!r}; known: {sorted(BACKENDS)}")
    return BACKENDS[name]()


def list_backends() -> dict[str, dict[str, Any]]:
    _load_all()
    out = {}
    for n, cls in BACKENDS.items():
        b = cls()
        ok = b.available()
        out[n] = {"available": ok, "version": b.version() if ok else None}
    return out


def _load_all() -> None:
    # import side-effect registration
    from . import boltz, chai, colabfold  # noqa: F401
