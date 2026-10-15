"""Backend-agnostic job specification.

A spec is a small YAML/JSON document:

    name: egfr_binder
    chains:
      - id: A
        type: protein
        sequence: MSKGEELFT...
        copies: 1
      - id: B
        type: protein
        sequence: ACDEFG...
    ligands:
      - id: L
        smiles: CC(=O)Oc1ccccc1C(=O)O
    msa: server          # server | none | cached | /path/to/dir
    seeds: [42]
    samples: 5
    reference: ref.pdb    # optional, enables DockQ / RMSD

Defaults: msa server, seeds [42], samples 1, recycles 3. Relative ``reference`` and ``msa``
paths are resolved against the spec file's directory by :func:`load_spec`. Unknown top-level
keys are passed to backends as ``extra``; one that looks like a typo of a known key
(``sampels``) triggers a ``UserWarning``.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

VALID_CHAIN_TYPES = {"protein", "dna", "rna"}
AA = set("ACDEFGHIKLMNPQRSTVWYX")
NT = set("ACGTUN")
FASTA_HEADER_KEYS = {"type", "copies", "msa"}
KNOWN_KEYS = ("name", "chains", "ligands", "msa", "seeds", "samples", "recycles", "reference", "extra")
MSA_MODES = ("server", "none", "cached")
# deliberate pass-through keys read by backends: never "did you mean"-warned about
PASSTHROUGH_KEYS = {"boltz", "chai", "colabfold", "boltz_args", "chai_args", "colabfold_args",
                    "constraints", "templates", "properties"}


def _warn_unknown_keys(keys: list[str]) -> None:
    """Warn about unknown top-level keys that look like typos of known ones."""
    for k in keys:
        if not isinstance(k, str) or k in PASSTHROUGH_KEYS:
            continue
        close = difflib.get_close_matches(k.lower(), KNOWN_KEYS, n=1, cutoff=0.75)
        if close:
            warnings.warn(f"unknown top-level key {k!r} (passed to backends as extra); "
                          f"did you mean {close[0]!r}?", UserWarning, stacklevel=3)


def _as_int(value: Any, what: str) -> int:
    """Coerce int-like values (YAML/FASTA strings such as "2", integral floats) to int."""
    if isinstance(value, bool):
        raise ValueError(f"{what} must be an integer, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            pass
    raise ValueError(f"{what} must be an integer, got {value!r}")


@dataclass
class Chain:
    id: str
    sequence: str
    type: str = "protein"
    copies: int = 1
    msa: str | None = None  # explicit a3m path overrides job-level msa mode

    def __post_init__(self) -> None:
        self.copies = _as_int(self.copies, f"chain {self.id}: copies")

    def validate(self) -> None:
        if self.type not in VALID_CHAIN_TYPES:
            raise ValueError(f"chain {self.id}: type must be one of {sorted(VALID_CHAIN_TYPES)}")
        seq = self.sequence.upper().replace("\n", "").replace(" ", "")
        self.sequence = seq
        alphabet = AA if self.type == "protein" else NT
        bad = set(seq) - alphabet
        if bad:
            raise ValueError(f"chain {self.id}: invalid characters {sorted(bad)}")
        if self.copies < 1:
            raise ValueError(f"chain {self.id}: copies must be >= 1")

    def chain_ids(self) -> list[str]:
        """IDs for each copy: A, A2, A3 ... (backends may remap)."""
        return [self.id] + [f"{self.id}{i}" for i in range(2, self.copies + 1)]


@dataclass
class Ligand:
    id: str
    smiles: str | None = None
    ccd: str | None = None
    copies: int = 1

    def __post_init__(self) -> None:
        self.copies = _as_int(self.copies, f"ligand {self.id}: copies")

    def validate(self) -> None:
        if not (self.smiles or self.ccd):
            raise ValueError(f"ligand {self.id}: need smiles or ccd")
        if self.copies < 1:
            raise ValueError(f"ligand {self.id}: copies must be >= 1")


@dataclass
class JobSpec:
    name: str
    chains: list[Chain] = field(default_factory=list)
    ligands: list[Ligand] = field(default_factory=list)
    msa: str = "server"
    seeds: list[int] = field(default_factory=lambda: [42])
    samples: int = 1
    recycles: int = 3
    reference: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)  # backend-specific passthrough

    def validate(self) -> None:
        if not self.name:
            raise ValueError("spec needs a name")
        if not self.chains:
            raise ValueError("spec needs at least one chain")
        for c in self.chains:
            c.validate()
        for lig in self.ligands:
            lig.validate()
        # expanded copy ids (A, A2, ...) must not collide with each other or with ligand ids
        ids = [cid for c in self.chains for cid in c.chain_ids()] + [lig.id for lig in self.ligands]
        dups = sorted({i for i in ids if ids.count(i) > 1})
        if dups:
            raise ValueError(f"duplicate ids in spec: {dups} (chain copy ids: {ids})")

    # ---- identity -------------------------------------------------------
    def canonical(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("reference", None)
        return d

    def content_hash(self, backend: str | None = None, backend_version: str | None = None) -> str:
        """Cache key. MSA paths enter by the sha256 of their contents, not their location, so a
        regenerated a3m changes the key and the same a3m in two projects shares it."""
        canon = self.canonical()
        canon["msa"] = _msa_identity(canon["msa"])
        for c in canon["chains"]:
            if c.get("msa"):
                c["msa"] = _msa_identity(c["msa"])
        payload = {"spec": canon, "backend": backend, "backend_version": backend_version}
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()[:16]

    def protein_sequences(self) -> list[tuple[str, str]]:
        return [(c.id, c.sequence) for c in self.chains if c.type == "protein"]

    def total_tokens(self) -> int:
        return sum(len(c.sequence) * c.copies for c in self.chains)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _msa_identity(value: str | None) -> str | None:
    """``sha256:<hex>`` of an a3m file (or of the *.a3m files in a directory); modes unchanged."""
    if not value or value in MSA_MODES:
        return value
    p = Path(value).expanduser()
    try:
        if p.is_file():
            return "sha256:" + _file_sha256(p)
        if p.is_dir():
            files = sorted(q for q in p.iterdir() if q.is_file() and q.suffix.lower() == ".a3m")
            inner = "\n".join(f"{q.name}:{_file_sha256(q)}" for q in files)
            return "sha256-dir:" + hashlib.sha256(inner.encode()).hexdigest()
    except OSError:
        pass
    return value  # not (yet) on disk: fall back to the path string


def _int_field(value: Any, what: str, minimum: int) -> int:
    v = _as_int(value, what)
    if v < minimum:
        raise ValueError(f"{what} must be >= {minimum}, got {v}")
    return v


def _seeds(value: Any) -> list[int]:
    """``42``, ``"42"``, ``"1,2"``, ``[1, "2"]`` -> list of ints (never a list of characters)."""
    if isinstance(value, str):
        items: list[Any] = [t for t in value.replace(",", " ").split() if t]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = [value]
    if not items:
        raise ValueError("seeds must hold at least one integer")
    return [_as_int(v, "seeds") for v in items]


def spec_from_dict(d: dict[str, Any]) -> JobSpec:
    if not d:
        raise ValueError("empty spec")
    if not isinstance(d, dict):
        raise ValueError(f"spec must be a mapping, got {type(d).__name__}")
    chains = [Chain(**c) for c in d.get("chains", [])]
    ligands = [Ligand(**lig) for lig in d.get("ligands", [])]
    unknown = [k for k in d if k not in KNOWN_KEYS]
    _warn_unknown_keys(unknown)
    extra = dict(d.get("extra", {}))
    extra.update({k: d[k] for k in unknown})
    spec = JobSpec(
        name=d["name"],
        chains=chains,
        ligands=ligands,
        msa=str(d.get("msa", "server")),
        seeds=_seeds(d.get("seeds", [42])),
        samples=_int_field(d.get("samples", 1), "samples", 1),
        recycles=_int_field(d.get("recycles", 3), "recycles", 0),
        reference=d.get("reference"),
        extra=extra,
    )
    spec.validate()
    return spec


def load_spec(path: str | Path) -> JobSpec:
    p = Path(path)
    text = p.read_text()
    if p.suffix.lower() in {".json"}:
        d = json.loads(text)
    elif p.suffix.lower() in {".fa", ".fasta"}:
        d = _spec_from_fasta(p.stem, text)
    else:
        d = yaml.safe_load(text)
    if not d:
        raise ValueError("empty spec")
    if not isinstance(d, dict):
        raise ValueError(f"spec must be a mapping, got {type(d).__name__}")
    _resolve_paths(d, p)
    return spec_from_dict(d)


def _resolve(base: Path, value: str) -> Path:
    q = Path(value).expanduser()
    return q if q.is_absolute() else (base / q).resolve()


def _resolve_paths(d: dict[str, Any], spec_path: Path) -> None:
    """Make relative ``reference`` / ``msa`` paths absolute w.r.t. the spec file's directory.

    Raises ``ValueError`` with the resolved path when an MSA path does not exist, so a typo
    surfaces here and not as a confusing backend error later.
    """
    base = spec_path.parent
    if d.get("reference") and isinstance(d["reference"], str):
        d["reference"] = str(_resolve(base, d["reference"]))
    msa = d.get("msa")
    if isinstance(msa, str) and msa not in MSA_MODES:
        m = _resolve(base, msa)
        if not m.exists():
            raise ValueError(
                f"msa: {msa!r} is not one of {'/'.join(MSA_MODES)} and no such a3m file or directory "
                f"exists (resolved to {m}, relative to the spec file {spec_path}); use msa: server, "
                f"msa: none, msa: cached, or a path to an .a3m file or a directory of <chain>.a3m files")
        d["msa"] = str(m)
    for c in d.get("chains") or []:
        if isinstance(c, dict) and isinstance(c.get("msa"), str) and c["msa"]:
            m = _resolve(base, c["msa"])
            if not m.is_file():
                raise ValueError(f"chain {c.get('id')}: msa file {c['msa']!r} not found "
                                 f"(resolved to {m}, relative to the spec file {spec_path})")
            c["msa"] = str(m)


def _spec_from_fasta(name: str, text: str) -> dict[str, Any]:
    """Plain FASTA -> spec. Header may carry `type=dna` / `copies=2` / `msa=path`.

    Other key=value tokens (e.g. UniProt OS=, OX=, GN=, PE=, SV=) are ignored.
    """
    chains = []
    cur_id, cur_meta, buf = None, {}, []
    def flush():
        if cur_id is not None:
            chains.append({"id": cur_id, "sequence": "".join(buf), **cur_meta})
    for line in text.splitlines():
        if line.startswith(">"):
            flush()
            toks = line[1:].split()
            cur_id = toks[0].split("|")[0]
            cur_meta = {}
            for t in toks[1:]:
                if "=" in t:
                    k, v = t.split("=", 1)
                    if k not in FASTA_HEADER_KEYS:
                        continue
                    cur_meta[k] = _as_int(v, f"chain {cur_id}: copies") if k == "copies" else v
            buf = []
        else:
            buf.append(line.strip())
    flush()
    return {"name": name, "chains": chains}
