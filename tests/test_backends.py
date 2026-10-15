"""Backend adapter tests: prepare / command / collect with fake tool outputs.

No network, no real folding tools: every external call is monkeypatched and
every output directory is hand-crafted to mimic the tool's native layout.
"""
from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

try:  # package __init__ imports .score, which may land later than the backends
    import foldbridge  # noqa: F401
except ImportError as exc:  # pragma: no cover - only while score.py is missing
    if "score" not in str(exc):
        raise
    _stub = types.ModuleType("foldbridge.score")
    _stub.score_structure = lambda *a, **k: None
    sys.modules["foldbridge.score"] = _stub

from foldbridge.backends import base as base_mod
from foldbridge.backends import chai as chai_mod
from foldbridge.backends.base import BACKENDS, ModelResult, get_backend
from foldbridge.backends.boltz import BoltzBackend
from foldbridge.backends.chai import ChaiBackend
from foldbridge.backends.colabfold import ColabFoldBackend
from foldbridge.spec import Chain, JobSpec, Ligand

SEQ_A = "MSKGEELFTG"
SEQ_B = "ACDEFGHIKL"


# ---- fixtures / helpers ----------------------------------------------------

def make_spec(**kw) -> JobSpec:
    d = dict(
        name="job1",
        chains=[Chain(id="A", sequence=SEQ_A, copies=2), Chain(id="B", sequence=SEQ_B)],
        ligands=[Ligand(id="L", smiles="CC(=O)O")],
        msa="server",
        seeds=[7, 8],
        samples=3,
        recycles=4,
    )
    d.update(kw)
    return JobSpec(**d)


def write_a3m(path: Path, query: str, n_hits: int = 2) -> Path:
    lines = [">query", query]
    for i in range(n_hits):
        lines += [f">hit{i}", query[:-1] + "-"]
    path.write_text("\n".join(lines) + "\n")
    return path


def write_cif(path: Path, bfactors: list[float]) -> Path:
    head = [
        "data_model",
        "loop_",
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_comp_id",
        "_atom_site.auth_asym_id",
        "_atom_site.auth_seq_id",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.B_iso_or_equiv",
        "_atom_site.pdbx_PDB_model_num",
    ]
    rows = [
        f"ATOM {i + 1} C CA ALA A {i + 1} {i * 3.8:.3f} 0.000 0.000 {b:.2f} 1"
        for i, b in enumerate(bfactors)
    ]
    path.write_text("\n".join(head + rows + ["#"]) + "\n")
    return path


def write_pdb(path: Path, n: int = 3) -> Path:
    lines = [
        f"ATOM  {i + 1:5d}  CA  ALA A{i + 1:4d}    {i * 3.8:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{90.0:6.2f}           C"
        for i in range(n)
    ]
    path.write_text("\n".join(lines + ["END"]) + "\n")
    return path


def fasta_records(text: str) -> list[tuple[str, str]]:
    recs, header = [], None
    for line in text.splitlines():
        if line.startswith(">"):
            header = line[1:]
        elif header is not None and line.strip():
            recs.append((header, line.strip()))
            header = None
    return recs


@pytest.fixture
def no_tools(monkeypatch, tmp_path_factory):
    """Pretend no folding CLI is installed (argv then uses the bare exe name)."""
    monkeypatch.setattr(base_mod.Backend, "which", staticmethod(lambda exe: None))
    monkeypatch.setattr(chai_mod, "_venv_bin_dir", lambda: tmp_path_factory.mktemp("emptybin"))
    monkeypatch.setattr(chai_mod, "module_runnable", lambda module: False)


# ---- registry --------------------------------------------------------------

def test_registry_has_all_backends():
    base_mod._load_all()
    assert {"boltz", "chai", "colabfold"} <= set(BACKENDS)
    assert isinstance(get_backend("chai"), ChaiBackend)
    assert isinstance(get_backend("colabfold"), ColabFoldBackend)


@pytest.mark.parametrize(
    "cls,exe,module",
    [(ChaiBackend, "chai-lab", "chai_lab.main"), (ColabFoldBackend, "colabfold_batch", "colabfold.batch"),
     (BoltzBackend, "boltz", "boltz.main")],
)
def test_available_detection(monkeypatch, tmp_path, cls, exe, module):
    import importlib.util

    venv_bin = tmp_path / "bin"
    venv_bin.mkdir()
    monkeypatch.setattr(chai_mod, "_venv_bin_dir", lambda: venv_bin)

    # 1. executable on PATH -> available, and command() runs that exact path
    monkeypatch.setattr(base_mod.Backend, "which", staticmethod(lambda e: f"/opt/x/{e}" if e == exe else None))
    b = cls()
    assert b.available() is True
    spec = make_spec(ligands=[])
    assert b.command(spec, tmp_path, tmp_path / "in.fasta")[0] == f"/opt/x/{exe}"

    # 2. nothing on PATH, nothing importable -> unavailable
    monkeypatch.setattr(base_mod.Backend, "which", staticmethod(lambda e: None))
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None)
    assert cls().available() is False

    # 3. a bare package import is not enough: only the `python -m` entry module counts
    top = module.split(".")[0]
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: object() if name == top else None)
    assert cls().available() is False
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: object() if name == module else None)
    assert cls().available() is True
    argv = cls().command(spec, tmp_path, tmp_path / "in.fasta")
    assert argv[:3] == [sys.executable, "-m", module]

    # 4. executable next to sys.executable (venv bin, not on PATH) wins over `python -m`
    tool = venv_bin / exe
    tool.write_text("#!/bin/sh\n")
    tool.chmod(0o755)
    assert cls().available() is True
    assert cls().command(spec, tmp_path, tmp_path / "in.fasta")[0] == str(tool)
    tool.chmod(0o644)  # not executable -> ignored
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None)
    assert cls().available() is False

    def boom(*a, **k):
        raise RuntimeError("broken env")

    monkeypatch.setattr(importlib.util, "find_spec", boom)
    assert cls().available() is False  # never raises


@pytest.mark.parametrize("cls", [ChaiBackend, ColabFoldBackend])
def test_version_ignores_usage_errors(monkeypatch, cls):
    monkeypatch.setattr(base_mod.Backend, "which", staticmethod(lambda e: f"/bin/{e}"))
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 2, "", "Error: No such option: --version"),
    )
    import importlib.metadata as md

    def nodist(name):
        raise md.PackageNotFoundError(name)

    monkeypatch.setattr(md, "version", nodist)
    assert cls().version() == "unknown"

    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a[0], 0, "tool 1.2.3\n", "")
    )
    assert cls().version() == "tool 1.2.3"


# ---- Chai ------------------------------------------------------------------

def test_chai_prepare_fasta(tmp_path, no_tools):
    spec = make_spec(
        chains=[
            Chain(id="A", sequence=SEQ_A, copies=2),
            Chain(id="B", sequence="ACGT", type="dna"),
            Chain(id="C", sequence="ACGU", type="rna"),
        ],
        ligands=[Ligand(id="L", smiles="CC(=O)O", copies=2)],
    )
    path = ChaiBackend().prepare(spec, tmp_path)
    assert path == tmp_path / "job1.fasta"
    recs = fasta_records(path.read_text())
    assert recs == [
        ("protein|name=A", SEQ_A),
        ("protein|name=A2", SEQ_A),
        ("dna|name=B", "ACGT"),
        ("rna|name=C", "ACGU"),
        ("ligand|name=L", "CC(=O)O"),
        ("ligand|name=L2", "CC(=O)O"),
    ]
    assert not (tmp_path / "msas").exists()


def test_chai_prepare_rejects_ccd_only_ligand(tmp_path, no_tools):
    spec = make_spec(ligands=[Ligand(id="L", ccd="ATP")])
    with pytest.raises(ValueError, match="SMILES"):
        ChaiBackend().prepare(spec, tmp_path / "w")


def test_chai_command_server(tmp_path, no_tools):
    spec = make_spec()
    b = ChaiBackend()
    inp = b.prepare(spec, tmp_path)
    argv = b.command(spec, tmp_path, inp, gpu=True)
    assert argv[:4] == ["chai-lab", "fold", str(inp.resolve()), str((tmp_path / "chai_out").resolve())]
    assert argv[argv.index("--num-trunk-recycles") + 1] == "4"
    assert argv[argv.index("--num-diffn-samples") + 1] == "3"
    assert argv[argv.index("--seed") + 1] == "7"
    assert "--use-msa-server" in argv
    assert "--device" not in argv and "--msa-directory" not in argv


def test_chai_command_none_cpu_passthrough(tmp_path, no_tools):
    spec = make_spec(msa="none", extra={"chai_args": ["--low-memory"]})
    b = ChaiBackend()
    inp = b.prepare(spec, tmp_path)
    argv = b.command(spec, tmp_path, inp, gpu=False)
    assert "--use-msa-server" not in argv
    assert argv[argv.index("--device") + 1] == "cpu"
    assert argv[-1] == "--low-memory"


def test_chai_explicit_a3m_staged(tmp_path, no_tools):
    a3m = write_a3m(tmp_path / "a.a3m", SEQ_A)
    spec = make_spec(msa="none", chains=[Chain(id="A", sequence=SEQ_A, copies=2, msa=str(a3m)),
                                         Chain(id="B", sequence=SEQ_B)])
    b = ChaiBackend()
    wd = tmp_path / "w"
    inp = b.prepare(spec, wd)
    # one staging dir per unique sequence (copies share it), raw a3m kept out of msas/
    (sub,) = (wd / "msa_a3m").iterdir()
    assert [p.name for p in sub.iterdir()] == ["a.a3m"]
    assert (sub / "a.a3m").read_text() == a3m.read_text()
    assert (wd / "msas").is_dir() and not any((wd / "msas").iterdir())  # no converter here
    argv = b.command(spec, wd, inp)
    assert argv[argv.index("--msa-directory") + 1] == str((wd / "msas").resolve())
    assert "--use-msa-server" not in argv


def _fake_a3m_to_pqt(calls, fail=False, wrong_query=False):
    """Stand-in for `chai-lab a3m-to-pqt DIR --output-directory OUT` (merges DIR/*.a3m)."""
    def run(argv, **kw):
        calls.append(argv)
        assert argv[1] == "a3m-to-pqt" and argv[3] == "--output-directory"
        src, out = Path(argv[2]), Path(argv[4])
        if fail:
            return subprocess.CompletedProcess(argv, 1, "", "AssertionError: bad a3m")
        queries = {chai_mod._a3m_query(p) for p in src.glob("*.a3m")}
        assert len(queries) == 1, f"a3m-to-pqt would merge different queries: {queries}"
        q = "WRONG" if wrong_query else queries.pop()
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{chai_mod._seq_hash(q)}.aligned.pqt").write_text("pqt")
        return subprocess.CompletedProcess(argv, 0, "", "")
    return run


def test_chai_msas_converted_per_sequence(tmp_path, no_tools, monkeypatch):
    monkeypatch.setattr(base_mod.Backend, "which", staticmethod(lambda e: f"/bin/{e}"))
    calls: list = []
    monkeypatch.setattr(subprocess, "run", _fake_a3m_to_pqt(calls))
    msa_dir = tmp_path / "msas_in"
    msa_dir.mkdir()
    write_a3m(msa_dir / "A.a3m", SEQ_A)
    write_a3m(msa_dir / "B.a3m", SEQ_B)
    spec = make_spec(msa="cached", ligands=[])  # A x2 + B: two unique sequences
    b = ChaiBackend()
    wd = tmp_path / "w"
    inp = b.prepare(spec, wd, msa_dir=msa_dir)
    assert len(calls) == 2  # one conversion per unique sequence, never a merged dir
    assert {Path(c[2]).parent for c in calls} == {wd / "msa_a3m"}
    assert all(c[0] == "/bin/chai-lab" and c[4] == str(wd / "msas") for c in calls)
    assert sorted(p.name for p in (wd / "msas").iterdir()) == sorted(
        f"{chai_mod._seq_hash(s)}.aligned.pqt" for s in (SEQ_A, SEQ_B))
    argv = b.command(spec, wd, inp)
    assert argv[argv.index("--msa-directory") + 1] == str((wd / "msas").resolve())

    # re-preparing starts from clean staging dirs
    b.prepare(make_spec(msa="none", ligands=[]), wd)
    assert not (wd / "msas").exists() and not (wd / "msa_a3m").exists()


def test_chai_msa_conversion_failure_raises(tmp_path, no_tools, monkeypatch):
    monkeypatch.setattr(base_mod.Backend, "which", staticmethod(lambda e: f"/bin/{e}"))
    a3m = write_a3m(tmp_path / "a.a3m", SEQ_A)
    spec = make_spec(msa="none", ligands=[], chains=[Chain(id="A", sequence=SEQ_A, msa=str(a3m))])
    monkeypatch.setattr(subprocess, "run", _fake_a3m_to_pqt([], fail=True))
    with pytest.raises(ValueError, match="a3m-to-pqt failed for chain A .*bad a3m"):
        ChaiBackend().prepare(spec, tmp_path / "w")
    monkeypatch.setattr(subprocess, "run", _fake_a3m_to_pqt([], wrong_query=True))
    with pytest.raises(ValueError, match="query sequence must equal"):
        ChaiBackend().prepare(spec, tmp_path / "w2")


def test_chai_explicit_msa_mixed_with_server_raises(tmp_path, no_tools):
    a3m = write_a3m(tmp_path / "a.a3m", SEQ_A)
    spec = make_spec(msa="server", ligands=[], chains=[Chain(id="A", sequence=SEQ_A, msa=str(a3m)),
                                                       Chain(id="B", sequence=SEQ_B)])
    with pytest.raises(ValueError, match=r"chains \['B'\] rely on msa: server"):
        ChaiBackend().prepare(spec, tmp_path / "w")
    # every protein chain explicit (identical sequences count) -> fine, no server flag
    spec = make_spec(msa="server", ligands=[], chains=[Chain(id="A", sequence=SEQ_A, msa=str(a3m)),
                                                       Chain(id="C", sequence=SEQ_A),
                                                       Chain(id="D", sequence="ACGT", type="dna")])
    b = ChaiBackend()
    inp = b.prepare(spec, tmp_path / "w2")
    argv = b.command(spec, tmp_path / "w2", inp)
    assert "--msa-directory" in argv and "--use-msa-server" not in argv


def test_chai_cached_msa_missing_raises(tmp_path, no_tools):
    spec = make_spec(msa="cached")
    with pytest.raises(ValueError, match="no a3m found for chain A"):
        ChaiBackend().prepare(spec, tmp_path / "w", msa_dir=tmp_path)


def _chai_outputs(out: Path, models: dict[int, dict], with_pae: set[int] = frozenset()):
    out.mkdir(parents=True, exist_ok=True)
    for i, sc in models.items():
        write_cif(out / f"pred.model_idx_{i}.cif", sc.pop("_bf", [0.9, 0.8, 0.7]))
        payload = {
            "aggregate_score": np.array([sc["agg"]]),
            "ptm": np.array([sc["ptm"]]),
            "iptm": np.array([sc["iptm"]]),
            "per_chain_ptm": np.array([[sc["ptm"], sc["ptm"] - 0.1]]),
            "has_inter_chain_clashes": np.array([False]),
        }
        if i in with_pae:
            payload["pae"] = np.full((1, 3, 3), float(i))
        np.savez(out / f"scores.model_idx_{i}.npz", **payload)


def test_chai_collect_parses_and_sorts(tmp_path):
    out = tmp_path / "chai_out"
    _chai_outputs(
        out,
        {
            0: {"agg": 0.50, "ptm": 0.6, "iptm": 0.4},
            1: {"agg": 0.90, "ptm": 0.8, "iptm": 0.7, "_bf": [70.0, 80.0, 90.0]},
            2: {"agg": 0.70, "ptm": 0.7, "iptm": 0.6},
        },
        with_pae={1},
    )
    # orphan score file without a structure must be skipped
    np.savez(out / "scores.model_idx_9.npz", aggregate_score=np.array([0.99]))
    # corrupt npz must be skipped, not crash
    write_cif(out / "pred.model_idx_8.cif", [0.5])
    (out / "scores.model_idx_8.npz").write_bytes(b"not an npz")

    res = ChaiBackend().collect(make_spec(), tmp_path)
    assert [r.sample for r in res] == [1, 2, 0]
    assert all(isinstance(r, ModelResult) and r.backend == "chai" for r in res)
    best = res[0]
    assert best.path == out / "pred.model_idx_1.cif"
    assert best.seed == 7
    assert best.scores["ranking"] == pytest.approx(0.90)
    assert best.scores["ptm"] == pytest.approx(0.8)
    assert best.scores["iptm"] == pytest.approx(0.7)
    assert best.scores["has_inter_chain_clashes"] == 0.0
    assert best.scores["plddt"] == pytest.approx(80.0)  # B-factors already 0-100
    assert res[2].scores["plddt"] == pytest.approx(80.0)  # 0-1 B-factors rescaled
    assert best.extra["per_chain_ptm"] == pytest.approx([0.8, 0.7])
    assert best.pae_path == tmp_path / "pae_1.npy"
    pae = np.load(best.pae_path)
    assert pae.shape == (3, 3) and pae[0, 0] == 1.0
    assert res[1].pae_path is None


def test_chai_collect_missing_outputs(tmp_path):
    assert ChaiBackend().collect(make_spec(), tmp_path) == []
    assert ChaiBackend().collect(make_spec(), tmp_path / "does_not_exist") == []


# ---- ColabFold -------------------------------------------------------------

def test_colabfold_prepare_fasta(tmp_path, no_tools):
    spec = make_spec(ligands=[])
    path = ColabFoldBackend().prepare(spec, tmp_path)
    assert path == tmp_path / "job1.fasta"
    assert path.read_text() == f">job1\n{SEQ_A}:{SEQ_A}:{SEQ_B}\n"


def test_colabfold_rejects_ligands_and_nucleic_acids(tmp_path, no_tools):
    with pytest.raises(ValueError, match="cannot model ligands"):
        ColabFoldBackend().prepare(make_spec(), tmp_path)
    spec = make_spec(ligands=[], chains=[Chain(id="A", sequence=SEQ_A),
                                         Chain(id="D", sequence="ACGT", type="dna")])
    with pytest.raises(ValueError, match="DNA/RNA"):
        ColabFoldBackend().prepare(spec, tmp_path)


def test_colabfold_command(tmp_path, no_tools):
    spec = make_spec(ligands=[], seeds=[11, 12, 13], samples=9, extra={"colabfold_args": ["--amber"]})
    b = ColabFoldBackend()
    inp = b.prepare(spec, tmp_path)
    argv = b.command(spec, tmp_path, inp)
    assert argv[:3] == ["colabfold_batch", str(inp.resolve()), str((tmp_path / "colabfold_out").resolve())]
    assert argv[argv.index("--num-recycle") + 1] == "4"
    assert argv[argv.index("--num-seeds") + 1] == "3"
    assert argv[argv.index("--random-seed") + 1] == "11"
    assert argv[argv.index("--num-models") + 1] == "5"  # clamped to AF2's 5 weights
    assert "--msa-mode" not in argv
    assert argv[-1] == "--amber"

    spec_none = make_spec(ligands=[], msa="none", samples=2)
    argv = b.command(spec_none, tmp_path, b.prepare(spec_none, tmp_path), gpu=False)
    assert argv[argv.index("--msa-mode") + 1] == "single_sequence"
    assert argv[argv.index("--num-models") + 1] == "2"


def test_colabfold_monomer_a3m_input(tmp_path, no_tools):
    a3m = write_a3m(tmp_path / "a.a3m", SEQ_A)
    spec = make_spec(ligands=[], chains=[Chain(id="A", sequence=SEQ_A, msa=str(a3m))])
    b = ColabFoldBackend()
    path = b.prepare(spec, tmp_path / "w")
    assert path.suffix == ".a3m"
    assert path.read_text() == a3m.read_text()
    assert "--msa-mode" not in b.command(spec, tmp_path / "w", path)


def test_colabfold_complex_a3m_layout(tmp_path, no_tools):
    msa_dir = tmp_path / "msas"
    msa_dir.mkdir()
    write_a3m(msa_dir / "A.a3m", SEQ_A, n_hits=1)
    write_a3m(msa_dir / "B.a3m", SEQ_B, n_hits=1)
    spec = make_spec(ligands=[], msa="cached")
    path = ColabFoldBackend().prepare(spec, tmp_path / "w", msa_dir=msa_dir)
    lines = path.read_text().splitlines()
    la, lb = len(SEQ_A), len(SEQ_B)
    assert lines[0] == f"#{la},{lb}\t2,1"
    assert lines[1] == ">101\t102"
    assert lines[2] == SEQ_A + SEQ_B
    assert lines[3:7] == [">101", SEQ_A + "-" * lb, ">hit0", SEQ_A[:-1] + "-" + "-" * lb]
    assert lines[7:11] == [">102", "-" * la + SEQ_B, ">hit0", "-" * la + SEQ_B[:-1] + "-"]
    # every row spans the full concatenated query length
    assert {len(l) for l in lines[2:] if not l.startswith(">")} == {la + lb}


def test_single_a3m_fallback_refuses_wrong_chain(tmp_path, no_tools):
    msa_dir = tmp_path / "msas"
    msa_dir.mkdir()
    write_a3m(msa_dir / "whatever.a3m", SEQ_A)
    spec = make_spec(ligands=[], msa="cached", chains=[Chain(id="A", sequence=SEQ_A),
                                                       Chain(id="B", sequence=SEQ_B)])
    with pytest.raises(ValueError, match="chain B"):
        ColabFoldBackend().prepare(spec, tmp_path / "w", msa_dir=msa_dir)
    mono = make_spec(ligands=[], msa="cached", chains=[Chain(id="A", sequence=SEQ_A)])
    assert ColabFoldBackend().prepare(mono, tmp_path / "w2", msa_dir=msa_dir).suffix == ".a3m"


def _cf_model(out: Path, rank: int, model: int, seed: int, ptm: float, iptm: float | None,
              plddt: list[float], relaxed: bool = False, pae: bool = True) -> None:
    tag = f"rank_{rank:03d}_alphafold2_multimer_v3_model_{model}_seed_{seed:03d}"
    write_pdb(out / f"job1_{'relaxed' if relaxed else 'unrelaxed'}_{tag}.pdb")
    d = {"plddt": plddt, "ptm": ptm, "max_pae": 31.75}
    if iptm is not None:
        d["iptm"] = iptm
    if pae:
        d["pae"] = [[float(rank)] * 3] * 3
    (out / f"job1_scores_{tag}.json").write_text(json.dumps(d))


def test_colabfold_collect_parses_and_sorts(tmp_path):
    out = tmp_path / "colabfold_out"
    out.mkdir()
    # ColabFold's own rank is by its multimer score; ours must be 0.8*iptm + 0.2*ptm
    _cf_model(out, 1, 2, 11, ptm=0.50, iptm=0.80, plddt=[80, 90, 100])   # 0.74
    _cf_model(out, 2, 4, 12, ptm=0.90, iptm=0.85, plddt=[60, 70, 80])    # 0.86
    _cf_model(out, 3, 1, 11, ptm=0.30, iptm=0.20, plddt=[50, 50, 50], relaxed=True, pae=False)
    (out / "job1_scores_rank_004_alphafold2_multimer_v3_model_3_seed_011.json").write_text("{}")  # no pdb
    (out / "job1_unrelaxed_rank_005_x_model_5_seed_011.pdb").write_text("END\n")
    (out / "job1_scores_rank_005_x_model_5_seed_011.json").write_text("{corrupt")

    res = ColabFoldBackend().collect(make_spec(ligands=[]), tmp_path)
    assert [r.extra["rank"] for r in res] == [2, 1, 3]
    best = res[0]
    assert best.backend == "colabfold"
    assert best.scores["ranking"] == pytest.approx(0.8 * 0.85 + 0.2 * 0.90)
    assert best.scores["plddt"] == pytest.approx(70.0)
    assert best.scores["max_pae"] == pytest.approx(31.75)
    assert best.seed == 12 and best.sample == 4
    assert best.path.name.startswith("job1_unrelaxed_rank_002")
    assert best.pae_path == tmp_path / "pae_1.npy"
    assert np.load(best.pae_path).shape == (3, 3)
    assert res[1].pae_path == tmp_path / "pae_0.npy"
    assert res[2].path.name.startswith("job1_relaxed_rank_003")  # relaxed fallback
    assert res[2].pae_path is None


def test_colabfold_collect_monomer_ranking_is_ptm(tmp_path):
    out = tmp_path / "colabfold_out"
    out.mkdir()
    _cf_model(out, 1, 1, 0, ptm=0.77, iptm=None, plddt=[0.9, 0.8])  # 0-1 plddt rescaled
    (res,) = ColabFoldBackend().collect(make_spec(ligands=[]), tmp_path)
    assert res.scores["ranking"] == pytest.approx(0.77)
    assert "iptm" not in res.scores
    assert res.scores["plddt"] == pytest.approx(85.0)


def test_colabfold_collect_missing_outputs(tmp_path):
    assert ColabFoldBackend().collect(make_spec(), tmp_path) == []


# ---- Boltz (recovered, previously untested) --------------------------------

def test_boltz_prepare_yaml(tmp_path, no_tools):
    spec = make_spec(ligands=[Ligand(id="L", smiles="CC(=O)O"), Ligand(id="M", ccd="ATP", copies=2)],
                     extra={"constraints": [{"bond": {"atom1": ["A", 1, "CA"], "atom2": ["B", 1, "CA"]}}]})
    path = BoltzBackend().prepare(spec, tmp_path)
    assert path == tmp_path / "job1.yaml"
    doc = yaml.safe_load(path.read_text())
    assert doc["version"] == 1
    seqs = doc["sequences"]
    assert seqs[0] == {"protein": {"id": ["A", "A2"], "sequence": SEQ_A}}  # server => no msa key
    assert seqs[1] == {"protein": {"id": "B", "sequence": SEQ_B}}
    assert seqs[2] == {"ligand": {"id": "L", "smiles": "CC(=O)O"}}
    assert seqs[3] == {"ligand": {"id": ["M", "M2"], "ccd": "ATP"}}
    assert doc["constraints"] == spec.extra["constraints"]


def test_boltz_prepare_msa_modes(tmp_path, no_tools):
    spec = make_spec(msa="none", ligands=[],
                     chains=[Chain(id="A", sequence=SEQ_A), Chain(id="D", sequence="ACGT", type="dna")])
    doc = yaml.safe_load(BoltzBackend().prepare(spec, tmp_path / "n").read_text())
    assert doc["sequences"][0]["protein"]["msa"] == "empty"
    assert "msa" not in doc["sequences"][1]["dna"]

    a3m = write_a3m(tmp_path / "A.a3m", SEQ_A)
    spec = make_spec(msa="cached", ligands=[], chains=[Chain(id="A", sequence=SEQ_A)])
    doc = yaml.safe_load(BoltzBackend().prepare(spec, tmp_path / "c", msa_dir=tmp_path).read_text())
    assert doc["sequences"][0]["protein"]["msa"] == str(a3m.resolve())

    spec = make_spec(msa="server", ligands=[], chains=[Chain(id="A", sequence=SEQ_A, msa=str(a3m))])
    doc = yaml.safe_load(BoltzBackend().prepare(spec, tmp_path / "e").read_text())
    assert doc["sequences"][0]["protein"]["msa"] == str(a3m.resolve())

    spec = make_spec(msa="cached", ligands=[])
    with pytest.raises(ValueError, match="no a3m found"):
        BoltzBackend().prepare(spec, tmp_path / "x", msa_dir=tmp_path / "empty")


def test_boltz_command(tmp_path, no_tools):
    spec = make_spec(extra={"boltz_args": ["--use_potentials"]})
    b = BoltzBackend()
    inp = b.prepare(spec, tmp_path)
    argv = b.command(spec, tmp_path, inp, gpu=False)
    assert argv[:3] == ["boltz", "predict", str(inp.resolve())]
    assert argv[argv.index("--diffusion_samples") + 1] == "3"
    assert argv[argv.index("--recycling_steps") + 1] == "4"
    assert argv[argv.index("--seed") + 1] == "7"
    assert argv[argv.index("--output_format") + 1] == "mmcif"
    assert "--write_full_pae" in argv and "--use_msa_server" in argv
    assert argv[argv.index("--accelerator") + 1] == "cpu"
    assert argv[-1] == "--use_potentials"
    argv = b.command(make_spec(msa="none"), tmp_path, inp, gpu=True)
    assert "--use_msa_server" not in argv and "--accelerator" not in argv


def _boltz_model(pred: Path, i: int, conf: float, plddt_json: bool = True) -> None:
    rec = pred.name
    write_cif(pred / f"{rec}_model_{i}.cif", [0.9])
    d = {"confidence_score": conf, "ptm": 0.6 + i / 100, "iptm": 0.5, "ligand_iptm": 0.4,
         "complex_pde": 0.8}
    if plddt_json:
        d["complex_plddt"] = 0.85
    (pred / f"confidence_{rec}_model_{i}.json").write_text(json.dumps(d))
    np.savez(pred / f"plddt_{rec}_model_{i}.npz", plddt=np.array([0.5, 0.7]))
    np.savez(pred / f"pae_{rec}_model_{i}.npz", pae=np.zeros((3, 3)))


def test_boltz_collect_parses_and_sorts(tmp_path):
    pred = tmp_path / "boltz_results_job1" / "predictions" / "job1"
    pred.mkdir(parents=True)
    _boltz_model(pred, 0, 0.61)
    _boltz_model(pred, 1, 0.83, plddt_json=False)
    _boltz_model(pred, 2, 0.72)
    (pred / "pae_job1_model_2.npz").unlink()
    (pred / "confidence_job1_model_7.json").write_text("{}")  # no structure
    write_cif(pred / "job1_model_8.cif", [0.9])
    (pred / "confidence_job1_model_8.json").write_text("not json")

    res = BoltzBackend().collect(make_spec(), tmp_path)
    assert [r.sample for r in res] == [1, 2, 0]
    best = res[0]
    assert best.backend == "boltz" and best.seed == 7
    assert best.path == pred / "job1_model_1.cif"
    assert best.scores["ranking"] == pytest.approx(0.83)
    assert best.scores["ptm"] == pytest.approx(0.61)
    assert best.scores["ligand_iptm"] == pytest.approx(0.4)
    assert best.scores["plddt"] == pytest.approx(60.0)  # npz fallback, rescaled to 0-100
    assert res[2].scores["plddt"] == pytest.approx(85.0)
    assert best.pae_path == pred / "pae_job1_model_1.npz"
    assert res[1].pae_path is None
    assert BoltzBackend().collect(make_spec(), tmp_path / "nothing") == []


def test_boltz_version_ignores_usage_errors(monkeypatch):
    monkeypatch.setattr(base_mod.Backend, "which", staticmethod(lambda e: "/bin/boltz"))
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 2, "", "Error: No such option: --version"),
    )
    import importlib.metadata as md

    def nodist(name):
        raise md.PackageNotFoundError(name)

    monkeypatch.setattr(md, "version", nodist)
    assert BoltzBackend().version() == "unknown"


def test_boltz_plddt_on_common_scale(tmp_path):
    pred = tmp_path / "boltz_results_job1" / "predictions" / "job1"
    pred.mkdir(parents=True)
    _boltz_model(pred, 0, 0.61)
    (res,) = BoltzBackend().collect(make_spec(), tmp_path)
    assert res.scores["plddt"] == pytest.approx(85.0)


# ---- chain renaming (output chain id -> spec copy id) ------------------------

def _hetero_spec(**kw):
    # A=s1 x2, B=s2, C=s1 again, then ligand L x2
    kw.setdefault("ligands", [Ligand(id="L", smiles="CC(=O)O", copies=2)])
    return make_spec(chains=[Chain(id="A", sequence=SEQ_A, copies=2), Chain(id="B", sequence=SEQ_B),
                             Chain(id="C", sequence=SEQ_A)], **kw)


def test_chai_chain_map(tmp_path):
    spec = _hetero_spec()
    # Chai letters FASTA records in order: chains x copies, then ligands
    assert ChaiBackend.chain_map(spec) == {"A": "A", "B": "A2", "C": "B", "D": "C", "E": "L", "F": "L2"}
    named = _hetero_spec(extra={"chai_args": ["--fasta-names-as-cif-chains"]})
    assert ChaiBackend.chain_map(named) == {c: c for c in ["A", "A2", "B", "C", "L", "L2"]}
    many = make_spec(ligands=[], chains=[Chain(id="X", sequence=SEQ_A, copies=28)])
    cm = ChaiBackend.chain_map(many)
    assert cm["Z"] == "X26" and cm["a"] == "X27" and cm["b"] == "X28"

    _chai_outputs(tmp_path / "chai_out", {0: {"agg": 0.5, "ptm": 0.6, "iptm": 0.4}})
    (res,) = ChaiBackend().collect(spec, tmp_path)
    assert res.extra["chain_map"] == ChaiBackend.chain_map(spec)


def test_colabfold_chain_map(tmp_path):
    spec = _hetero_spec(ligands=[])
    # ColabFold groups identical sequences (first appearance), then letters A, B, ...
    assert ColabFoldBackend.chain_map(spec) == {"A": "A", "B": "A2", "C": "C", "D": "B"}
    out = tmp_path / "colabfold_out"
    out.mkdir()
    _cf_model(out, 1, 1, 0, ptm=0.5, iptm=0.5, plddt=[80.0])
    (res,) = ColabFoldBackend().collect(spec, tmp_path)
    assert res.extra["chain_map"] == {"A": "A", "B": "A2", "C": "C", "D": "B"}


def test_colabfold_chain_map_matches_complex_a3m_order(tmp_path, no_tools):
    msa_dir = tmp_path / "m"
    msa_dir.mkdir()
    for cid, seq in (("A", SEQ_A), ("B", SEQ_B), ("C", SEQ_A)):
        write_a3m(msa_dir / f"{cid}.a3m", seq, n_hits=0)
    spec = _hetero_spec(ligands=[], msa="cached")
    lines = ColabFoldBackend().prepare(spec, tmp_path / "w", msa_dir=msa_dir).read_text().splitlines()
    assert lines[0] == f"#{len(SEQ_A)},{len(SEQ_B)}\t3,1"  # s1 x3 first -> output A, B, C are s1


def test_boltz_chain_map_identity(tmp_path):
    spec = _hetero_spec()
    assert BoltzBackend.chain_map(spec) == {c: c for c in ["A", "A2", "B", "C", "L", "L2"]}
    pred = tmp_path / "boltz_results_job1" / "predictions" / "job1"
    pred.mkdir(parents=True)
    _boltz_model(pred, 0, 0.61)
    (res,) = BoltzBackend().collect(spec, tmp_path)
    assert res.extra["chain_map"] == BoltzBackend.chain_map(spec)


# ---- MSA lookup error messages ------------------------------------------------

@pytest.mark.parametrize("cls", [BoltzBackend, ChaiBackend, ColabFoldBackend])
def test_missing_msa_dir_message(tmp_path, no_tools, cls):
    spec = make_spec(msa="cached", ligands=[])
    with pytest.raises(ValueError) as ei:
        cls().prepare(spec, tmp_path / "w", msa_dir=None)
    msg = str(ei.value)
    assert "None" not in msg and "no MSA directory was given" in msg

    with pytest.raises(ValueError, match=r"MSA directory .*nope does not exist"):
        cls().prepare(spec, tmp_path / "w", msa_dir=tmp_path / "nope")

    (tmp_path / "empty").mkdir(exist_ok=True)
    with pytest.raises(ValueError, match=r"looked for <chain>.a3m .* in .*empty"):
        cls().prepare(spec, tmp_path / "w", msa_dir=tmp_path / "empty")


def test_boltz_explicit_msa_must_exist(tmp_path, no_tools):
    spec = make_spec(ligands=[], chains=[Chain(id="A", sequence=SEQ_A, msa=str(tmp_path / "gone.a3m"))])
    with pytest.raises(ValueError, match="chain A msa .* does not exist"):
        BoltzBackend().prepare(spec, tmp_path / "w")
