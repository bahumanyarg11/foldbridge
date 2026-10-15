"""CLI: backends, run (dry-run + fake backend + cache hit), score, rank, prep-md, mcp, errors."""
from __future__ import annotations

import csv
import json
import shlex
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from foldbridge import cli, mdprep
from foldbridge.structure import Structure

BACKBONE = {"N": (-1.2, 0.8, 0.0), "CA": (0.0, 0.0, 0.0), "C": (1.2, 0.8, 0.0), "O": (1.2, 2.0, 0.0),
            "CB": (0.0, -1.0, 1.2)}
RES = ["MET", "LYS", "ALA", "GLU", "LEU", "ARG", "SER", "ASP", "VAL", "LYS", "GLY", "THR"]
SEQ_A = "MKAELRSDVKGT"


def make_structure(n=12, sep=7.0, plddt=85.0) -> Structure:
    rows = []
    for ci, ch in enumerate("AB"):
        flip = -1 if ci else 1
        for i in range(n):
            rn = RES[i % len(RES)]
            ca = np.array([3.8 * i, -sep * ci, 0.0])
            for an, off in BACKBONE.items():
                if an == "CB" and rn == "GLY":
                    continue
                rows.append((ca + np.array(off) * np.array([1, flip, 1]), an, an[0], rn, i + 1, ch,
                             plddt - (45 if i >= n - 2 else 0), False))
    coords, names, elems, rn, ri, ch, bf, het = zip(*rows)
    return Structure(np.asarray(coords, np.float32), np.asarray(names, object), np.asarray(elems, object),
                     np.asarray(rn, object), np.asarray(ri, np.int64), np.asarray(ch, object),
                     np.asarray(bf, np.float32), np.asarray(het, bool), "synthetic")


def run(capsys, *argv):
    code = cli.main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("FOLDBRIDGE_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("FOLDBRIDGE_DEBUG", raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def spec_file(tmp_path) -> Path:
    p = tmp_path / "job.yaml"
    p.write_text(textwrap.dedent(f"""\
        name: t
        chains:
          - id: A
            sequence: {SEQ_A}
          - id: B
            sequence: {SEQ_A}
        msa: none
        samples: 2
    """))
    return p


@pytest.fixture
def models(tmp_path) -> Path:
    d = tmp_path / "models"
    (d / "sub").mkdir(parents=True)
    make_structure().write_pdb(d / "m1.pdb")
    make_structure(sep=9.0, plddt=70).write_pdb(d / "sub" / "m2.pdb")
    rng = np.random.default_rng(0)
    pae = rng.uniform(1, 5, size=(24, 24))
    np.savez(d / "pae_m1.npz", pae=pae)
    return d


# ---- basics -----------------------------------------------------------------------------

def test_version(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0
    assert "foldbridge 0.1.0" in capsys.readouterr().out


def test_no_command_is_usage_error(capsys):
    code, _, err = run(capsys)
    assert code == 2 and "usage" in err


def test_backends(capsys):
    code, out, _ = run(capsys, "backends")
    assert code == 0
    for name in ("boltz", "chai", "colabfold"):
        assert name in out
    code, out, _ = run(capsys, "backends", "--json")
    assert code == 0 and set(json.loads(out)) >= {"boltz", "chai", "colabfold"}


# ---- run --------------------------------------------------------------------------------

def test_run_dry_run_boltz(capsys, spec_file, tmp_path):
    out_dir = tmp_path / "out"
    code, out, err = run(capsys, "run", str(spec_file), "--backend", "boltz", "--out", str(out_dir),
                         "--dry-run", "--cpu")
    assert code == 0, err
    res = json.loads((out_dir / "results.json").read_text())
    work = out_dir / "work" / res["cache_key"]  # per-key workdir
    cmd = work / "command.txt"
    assert cmd.is_file() and res["workdir"] == str(work)
    text = cmd.read_text()
    assert "predict" in text and "--accelerator cpu" in text and "--diffusion_samples 2" in text
    assert (work / "t.yaml").is_file()
    assert res["dry_run"] is True and res["results"] == [] and res["backend"] == "boltz"
    assert res["command"].strip() == text.strip()
    assert "command:" in out
    assert not (tmp_path / "cache" / "results").exists()  # dry runs never touch the cache


FAKE_BOLTZ = r"""
import json, sys, shutil
from pathlib import Path
src = Path(sys.argv[1])
pred = Path("boltz_results_t/predictions/t"); pred.mkdir(parents=True, exist_ok=True)
for i, (rk, iptm) in enumerate([(0.6, 0.55), (0.9, 0.85)]):
    shutil.copy(src, pred / f"t_model_{i}.pdb")
    (pred / f"confidence_t_model_{i}.json").write_text(json.dumps(
        {"confidence_score": rk, "ptm": 0.7, "iptm": iptm, "complex_plddt": 0.8}))
"""


@pytest.fixture
def fake_boltz(monkeypatch, tmp_path):
    from foldbridge.backends import boltz

    model = tmp_path / "fake_model.pdb"
    make_structure().write_pdb(model)
    script = tmp_path / "fake_boltz.py"
    script.write_text(FAKE_BOLTZ)
    calls = []

    def command(self, spec, workdir, input_path, gpu=True):
        calls.append(input_path)
        return [sys.executable, str(script), str(model)]

    monkeypatch.setattr(boltz.BoltzBackend, "available", lambda self: True)
    monkeypatch.setattr(boltz.BoltzBackend, "version", lambda self: "fake-1.0")
    monkeypatch.setattr(boltz.BoltzBackend, "command", command)
    return calls


def test_run_fake_backend_scores_and_caches(capsys, spec_file, tmp_path, fake_boltz):
    out1 = tmp_path / "o1"
    code, out, err = run(capsys, "run", str(spec_file), "-b", "boltz", "--out", str(out1), "--score")
    assert code == 0, err
    assert len(fake_boltz) == 1 and "cached as" in out
    res = json.loads((out1 / "results.json").read_text())
    assert res["n_models"] == 2 and res["cache_hit"] is False
    rows = res["results"]
    assert rows[0]["rank"] == 1 and rows[0]["iptm"] == pytest.approx(0.85)
    assert (out1 / "report.html").is_file()

    # identical spec + backend version -> cache hit, backend not invoked again
    out2 = tmp_path / "o2"
    code, out, err = run(capsys, "run", str(spec_file), "-b", "boltz", "--out", str(out2))
    assert code == 0, err
    assert len(fake_boltz) == 1 and "cache hit" in out
    res2 = json.loads((out2 / "results.json").read_text())
    assert res2["cache_hit"] is True and res2["n_models"] == 2
    assert all(Path(r["path"]).parent == out2 / "models" for r in res2["results"])

    # --no-cache forces a re-run
    code, out, err = run(capsys, "run", str(spec_file), "-b", "boltz", "--out", str(tmp_path / "o3"), "--no-cache")
    assert code == 0 and len(fake_boltz) == 2


def test_run_backend_missing_exit_3(capsys, spec_file, monkeypatch):
    from foldbridge.backends import boltz

    monkeypatch.setattr(boltz.BoltzBackend, "available", lambda self: False)
    code, _, err = run(capsys, "run", str(spec_file), "-b", "boltz")
    assert code == 3 and "not installed" in err and "Traceback" not in err


def test_run_bad_spec_clean_error(capsys, tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\nchains:\n  - id: A\n    sequence: MK1Z\n")
    code, _, err = run(capsys, "run", str(bad), "-b", "boltz", "--dry-run")
    assert code == 2 and "invalid" in err and "Traceback" not in err
    code, _, err = run(capsys, "run", str(tmp_path / "nope.yaml"), "-b", "boltz", "--dry-run")
    assert code == 2 and "not found" in err


def test_run_unknown_backend_is_usage_error(capsys, spec_file):
    with pytest.raises(SystemExit) as e:
        cli.main(["run", str(spec_file), "-b", "alphafold9"])
    assert e.value.code == 2


# ---- score / rank -----------------------------------------------------------------------------

def test_find_pae(models):
    assert cli.find_pae(models / "m1.pdb") == models / "pae_m1.npz"
    assert cli.find_pae(models / "sub" / "m2.pdb") is None
    cf = models / "x_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb"
    sc = models / "x_scores_rank_001_alphafold2_multimer_v3_model_1_seed_000.json"
    cf.write_text("END\n"); sc.write_text("{}")
    assert cli.find_pae(cf) == sc


def test_score_and_rank(capsys, models, tmp_path):
    pytest.importorskip("foldbridge.score")
    out_csv = tmp_path / "scores.csv"
    code, out, err = run(capsys, "score", str(models), "--out", str(out_csv))
    assert code == 0, err
    with open(out_csv) as fh:
        rows = list(csv.DictReader(fh))
    assert {r["id"] for r in rows} == {"m1", "m2"}
    by_id = {r["id"]: r for r in rows}
    assert by_id["m1"]["pae_path"].endswith("pae_m1.npz")
    assert "wrote" in out

    code, out, err = run(capsys, "rank", str(out_csv), "--top", "1", "--html", str(tmp_path / "r.html"),
                         "--out", str(tmp_path / "ranked.json"))
    assert code == 0, err
    assert (tmp_path / "r.html").is_file()
    ranked = json.loads((tmp_path / "ranked.json").read_text())
    ranked = ranked["rows"] if isinstance(ranked, dict) and "rows" in ranked else ranked
    assert len(ranked) == 1 and ranked[0]["rank"] == 1

    # json input, --no-length-norm, explicit metric
    code, out, err = run(capsys, "rank", str(tmp_path / "ranked.json"), str(out_csv), "--by", "plddt",
                         "--no-length-norm")
    assert code == 0, err
    assert "plddt" in out


def test_rank_errors(capsys, tmp_path, models):
    code, _, err = run(capsys, "rank", str(tmp_path / "missing.csv"))
    assert code == 2 and "not found" in err
    p = tmp_path / "r.csv"
    p.write_text("id,iptm\na,0.5\nb,0.7\n")
    code, _, err = run(capsys, "rank", str(p), "--by", "nonexistent_metric")
    assert code == 2 and "not present" in err
    code, out, err = run(capsys, "rank", str(p), "--by", "iptm")
    assert code == 0, err
    lines = [ln for ln in out.splitlines() if ln.startswith("1 ")]
    assert lines and " b " in lines[0] + " "


def test_score_errors(capsys, models, tmp_path):
    code, _, err = run(capsys, "score", str(models), "--pae", str(models / "pae_m1.npz"))
    assert code == 2 and "exactly one" in err
    code, _, err = run(capsys, "score", str(tmp_path / "nothing_here"))
    assert code == 2 and "no such file" in err
    empty = tmp_path / "empty"; empty.mkdir()
    code, _, err = run(capsys, "score", str(empty))
    assert code == 2 and "no structure files" in err


# ---- prep-md / mcp -----------------------------------------------------------------------------

def test_prep_md_fallback(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(mdprep, "openmm_available", lambda: False)
    p = tmp_path / "m.pdb"
    make_structure().write_pdb(p)
    code, out, err = run(capsys, "prep-md", str(p), "--out", str(tmp_path / "md"), "--ph", "7.4")
    assert code == 0, err
    rep = json.loads((tmp_path / "md" / "system_report.json").read_text())
    assert rep["openmm"] is False and rep["ph"] == 7.4
    assert (tmp_path / "md" / "prepared.pdb").is_file()
    assert "numpy fallback" in out and "openmm" in err  # install hint as a warning
    code, _, err = run(capsys, "prep-md", str(tmp_path / "missing.pdb"))
    assert code == 2 and "not found" in err


def test_mcp_missing_dependency(monkeypatch, capsys):
    from foldbridge import mcp_server

    def boom():
        raise ImportError("no mcp")

    monkeypatch.setattr(mcp_server, "_server_class", boom)
    assert mcp_server.main() == 1
    assert "pip install" in capsys.readouterr().err
    assert cli.main(["mcp"]) == 1


def test_mcp_tools_register():
    pytest.importorskip("mcp")
    import asyncio

    from foldbridge import mcp_server

    server = mcp_server.build_server()
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert names == {"list_backends", "score_structure", "rank_results", "run_prediction", "prep_md"}


def test_mcp_tool_functions(tmp_path, models, spec_file):
    from foldbridge import mcp_server as m

    assert "boltz" in m.tool_list_backends()
    s = m.tool_score_structure(str(models / "m1.pdb"))
    assert "error" not in s and s["id"] == "m1"
    with pytest.raises(m.ToolError, match="no such file"):
        m.tool_score_structure(str(tmp_path / "nope.pdb"))
    with pytest.raises(m.ToolError, match="directory"):
        m.tool_score_structure(str(models))
    # chains restricts the metrics
    one = m.tool_score_structure(str(models / "m1.pdb"), chains=["A"])
    assert one.get("ipsae") is None and m.tool_score_structure(str(models / "m1.pdb"), chains="A,B")["id"] == "m1"
    ranked = m.tool_rank_results([str(models)])
    assert [r["rank"] for r in ranked] == [1, 2]
    with pytest.raises(m.ToolError, match="not present"):
        m.tool_rank_results([str(models)], by="nonexistent_metric")
    dr = m.tool_run_prediction(str(spec_file), "boltz", out_dir=str(tmp_path / "mcp_out"))
    assert dr["dry_run"] is True and "predict" in dr["command"]
    json.dumps(dr)
    with pytest.raises(m.ToolError):
        m.tool_run_prediction(str(tmp_path / "missing.yaml"), "boltz")
    with pytest.raises(m.ToolError):
        m.tool_prep_md(str(tmp_path / "missing.pdb"))


def test_mcp_errors_are_is_error_results(models, tmp_path):
    """Through a real MCP client: failures come back as isError results, not {"error": ...} payloads."""
    pytest.importorskip("mcp")
    import asyncio

    try:
        from mcp import Client
    except ImportError:
        pytest.skip("mcp SDK without the in-process Client")
    from foldbridge import mcp_server

    async def go():
        async with Client(mcp_server.build_server()) as c:
            tools = {t.name: t for t in (await c.list_tools()).tools}
            assert "chains" in tools["score_structure"].input_schema["properties"]
            assert "Parameters:" in tools["score_structure"].description
            bad = await c.call_tool("score_structure", {"path": str(tmp_path / "nope.pdb")})
            good = await c.call_tool("score_structure", {"path": str(models / "m1.pdb"), "chains": ["A", "B"]})
            return bad, good

    bad, good = asyncio.run(go())
    assert bad.is_error is True and "nope.pdb" in bad.content[0].text
    assert good.is_error is False


# ---- review fixes: score -----------------------------------------------------------------------

DATA = Path(__file__).resolve().parent / "data"


def test_score_missing_chains_is_clear_error(capsys, models):
    code, _, err = run(capsys, "score", str(models / "m1.pdb"), "--chains", "E,I", "--no-progress")
    assert code == 2 and "chains E,I not found in m1.pdb (available: A, B)" in err
    code, _, err = run(capsys, "score", str(models / "m1.pdb"), "--chains", "A,Z", "--no-progress")
    assert code == 2 and "chain Z not found in m1.pdb (available: A, B)" in err


def test_score_prints_per_file_warnings_and_notes(capsys, models, tmp_path):
    bad = tmp_path / "bad_pae.npy"
    np.save(bad, np.ones((5, 5)))
    code, out, err = run(capsys, "score", str(models / "m1.pdb"), "--pae", str(bad), "--no-progress",
                         "-o", str(tmp_path / "a.csv"))
    assert code == 0, err
    assert f"warning: {models / 'm1.pdb'}: PAE size 5 != 24 tokens; skipping PAE metrics" in err
    txt = tmp_path / "pae.txt"
    txt.write_text("1 2\n3 4\n")
    code, _, err = run(capsys, "score", str(models / "m1.pdb"), "--pae", str(txt), "--no-progress",
                       "-o", str(tmp_path / "b.csv"))
    assert code == 0 and "warning: " in err and "unsupported PAE file" in err

    xtal = tmp_path / "xtal"
    xtal.mkdir()
    (xtal / "1BRS.pdb").write_text((DATA / "1BRS.pdb").read_text())
    code, _, err = run(capsys, "score", str(xtal), "--chains", "A,D", "--no-progress", "-o", str(tmp_path / "c.csv"))
    assert code == 0, err
    assert err.count("experimental structure: B-factors are not pLDDT") == 1
    with open(tmp_path / "c.csv") as fh:
        row = next(csv.DictReader(fh))
    assert not any(k.startswith("plddt") for k in row) and "experimental" in row["note"]
    assert "pdockq" not in row and row["n_chains"] == "2" and row["n_contacts"].isdigit()

    ref = tmp_path / "ref.pdb"
    st = make_structure()
    st.chain_ids = np.where(st.chain_ids == "A", "X", "Y").astype(object)
    st.write_pdb(ref)
    code, _, err = run(capsys, "score", str(models / "m1.pdb"), "--reference", str(ref), "--no-progress",
                       "-o", str(tmp_path / "d.csv"))
    assert code == 0, err
    assert "reference matched only 0 of 24 residues; check chain ids/numbering" in err
    with open(tmp_path / "d.csv") as fh:
        assert "reference matched only" in next(csv.DictReader(fh))["warnings"]


def test_score_output_extension_checked_first(capsys, models, monkeypatch):
    from foldbridge import batch

    monkeypatch.setattr(batch, "score_many", lambda *a, **k: pytest.fail("scored before validating -o"))
    code, _, err = run(capsys, "score", str(models), "-o", "results.xlsx")
    assert code == 2 and "unsupported output type '.xlsx'" in err
    code, _, err = run(capsys, "rank", "x.csv", "--html", "r.csv")
    assert code == 2 and "--html" in err


def test_score_id_collisions(capsys, tmp_path):
    d = tmp_path / "pair"
    d.mkdir()
    (d / "1A2K.pdb").write_text((DATA / "1A2K.pdb").read_text())
    (d / "1A2K.cif").write_text((DATA / "1A2K.cif").read_text())
    code, _, err = run(capsys, "score", str(d), "--no-progress", "-o", str(tmp_path / "s.json"))
    assert code == 0, err
    ids = sorted(r["id"] for r in json.loads((tmp_path / "s.json").read_text()))
    assert ids == ["1A2K.cif", "1A2K.pdb"]
    rows = cli.score_paths([d])
    assert sorted(r["id"] for r in rows) == ["1A2K.cif", "1A2K.pdb"]


def test_debug_flag_anywhere(capsys, tmp_path):
    p = cli.build_parser()
    assert p.parse_args(["score", "x", "--debug"]).debug is True
    assert p.parse_args(["--debug", "score", "x"]).debug is True
    assert p.parse_args(["score", "x"]).debug is False
    with pytest.raises(cli.CLIError):
        cli.main(["score", str(tmp_path / "missing.pdb"), "--debug"])


def test_outputs_round_tiny_values_and_int_counts(tmp_path):
    rows = [{"id": "a", "composite": 1e-17, "x": -3e-12, "n_contacts": 4.0, "rank": 1.0, "bsa": 12.5}]
    cli.write_rows(rows, tmp_path / "r.csv")
    cli.write_rows(rows, tmp_path / "r.json")
    with open(tmp_path / "r.csv") as fh:
        r = next(csv.DictReader(fh))
    assert r["composite"] == "0.0" and r["x"] == "0.0" and r["n_contacts"] == "4" and r["rank"] == "1"
    j = json.loads((tmp_path / "r.json").read_text())[0]
    assert j["composite"] == 0.0 and j["n_contacts"] == 4 and isinstance(j["n_contacts"], int)
    assert cli._clean_json({"v": -1e-12, "w": 0.5}) == {"v": 0.0, "w": 0.5}


# ---- review fixes: rank ------------------------------------------------------------------------

def test_rank_error_messages(capsys, tmp_path, models):
    dry = tmp_path / "dry.json"
    dry.write_text(json.dumps({"dry_run": True, "n_models": 0, "results": []}))
    code, _, err = run(capsys, "rank", str(dry))
    assert code == 2 and f"{dry} is from a --dry-run (0 models); run without --dry-run first" in err
    code, _, err = run(capsys, "rank", str(models))
    assert code == 2 and f"{models} is a directory; rank takes results .csv/.json (use foldbridge score DIR first)" in err
    p = tmp_path / "r.csv"
    p.write_text("id,iptm,bsa,path\na,0.5,100,/m/a.cif\nb,0.7,200,/m/b.cif\n")
    code, _, err = run(capsys, "rank", str(p), "--by", "nope")
    assert code == 2 and "not present" in err and "available: composite, bsa, iptm" in err


def test_rank_dedupes_and_leaves_metricless_rows_unranked(capsys, tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("id,iptm,path,status\n007,0.5,/m/a.cif,done\nb,0.7,/m/b.cif,done\nc,,/m/c.cif,failed\n")
    out = tmp_path / "ranked.csv"
    code, o, err = run(capsys, "rank", str(p), str(p), "--out", str(out))  # same file twice
    assert code == 0, err
    with open(out) as fh:
        rows = list(csv.DictReader(fh))
    assert [r["id"] for r in rows] == ["b", "007", "c"]  # "007" stays a string
    assert [r["rank"] for r in rows] == ["1", "2", ""]
    assert "1 row(s) without metrics left unranked" in err


def test_csv_reload_keeps_string_columns(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("id,sequence,design_id,backend,iptm,n_models\n007,1,2e5,nan,0.5,2\n")
    r = cli.load_rows(p)[0]
    assert r["id"] == "007" and r["sequence"] == "1" and r["design_id"] == "2e5" and r["backend"] == "nan"
    assert r["iptm"] == 0.5 and r["n_models"] == 2


# ---- review fixes: prep-md -----------------------------------------------------------------------

def test_prep_md_default_out_in_cwd(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(mdprep, "openmm_available", lambda: False)
    src_dir = tmp_path / "inputs"
    src_dir.mkdir()
    p = src_dir / "m.pdb"
    make_structure().write_pdb(p)
    work = tmp_path / "elsewhere"
    work.mkdir()
    monkeypatch.chdir(work)
    code, out, err = run(capsys, "prep-md", str(p))
    assert code == 0, err
    assert (work / "m_md" / "prepared.pdb").is_file() and not (src_dir / "m_md").exists()
    assert f"prepared:      {(work / 'm_md' / 'prepared.pdb').resolve()}" in out
    assert str((work / "m_md" / "system_report.json").resolve()) in out
    code, _, err = run(capsys, "prep-md", str(p), "--solvate")
    assert code == 2 and "--solvate needs openmm" in err
    assert "pH" in cli.build_parser()._subparsers._group_actions[0].choices["prep-md"].format_help()


# ---- review fixes: run / backends --------------------------------------------------------------

def test_backend_missing_executable_is_runtime_error(tmp_path, spec_file, monkeypatch, capsys):
    from foldbridge.backends import boltz
    from foldbridge.spec import load_spec

    monkeypatch.setattr(boltz.BoltzBackend, "command",
                        lambda self, spec, workdir, inp, gpu=True: [str(tmp_path / "no-such-boltz"), "predict"])
    with pytest.raises(RuntimeError, match="not found"):
        boltz.BoltzBackend().run(load_spec(spec_file), tmp_path / "w")
    monkeypatch.setattr(boltz.BoltzBackend, "available", lambda self: True)
    code, _, err = run(capsys, "run", str(spec_file), "-b", "boltz", "--out", str(tmp_path / "o"))
    assert code == 1 and "no-such-boltz" in err and "not found" in err and "Traceback" not in err


def test_run_relative_out_writes_cache(capsys, spec_file, tmp_path, fake_boltz):
    code, out, err = run(capsys, "run", str(spec_file), "-b", "boltz", "--out", "rel_out")  # cwd = tmp_path
    assert code == 0, err
    assert "cached as" in out and "could not write cache" not in out + err
    res = json.loads((tmp_path / "rel_out" / "results.json").read_text())
    assert Path(res["out_dir"]).is_absolute() and all(Path(r["path"]).is_absolute() for r in res["results"])
    cmd = (Path(res["workdir"]) / "command.txt").read_text().strip()
    assert shlex.split(cmd)[0] == sys.executable  # shlex-quoted argv


def test_run_rerun_ignores_stale_workdir(capsys, spec_file, tmp_path, fake_boltz):
    out_dir = tmp_path / "o"
    code, _, err = run(capsys, "run", str(spec_file), "-b", "boltz", "--out", str(out_dir), "--no-cache")
    assert code == 0, err
    res = json.loads((out_dir / "results.json").read_text())
    stale = Path(res["workdir"]) / "boltz_results_t" / "predictions" / "t"
    (stale / "t_model_7.pdb").write_text((stale / "t_model_0.pdb").read_text())
    (stale / "confidence_t_model_7.json").write_text(json.dumps({"confidence_score": 0.99, "iptm": 0.99}))
    code, _, err = run(capsys, "run", str(spec_file), "-b", "boltz", "--out", str(out_dir), "--no-cache")
    assert code == 0, err
    res = json.loads((out_dir / "results.json").read_text())
    assert res["n_models"] == 2 and not (stale / "t_model_7.pdb").exists()


def test_run_multi_seed_boltz_runs_each_seed(capsys, tmp_path, fake_boltz):
    spec = tmp_path / "ms.yaml"
    spec.write_text(f"name: t\nchains:\n  - id: A\n    sequence: {SEQ_A}\nmsa: none\nseeds: [3, 8]\nsamples: 2\n")
    code, _, err = run(capsys, "run", str(spec), "-b", "boltz", "--out", str(tmp_path / "o"))
    assert code == 0, err
    assert len(fake_boltz) == 2
    res = json.loads((tmp_path / "o" / "results.json").read_text())
    assert res["n_models"] == 4 and sorted({r["seed"] for r in res["results"]}) == [3, 8]
    assert len({Path(r["path"]).parent for r in res["results"]}) == 2  # one workdir per seed
    assert len(res["command"].splitlines()) == 2


def test_default_out_dir_sanitizes_spec_name(capsys, tmp_path):
    assert cli.safe_name("my job:1") == "my_job_1" and cli.safe_name("a/b") == "a_b"
    for bad in ("../evil", "..", "...", "a/../b"):
        with pytest.raises(cli.CLIError):
            cli.safe_name(bad)
    spec = tmp_path / "bad.yaml"
    spec.write_text(f"name: ../evil\nchains:\n  - id: A\n    sequence: {SEQ_A}\nmsa: none\n")
    code, _, err = run(capsys, "run", str(spec), "-b", "boltz", "--dry-run")
    assert code == 2 and "path traversal" in err
    assert not (tmp_path / "evil-boltz").exists()
