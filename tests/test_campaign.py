"""campaign: binder screen end to end with a stub Boltz executable (no real folding)."""
from __future__ import annotations

import csv
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from foldbridge import campaign, cli
from foldbridge.spec import load_spec, spec_from_dict
from tests.test_cli import SEQ_A, make_structure

GOOD, MID, BAD, BROKEN = "MKVLAAGIIGHW", "MSTNPKPQRKTK", "MDDDEEEDDEEG", "MWWWWWWWWWWW"

# Stub "boltz predict": reads the Boltz YAML foldbridge wrote, looks the binder sequence up in
# models.json (prepared by the test) and lays out Boltz-style predictions. Unknown binder -> exit 1.
FAKE = r"""
import json, shutil, sys
from pathlib import Path
import yaml
inp, table, runlog = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text()), Path(sys.argv[3])
doc = yaml.safe_load(inp.read_text())
binder = [e["protein"]["sequence"] for e in doc["sequences"] if "protein" in e][-1]
with open(runlog, "a") as fh:
    fh.write(binder + "\n")
if binder not in table:
    sys.exit("fake boltz: cannot fold this one")
spec = table[binder]
rec = inp.stem
pred = Path(f"boltz_results_{rec}/predictions/{rec}"); pred.mkdir(parents=True, exist_ok=True)
for i, m in enumerate(spec["models"]):
    shutil.copy(m["pdb"], pred / f"{rec}_model_{i}.pdb")
    shutil.copy(m["pae"], pred / f"pae_{rec}_model_{i}.npz")
    (pred / f"confidence_{rec}_model_{i}.json").write_text(json.dumps(
        {"confidence_score": m["iptm"], "ptm": 0.7, "iptm": m["iptm"], "complex_plddt": 0.8}))
"""


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("FOLDBRIDGE_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("FOLDBRIDGE_DEBUG", raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def fake_boltz(monkeypatch, tmp_path):
    """Patch BoltzBackend to run the stub; returns a callable giving the list of folded binders."""
    from foldbridge.backends import boltz

    mdir = tmp_path / "fake_models"
    mdir.mkdir()
    table = {}
    # (sep between chains, pLDDT, PAE level, iptm) per sample: good < mid < bad interfaces
    quality = {GOOD: [(5.0, 92, 1.5, 0.88), (5.3, 90, 2.0, 0.84)],
               MID: [(6.0, 78, 5.0, 0.62), (6.3, 75, 6.0, 0.58)],
               BAD: [(14.0, 45, 20.0, 0.21), (15.0, 40, 22.0, 0.18)]}
    rng = np.random.default_rng(0)
    for seq, samples in quality.items():
        models = []
        for k, (sep, plddt, pae_level, iptm) in enumerate(samples):
            pdb = mdir / f"{seq}_{k}.pdb"
            make_structure(sep=sep, plddt=plddt).write_pdb(pdb)
            pae = mdir / f"{seq}_{k}.npz"
            np.savez(pae, pae=np.clip(rng.normal(pae_level, 0.5, size=(24, 24)), 0.5, 31))
            models.append({"pdb": str(pdb), "pae": str(pae), "iptm": iptm})
        table[seq] = {"models": models}
    tfile = tmp_path / "models.json"
    tfile.write_text(json.dumps(table))
    script = tmp_path / "fake_boltz.py"
    script.write_text(FAKE)
    runlog = tmp_path / "folded.txt"

    def command(self, spec, workdir, input_path, gpu=True):
        return [sys.executable, str(script), str(Path(input_path).resolve()), str(tfile), str(runlog)]

    monkeypatch.setattr(boltz.BoltzBackend, "available", lambda self: True)
    monkeypatch.setattr(boltz.BoltzBackend, "version", lambda self: "fake-2.0")
    monkeypatch.setattr(boltz.BoltzBackend, "command", command)
    return lambda: runlog.read_text().split() if runlog.exists() else []


@pytest.fixture
def target(tmp_path) -> Path:
    p = tmp_path / "target.yaml"
    p.write_text(textwrap.dedent(f"""\
        name: tgt
        chains:
          - id: A
            sequence: {SEQ_A}
        msa: none
        samples: 2
    """))
    return p


def fasta(path: Path, records) -> Path:
    path.write_text("".join(f">{h}\n{s}\n" for h, s in records))
    return path


def run(capsys, *argv):
    code = cli.main([str(a) for a in argv])
    out, err = capsys.readouterr()
    return code, out, err


def read_csv(p: Path):
    with open(p) as fh:
        return list(csv.DictReader(fh))


# ---- end to end -----------------------------------------------------------------------

def test_campaign_end_to_end_and_resume(capsys, tmp_path, target, fake_boltz):
    designs = fasta(tmp_path / "designs.fasta",
                    [("bad_design desc", BAD), ("mid|v2 binder", MID), ("good/1", GOOD)])
    out = tmp_path / "screen"
    code, o, e = run(capsys, "campaign", "--target", target, "--designs", designs, "--backend", "boltz",
                     "--out", out, "--top", "2", "--workers", "2", "--no-progress")
    assert code == 0, e
    assert sorted(fake_boltz()) == sorted([BAD, MID, GOOD])

    # per-design spec = target + binder chain B, named after the sanitised design id
    spec = load_spec(out / "designs" / "good_1" / "spec.yaml")
    assert spec.name == "good_1" and [c.id for c in spec.chains] == ["A", "B"]
    assert spec.chains[1].sequence == GOOD and spec.samples == 2

    rows = read_csv(out / "campaign.csv")
    assert [r["id"] for r in rows] == ["good_1", "mid_v2", "bad_design"]
    assert [r["rank"] for r in rows] == ["1", "2", "3"]
    for col in ("ipsae", "iptm", "pdockq", "ipae", "plddt_interface", "clash_fraction", "composite"):
        assert col in rows[0], col
    assert float(rows[0]["iptm"]) == pytest.approx(0.88)
    assert float(rows[0]["ipsae"]) > float(rows[1]["ipsae"]) > float(rows[2]["ipsae"])
    assert rows[0]["sequence"] == GOOD and rows[0]["status"] == "done" and rows[0]["n_models"] == "2"
    assert rows[0]["design_id"] == "good/1"

    js = json.loads((out / "campaign.json").read_text())
    assert js["campaign"]["backend"] == "boltz" and [r["id"] for r in js["rows"]] == ["good_1", "mid_v2", "bad_design"]
    models = read_csv(out / "campaign_models.csv")
    assert len(models) == 6 and models[0]["design"] == "good_1" and models[0]["pae_path"]
    html = (out / "report.html").read_text()
    assert html.index("good_1") < html.index("mid_v2") < html.index("bad_design")
    top = (out / "top_2.fasta").read_text().splitlines()
    assert top[0].startswith(">good_1 rank=1") and top[1] == GOOD and top[3] == MID and len(top) == 4
    state = json.loads((out / "campaign_state.json").read_text())
    assert {d["status"] for d in state["designs"].values()} == {"done"}
    assert "good_1" in o and "3 done" in o

    # rerun: everything already done -> nothing folded, nothing rescored, same ranking
    ck_lines = len((out / "scores.jsonl").read_text().splitlines())
    code, o, e = run(capsys, "campaign", "-t", target, "-d", designs, "-b", "boltz", "-o", out,
                     "--top", "2", "--no-progress")
    assert code == 0, e
    assert len(fake_boltz()) == 3 and "3 already done" in o
    assert len((out / "scores.jsonl").read_text().splitlines()) == ck_lines
    assert [r["id"] for r in read_csv(out / "campaign.csv")] == ["good_1", "mid_v2", "bad_design"]

    # lose the campaign state: designs are re-run through `run`, but the result cache answers
    (out / "campaign_state.json").unlink()
    code, o, e = run(capsys, "campaign", "-t", target, "-d", designs, "-b", "boltz", "-o", out, "--no-progress")
    assert code == 0, e
    assert len(fake_boltz()) == 3 and o.count("cache hit") == 3

    # a fresh campaign dir with the same designs is also served from the cache
    code, o, e = run(capsys, "campaign", "-t", target, "-d", designs, "-b", "boltz", "-o", tmp_path / "s2",
                     "--no-progress")
    assert code == 0, e
    assert len(fake_boltz()) == 3
    assert [r["id"] for r in read_csv(tmp_path / "s2" / "campaign.csv")][0] == "good_1"


def test_campaign_failures_continue_and_retry(capsys, tmp_path, target, fake_boltz):
    designs = fasta(tmp_path / "d.fasta", [("good", GOOD), ("broken", BROKEN), ("invalid", "MK1Z"),
                                           ("mid", MID)])
    out = tmp_path / "screen"
    code, o, e = run(capsys, "campaign", "-t", target, "-d", designs, "-b", "boltz", "-o", out, "--no-progress")
    assert code == 0, e
    assert "[broken] failed" in o and "[invalid] invalid design" in o
    rows = {r["id"]: r for r in read_csv(out / "campaign.csv")}
    assert rows["good"]["rank"] == "1" and rows["mid"]["rank"] == "2"
    assert rows["broken"]["status"] == "failed" and rows["broken"]["rank"] == ""
    assert "boltz failed" in rows["broken"]["error"]
    assert rows["invalid"]["status"] == "failed" and "invalid" in rows["invalid"]["error"]
    assert [r["id"] for r in read_csv(out / "campaign.csv")][:2] == ["good", "mid"]
    assert sorted(fake_boltz()) == sorted([GOOD, BROKEN, MID])

    # rerun retries only the failed design
    code, o, e = run(capsys, "campaign", "-t", target, "-d", designs, "-b", "boltz", "-o", out, "--no-progress")
    assert code == 0, e
    assert sorted(fake_boltz()) == sorted([GOOD, BROKEN, BROKEN, MID])


def test_campaign_max_jobs_slices(capsys, tmp_path, target, fake_boltz):
    designs = fasta(tmp_path / "d.fasta", [("a", GOOD), ("b", MID), ("c", BAD)])
    out = tmp_path / "screen"
    for n_done in (1, 2, 3):
        code, o, e = run(capsys, "campaign", "-t", target, "-d", designs, "-b", "boltz", "-o", out,
                         "--max-jobs", "1", "--no-progress")
        assert code == 0, e
        state = json.loads((out / "campaign_state.json").read_text())["designs"]
        assert sum(d["status"] == "done" for d in state.values()) == n_done
        assert sum(d["status"] == "pending" for d in state.values()) == 3 - n_done
        assert len(fake_boltz()) == n_done
        assert len([r for r in read_csv(out / "campaign.csv") if r["rank"]]) == n_done


def test_campaign_dry_run_and_seeds(capsys, tmp_path, target, fake_boltz):
    designs = fasta(tmp_path / "d.fasta", [("a", GOOD), ("b", MID)])
    out = tmp_path / "screen"
    code, o, e = run(capsys, "campaign", "-t", target, "-d", designs, "-b", "boltz", "-o", out,
                     "--dry-run", "--seeds", "1,2", "--samples", "3", "--no-progress")
    assert code == 0, e
    assert fake_boltz() == []  # nothing executed
    cmds = (out / "commands.txt").read_text().splitlines()
    assert len(cmds) == 4  # 2 designs x 2 seeds (Boltz takes one seed per run)
    assert (out / "designs" / "a" / "s1" / "spec.yaml").is_file()
    s2 = load_spec(out / "designs" / "a" / "s2" / "spec.yaml")
    assert s2.seeds == [2] and s2.samples == 3
    state = json.loads((out / "campaign_state.json").read_text())["designs"]
    assert {d["status"] for d in state.values()} == {"dry_run"}
    assert not (out / "top_10.fasta").exists()

    # the real run after a dry run folds everything: 2 designs x 2 seeds, 2 models each
    code, o, e = run(capsys, "campaign", "-t", target, "-d", designs, "-b", "boltz", "-o", out,
                     "--seeds", "1,2", "--no-progress")
    assert code == 0, e
    assert len(fake_boltz()) == 4
    rows = read_csv(out / "campaign.csv")
    assert [r["id"] for r in rows] == ["a", "b"] and rows[0]["n_models"] == "4"
    assert len(read_csv(out / "campaign_models.csv")) == 8


def test_campaign_errors(capsys, tmp_path, target, monkeypatch):
    from foldbridge.backends import boltz

    designs = fasta(tmp_path / "d.fasta", [("a", GOOD)])
    code, _, err = run(capsys, "campaign", "-t", tmp_path / "nope.yaml", "-d", designs, "-b", "boltz", "-o", "x")
    assert code == 2 and "target spec not found" in err
    empty = tmp_path / "empty.fasta"
    empty.write_text("")
    code, _, err = run(capsys, "campaign", "-t", target, "-d", empty, "-b", "boltz", "-o", "x")
    assert code == 2 and "no sequences" in err
    code, _, err = run(capsys, "campaign", "-t", target, "-d", designs, "-b", "boltz", "-o", "x",
                       "--binder-chain", "A")
    assert code == 2 and "binder chain" in err
    monkeypatch.setattr(boltz.BoltzBackend, "available", lambda self: False)
    code, _, err = run(capsys, "campaign", "-t", target, "-d", designs, "-b", "boltz", "-o", tmp_path / "o")
    assert code == 3 and "not installed" in err
    state = json.loads((tmp_path / "o" / "campaign_state.json").read_text())
    assert state["designs"]["a"]["status"] == "pending"


# ---- helpers ----------------------------------------------------------------------------

def test_read_designs_and_names(tmp_path):
    p = tmp_path / "x.fasta"
    p.write_text(">d1 some text\nmkv\nLAA*\n\n>d1\nGG\n>weird name/with:chars\nAC\n")
    recs = campaign.read_designs(p)
    assert recs == [("d1", "MKVLAA"), ("d1", "GG"), ("weird", "AC")]
    assert campaign._unique_names(["d1", "d1", "a/b", "a_b"]) == ["d1", "d1_1", "a_b", "a_b_1"]
    assert campaign.sanitize_name("..x y..") == "x_y"
    bad = tmp_path / "bad.fasta"
    bad.write_text("MKV\n>a\nGG\n")
    with pytest.raises(ValueError):
        campaign.read_designs(bad)


def test_free_chain_id_and_design_spec():
    t = spec_from_dict({"name": "t", "chains": [{"id": "A", "sequence": "MK", "copies": 2},
                                               {"id": "B", "sequence": "GG"}],
                        "ligands": [{"id": "C", "ccd": "HEM"}], "msa": "cached"})
    assert campaign.free_chain_id(t) == "D"
    s = campaign.design_spec(t, "x", "MKV", "D", samples=4, seeds=[7, 8], binder_msa="/tmp/x.a3m")
    assert [c.id for c in s.chains] == ["A", "B", "D"] and s.samples == 4 and s.seeds == [7, 8]
    assert s.chains[-1].msa == "/tmp/x.a3m" and t.samples == 1 and len(t.chains) == 2
    with pytest.raises(ValueError):
        campaign.design_spec(t, "x", "", "D")


def test_binder_single_sequence_msa(capsys, tmp_path, fake_boltz):
    """With msa: cached the binder gets a stable single-sequence a3m instead of failing."""
    t = tmp_path / "t.yaml"
    t.write_text(f"name: t\nchains:\n  - id: A\n    sequence: {SEQ_A}\n    msa: {tmp_path / 'A.a3m'}\nmsa: cached\n")
    (tmp_path / "A.a3m").write_text(f">A\n{SEQ_A}\n")
    designs = fasta(tmp_path / "d.fasta", [("g", GOOD)])
    code, o, e = run(capsys, "campaign", "-t", t, "-d", designs, "-b", "boltz", "-o", tmp_path / "s",
                     "--no-progress")
    assert code == 0, e
    spec = load_spec(tmp_path / "s" / "designs" / "g" / "spec.yaml")
    a3m = Path(spec.chains[1].msa)
    assert a3m.is_file() and a3m.read_text().split() == [">binder", GOOD]
    assert str(a3m).startswith(str(tmp_path / "cache"))


# ---- chain renaming, duplicate ids, column contract --------------------------------------

def _model(chain_map=None):
    return {"design": "d", "id": "d_chai_s1_m0", "backend": "chai", "seed": 1, "sample": 0, "path": "x.cif",
            "pae_path": None, "scores": {}, "chain_map": chain_map}


def test_merge_translates_chain_map():
    """Chai/ColabFold relabel A, A2, B (target copies + binder) as A, B, C: B is a target chain."""
    scored = {"ipsae": 0.9, "ipsae_A_B": 0.9, "ipsae_A_C": 0.2, "ipsae_B_C": 0.25,
              "plddt_B": 95.0, "plddt_C": 61.0, "plddt_interface": 70.0, "bsa_C": 400.0}
    row = campaign._merge(_model({"A": "A", "B": "A2", "C": "B"}), scored, "B")
    assert row["ipsae"] == 0.25 and row["ipsae_complex"] == 0.9  # binder interface, not target-target
    assert row["ipsae_A_A2"] == 0.9 and row["ipsae_A_B"] == 0.2 and row["ipsae_A2_B"] == 0.25
    assert row["plddt_binder"] == 61.0 and row["plddt_A2"] == 95.0 and row["bsa_B"] == 400.0
    assert row["plddt_interface"] == 70.0  # not a chain key
    # no chain_map: identity (Boltz keeps spec ids)
    row = campaign._merge(_model(None), {"ipsae": 0.9, "ipsae_A_A2": 0.9, "ipsae_A_B": 0.2,
                                         "ipsae_A2_B": 0.25, "plddt_B": 61.0}, "B")
    assert row["ipsae"] == 0.25 and row["plddt_binder"] == 61.0
    assert campaign.translate_chain_keys({"plddt_A": 1}, None) == {"plddt_A": 1}


def test_duplicate_fasta_ids_warn(capsys, tmp_path, target, fake_boltz):
    designs = fasta(tmp_path / "d.fasta", [("x", GOOD), ("x", MID), ("x!", BAD)])
    code, o, e = run(capsys, "campaign", "-t", target, "-d", designs, "-b", "boltz", "-o", tmp_path / "s",
                     "--dry-run", "--no-progress")
    assert code == 0, e
    assert "duplicate" in e and "'x' (record 2) duplicates record 1" in e and "renamed to 'x_1'" in e
    assert "(record 3)" in e and str(designs) in e
    assert sorted(r["id"] for r in read_csv(tmp_path / "s" / "campaign.csv")) == ["x", "x_1", "x_2"]


def test_campaign_csv_columns_and_dry_run_unranked(capsys, tmp_path, target, fake_boltz):
    designs = fasta(tmp_path / "d.fasta", [("a", GOOD), ("b", MID)])
    out = tmp_path / "s"
    code, o, e = run(capsys, "campaign", "-t", target, "-d", designs, "-b", "boltz", "-o", out,
                     "--dry-run", "--no-progress")
    assert code == 0, e
    rows = read_csv(out / "campaign.csv")
    assert set(rows[0]) == set(campaign.DESIGN_COLS)
    assert all(r["rank"] == "" and r["composite"] == "" and r["status"] == "dry_run" for r in rows)
    code, o, e = run(capsys, "campaign", "-t", target, "-d", designs, "-b", "boltz", "-o", out, "--no-progress")
    assert code == 0, e
    rows = read_csv(out / "campaign.csv")
    assert set(rows[0]) == set(campaign.DESIGN_COLS)
    for col in ("rank", "best_model", "path", "composite", "ipsae", "binder_length", "n_models", "sequence"):
        assert rows[0][col] != "", col
    assert rows[0]["best_model"].startswith("a_boltz_") and Path(rows[0]["path"]).is_file()


def test_backend_chain_map_fallback(monkeypatch):
    """Without chain_map in the run rows, campaign asks the backend's chain_map(spec)."""
    from foldbridge.backends import base

    class Fake:
        @staticmethod
        def chain_map(spec):
            return {"A": "A", "B": "A2", "C": "B"}

    monkeypatch.setattr(base, "get_backend", lambda name: Fake())
    t = spec_from_dict({"name": "t", "chains": [{"id": "A", "sequence": "MKV", "copies": 2}]})
    assert campaign._backend_chain_map("chai", t) == {"A": "A", "B": "A2", "C": "B"}
    monkeypatch.setattr(base, "get_backend", lambda name: object())
    assert campaign._backend_chain_map("chai", t) == {}
