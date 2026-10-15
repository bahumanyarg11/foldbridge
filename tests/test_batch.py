"""batch.score_many: parallel == serial, PAE pairing, checkpoint resume, per-file errors, CLI."""
from __future__ import annotations

import csv
import io
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from foldbridge import batch, cli

DATA = Path(__file__).resolve().parent / "data"


def perturb_pdb(src: Path, dst: Path, sigma: float, seed: int) -> Path:
    """Copy a PDB file, adding N(0, sigma) noise to every ATOM/HETATM coordinate."""
    rng = np.random.default_rng(seed)
    out = []
    for line in src.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")) and len(line) >= 54:
            xyz = [float(line[30 + 8 * k: 38 + 8 * k]) + rng.normal(0, sigma) for k in range(3)]
            line = line[:30] + "".join(f"{v:8.3f}" for v in xyz) + line[54:]
        out.append(line)
    dst.write_text("\n".join(out) + "\n")
    return dst


def same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    if isinstance(a, float) or isinstance(b, float):
        return a == pytest.approx(b, rel=1e-12, abs=1e-12)
    return a == b


def assert_rows_equal(r1, r2):
    assert len(r1) == len(r2)
    for a, b in zip(r1, r2):
        assert a.keys() == b.keys(), set(a) ^ set(b)
        for k in a:
            assert same(a[k], b[k]), (k, a[k], b[k])


@pytest.fixture
def copies(tmp_path) -> list[Path]:
    d = tmp_path / "copies"
    d.mkdir()
    return [perturb_pdb(DATA / "1BRS.pdb", d / f"brs_{i:02d}.pdb", 0.3, i) for i in range(6)]


@pytest.fixture
def with_pae(tmp_path) -> Path:
    """A two-chain synthetic model with a Boltz-style pae_<stem>.npz next to it."""
    from tests.test_cli import make_structure

    d = tmp_path / "paired"
    d.mkdir()
    make_structure().write_pdb(d / "m1.pdb")
    np.savez(d / "pae_m1.npz", pae=np.random.default_rng(1).uniform(1, 5, size=(24, 24)))
    return d / "m1.pdb"


def test_parallel_equals_serial(copies, with_pae):
    files = copies + [with_pae]
    serial = batch.score_many(files, workers=1, progress=False)
    parallel = batch.score_many(files, workers=3, progress=False)
    assert_rows_equal(serial, parallel)
    assert [r["path"] for r in serial] == [str(f) for f in files]  # input order kept
    assert all("error" not in r for r in serial)
    paired = serial[-1]
    assert paired["pae_path"].endswith("pae_m1.npz") and "ipsae" in paired
    # the perturbed copies really differ
    assert len({round(r["bsa"], 6) for r in serial[:-1]}) == len(copies)


def test_matches_score_structure_and_reference(copies):
    from foldbridge.score import score_structure

    rows = batch.score_many(copies[:2], reference=DATA / "1BRS.pdb", chains=["A", "D"], workers=2,
                            progress=False)
    direct = score_structure(copies[0], reference=DATA / "1BRS.pdb", chains=["A", "D"])
    for k, v in direct.items():
        assert same(rows[0][k], v), k
    assert rows[0]["n_chains"] == 2 and 0 < rows[0]["rmsd"] < 1.5


def test_directory_input_and_pae_mapping(with_pae, tmp_path):
    rows = batch.score_many([with_pae.parent], progress=False)
    assert [r["id"] for r in rows] == ["m1"] and rows[0]["pae_path"]
    rows = batch.score_many([with_pae], pae={str(with_pae): None}, progress=False)
    assert rows[0]["pae_path"] is None and "ipsae" not in rows[0]
    with pytest.raises(ValueError):
        batch.score_many([with_pae, with_pae.parent / "x.pdb"], pae=str(with_pae), progress=False)


@pytest.mark.parametrize("workers", [1, 2])
def test_error_capture(copies, tmp_path, workers):
    bad_text = tmp_path / "corrupt.pdb"
    bad_text.write_text("ATOM      1  CA  ALA A   1      xx.xxx  11.104   6.134  1.00  0.00           C\n")
    bad_bin = tmp_path / "binary.cif"
    bad_bin.write_bytes(bytes(range(256)) * 8)
    empty = tmp_path / "empty.pdb"
    empty.write_text("REMARK nothing here\n")
    missing = tmp_path / "missing.pdb"
    files = [copies[0], bad_text, bad_bin, empty, missing, copies[1]]
    rows = batch.score_many(files, workers=workers, progress=False)
    assert len(rows) == len(files)
    assert [("error" in r) for r in rows] == [False, True, True, True, True, False]
    assert "ValueError" in rows[1]["error"] and "no atoms" in rows[3]["error"]
    assert "FileNotFoundError" in rows[4]["error"]
    assert rows[0]["n_atoms"] > 0 and rows[5]["n_atoms"] > 0


def test_checkpoint_resume(copies, tmp_path, monkeypatch):
    ck = tmp_path / "ck" / "scores.jsonl"
    calls = []
    real = batch._score_one

    def counting(path, *a):
        calls.append(Path(path).name)
        return real(path, *a)

    monkeypatch.setattr(batch, "_score_one", counting)
    first = batch.score_many(copies[:3], workers=1, progress=False, checkpoint=ck)
    assert len(calls) == 3 and len(ck.read_text().splitlines()) == 3

    # simulate a crash that left a torn half-line at the end
    with open(ck, "a") as fh:
        fh.write('{"v":1,"key":"/x","sig":{},"row":{"id"')
    calls.clear()
    full = batch.score_many(copies, workers=1, progress=False, checkpoint=ck)
    assert sorted(calls) == [p.name for p in copies[3:]]  # only the new files were scored
    assert_rows_equal(first, full[:3])  # checkpointed rows round-trip exactly (NaN included)
    fresh = batch.score_many(copies, workers=1, progress=False)
    assert_rows_equal(full, fresh)

    # nothing left to do -> nothing scored, checkpoint untouched
    calls.clear()
    n_lines = len(ck.read_text().splitlines())
    again = batch.score_many(copies, workers=1, progress=False, checkpoint=ck)
    assert calls == [] and len(ck.read_text().splitlines()) == n_lines
    assert_rows_equal(again, fresh)

    # a modified file is rescored; a different reference invalidates every entry
    os.utime(copies[0], ns=(1, 1))
    calls.clear()
    batch.score_many(copies, workers=1, progress=False, checkpoint=ck)
    assert calls == [copies[0].name]
    calls.clear()
    batch.score_many(copies[:2], reference=DATA / "1BRS.pdb", workers=1, progress=False, checkpoint=ck)
    assert len(calls) == 2


def test_checkpoint_retries_errors(tmp_path, copies):
    bad = tmp_path / "late.pdb"
    bad.write_text("garbage\n")
    ck = tmp_path / "s.jsonl"
    rows = batch.score_many([bad, copies[0]], workers=1, progress=False, checkpoint=ck)
    assert "error" in rows[0]
    bad.write_text(copies[1].read_text())  # file fixed (size changes too)
    rows = batch.score_many([bad, copies[0]], workers=1, progress=False, checkpoint=ck)
    assert "error" not in rows[0]


def test_parallel_checkpoint(copies, tmp_path):
    ck = tmp_path / "p.jsonl"
    rows = batch.score_many(copies, workers=3, progress=False, checkpoint=ck)
    recs = [json.loads(line) for line in ck.read_text().splitlines()]
    assert sorted(r["key"] for r in recs) == sorted(str(p.resolve()) for p in copies)
    again = batch.score_many(copies, workers=3, progress=False, checkpoint=ck)
    assert_rows_equal(rows, again)


def test_progress_line():
    buf = io.StringIO()
    bar = batch.Progress(20, stream=buf)
    for k in range(20):
        bar.update(error=(k == 3))
    bar.close()
    lines = buf.getvalue().strip().splitlines()
    assert lines[-1].startswith("score 20/20") and "/s" in lines[-1] and "ETA" in lines[-1]
    assert "1 failed" in lines[-1]
    assert 2 <= len(lines) <= 12  # non-tty: about one line per 10 %
    silent = io.StringIO()
    batch.Progress(5, stream=silent, enabled=False).update()
    assert silent.getvalue() == ""


def test_default_workers():
    assert batch.default_workers() == max(1, (os.cpu_count() or 2) - 1)


def test_cli_score_workers_checkpoint(capsys, copies, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    bad = copies[0].parent / "zz_bad.pdb"
    bad.write_text("ATOM      1  CA  ALA A   1      xx.xxx  11.104   6.134  1.00  0.00           C\n")
    ck = tmp_path / "ck.jsonl"
    out = tmp_path / "s.csv"
    code = cli.main(["score", str(copies[0].parent), "--workers", "2", "--checkpoint", str(ck),
                     "-o", str(out), "--chains", "A,D"])
    o, e = capsys.readouterr()
    assert code == 0, e
    assert "failed to score" in e and "zz_bad.pdb" in e and "1 failed" in o
    with open(out) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(copies) and all(r["n_chains"] == "2" for r in rows)
    assert len(ck.read_text().splitlines()) == len(copies) + 1
    code = cli.main(["score", str(copies[0].parent), "--workers", "0"])
    assert code == 2 and "--workers" in capsys.readouterr().err


def test_broken_pool_falls_back_to_serial(copies, monkeypatch, capsys):
    from concurrent.futures import Future
    from concurrent.futures.process import BrokenProcessPool

    class DeadPool:  # what an unguarded script under the spawn start method produces
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def submit(self, *a, **k):
            f = Future()
            f.set_exception(BrokenProcessPool("A process in the process pool was terminated abruptly"))
            return f

    monkeypatch.setattr(batch, "ProcessPoolExecutor", DeadPool)
    rows = batch.score_many(copies[:3], workers=3, progress=False)
    err = capsys.readouterr().err
    assert len(rows) == 3 and all("error" not in r and r["n_atoms"] > 0 for r in rows)
    assert err.count("warning:") == 1 and 'if __name__ == "__main__":' in err and "serially" in err
    assert_rows_equal(rows, batch.score_many(copies[:3], workers=1, progress=False))


def test_checkpoint_dedupes_failed_files(tmp_path, copies):
    bad = tmp_path / "bad.pdb"
    bad.write_text("garbage\n")
    ck = tmp_path / "ck.jsonl"
    for _ in range(3):  # the failure is retried each time but not re-appended
        rows = batch.score_many([bad, copies[0]], workers=1, progress=False, checkpoint=ck)
        assert "error" in rows[0]
        assert len(ck.read_text().splitlines()) == 2
    # duplicate lines (e.g. from an older version) are compacted, last write wins
    lines = ck.read_text().splitlines()
    rec = json.loads(lines[1])
    rec["row"]["bsa"] = -1.0
    with open(ck, "a") as fh:
        fh.write(lines[1] + "\n" + json.dumps(rec) + "\n")
    assert batch._load_checkpoint(ck)[rec["key"]]["row"]["bsa"] == -1.0
    batch.score_many([bad, copies[0], copies[1]], workers=1, progress=False, checkpoint=ck)
    keys = [json.loads(x)["key"] for x in ck.read_text().splitlines()]
    assert len(keys) == len(set(keys)) == 3
    assert batch.compact_checkpoint(ck) is False


def test_progress_counts_cached_rows(copies, tmp_path, monkeypatch):
    ck = tmp_path / "ck.jsonl"
    batch.score_many(copies[:4], workers=1, progress=False, checkpoint=ck)
    buf = io.StringIO()
    real = batch.Progress
    monkeypatch.setattr(batch, "Progress", lambda total, enabled=True: real(total, stream=buf, enabled=enabled))
    batch.score_many(copies, workers=1, progress=True, checkpoint=ck)
    lines = buf.getvalue().strip().splitlines()
    assert lines[0].startswith("score 4/6 (4 cached)")
    assert lines[-1].startswith("score 6/6 (4 cached)")
    bar = real(6, stream=io.StringIO())
    bar.skip(4)
    assert bar.line().startswith("score 4/6 (4 cached)  0.0/s")  # cached rows don't inflate the rate


def test_checkpoint_signature_tracks_pae_and_reference_content(with_pae, tmp_path, monkeypatch):
    ck = tmp_path / "ck.jsonl"
    ref = tmp_path / "ref.pdb"
    ref.write_text(with_pae.read_text())
    first = batch.score_many([with_pae], reference=ref, workers=1, progress=False, checkpoint=ck)
    calls = []
    real = batch._score_one
    monkeypatch.setattr(batch, "_score_one", lambda *a: calls.append(a) or real(*a))
    batch.score_many([with_pae], reference=ref, workers=1, progress=False, checkpoint=ck)
    assert calls == []
    pae = with_pae.parent / "pae_m1.npz"
    np.savez(pae, pae=np.full((24, 24), 25.0))  # same path, new content
    again = batch.score_many([with_pae], reference=ref, workers=1, progress=False, checkpoint=ck)
    assert len(calls) == 1 and again[0]["ipsae"] != first[0]["ipsae"]
    os.utime(ref, ns=(5, 5))  # reference touched -> rescored
    batch.score_many([with_pae], reference=ref, workers=1, progress=False, checkpoint=ck)
    assert len(calls) == 2


def test_ids_disambiguated_when_stems_collide(tmp_path):
    a, b = tmp_path / "1A2K.pdb", tmp_path / "1A2K.cif"
    a.write_text((DATA / "1A2K.pdb").read_text())
    b.write_text((DATA / "1A2K.cif").read_text())
    (tmp_path / "sub").mkdir()
    c = tmp_path / "sub" / "1A2K.pdb"
    c.write_text(a.read_text())
    d = tmp_path / "other.pdb"
    d.write_text(a.read_text())
    rows = batch.score_many([a, b, d], workers=1, progress=False)
    assert [r["id"] for r in rows] == ["1A2K.pdb", "1A2K.cif", "other"]
    rows = batch.score_many([a, c], workers=1, progress=False)
    assert [r["id"] for r in rows] == [str(a), str(c)]
