"""Result + MSA cache: roundtrip, atomic replace, corrupt entries, serialisation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from foldbridge.backends.base import ModelResult
from foldbridge.cache import Cache, a3m_query, harvest_msas, result_from_dict, result_to_dict, sequence_hash
from foldbridge.spec import spec_from_dict

SEQ = "MKTAYIAKQRQISFVKSHFSRQ"


def _spec(seq=SEQ):
    return spec_from_dict({"name": "t", "chains": [{"id": "A", "sequence": seq}], "msa": "none"})


def _fake_results(workdir: Path, n=2):
    out = []
    for i in range(n):
        m = workdir / "pred" / f"t_model_{i}.cif"
        m.parent.mkdir(parents=True, exist_ok=True)
        m.write_text(f"data_model_{i}\n")
        pae = workdir / "pred" / f"pae_t_model_{i}.npz"
        pae.write_bytes(b"PK-fake" + bytes([i]))
        out.append(ModelResult("boltz", m, seed=42, sample=i, scores={"iptm": 0.8 - i / 10, "ptm": 0.7},
                               pae_path=pae, extra={"record": f"t_model_{i}", "p": Path("/x")}))
    return out


def test_key_depends_on_backend_and_version():
    s = _spec()
    k1 = Cache.key_for(s, "boltz", "2.1")
    assert k1 == s.content_hash("boltz", "2.1")
    assert k1 != Cache.key_for(s, "boltz", "2.2")
    assert k1 != Cache.key_for(s, "chai", "2.1")
    assert k1 != Cache.key_for(_spec(SEQ + "A"), "boltz", "2.1")


def test_default_root_env(monkeypatch, tmp_path):
    monkeypatch.setenv("FOLDBRIDGE_CACHE", str(tmp_path / "c"))
    assert Cache().root == tmp_path / "c"
    monkeypatch.delenv("FOLDBRIDGE_CACHE")
    assert Cache().root == Path("~/.cache/foldbridge").expanduser()


def test_roundtrip(tmp_path):
    cache = Cache(tmp_path / "cache")
    work = tmp_path / "work"
    res = _fake_results(work)
    key = Cache.key_for(_spec(), "boltz", "x")
    assert not cache.has(key) and cache.load(key) is None

    stored = cache.store(key, res, work, meta={"backend": "boltz"})
    assert cache.has(key)
    assert cache.keys() == [key]
    loaded = cache.load(key)
    assert loaded is not None and len(loaded) == 2
    for orig, got in zip(res, loaded):
        assert got.path.is_file() and got.path.read_text() == orig.path.read_text()
        assert cache.path_for(key) in got.path.parents
        assert got.pae_path.read_bytes() == orig.pae_path.read_bytes()
        assert (got.backend, got.seed, got.sample, got.scores) == (orig.backend, orig.seed, orig.sample, orig.scores)
        assert got.extra["record"] == orig.extra["record"]
    assert [r.path for r in stored] == [r.path for r in loaded]

    man = json.loads((cache.path_for(key) / "manifest.json").read_text())
    assert man["key"] == key and man["meta"]["backend"] == "boltz"
    assert not Path(man["results"][0]["path"]).is_absolute()  # relocatable cache

    # the cache survives deletion of the workdir
    for r in res:
        r.path.unlink()
    assert cache.load(key) is not None


def test_store_replaces_atomically_and_leaves_no_temp(tmp_path):
    cache = Cache(tmp_path / "cache")
    work = tmp_path / "w"
    key = "abc123"
    cache.store(key, _fake_results(work, 2), work)
    cache.store(key, _fake_results(work, 1), work)
    assert len(cache.load(key)) == 1
    leftovers = [p.name for p in cache.results_dir.iterdir() if p.name.startswith(".")]
    assert leftovers == []


def test_store_missing_file_leaves_no_entry(tmp_path):
    cache = Cache(tmp_path / "cache")
    bad = [ModelResult("boltz", tmp_path / "nope.cif")]
    with pytest.raises(FileNotFoundError):
        cache.store("k1", bad, tmp_path)
    assert not cache.has("k1")
    assert [p for p in cache.results_dir.iterdir()] == []


def test_relative_paths_resolved_against_workdir(tmp_path):
    work = tmp_path / "w"
    (work / "m.pdb").parent.mkdir(parents=True)
    (work / "m.pdb").write_text("END\n")
    cache = Cache(tmp_path / "c")
    out = cache.store("k2", [ModelResult("chai", Path("m.pdb"))], work)
    assert out[0].path.read_text() == "END\n" and out[0].pae_path is None


def test_corrupt_or_incomplete_entry_is_a_miss(tmp_path):
    cache = Cache(tmp_path / "c")
    work = tmp_path / "w"
    cache.store("k", _fake_results(work, 1), work)
    loaded = cache.load("k")
    loaded[0].path.unlink()
    assert cache.load("k") is None
    (cache.path_for("k") / "manifest.json").write_text("{not json")
    assert cache.load("k") is None and cache.manifest("k") is None


def test_invalid_key(tmp_path):
    with pytest.raises(ValueError):
        Cache(tmp_path).path_for("../escape")


def test_result_dict_roundtrip(tmp_path):
    r = ModelResult("colabfold", tmp_path / "a" / "m.pdb", 1, 2, {"plddt": 88.0}, tmp_path / "a" / "p.json",
                    {"rank": 1})
    d = result_to_dict(r, base=tmp_path)
    assert d["path"] == "a/m.pdb" and d["pae_path"] == "a/p.json"
    json.dumps(d)
    back = result_from_dict(d, base=tmp_path)
    assert back == r
    d_abs = result_to_dict(r)
    assert result_from_dict(d_abs) == r


def test_msa_cache(tmp_path):
    cache = Cache(tmp_path / "c")
    assert cache.msa_path(SEQ) is None
    a3m = tmp_path / "q.a3m"
    a3m.write_text(f">query\n{SEQ}\n>hit1\n{SEQ[:5]}--aa{SEQ[7:]}\n")
    p = cache.store_msa(SEQ, a3m)
    assert p == cache.msa_path(SEQ) == cache.msa_path(SEQ.lower() + "\n")
    assert p.name == sequence_hash(SEQ) + ".a3m"
    assert p.read_text() == a3m.read_text()
    assert cache.msa_path(SEQ + "W") is None
    with pytest.raises(FileNotFoundError):
        cache.store_msa(SEQ, tmp_path / "missing.a3m")


def test_harvest_msas(tmp_path):
    work = tmp_path / "w" / "msas"
    work.mkdir(parents=True)
    (work / "0.a3m").write_text(f"#101\t1\n>101\n{SEQ}\n>hit\n{SEQ}\n")
    (work / "1.a3m").write_text(">other\nAAAAAA\n")
    assert a3m_query(work / "0.a3m") == SEQ
    cache = Cache(tmp_path / "c")
    stored = harvest_msas(cache, tmp_path / "w", [SEQ, "CCCC"])
    assert len(stored) == 1 and cache.msa_path(SEQ) is not None


def test_harvest_skips_mmseqs_scratch(tmp_path):
    """Env-only / uniref-only / paired-only server scratch must never become the cached MSA."""
    seq = "MKTAYIAKQR"
    w = tmp_path / "w"
    msa = w / "boltz_results_x" / "msa"
    tmp_env = msa / "x_unpaired_tmp_env"
    tmp_env.mkdir(parents=True)
    (tmp_env / "uniref.a3m").write_text(f">101\n{seq}\n>u1\nMKTAYIAKQK\n")
    (tmp_env / "bfd.mgnify30.metaeuk30.smag30.a3m").write_text(f">101\n{seq}\n>env1\nMKSAYIAKQR\n")
    pair = msa / "x_paired_tmp_pairgreedy"
    pair.mkdir()
    (pair / "pair.a3m").write_text(f">101\n{seq}\n>p1\nMKTAYIAKQA\n")
    cf = w / "colabfold_out"
    (cf / "job_env").mkdir(parents=True)
    (cf / "job_env" / "uniref.a3m").write_text(f">101\n{seq}\n")
    (cf / "job_pair.a3m").write_text(f">101\n{seq}\n")
    cache = Cache(tmp_path / "c")
    assert harvest_msas(cache, w, [seq]) == []
    assert cache.msa_path(seq) is None


def test_harvest_boltz_final_csv(tmp_path):
    """Boltz's final per-entity msa/<target>_<entity>.csv (paired block first) is harvested as a3m."""
    s0, s1 = "MKTAYIAKQR", "GSHMLEDPVA"
    msa = tmp_path / "w" / "boltz_results_job" / "msa"
    (msa / "job_unpaired_tmp_env").mkdir(parents=True)
    (msa / "job_unpaired_tmp_env" / "uniref.a3m").write_text(f">101\n{s0}\n>bad\nAAAAAAAAAA\n")
    (msa / "job_0.csv").write_text(f"key,sequence\n0,{s0}\n3,MKTAYIAKQK\n-1,MK-AYiaIAKQR")
    (msa / "job_1.csv").write_text(f"key,sequence\n-1,{s1}\n-1,GSHMLEDPVV")
    (msa / "other.csv").write_text("a,b\n1,2\n")  # not a Boltz MSA
    cache = Cache(tmp_path / "c")
    stored = harvest_msas(cache, tmp_path / "w", [s0, s1])
    assert len(stored) == 2
    a3m0 = cache.msa_path(s0).read_text()
    assert a3m_query(cache.msa_path(s0)) == s0
    assert a3m0.splitlines()[1::2] == [s0, "MKTAYIAKQK", "MK-AYiaIAKQR"]
    assert "AAAAAAAAAA" not in a3m0
    assert a3m_query(cache.msa_path(s1)) == s1


def test_shared_cache_respects_umask(tmp_path):
    import os
    import stat

    old = os.umask(0o002)
    try:
        cache = Cache(tmp_path / "c")
        work = tmp_path / "w"
        cache.store("k", _fake_results(work, 1), work)
        entry = cache.path_for("k")
        assert stat.S_IMODE(entry.stat().st_mode) == 0o775  # mkdtemp alone would be 0o700
        a3m = tmp_path / "q.a3m"
        a3m.write_text(f">q\n{SEQ}\n")
        assert stat.S_IMODE(cache.store_msa(SEQ, a3m).stat().st_mode) == 0o664  # mkstemp: 0o600
        os.umask(0o077)
        cache.store("k2", _fake_results(work, 1), work)
        assert stat.S_IMODE(cache.path_for("k2").stat().st_mode) == 0o700
    finally:
        os.umask(old)
