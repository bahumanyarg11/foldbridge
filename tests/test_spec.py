"""JobSpec loading, validation and hashing."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from foldbridge.spec import Chain, JobSpec, Ligand, load_spec, spec_from_dict

BARNASE = "AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR"
BARSTAR = "KKAVINGEQIRSISDLHQTLKKELALPEYYGENLDALWDALTGWVEYPLVLEWRQFEQSKQLTENGAESVLQVFREAKAEGADITIILS"

BASE = {
    "name": "toy",
    "chains": [
        {"id": "A", "type": "protein", "sequence": "MKTAYIAKQR"},
        {"id": "B", "type": "dna", "sequence": "ACGTACGT", "copies": 2},
    ],
    "ligands": [{"id": "L", "smiles": "CC(=O)Oc1ccccc1C(=O)O"}],
    "msa": "none",
    "seeds": [1, 2],
    "samples": 3,
}


def _d(**over):
    d = copy.deepcopy(BASE)
    d.update(over)
    return d


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def test_load_yaml_json_equivalent(tmp_path):
    y = tmp_path / "job.yaml"
    j = tmp_path / "job.json"
    y.write_text(yaml.safe_dump(BASE))
    j.write_text(json.dumps(BASE))
    sy, sj = load_spec(y), load_spec(j)
    assert sy == sj
    assert sy.content_hash() == sj.content_hash()
    assert sy.msa == "none" and sy.seeds == [1, 2] and sy.samples == 3 and sy.recycles == 3
    assert [c.type for c in sy.chains] == ["protein", "dna"]
    assert sy.ligands[0].smiles.startswith("CC(")


def test_yml_suffix_and_save_roundtrip(tmp_path):
    p = tmp_path / "job.yml"
    p.write_text(yaml.safe_dump(BASE))
    s = load_spec(p)
    out = tmp_path / "saved.yaml"
    s.save(out)
    assert load_spec(out) == s


def test_example_barnase_barstar(examples_dir):
    s = load_spec(examples_dir / "barnase_barstar.yaml")
    assert s.name == "barnase_barstar"
    assert [(c.id, len(c.sequence)) for c in s.chains] == [("A", 110), ("B", 89)]
    assert s.chains[0].sequence == BARNASE and s.chains[1].sequence == BARSTAR
    assert s.total_tokens() == 199
    assert s.protein_sequences() == [("A", BARNASE), ("B", BARSTAR)]


def test_example_hemoglobin(examples_dir):
    s = load_spec(examples_dir / "hemoglobin_heme.yaml")
    assert [(c.id, c.copies, len(c.sequence)) for c in s.chains] == [("A", 2, 141), ("B", 2, 146)]
    assert s.ligands[0].ccd == "HEM" and s.ligands[0].copies == 4
    assert s.total_tokens() == 2 * 141 + 2 * 146
    assert [cid for c in s.chains for cid in c.chain_ids()] == ["A", "A2", "B", "B2"]


def test_example_fasta(examples_dir):
    s = load_spec(examples_dir / "single.fasta")
    assert s.name == "single"
    assert len(s.chains) == 1 and s.chains[0].id == "A"
    assert s.chains[0].sequence == "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN"


def test_fasta_header_metadata(tmp_path):
    p = tmp_path / "complex.fa"
    p.write_text(
        ">A|heavy chain\nmktay\niakqr\n\n"
        ">D type=dna copies=2\nACGT\nACGT\n"
        ">R type=rna\nACGU\n"
    )
    s = load_spec(p)
    assert s.name == "complex"
    a, d, r = s.chains
    assert (a.id, a.type, a.copies, a.sequence) == ("A", "protein", 1, "MKTAYIAKQR")
    assert (d.id, d.type, d.copies, d.sequence) == ("D", "dna", 2, "ACGTACGT")
    assert isinstance(d.copies, int)
    assert d.chain_ids() == ["D", "D2"]
    assert (r.type, r.sequence) == ("rna", "ACGU")


def test_fasta_uniprot_header_tolerated(tmp_path):
    p = tmp_path / "barnase.fasta"
    p.write_text(">A Barnase OS=Bacillus amyloliquefaciens OX=1390 GN=bar\n" + BARNASE + "\n")
    s = load_spec(p)
    assert s.chains[0].sequence == BARNASE
    # known keys still honoured next to unknown ones; copies coerced to int
    p.write_text(">D PE=1 SV=2 type=dna copies=3 OS=Homo\nACGT\n")
    d = load_spec(p).chains[0]
    assert (d.type, d.copies) == ("dna", 3) and isinstance(d.copies, int)
    p.write_text(">D copies=two\nACGT\n")
    with pytest.raises(ValueError, match="copies"):
        load_spec(p)


def test_reference_resolved_relative_to_spec(tmp_path):
    sub = tmp_path / "jobs"
    sub.mkdir()
    p = sub / "job.yaml"
    p.write_text(yaml.safe_dump(_d(reference="../ref/1brs.pdb")))
    s = load_spec(p)
    assert Path(s.reference).is_absolute()
    assert Path(s.reference) == (tmp_path / "ref" / "1brs.pdb").resolve()


def test_unknown_top_level_keys_go_to_extra():
    s = spec_from_dict(_d(boltz={"use_potentials": True}, extra={"x": 1}))
    assert s.extra == {"x": 1, "boltz": {"use_potentials": True}}


def test_sequence_normalised():
    s = spec_from_dict(_d(chains=[{"id": "A", "sequence": "mkt ay\niak"}]))
    assert s.chains[0].sequence == "MKTAYIAK"


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chain,msg", [
    ({"id": "A", "sequence": "MKTJB"}, "invalid characters"),            # J, B not amino acids
    ({"id": "A", "sequence": "MKT*"}, "invalid characters"),
    ({"id": "A", "type": "dna", "sequence": "ACGTX"}, "invalid characters"),
    ({"id": "A", "type": "rna", "sequence": "ACGUQ"}, "invalid characters"),
    ({"id": "A", "type": "glycan", "sequence": "AAA"}, "type must be"),
    ({"id": "A", "sequence": "MKT", "copies": 0}, "copies"),
])
def test_bad_chains_rejected(chain, msg):
    with pytest.raises(ValueError, match=msg):
        spec_from_dict(_d(chains=[chain], ligands=[]))


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        spec_from_dict(_d(chains=[{"id": "A", "sequence": "MKT"}, {"id": "A", "sequence": "GGG"}]))
    with pytest.raises(ValueError, match="duplicate"):  # chain/ligand clash
        spec_from_dict(_d(chains=[{"id": "A", "sequence": "MKT"}], ligands=[{"id": "A", "ccd": "ATP"}]))


def test_copy_ids_collision_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        spec_from_dict(_d(chains=[{"id": "A", "sequence": "MKT", "copies": 2},
                                  {"id": "A2", "sequence": "GGG"}], ligands=[]))


def test_ligand_needs_smiles_or_ccd():
    with pytest.raises(ValueError, match="smiles or ccd"):
        spec_from_dict(_d(ligands=[{"id": "L"}]))
    assert spec_from_dict(_d(ligands=[{"id": "L", "ccd": "HEM"}])).ligands[0].ccd == "HEM"


def test_ligand_copies_validated():
    with pytest.raises(ValueError, match="copies"):
        spec_from_dict(_d(ligands=[{"id": "L", "ccd": "HEM", "copies": 0}]))
    with pytest.raises(ValueError, match="copies"):
        spec_from_dict(_d(ligands=[{"id": "L", "ccd": "HEM", "copies": -2}]))


def test_copy_ids_collide_with_ligand():
    with pytest.raises(ValueError, match="duplicate"):
        spec_from_dict(_d(chains=[{"id": "A", "sequence": "MKT", "copies": 2}],
                          ligands=[{"id": "A2", "ccd": "ATP"}]))


@pytest.mark.parametrize("value,expected", [("2", 2), (" 3 ", 3), (2.0, 2), (4, 4)])
def test_copies_string_numerics_coerced(value, expected):
    s = spec_from_dict(_d(chains=[{"id": "A", "sequence": "MKT", "copies": value}],
                          ligands=[{"id": "L", "ccd": "HEM", "copies": value}]))
    assert s.chains[0].copies == expected and isinstance(s.chains[0].copies, int)
    assert s.ligands[0].copies == expected and isinstance(s.ligands[0].copies, int)
    assert len(s.chains[0].chain_ids()) == expected


@pytest.mark.parametrize("value", ["two", "2.5", 2.5, True, None, [2]])
def test_copies_garbage_rejected(value):
    with pytest.raises(ValueError, match="copies must be an integer"):
        spec_from_dict(_d(chains=[{"id": "A", "sequence": "MKT", "copies": value}], ligands=[]))
    with pytest.raises(ValueError, match="copies must be an integer"):
        spec_from_dict(_d(ligands=[{"id": "L", "ccd": "HEM", "copies": value}]))


def test_yaml_string_copies(tmp_path):
    p = tmp_path / "job.yaml"
    p.write_text('name: x\nchains:\n  - id: A\n    sequence: MKT\n    copies: "2"\n')
    assert load_spec(p).chains[0].chain_ids() == ["A", "A2"]


@pytest.mark.parametrize("text", ["", "   \n# only a comment\n", "null\n", "{}\n"])
def test_empty_spec_file_rejected(tmp_path, text):
    p = tmp_path / "empty.yaml"
    p.write_text(text)
    with pytest.raises(ValueError, match="empty spec"):
        load_spec(p)


def test_non_mapping_spec_rejected(tmp_path):
    p = tmp_path / "list.yaml"
    p.write_text("- a\n- b\n")
    with pytest.raises(ValueError, match="mapping"):
        load_spec(p)


def test_missing_name_or_chains():
    with pytest.raises(ValueError, match="name"):
        spec_from_dict(_d(name=""))
    with pytest.raises(ValueError, match="chain"):
        spec_from_dict(_d(chains=[]))
    with pytest.raises(KeyError):
        spec_from_dict({"chains": BASE["chains"]})


def test_unknown_chain_field_rejected():
    with pytest.raises(TypeError):
        spec_from_dict(_d(chains=[{"id": "A", "sequence": "MKT", "colour": "red"}]))


# ---------------------------------------------------------------------------
# identity / hashing
# ---------------------------------------------------------------------------


def test_content_hash_stable():
    a, b = spec_from_dict(_d()), spec_from_dict(_d())
    assert a.content_hash() == b.content_hash()
    assert len(a.content_hash()) == 16 and int(a.content_hash(), 16) >= 0
    # key order in the input does not matter
    rev = {k: BASE[k] for k in reversed(list(BASE))}
    assert spec_from_dict(copy.deepcopy(rev)).content_hash() == a.content_hash()
    # normalisation happens before hashing: lowercase / whitespace input hashes the same
    lower = _d(chains=[{"id": "A", "sequence": "mktay iakqr"}, BASE["chains"][1]])
    assert spec_from_dict(lower).content_hash() == a.content_hash()


@pytest.mark.parametrize("field,value", [
    ("seeds", [1, 3]),
    ("samples", 4),
    ("recycles", 10),
    ("msa", "server"),
    ("name", "toy2"),
    ("ligands", [{"id": "L", "smiles": "CCO"}]),
    ("chains", [{"id": "A", "sequence": "MKTAYIAKQK"}, BASE["chains"][1]]),
    ("extra", {"boltz": {"use_potentials": True}}),
])
def test_content_hash_sensitive(field, value):
    base = spec_from_dict(_d()).content_hash()
    assert spec_from_dict(_d(**{field: value})).content_hash() != base


def test_content_hash_copies_and_backend():
    s = spec_from_dict(_d())
    more = _d()
    more["chains"][1]["copies"] = 3
    assert spec_from_dict(more).content_hash() != s.content_hash()
    assert s.content_hash("boltz") != s.content_hash("chai")
    assert s.content_hash("boltz", "2.1.0") != s.content_hash("boltz", "2.2.0")
    assert s.content_hash() == s.content_hash(None, None)


def test_content_hash_ignores_reference(tmp_path):
    s1 = spec_from_dict(_d())
    s2 = spec_from_dict(_d(reference="/some/where/ref.pdb"))
    s3 = spec_from_dict(_d(reference=str(tmp_path / "other.cif")))
    assert s1.content_hash() == s2.content_hash() == s3.content_hash()
    assert "reference" not in s1.canonical()


# ---------------------------------------------------------------------------
# chain copies
# ---------------------------------------------------------------------------


def test_chain_ids_copies():
    assert Chain(id="A", sequence="MK").chain_ids() == ["A"]
    assert Chain(id="A", sequence="MK", copies=3).chain_ids() == ["A", "A2", "A3"]
    assert Chain(id="H", sequence="AC", type="dna", copies=2).chain_ids() == ["H", "H2"]


def test_total_tokens_counts_copies():
    s = spec_from_dict(_d())
    assert s.total_tokens() == 10 + 2 * 8
    assert s.protein_sequences() == [("A", "MKTAYIAKQR")]


def test_direct_construction_and_validate():
    s = JobSpec(name="x", chains=[Chain(id="A", sequence="acd")], ligands=[Ligand(id="L", ccd="ATP")])
    s.validate()
    assert s.chains[0].sequence == "ACD"
    assert s.to_dict()["ligands"][0]["ccd"] == "ATP"


# ---------------------------------------------------------------------------
# typo warnings, MSA paths, int coercion
# ---------------------------------------------------------------------------


def test_typo_key_warns_with_suggestion():
    with pytest.warns(UserWarning, match=r"unknown top-level key 'sampels' \(passed to backends as extra\); "
                                         r"did you mean 'samples'\?"):
        s = spec_from_dict(_d(sampels=5))
    assert s.extra == {"sampels": 5} and s.samples == 3


def test_passthrough_keys_do_not_warn(recwarn):
    spec_from_dict(_d(boltz={"x": 1}, chai={"y": 2}, chai_args=["--low-memory"], constraints=[], my_note="hi"))
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


def test_msa_paths_resolved_relative_to_spec(tmp_path):
    (tmp_path / "msas").mkdir()
    (tmp_path / "msas" / "A.a3m").write_text(">A\nMKTAYIAKQR\n")
    sub = tmp_path / "specs"
    sub.mkdir()
    d = _d(msa="../msas")
    d["chains"][0]["msa"] = "../msas/A.a3m"
    (sub / "job.yaml").write_text(yaml.safe_dump(d))
    s = load_spec(sub / "job.yaml")
    assert s.msa == str((tmp_path / "msas").resolve())
    assert s.chains[0].msa == str((tmp_path / "msas" / "A.a3m").resolve())


def test_missing_msa_dir_clear_error(tmp_path):
    (tmp_path / "job.yaml").write_text(yaml.safe_dump(_d(msa="no_such_dir")))
    with pytest.raises(ValueError) as e:
        load_spec(tmp_path / "job.yaml")
    msg = str(e.value)
    assert "None" not in msg and str((tmp_path / "no_such_dir").resolve()) in msg and "server" in msg
    d = _d()
    d["chains"][0]["msa"] = "missing.a3m"
    (tmp_path / "job2.yaml").write_text(yaml.safe_dump(d))
    with pytest.raises(ValueError, match="chain A: msa file 'missing.a3m' not found"):
        load_spec(tmp_path / "job2.yaml")


def test_content_hash_uses_msa_contents(tmp_path):
    a, b = tmp_path / "p1" / "A.a3m", tmp_path / "p2" / "A.a3m"
    for f in (a, b):
        f.parent.mkdir()
        f.write_text(">A\nMKTAYIAKQR\n")

    def spec(path):
        d = _d()
        d["chains"][0]["msa"] = str(path)
        return spec_from_dict(d)

    assert spec(a).content_hash() == spec(b).content_hash()  # same a3m, different project
    b.write_text(">A\nMKTAYIAKQR\n>hit\nMKTAYIAKQK\n")
    assert spec(a).content_hash() != spec(b).content_hash()  # regenerated a3m -> new key
    # directory msa: hashed by the a3m files inside it
    h1 = spec_from_dict(_d(msa=str(a.parent))).content_hash()
    a.write_text(">A\nMKTAYIAKQR\n>hit\nMKTAYIAKQA\n")
    assert spec_from_dict(_d(msa=str(a.parent))).content_hash() != h1
    assert spec_from_dict(_d(msa="server")).content_hash() == spec_from_dict(_d(msa="server")).content_hash()


@pytest.mark.parametrize("seeds, want", [("42", [42]), (42, [42]), ("1,2", [1, 2]), ([1, "2"], [1, 2]),
                                          (7.0, [7])])
def test_seeds_coerced(seeds, want):
    assert spec_from_dict(_d(seeds=seeds)).seeds == want


@pytest.mark.parametrize("over, msg", [({"seeds": "4x"}, "seeds"), ({"seeds": []}, "seeds"),
                                       ({"samples": "five"}, "samples"), ({"samples": 0}, "samples"),
                                       ({"recycles": -1}, "recycles"), ({"seeds": [True]}, "seeds")])
def test_bad_int_fields(over, msg):
    with pytest.raises(ValueError, match=msg):
        spec_from_dict(_d(**over))


def test_samples_recycles_strings():
    s = spec_from_dict(_d(samples="5", recycles="10"))
    assert s.samples == 5 and s.recycles == 10
