"""Tests for foldbridge.rank and foldbridge.report."""
from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest

from foldbridge.rank import pareto_front, rank
from foldbridge.report import write_csv, write_html, write_json


def rows_basic():
    return [
        {"id": "m0", "iptm": 0.50, "ptm": 0.60, "plddt": 70.0, "ipae": 12.0, "clash_fraction": 0.02, "n_residues": 100},
        {"id": "m1", "iptm": 0.90, "ptm": 0.85, "plddt": 90.0, "ipae": 3.0, "clash_fraction": 0.00, "n_residues": 100},
        {"id": "m2", "iptm": 0.70, "ptm": 0.70, "plddt": 80.0, "ipae": 6.0, "clash_fraction": 0.01, "n_residues": 100},
    ]


def test_rank_composite_order_and_fields():
    rows = rows_basic()
    out = rank(rows)
    assert [r["id"] for r in out] == ["m1", "m2", "m0"]
    assert [r["rank"] for r in out] == [1, 2, 3]
    assert out[0]["composite"] > out[1]["composite"] > out[2]["composite"]
    assert "rank" not in rows[0]  # input untouched


def test_rank_by_lower_better_metric():
    rows = rows_basic()
    rows[0]["ipae"] = 1.0
    assert rank(rows, by="ipae")[0]["id"] == "m0"
    assert rank(rows, by="iptm")[0]["id"] == "m1"


def test_lower_better_direction_in_composite():
    rows = [{"id": "a", "ipae": 2.0}, {"id": "b", "ipae": 20.0}]
    assert rank(rows, weights={"ipae": 1.0})[0]["id"] == "a"
    rows = [{"id": "a", "clash_fraction": 0.3}, {"id": "b", "clash_fraction": 0.0}]
    assert rank(rows)[0]["id"] == "b"


def test_missing_metrics_are_neutral_and_missing_sort_key_last():
    rows = [{"id": "a", "iptm": 0.9, "ipae": 3.0}, {"id": "b", "ipae": 5.0}, {"id": "c", "iptm": 0.1, "ipae": 9.0}]
    out = rank(rows)
    assert [r["id"] for r in out] == ["a", "b", "c"]  # b: missing iptm is neutral, ipae counts
    assert rank(rows, by="iptm")[-1]["id"] == "b"
    assert rank(rows, by="iptm")[-1]["rank"] == 3  # has metrics, just not the sort key


def test_rows_without_metrics_unranked_and_last():
    """Dry-run / failed rows carry no metrics: rank None, composite None, after all ranked rows."""
    rows = [{"id": "dry1", "path": "x.cif", "seed": 1}, {"id": "a", "iptm": 0.2},
            {"id": "dry2", "iptm": None, "ipsae": float("nan")}, {"id": "b", "iptm": 0.8}]
    out = rank(rows)
    assert [r["id"] for r in out] == ["b", "a", "dry1", "dry2"]
    assert [r["rank"] for r in out] == [1, 2, None, None]
    assert out[2]["composite"] is None and out[3]["composite"] is None
    # unranked rows do not shift the z-scores of the others
    alone = rank([rows[1], rows[3]])
    assert [r["composite"] for r in out[:2]] == pytest.approx([r["composite"] for r in alone])
    assert [r["rank"] for r in rank([{"id": "x"}, {"id": "y"}])] == [None, None]
    assert rank(rows, by="iptm")[0]["id"] == "b"


def test_single_row_and_empty():
    assert rank([]) == []
    out = rank([{"id": "x", "iptm": 0.5}])
    assert out[0]["rank"] == 1 and out[0]["composite"] == 0.0


def test_length_normalization_removes_size_trend():
    rng = np.random.default_rng(0)
    L = np.array([50, 80, 120, 200, 300, 450, 600, 800, 1000, 1500])
    # bsa is a pure function of size; iptm carries the real signal
    iptm = rng.uniform(0.3, 0.9, size=len(L))
    rows = [{"id": f"m{k}", "bsa": 300.0 * math.log(l), "iptm": float(iptm[k]), "n_residues": int(l)}
            for k, l in enumerate(L)]
    w = {"iptm": 1.0, "bsa": 1.0}
    normed = rank(rows, weights=w, length_normalize=True)
    raw = rank(rows, weights=w, length_normalize=False)
    iptm_only = rank(rows, weights={"iptm": 1.0}, length_normalize=True)
    # once size is regressed out, bsa carries no information and drops out of the composite
    assert [r["id"] for r in normed] == [r["id"] for r in iptm_only]
    for a, b in zip(normed, iptm_only):
        assert a["composite"] == pytest.approx(b["composite"])
    assert [r["id"] for r in raw] != [r["id"] for r in normed]


def test_length_normalization_needs_batch_of_8():
    rows = [{"id": f"m{k}", "bsa": float(k), "n_residues": 10 * (k + 1)} for k in range(5)]
    assert rank(rows, weights={"bsa": 1.0})[0]["id"] == "m4"


def test_deterministic_and_stable_ties():
    rows = [{"id": "a", "iptm": 0.5}, {"id": "b", "iptm": 0.5}]
    assert [r["id"] for r in rank(rows)] == ["a", "b"]
    assert rank(rows_basic()) == rank(rows_basic())


def test_pareto_front():
    rows = [
        {"id": "a", "iptm": 0.9, "ipae": 8.0},
        {"id": "b", "iptm": 0.7, "ipae": 3.0},
        {"id": "c", "iptm": 0.6, "ipae": 9.0},   # dominated by a and b
        {"id": "d", "iptm": 0.9, "ipae": 8.0},   # duplicate of a: not dominated
        {"id": "e", "iptm": 0.95},               # missing objective -> excluded
    ]
    front = pareto_front(rows, ["iptm", "ipae"])
    assert [r["id"] for r in front] == ["a", "b", "d"]
    front2 = pareto_front(rows, {"iptm": "max", "ipae": "max"})
    assert {r["id"] for r in front2} == {"a", "d", "c"}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def test_reports(tmp_path):
    from pathlib import Path
    out = rank(rows_basic())
    out[0]["path"] = Path("/x/model.cif")
    out[1]["dockq"] = float("nan")
    out[2]["note"] = "<b>&"
    c = write_csv(out, tmp_path / "r.csv")
    j = write_json(out, tmp_path / "r.json")
    h = write_html(out, tmp_path / "r.html", title="T <1>")

    with open(c) as fh:
        table = list(csv.DictReader(fh))
    assert len(table) == 3
    head = list(table[0].keys())
    assert head[:3] == ["rank", "id", "composite"] and head[-1] == "path"
    assert table[0]["id"] == "m1" and float(table[0]["iptm"]) == 0.9

    data = json.loads(j.read_text())
    assert data[1]["dockq"] is None and data[0]["path"] == "/x/model.cif"

    doc = h.read_text()
    assert doc.startswith("<!doctype html>")
    assert "prefers-color-scheme: dark" in doc and "<script>" in doc
    assert "T &lt;1&gt;" in doc and "&lt;b&gt;&amp;" in doc
    assert "http" not in doc  # no external assets
    assert doc.count("<tr") == 4


def test_html_formatting_columns_legend_and_escaping(tmp_path):
    rows = rank(rows_basic())
    for r in rows:
        r["dockq"] = None              # all-empty column -> dropped
        r["n_contacts"] = 17.0         # count arriving as float -> "17"
        r["zeta"] = "z"
        r["alpha"] = 1.23456
        r["path"] = f"/m/{r['id']}.cif"
    rows[0]["id"] = "<script>alert(1)</script>"
    rows[1]["note"] = '"><img src=x onerror=alert(2)>'
    h = write_html(rows, tmp_path / "r.html", title="<script>t</script>").read_text()
    head = h[h.index("<thead>"):h.index("</thead>")]
    cols = [c.split("</th>")[0] for c in head.split("<th>")[1:]]
    assert cols[:4] == ["rank", "id", "composite", "ipsae"] or cols[:3] == ["rank", "id", "composite"]
    key = [c for c in cols if c in ("rank", "id", "composite", "dockq", "ipsae", "iptm", "pdockq", "ipae",
                                    "plddt_interface", "bsa", "clash_fraction")]
    assert cols[:len(key)] == key == ["rank", "id", "composite", "iptm", "ipae", "clash_fraction"]
    rest = cols[len(key):]
    assert rest[-1] == "path" and rest[:-2] == sorted(rest[:-2]) and "note" in rest
    assert "dockq" not in cols
    assert ">17<" in h and ">17.000<" not in h and ">1.235<" in h and ">0.900<" in h
    assert ">1<" in h  # rank as int
    # escaping: no live markup from data or title
    assert "<script>alert(1)" not in h and "&lt;script&gt;alert(1)&lt;/script&gt;" in h
    assert "<script>t</script>" not in h and "<img" not in h
    assert h.count("<script>") == 1  # only the sorter
    assert 'class="legend"' in h and "higher is better" in h and "lower is better" in h
    leg = h[h.index('class="legend"'):]
    leg = leg[:leg.index("</p>")]
    assert "iptm" in leg and "ipae" in leg and "clash_fraction" in leg and "dockq" not in leg
