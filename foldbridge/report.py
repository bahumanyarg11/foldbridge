"""Write ranked score tables as CSV, JSON or a single self-contained HTML page."""
from __future__ import annotations

import csv
import html
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = ["write_csv", "write_json", "write_html", "columns"]

# key metrics first (in this order), then everything else alphabetically, long text last
PREFERRED = (
    "rank", "id", "composite", "dockq", "ipsae", "iptm", "pdockq", "ipae", "plddt_interface", "bsa",
    "clash_fraction",
)
TRAILING = ("source_file", "note", "warnings", "error", "pae_path", "path")
# integer-valued columns (written without decimals even if they arrive as floats, e.g. from an old CSV)
COUNT_KEYS = {"rank", "seed", "sample", "clash_count", "pdockq_contacts", "binder_length", "length"}
ZERO_EPS = 1e-9  # |x| below this is written as 0 (no -1.2e-17 round-off noise in outputs)

try:  # direction legend; rank.py is optional for the writers
    from .rank import HIGHER_BETTER as _HB
    from .rank import LOWER_BETTER as _LB
except Exception:  # pragma: no cover
    _HB, _LB = (), ()
HIGHER_BETTER = ("composite",) + tuple(_HB)
LOWER_BETTER = tuple(_LB)


def is_count(key: str) -> bool:
    return key in COUNT_KEYS or key.startswith("n_")


def columns(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Union of keys: key metrics first (rank, id, composite, dockq, ipsae, iptm, pdockq, ipae,
    plddt_interface, bsa, clash_fraction), then the rest alphabetically, paths / notes last."""
    seen: set[str] = set()
    for r in rows:
        seen.update(str(k) for k in r)
    head = [k for k in PREFERRED if k in seen]
    tail = [k for k in TRAILING if k in seen]
    return head + sorted(seen - set(head) - set(tail)) + tail


def _plain(v: Any, key: str | None = None) -> Any:
    """JSON/CSV-safe value: numpy -> python, Path -> str, NaN/inf -> None, |x| < 1e-9 -> 0,
    integral floats in count columns -> int."""
    if v is None:
        return None
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        try:
            v = v.item()
        except Exception:
            pass
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            return None
        if abs(v) < ZERO_EPS:
            v = 0.0
        if key is not None and is_count(key) and v.is_integer():
            return int(v)
        return v
    if isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, Mapping):
        return {str(k): _plain(x, str(k)) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return str(v)


def write_csv(rows: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    rows = list(rows)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cols = columns(rows)
    with p.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            out = []
            for c in cols:
                v = _plain(r.get(c), c)
                out.append("" if v is None else (repr(v) if isinstance(v, float) else v))
            w.writerow(out)
    return p


def write_json(rows: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([_plain(dict(r)) for r in rows], indent=2, allow_nan=False) + "\n")
    return p


def _fmt(v: Any, key: str | None = None) -> tuple[str, str]:
    """(display text, sort key) for one cell: ints without decimals, floats with 3 decimals."""
    v = _plain(v, key)
    if v is None:
        return "", ""
    if isinstance(v, bool):
        return str(v), str(int(v))
    if isinstance(v, int):
        return str(v), repr(float(v))
    if isinstance(v, float):
        txt = f"{v:.3f}"
        return ("0.000" if txt == "-0.000" else txt), repr(v)
    if isinstance(v, (list, dict)):
        v = json.dumps(v)
    return str(v), str(v)


def _empty(v: Any) -> bool:
    v = _plain(v)
    return v is None or v == "" or v == [] or v == {}


def _legend(cols: Sequence[str]) -> str:
    hb = [c for c in cols if c in HIGHER_BETTER]
    lb = [c for c in cols if c in LOWER_BETTER]
    parts = []
    if hb:
        parts.append("<b>higher is better</b>: " + ", ".join(html.escape(c) for c in hb))
    if lb:
        parts.append("<b>lower is better</b>: " + ", ".join(html.escape(c) for c in lb))
    if not parts:
        return ""
    return f'<p class="legend">{" &middot; ".join(parts)}</p>'


_CSS = """
:root{--bg:#ffffff;--fg:#1d2433;--muted:#5d6678;--line:#e3e6ec;--head:#f4f6f9;--hover:#eef3ff;--best:#e7f6ec;--accent:#2f6fde}
@media (prefers-color-scheme: dark){:root{--bg:#14171c;--fg:#e4e7ec;--muted:#9aa3b2;--line:#2a2f38;--head:#1c2027;--hover:#222a38;--best:#16301f;--accent:#7aa7ff}}
*{box-sizing:border-box}
body{margin:0;padding:24px 16px;background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
h1{font-size:20px;margin:0 0 4px}
p.meta{color:var(--muted);margin:0 0 8px}
p.legend{color:var(--muted);margin:0 0 16px;font-size:13px}
.wrap{overflow-x:auto;border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);white-space:nowrap;text-align:right}
th:first-child,td:first-child{text-align:left}
td.s{text-align:left}
th{position:sticky;top:0;background:var(--head);cursor:pointer;user-select:none;font-weight:600}
th:hover{color:var(--accent)}
th[aria-sort=ascending]::after{content:" \\25B2";font-size:10px}
th[aria-sort=descending]::after{content:" \\25BC";font-size:10px}
tbody tr:hover{background:var(--hover)}
tbody tr.best{background:var(--best)}
td.path{color:var(--muted);font-size:12px;text-align:left}
"""

_JS = """
(function(){var t=document.getElementById('t'),b=t.tBodies[0],hs=t.tHead.rows[0].cells;
for(var c=0;c<hs.length;c++){(function(c){hs[c].addEventListener('click',function(){
var asc=hs[c].getAttribute('aria-sort')!=='ascending';for(var k=0;k<hs.length;k++)hs[k].removeAttribute('aria-sort');
hs[c].setAttribute('aria-sort',asc?'ascending':'descending');var rs=Array.prototype.slice.call(b.rows);
rs.sort(function(x,y){var a=x.cells[c].getAttribute('data-v'),d=y.cells[c].getAttribute('data-v');
if(a===''&&d!=='')return 1;if(d===''&&a!=='')return -1;var na=parseFloat(a),nd=parseFloat(d),r;
if(!isNaN(na)&&!isNaN(nd))r=na-nd;else r=a<d?-1:a>d?1:0;return asc?r:-r;});
rs.forEach(function(r){b.appendChild(r);});});})(c);}})();
"""


def write_html(rows: Iterable[Mapping[str, Any]], path: str | Path, title: str = "foldbridge report") -> Path:
    """Single self-contained HTML file: sortable table (click headers), light/dark via CSS."""
    rows = list(rows)
    # all-empty columns (e.g. dockq without a reference) are dropped
    cols = [c for c in columns(rows) if not all(_empty(r.get(c)) for r in rows)]
    esc = html.escape
    head = "".join(f"<th>{esc(c)}</th>" for c in cols)
    body = []
    for r in rows:
        cls = ' class="best"' if _plain(r.get("rank"), "rank") == 1 else ""
        cells = []
        for c in cols:
            txt, key = _fmt(r.get(c), c)
            pv = _plain(r.get(c), c)
            num = isinstance(pv, (int, float)) and not isinstance(pv, bool)
            klass = "path" if c in ("path", "pae_path") else ("" if num else "s")
            ca = f' class="{klass}"' if klass else ""
            cells.append(f'<td{ca} data-v="{esc(key, quote=True)}">{esc(txt)}</td>')
        body.append(f"<tr{cls}>{''.join(cells)}</tr>")
    doc = (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<meta name=\"color-scheme\" content=\"light dark\">"
        f"<title>{esc(title)}</title><style>{_CSS}</style></head><body>"
        f"<h1>{esc(title)}</h1><p class=\"meta\">{len(rows)} models &middot; click a column header to sort</p>"
        f"{_legend(cols)}"
        f"<div class=\"wrap\"><table id=\"t\"><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
        f"<script>{_JS}</script></body></html>\n"
    )
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(doc, encoding="utf-8")
    return p
