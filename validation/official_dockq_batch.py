"""Score a batch of (model, native, receptor, ligand) jobs with the official DockQ v2 API.

Runs inside a separate venv because DockQ 2.1.x pins numpy<2 while foldbridge's venv uses
numpy 2.x:

    python3 -m venv validation/.dockq-venv
    validation/.dockq-venv/bin/pip install DockQ==2.1.3
    validation/.dockq-venv/bin/python validation/official_dockq_batch.py jobs.json out.json

jobs.json: [{"id":..., "model":..., "native":..., "receptor": "A", "ligand": "D"}, ...]
"""
from __future__ import annotations

import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

from DockQ.DockQ import load_PDB, run_on_all_native_interfaces


def score(model_path: str, native_path: str, rec: str, lig: str) -> dict:
    model = load_PDB(model_path)
    native = load_PDB(native_path)
    res, _ = run_on_all_native_interfaces(model, native, chain_map={rec: rec, lig: lig})
    if not res:
        return {"fnat": None, "irmsd": None, "lrmsd": None, "dockq": None}
    (info,) = res.values()
    return {"fnat": float(info["fnat"]), "irmsd": float(info["iRMSD"]), "lrmsd": float(info["LRMSD"]),
            "dockq": float(info["DockQ"]), "nat_total": int(info["nat_total"]),
            "nat_correct": int(info["nat_correct"]), "len1": int(info["len1"]), "len2": int(info["len2"])}


def cli_dockq(model_path: str, native_path: str, rec: str, lig: str) -> float | None:
    exe = Path(sys.executable).with_name("DockQ")
    out = subprocess.run([str(exe), model_path, native_path, "--mapping", f"{rec}{lig}:{rec}{lig}", "--short"],
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("DockQ "):
            return float(line.split()[1])
    return None


def main() -> None:
    jobs = json.loads(Path(sys.argv[1]).read_text())
    out = {"dockq_version": version("DockQ"), "results": {}}
    for j in jobs:
        r = score(j["model"], j["native"], j["receptor"], j["ligand"])
        if j.get("cli"):
            r["dockq_cli"] = cli_dockq(j["model"], j["native"], j["receptor"], j["ligand"])
        out["results"][j["id"]] = r
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
