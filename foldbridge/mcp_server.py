"""MCP server exposing foldbridge to LLM agents (stdio transport).

    pip install "foldbridge[mcp]"
    foldbridge mcp            # or: python -m foldbridge.mcp_server

Tools (parameters in each tool's description):

    list_backends()
    score_structure(path, reference=None, pae=None, chains=None)
    rank_results(paths, by="composite", length_normalize=True, top=None)
    run_prediction(spec_path, backend, dry_run=True, out_dir=None, cpu=False, score=True, use_cache=True)
    prep_md(path, out_dir=None, ph=7.0, keep_ligands=False, solvate=False)

Every tool returns plain JSON (NaN -> null). Failures raise ``ToolError``, which the SDK turns
into a tool result with ``isError: true`` carrying the message. Works with the ``mcp`` SDK 1.x
(``FastMCP``) and 2.x (``MCPServer``).
"""
from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path
from typing import Any

INSTALL_HINT = ('foldbridge mcp needs the MCP Python SDK: pip install "foldbridge[mcp]"  '
                "(or: pip install mcp)")

INSTRUCTIONS = (
    "foldbridge runs protein-complex structure prediction (Boltz-2, Chai-1, ColabFold) from one "
    "job spec, scores models uniformly (ipTM, ipSAE, pDockQ, iPAE, clashes, RMSD/DockQ vs a "
    "reference) and prepares structures for MD. run_prediction defaults to dry_run=True: it only "
    "writes backend inputs and the command; pass dry_run=False to actually fold (can take minutes "
    "to hours on a GPU)."
)


def _server_class():
    """(class, flavour) for the installed mcp SDK, or raise ImportError."""
    try:
        from mcp.server.mcpserver import MCPServer  # mcp >= 2

        return MCPServer, "mcpserver"
    except ImportError:
        from mcp.server.fastmcp import FastMCP  # mcp 1.x

        return FastMCP, "fastmcp"


def _tool_error_class() -> type[Exception]:
    """The SDK's ToolError (a raised ToolError becomes an ``isError: true`` tool result with our
    message); a local stand-in when mcp is not installed so the tool functions stay importable."""
    for mod in ("mcp.server.mcpserver.exceptions", "mcp.server.fastmcp.exceptions"):
        try:
            return importlib.import_module(mod).ToolError
        except (ImportError, AttributeError):
            continue

    class ToolError(Exception):
        """Anticipated tool failure (mcp not installed)."""

    return ToolError


ToolError = _tool_error_class()


# ---- tool implementations (plain functions, importable and testable without mcp) -----------
# Failures raise ToolError, so MCP clients get a tool result with isError=true and the message.

def _clean(x: Any) -> Any:
    from .cli import _clean_json

    return _clean_json(x)


def _chain_list(chains: list[str] | str | None) -> list[str] | None:
    if chains is None:
        return None
    if isinstance(chains, str):
        chains = chains.replace(",", " ").split()
    out = [str(c).strip() for c in chains if str(c).strip()]
    return out or None


def tool_list_backends() -> dict[str, Any]:
    """Prediction backends known to foldbridge (boltz, chai, colabfold), whether each is
    installed, and its version. No parameters."""
    from .backends.base import list_backends

    return _clean(list_backends())


def tool_score_structure(path: str, reference: str | None = None, pae: str | None = None,
                         chains: list[str] | None = None) -> dict[str, Any]:
    """Score one structure file with foldbridge's uniform metrics (pLDDT, ipSAE, pDockQ, iPAE,
    buried SASA, clashes; RMSD and DockQ-lite when a reference is given).

    Parameters:
    - path: structure file (.cif / .mmcif / .pdb).
    - reference: optional reference structure (e.g. a crystal structure) for RMSD / DockQ-lite.
    - pae: optional PAE file (.npz / .npy / .json). Default: a PAE file next to the model is
      found automatically (Boltz pae_*.npz, Chai-1 pae.*.npy, ColabFold / AF3 JSON).
    - chains: optional chain ids to restrict the metrics to, e.g. ["A", "B"] (or "A,B").

    Returns one row of metrics. Errors (missing file, unreadable structure) are tool errors.
    """
    from .cli import CLIError, score_paths

    p = Path(path)
    if p.is_dir():
        raise ToolError(f"{path} is a directory; score_structure takes one file "
                        "(use rank_results to score and rank a directory of structures)")
    try:
        rows = score_paths([p], pae=pae, reference=reference, chains=_chain_list(chains))
    except (CLIError, FileNotFoundError, ValueError) as e:
        raise ToolError(str(e)) from e
    return _clean(rows[0])


def tool_rank_results(paths: list[str], by: str = "composite", length_normalize: bool = True,
                      top: int | None = None) -> list[dict[str, Any]]:
    """Rank models best-first, adding 'rank' (1 = best) and 'composite' to each row.

    Parameters:
    - paths: results files (.csv / .json from `foldbridge score` or `run`) and/or structure
      files or directories of structures (these are scored first). A single string is accepted.
    - by: "composite" (default; weighted z-scores of ipTM, ipSAE, pTM, pLDDT, pDockQ, iPAE,
      BSA, clash fraction) or any metric column, e.g. "ipsae", "dockq"; lower-is-better
      metrics (ipae, rmsd, clash_fraction, ...) sort ascending.
    - length_normalize: regress metrics on log(length) before z-scoring (batches of >= 8 rows).
    - top: return only the N best rows.

    Rows without any metric (e.g. dry-run results) get rank null and come last.
    """
    from .cli import CLIError, load_rows, score_paths
    from .rank import rank

    if isinstance(paths, str):
        paths = [paths]
    rows: list[dict[str, Any]] = []
    structs: list[str] = []
    try:
        for p in paths:
            if Path(p).is_file() and Path(p).suffix.lower() in {".csv", ".json"}:
                rows.extend(load_rows(p))
            else:
                structs.append(p)
        if structs:
            rows.extend(score_paths(structs, on_error=lambda f, e: None))
    except (CLIError, FileNotFoundError, ValueError) as e:
        raise ToolError(str(e)) from e
    if not rows:
        raise ToolError("nothing to rank: no rows in the given results files / structures")
    if by != "composite" and not any(by in r for r in rows):
        raise ToolError(f"metric {by!r} not present in the results")
    ranked = rank(rows, by=by, length_normalize=length_normalize)
    return _clean(ranked[:top] if top else ranked)


def tool_run_prediction(spec_path: str, backend: str, dry_run: bool = True, out_dir: str | None = None,
                        cpu: bool = False, score: bool = True, use_cache: bool = True) -> dict[str, Any]:
    """Run a structure prediction from a foldbridge job spec. Dry run by default.

    Parameters:
    - spec_path: job spec (.yaml / .json / .fasta).
    - backend: "boltz", "chai" or "colabfold".
    - dry_run: default True: only write the backend's native inputs and command (no GPU, no
      models). Pass False to actually fold (needs the predictor installed; minutes to hours).
    - out_dir: output directory (default foldbridge_out/<name>-<backend>).
    - cpu: run the predictor on CPU.
    - score: compute uniform metrics for every model (ignored for dry runs).
    - use_cache: reuse / store results in the content-addressed cache.

    Returns the run summary: backend command, cache key, results.json path and (if run) the
    scored, ranked models. Errors (bad spec, backend missing, prediction failed) are tool errors.
    """
    from .cli import CLIError, run_job

    logs: list[str] = []
    try:
        summary = run_job(spec_path, backend, out=out_dir, cpu=cpu, dry_run=dry_run, use_cache=use_cache,
                          score=score, log=logs.append)
    except CLIError as e:
        tail = ("\nlog:\n" + "\n".join(logs[-20:])) if logs else ""
        raise ToolError(f"{e}{tail}") from e
    summary["log"] = logs
    return _clean(summary)


def tool_prep_md(path: str, out_dir: str | None = None, ph: float = 7.0, keep_ligands: bool = False,
                 solvate: bool = False) -> dict[str, Any]:
    """Prepare a structure for molecular dynamics.

    With openmm + pdbfixer installed (pip install "foldbridge[md]"): rebuild missing atoms,
    add hydrogens at the given pH, optionally solvate, and write GROMACS position restraints
    for confident (high-pLDDT) CA atoms. Otherwise a numpy-only cleanup runs and the report
    says so in 'warnings'.

    Parameters:
    - path: structure file (.cif / .pdb).
    - out_dir: output directory (default <stem>_md next to the input).
    - ph: protonation pH (default 7.0).
    - keep_ligands: export ligands to ligands/ for separate parameterisation.
    - solvate: add TIP3P water + 0.15 M NaCl (needs openmm).

    Returns the system report (prepared_pdb, charge, low-pLDDT residues, restraint files, warnings).
    """
    from .mdprep import prepare_for_md

    p = Path(path)
    out = Path(out_dir) if out_dir else p.parent / f"{p.stem}_md"
    try:
        return _clean(prepare_for_md(p, out, ph=ph, keep_ligands=keep_ligands, solvate=solvate))
    except (FileNotFoundError, ValueError) as e:
        raise ToolError(str(e)) from e


TOOLS = {
    "list_backends": tool_list_backends,
    "score_structure": tool_score_structure,
    "rank_results": tool_rank_results,
    "run_prediction": tool_run_prediction,
    "prep_md": tool_prep_md,
}


def build_server():
    """Create the MCP server with all tools registered. Raises ImportError if mcp is missing."""
    from . import __version__

    cls, _ = _server_class()
    try:
        server = cls(name="foldbridge", instructions=INSTRUCTIONS, version=__version__)
    except TypeError:  # older FastMCP without version kwarg
        server = cls("foldbridge", instructions=INSTRUCTIONS)
    for name, fn in TOOLS.items():
        server.tool(name=name, description=inspect.cleandoc(fn.__doc__ or ""))(fn)
    return server


def main() -> int:
    try:
        server = build_server()
    except ImportError:
        print(INSTALL_HINT, file=sys.stderr)
        return 1
    server.run()  # stdio
    return 0


if __name__ == "__main__":
    sys.exit(main())
