# Changelog

All notable changes to foldbridge are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-09

First release.

### Added

- **One job spec** (YAML / JSON / FASTA) for protein, DNA and RNA chains, homo-oligomer `copies`,
  SMILES / CCD ligands, MSA mode (`server`, `none`, `cached`, or an a3m file or directory), seeds,
  samples, recycles and an optional reference structure. Relative paths resolve against the spec
  file's directory. Typo-like unknown keys (`sampels`) get a "did you mean" warning. Scalar seeds
  (`seeds: 42`) and numeric strings are coerced to ints.
- **Backends**: Boltz-2, Chai-1 and ColabFold. Each one writes its native input, builds the
  command line, runs it and collects models, confidence scores and PAE files. `--dry-run` writes
  the inputs and `command.txt` without running anything.
- **Uniform scoring** of any `.cif` / `.pdb`, with or without a PAE matrix: pLDDT (per chain and
  interface), backend ipTM/pTM, **ipSAE** (d0res variant, per chain pair, max over both
  directions), **pDockQ**, iPAE, **DockQ-lite** (Fnat, iRMSD, LRMSD), Kabsch RMSD, **buried SASA**
  (Shrake–Rupley) and **clashes**. Crystal structures are detected, and their B-factors are not
  treated as pLDDT. Insertion codes are respected throughout.
- **Ranking** across backends, seeds and samples by a weighted z-score composite, optionally
  length-normalised. Also a Pareto front and CSV / JSON / self-contained HTML reports. Rows with
  no metrics (dry runs, failed models) are not ranked and sort last.
- **Batch scoring** (`foldbridge score -j N --checkpoint FILE`, `foldbridge.batch.score_many`):
  parallel worker processes, automatic PAE pairing for Boltz / Chai-1 / ColabFold / AF3 outputs,
  per-file error isolation, a progress line, and a JSONL checkpoint to resume from.
  **4–5× faster** than serial on desktop CPUs (Apple M4: 4.0×; Intel i9-12900: 5.0×, 38 structures/s).
- **Binder-design campaigns** (`foldbridge campaign`): fold every FASTA design against one
  target, resumably (`campaign_state.json`, `--max-jobs`). Then score, rank by best model per
  design, and write `campaign.csv` / `.json`, `campaign_models.csv`, `report.html` and
  `top_<N>.fasta`. Binder-specific ipSAE uses the backend's chain renaming when there is one.
  Duplicate FASTA ids are warned about and renamed.
- **Content-addressed cache** of results and MSAs, keyed by spec + backend + backend version.
  MSA files enter the key by content hash, and re-runs are served from the cache.
- **MD preparation** (`foldbridge prep-md`): with OpenMM + PDBFixer, rebuild missing atoms, add
  hydrogens at a chosen pH, optionally solvate (amber14 + TIP3P, 0.15 M NaCl), and write GROMACS
  position restraints for high-pLDDT CA atoms. Without them, a numpy fallback cleans and renumbers
  the structure and lists the atoms to restrain by residue identity, because atom numbers change
  once `pdb2gmx` adds hydrogens.
- **MCP server** (`foldbridge mcp`) with the tools `list_backends`, `score_structure` (with
  `chains`), `rank_results`, `run_prediction` (dry run by default) and `prep_md`. Failures are
  returned as MCP tool errors (`isError: true`).
- CLI with clean one-line errors (`--debug` for tracebacks), MIT licence, GitHub Actions CI (Linux +
  macOS, Python 3.10 / 3.12 / 3.13) and ruff-clean code.

### Validation

- **DockQ-lite vs the official DockQ v2** (2.1.3) on 180 decoys from 3 crystal complexes:
  max |ΔDockQ| = **1.7×10⁻⁶**, Fnat identical, every decoy in the same CAPRI class
  ([docs/VALIDATION.md](docs/VALIDATION.md)).
- **ipSAE vs the reference `ipsae.py`** (Dunbrack lab) on 5 real Boltz-2 models: **identical** to 6
  decimals; pDockQ within 4×10⁻⁵ ([docs/REAL_RUN.md](docs/REAL_RUN.md)).
- **Real Boltz-2 (2.2.1) run** of barnase–barstar on an RTX A6000, driven end to end by foldbridge:
  all 5 models are CAPRI high quality, **DockQ 0.935–0.945** against crystal 1BRS
  ([docs/REAL_RUN.md](docs/REAL_RUN.md)).

### Known limitations

- The Chai-1 and ColabFold backends have not yet been run end to end against a live install. Flags
  marked `TODO verify` in `foldbridge/backends/` should be checked with `--dry-run` first.
- DockQ-lite matches chains by id (no permutation search) and handles protein–protein interfaces only.
