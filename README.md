# foldbridge

One job spec, many structure predictors. You describe a complex once in a small YAML/JSON/FASTA
file. foldbridge then runs it on [Boltz-2](https://github.com/jwohlwend/boltz),
[Chai-1](https://github.com/chaidiscovery/chai-lab) or
[ColabFold](https://github.com/sokrypton/ColabFold), writing each backend's native input for you.
It scores every model the same way, whatever produced it: pLDDT, ipTM/pTM as reported, **ipSAE**,
**pDockQ**, iPAE, **DockQ-lite**/RMSD against a reference, **buried SASA** and **clashes**. It ranks
models across backends, seeds and samples, screens libraries of designed binders, caches results
and MSAs, prepares the best model for molecular dynamics, and exposes all of this to LLM agents
through an MCP server. Scoring only needs a structure file and, optionally, a PAE matrix, so it
also works on AlphaFold3, AF2-multimer, RoseTTAFold or crystal structures.

> **Validated**
>
> - **DockQ-lite vs official DockQ v2**: max |ΔDockQ| = **1.7×10⁻⁶** over **180 decoys** of
>   3 crystal complexes, all in the same CAPRI class. See [docs/VALIDATION.md](docs/VALIDATION.md).
> - **ipSAE vs the reference `ipsae.py`** (Dunbrack lab): **identical** to 6 decimals on 5 real
>   Boltz-2 models; pDockQ within 4×10⁻⁵. See [docs/REAL_RUN.md](docs/REAL_RUN.md).
> - **Real Boltz-2 run through foldbridge** (barnase–barstar, A6000): all 5 models reach
>   **DockQ 0.935–0.945** against crystal 1BRS (CAPRI high quality). See [docs/REAL_RUN.md](docs/REAL_RUN.md).
> - **Batch scoring**: 4–5× faster than serial on a desktop CPU (about 34–38 structures/s), resumable.

## Install

From PyPI:

```bash
pip install foldbridge            # core
pip install "foldbridge[md,mcp]"  # + MD prep and the MCP server
```

From source:

```bash
git clone https://github.com/bahumanyarg11/foldbridge.git && cd foldbridge
python -m venv .venv && . .venv/bin/activate
pip install -e .                 # core: numpy, pyyaml, requests
pip install -e ".[md]"           # + OpenMM / PDBFixer for `prep-md` (protonation, solvation)
pip install -e ".[mcp]"          # + MCP server for agents
pip install -e ".[score]"        # + official DockQ / PoseBusters, for cross-checking
pip install -e ".[dev]"          # + pytest, ruff
```

The predictors are separate installs (`pip install boltz`, `pip install chai_lab`,
`pip install "colabfold[alphafold]"` or LocalColabFold), usually on a GPU machine.
`foldbridge backends` shows which ones it can find. **You do not need any of them to try the
scoring, ranking and MD-prep steps below.**

## Quickstart

### 1. Score a structure (no predictor needed)

```bash
foldbridge score tests/data/1BRS.pdb --chains A,D --reference tests/data/1BRS.pdb -o results.csv
```

This scores the barnase (A) – barstar (D) interface of the 1BRS crystal structure against itself
(DockQ 1.0, RMSD 0). foldbridge recognises it as an experimental structure: crystal B-factors are
not pLDDT, so pLDDT, interface pLDDT and pDockQ are left out and the `note` column says why.

### 2. Score real Boltz-2 models with their PAE

The repository ships the five Boltz-2 models from the validation run, with their PAE files:

```bash
foldbridge score validation/boltz_barnase_barstar/barnase_barstar/ \
    --reference tests/data/1BRS.pdb -o boltz.csv
```

Each `*.cif` is paired with the `pae_<stem>.npz` next to it, so you get pLDDT, ipSAE, pDockQ, iPAE
and, against the crystal structure, DockQ 0.935–0.945.

### 3. Rank and write an HTML report

```bash
foldbridge rank boltz.csv --html report.html
```

`report.html` is a single self-contained, sortable table. `--by ipsae` (or any column) sorts by one
metric instead of the composite score.

### 4. Run a predictor

This step needs the predictor installed (`foldbridge backends`).

```bash
# Inspect what would run, without a GPU: writes the native input + work/<cache key>/command.txt
foldbridge run examples/barnase_barstar.yaml --backend boltz --dry-run -o out/bb_boltz

# Fold and score every model. A second identical call is a cache hit and returns at once.
foldbridge run examples/barnase_barstar.yaml --backend boltz --score -o out/bb_boltz
foldbridge run examples/barnase_barstar.yaml --backend chai  --score -o out/bb_chai

foldbridge rank out/bb_boltz/results.json out/bb_chai/results.json --html report.html
```

Layout of `-o out/bb_boltz/`: `results.json` (run summary + scored models), `models/` (models
copied from the cache on a cache hit), and `work/<cache key>/` with the backend's native input,
`command.txt`, `run.log` and raw outputs. Multi-seed Boltz-2 / Chai-1 runs (one seed per
predictor call) get one `work/<cache key>/s<seed>/` per seed.

A `--dry-run` writes `results.json` with **no models**: there is nothing to score or rank until
the real run has finished (`foldbridge rank` says so).

Example specs in `examples/`:

| file | what |
|---|---|
| `barnase_barstar.yaml` | two-chain protein complex (1BRS SEQRES) |
| `hemoglobin_heme.yaml` | α2β2 haemoglobin, `copies: 2` per chain, 4 haem ligands (`ccd: HEM` plus a SMILES, so Chai-1 can run it) |
| `single.fasta` | one chain (crambin) as plain FASTA |
| `target.yaml`, `designs.fasta` | binder-campaign example: barnase target, barstar plus 4 interface alanine mutants |

### 5. Prepare a model for MD

```bash
pip install -e ".[md]"     # OpenMM + PDBFixer; without them a numpy-only cleanup runs
foldbridge prep-md tests/data/1CRN.pdb -o md_1crn/
foldbridge prep-md validation/boltz_barnase_barstar/barnase_barstar/barnase_barstar_model_0.cif -o md_bb/
```

Outputs: `prepared.pdb`, `system_report.json` (charge, gaps, low-pLDDT residues, warnings) and,
for predicted models (pLDDT in the B-factor column), position restraints for confident CA atoms:

- **with OpenMM/PDBFixer**: missing atoms rebuilt, hydrogens added at `--ph`, optional
  `--solvate` (amber14 + TIP3P, 0.15 M NaCl), and GROMACS `posre_confident_<chain>.itp` files
  (`#ifdef POSRES_CONFIDENT`).
- **without them** (numpy fallback): waters/ligands removed and residues renumbered, but no hydrogens.
  Atom numbers will change when `pdb2gmx` adds hydrogens, so no `.itp` is written. The confident
  CA atoms are listed by (chain, resid, icode, atom name) in `restraints_confident.tsv`, and a
  warning explains how to regenerate the restraints.

Crystal structures such as 1CRN get no restraints and no low-pLDDT list, because their B-factors are
not pLDDT.

## Spec format

```yaml
name: my_complex
chains:
  - id: A
    type: protein          # protein | dna | rna
    sequence: MKTAYIAKQR...
    copies: 1              # homo-oligomers: copies get ids A, A2, A3 ...
    msa: msas/A.a3m        # optional per-chain a3m
  - id: B
    sequence: ACDEFGHIK...
ligands:
  - id: L
    smiles: CC(=O)Oc1ccccc1C(=O)O
  - id: H
    ccd: HEM               # CCD code: Boltz-2 only
    copies: 2
msa: server                # server | none | cached | path/to/a3m_dir_or_file
seeds: [42]
samples: 5
recycles: 3
reference: ref.pdb         # optional; enables RMSD / DockQ-lite (not part of the cache key)
boltz_args: ["--use_potentials"]   # any other top-level key is passed to backends as `extra`
```

- **Defaults:** `samples: 1`, `seeds: [42]`, `recycles: 3`, `msa: server`, `type: protein`, `copies: 1`.
  `seeds: 7` and `seeds: "1,2"` are accepted and become lists of ints.
- **Relative paths** (`reference`, `msa`, per-chain `msa`) are resolved against the **spec file's
  directory**, not the current directory. A missing MSA path is an error that names the resolved path.
- **MSA modes:** `server` (ColabFold MSA server), `none` (single sequence), `cached` (MSAs
  foldbridge harvested from an earlier run of the same sequence), or a path to an `.a3m` file or
  a directory of `<chain>.a3m` files. The cache key hashes MSA *contents*, so regenerating an a3m
  invalidates cached results.
- **Ligands:** `ccd:` is understood by Boltz-2 only. Chai-1 needs `smiles:`. Give both to run one
  spec on both backends (Boltz-2 then uses the SMILES). **ColabFold rejects ligands and DNA/RNA chains.**
- **Unknown top-level keys** go to `extra` (`boltz_args`, `chai_args`, `colabfold_args` are
  appended to the backend command line; `constraints`, `templates`, `properties` are copied into
  the Boltz YAML). A key that looks like a typo of a known one gets a warning:
  `unknown top-level key 'sampels' (passed to backends as extra); did you mean 'samples'?`
- **FASTA** works too. Headers may carry `type=` and `copies=` (`>B type=dna copies=2`).

## CLI reference

Global: `--debug` (or `FOLDBRIDGE_DEBUG=1`) shows tracebacks; `FOLDBRIDGE_CACHE` sets the cache
root (default `~/.cache/foldbridge`); `foldbridge --version`.

| command | purpose | options |
|---|---|---|
| `foldbridge backends` | list backends, whether installed, version | `--json` |
| `foldbridge run SPEC -b {boltz,chai,colabfold}` | prepare and run a prediction | `-o/--out DIR` (default `foldbridge_out/<name>-<backend>`), `--dry-run`, `--score`, `--reference REF` (overrides the spec), `--cpu`, `--no-cache`, `--cache-dir DIR` |
| `foldbridge score PATH...` | score structure files or directories (recursive `*.cif` / `*.pdb`) | `-o/--out FILE` (`.csv`/`.json`/`.html`, default `results.csv`), `--reference REF`, `--chains A,B`, `--pae FILE` (single structure only; otherwise PAE files are paired automatically), `-j/--workers N`, `--checkpoint FILE`, `--no-progress` |
| `foldbridge rank RESULTS...` | rank rows from `.csv`/`.json` results files | `--by METRIC` (default `composite`), `--no-length-norm`, `--top N`, `--html FILE`, `-o/--out FILE` |
| `foldbridge prep-md STRUCTURE` | clean, protonate, optionally solvate for MD; pLDDT-based restraints | `-o/--out DIR` (default `./<stem>_md`), `--ph 7.0`, `--keep-ligands`, `--solvate`, `--padding 1.0` (nm), `--plddt-cutoff 70` |
| `foldbridge campaign` | fold many binders against one target, score, rank | `-t/--target SPEC`, `-d/--designs FASTA`, `-b/--backend`, `-o/--out DIR` (all required), `--samples N`, `--seeds 1,2`, `--dry-run`, `--max-jobs N`, `--top N` (default 10), `--binder-chain ID`, `--binder-msa {auto,single,inherit}`, `--cpu`, `--no-cache`, `--cache-dir DIR`, `-j/--workers N`, `--no-progress` |
| `foldbridge mcp` | start the MCP server on stdio | |

## Python API

```python
from pathlib import Path
from foldbridge import load_spec, read_structure, score_structure
from foldbridge.backends.base import get_backend
from foldbridge.score import score_result
from foldbridge.rank import rank

# score any model, e.g. a shipped Boltz-2 model with its PAE, against the crystal structure
d = "validation/boltz_barnase_barstar/barnase_barstar/"
m = score_structure(d + "barnase_barstar_model_0.cif", pae=d + "pae_barnase_barstar_model_0.npz",
                    reference="tests/data/1BRS.pdb")
print(m.get("ipsae"), m.get("pdockq"), m.get("dockq"), m.get("bsa"))   # 0.935 0.499 0.945 1565

# DockQ-lite of a crystal structure against itself
xtal = read_structure("tests/data/1BRS.pdb")
print(score_structure(xtal, chains=["A", "D"], reference=xtal)["dockq"])   # 1.0

# run a predictor (needs it installed), then score and rank
spec = load_spec("examples/barnase_barstar.yaml")
models = get_backend("boltz").run(spec, workdir=Path("out/bb"), gpu=True)   # list[ModelResult]
rows = [score_result(r, reference=spec.reference) for r in models]
best = rank(rows)[0]
```

`score_structure` returns a dict with **only the metrics it could compute**. Use `m.get(...)`:
without a PAE there is no `ipsae`/`ipae`, with one chain there are no interface metrics, without
a reference there is no `dockq`/`rmsd`, and for crystal structures there is no `plddt`/`pdockq`.

### Batch scoring (parallel, resumable)

```python
from foldbridge.batch import score_many

if __name__ == "__main__":   # required: workers are separate processes (spawn on macOS/Windows)
    rows = score_many(["models/"], workers=8, checkpoint="scores.jsonl")  # one row per file, input order
    bad = [r for r in rows if "error" in r]                               # per-file failures, never raised
```

`score_many(paths, reference=None, workers=None, chains=None, progress=True, checkpoint=None, pae=None)`.
`pae` is one file (single structure) or a `{structure: pae}` mapping; anything not in the mapping is
paired automatically. Score warnings (e.g. PAE size mismatch) land in a `warnings` column, joined with `" | "`.
Chain ids passed to `--chains` / `chains=` must exist in every structure, otherwise that file is an error.

From the CLI:

```bash
foldbridge score models/ -j 8 --checkpoint scores.jsonl -o scores.csv   # Ctrl-C, rerun: resumes
foldbridge score models/ --chains A,B --reference ref.pdb -o scores.csv
```

Files are scored in worker processes (default CPUs − 1; fewer than 4 files stay in-process). Each
model is paired with the PAE file next to it (Boltz `pae_<stem>.npz`, Chai-1 `pae.<...>.npy`,
ColabFold `*_scores_*.json`, AF3 `*_confidences.json`, ...). Unreadable or empty structures are
reported as warnings and left out; the run never stops on a bad file. `--checkpoint FILE` appends one
JSON line per finished file. A rerun skips every file already recorded with the same size, mtime,
PAE, reference and chains. A progress line (`score n/N  rate/s  ETA`) goes to stderr.

## Binder-design campaigns

Fold many designed binders against one target, score and rank them, and get the best sequences
back. With the shipped example (barnase target, barstar + 4 alanine mutants as "designs"):

```bash
foldbridge campaign --target examples/target.yaml --designs examples/designs.fasta \
    --backend boltz --out screen/ --samples 5 --seeds 1,2 --top 3

foldbridge campaign -t examples/target.yaml -d examples/designs.fasta -b boltz -o screen/ --dry-run
foldbridge campaign -t examples/target.yaml -d examples/designs.fasta -b boltz -o screen/ --max-jobs 50
```

- `target.yaml` is an ordinary job spec holding the target chain(s) and any ligands. `designs.fasta`
  has one record per binder. Each design becomes `target + chain B` (or the next free id), named
  after the sanitised FASTA id, under `screen/designs/<design>/`. Repeated FASTA ids get a warning
  on stderr and are renamed `<id>_1`, `<id>_2`, ...
- Every design goes through the same code path as `foldbridge run`, including the result cache.
  Boltz-2 and Chai-1 take one seed per run, so each seed is its own job (`<design>/s<seed>/`).
- `--binder-msa`: `auto` (default) uses the target spec's `msa` mode for `server` / `none` and a
  single-sequence a3m for the binder otherwise (designed binders have no homologues); `single`
  always uses the single-sequence a3m; `inherit` never does.
- Failures (an invalid sequence, a crashed prediction) are recorded and the campaign carries on.
  `screen/campaign_state.json` tracks each design (`pending`, `running`, `done`, `failed`,
  `dry_run`). Rerunning the same command skips finished designs, retries failed or interrupted
  ones, and folds new FASTA records. `--max-jobs N` folds at most N new designs per invocation.
- When folding is done, all models are scored with the batch scorer (checkpointed to
  `screen/scores.jsonl`), backend ipTM/pTM are merged in, and all models are ranked together by the
  length-normalised composite. Each design is represented by its best model. In a campaign, `ipsae`
  is the best interface **involving the binder**. When the target has several chains, the
  whole-complex maximum is kept as `ipsae_complex`. When a backend renames chains (Chai-1 and
  ColabFold write A, B, C... in spec order), per-chain metrics are mapped back to spec chain ids
  before the binder is picked out.

Outputs in `--out`:

| file | content |
|---|---|
| `campaign.csv`, `campaign.json` | one row per design, best first. Every row has the same columns: `rank`, `id`, `design_id` (original FASTA id), `composite`, `ipsae`, `ipsae_complex`, `iptm`, `ptm`, `pdockq`, `ipae`, `plddt`, `plddt_interface`, `plddt_binder`, `clash_fraction`, `binder_length`, `n_residues`, `n_models`, `best_model`, `status`, `error`, `sequence`, `path` (best model file). Failed, pending and dry-run designs have no metrics and no rank, and come last |
| `campaign_models.csv` | every scored model, ranked |
| `report.html` | sortable table, best design first |
| `top_<N>.fasta` | the N best binders (`--top`, default 10), with rank and key scores in the header |
| `campaign_state.json`, `scores.jsonl`, `commands.txt` | resume state, scoring checkpoint, backend command lines |

## MCP server (LLM agents)

```bash
pip install -e ".[mcp]"
which foldbridge          # e.g. /home/me/foldbridge/.venv/bin/foldbridge
```

Client config for any MCP-compatible client. Use the **absolute path** to the venv's
`foldbridge`, because the client does not activate your virtualenv:

```json
{"mcpServers": {"foldbridge": {"command": "/home/me/foldbridge/.venv/bin/foldbridge", "args": ["mcp"]}}}
```

| tool | parameters | returns |
|---|---|---|
| `list_backends` | none | backends, availability, versions |
| `score_structure` | `path`, `reference=None`, `pae=None` (auto-paired if omitted), `chains=None` (e.g. `["A","B"]`) | one row of metrics |
| `rank_results` | `paths` (results `.csv`/`.json`, structures or directories), `by="composite"`, `length_normalize=True`, `top=None` | rows best-first with `rank` and `composite` |
| `run_prediction` | `spec_path`, `backend` (`boltz`/`chai`/`colabfold`), `dry_run=True`, `out_dir=None`, `cpu=False`, `score=True`, `use_cache=True` | run summary: command, cache key, scored models |
| `prep_md` | `path`, `out_dir=None` (default `<stem>_md` next to the input), `ph=7.0`, `keep_ligands=False`, `solvate=False` | MD-prep system report |

`run_prediction` is a **dry run by default**. An agent must pass `dry_run=false` to actually fold.
Failures (missing file, bad spec, backend not installed) come back as MCP tool errors
(`isError: true` with the message), not as normal results.

## Metrics

| key | definition |
|---|---|
| `plddt`, `plddt_<chain>`, `plddt_interface` | mean pLDDT read from B-factors (0–1 inputs rescaled to 0–100) over tokens / interface residues. Left out for experimental structures |
| `iptm`, `ptm` | as reported by the backend (not recomputed) |
| `ipsae_<A>_<B>` | ipSAE, d0res variant (Dunbrack 2025), for chains A and B. For each residue i in A, take the residues j in B with PAE<sub>ij</sub> < 10 Å; d0 comes from their count; score<sub>i</sub> is the TM-like mean over those j. The A→B score is the max over i. **`ipsae_<A>_<B>` is the max of the A→B and B→A directions** (the `max` row of the reference `ipsae.py`) |
| `ipsae` | max of `ipsae_<A>_<B>` over all chain pairs (in a campaign: over pairs involving the binder) |
| `pdockq` | 0.724 / (1 + e<sup>−0.052(x−152.611)</sup>) + 0.018, x = mean interface pLDDT × log10(#CB–CB contacts < 8 Å). Bryant et al. (2022) |
| `ipae`, `pae_interchain` | mean PAE over contacting / all inter-chain token pairs |
| `n_interface_residues`, `n_contacts` | residues / residue pairs with any inter-chain heavy-atom pair < 5 Å |
| `clash_count`, `clash_fraction` | inter-residue heavy-atom pairs < 2.2 Å (excluding i,i+1, disulphides and metal coordination) |
| `bsa`, `bsa_<chain>` | buried SASA, Σ SASA(chain) − SASA(complex); Shrake–Rupley, 100 points, 1.4 Å probe |
| `rmsd`, `n_aligned` | Kabsch CA/C1′ RMSD vs reference over matched (chain, resid, icode) |
| `fnat`, `irmsd`, `lrmsd`, `dockq` | DockQ-lite for two-chain complexes: DockQ = (Fnat + 1/(1+(iRMS/1.5)²) + 1/(1+(LRMS/8.5)²))/3. Basu & Wallner (2016); matches DockQ v2 to 1.7×10⁻⁶ |
| `composite` | weighted mean of per-metric z-scores (weights: ipTM 1, ipSAE 1, pTM 0.5, pLDDT 0.5, pDockQ 0.5, iPAE 0.5, clash fraction 0.5, BSA 0.25; lower-is-better metrics sign-flipped; a missing metric counts as z = 0). With ≥ 8 rows of differing size, each metric is first regressed on log(length) so big constructs are not rewarded for size. Rows with no metric at all (dry runs, failed models) get no composite and no rank, and sort last |

References

- Dunbrack RL. *Rēs ipSAE loquunt: What's wrong with AlphaFold's ipTM score and how to fix it.* bioRxiv 2025. doi:10.1101/2025.02.10.637595
- Bryant P, Pozzati G, Elofsson A. *Improved prediction of protein–protein interactions using AlphaFold2.* Nat Commun 13, 1265 (2022). doi:10.1038/s41467-022-28865-w
- Basu S, Wallner B. *DockQ: a quality measure for protein–protein docking models.* PLoS ONE 11, e0161879 (2016). doi:10.1371/journal.pone.0161879; DockQ v2: Mirabello C, Wallner B, Bioinformatics 40, btae586 (2024)
- Shrake A, Rupley JA. J Mol Biol 79, 351–371 (1973)

## Performance

Benchmark: score 1000 copies of `tests/data/1BRS.pdb` (6 chains, 4638 heavy atoms, 588 residues;
every ATOM coordinate perturbed by N(0, 0.2 Å)). No PAE or reference. All metrics are computed,
including Shrake–Rupley buried SASA, which is most of the cost.

| machine | workers | wall time | throughput | speed-up |
|---|---|---|---|---|
| Apple M4 (4P + 6E cores), macOS | 1 | 118.3 s | 8.5 /s | 1.0× |
| | 4 | 42.0 s | 23.8 /s | 2.8× |
| | 9 (default) | 29.5 s | 33.9 /s | 4.0× |
| Intel i9-12900 (8P + 8E cores, 24 threads), Linux | 1 | 128.8 s | 7.8 /s | 1.0× |
| | 4 | 39.1 s | 25.6 /s | 3.3× |
| | 23 (default) | 26.0 s | 38.4 /s | 5.0× |

- Parallel rows are identical to serial ones.
- Resume works: with the checkpoint holding 500 rows, finishing the batch took about the same
  time as the first 500 (13 s on the i9). A rerun with every file already done takes 0.1–0.2 s.
- Throughput levels off at about 8 workers on both machines, so workers beyond the performance
  cores add little.
- On the M4 run, low disk space meant 100 distinct perturbed files plus hard links up to 1000
  paths; every path is still parsed and scored separately. The i9 run used 1000 distinct files.

Rough rule: ~0.12 s per 600-residue complex per core. A 10k-model campaign scores in about
4–5 minutes on a desktop CPU, which is small next to the folding time.

## Limitations

- **DockQ-lite approximates DockQ v2** for protein–protein pairs: chains are matched by id (no
  permutation search), residues by (chain, resid, icode) (no sequence alignment), one interface at
  a time, no small-molecule / nucleic-acid special cases. Use the `DockQ` package
  (`pip install -e ".[score]"`) for publication numbers.
- **ColabFold cannot model ligands or nucleic acids**; such specs are rejected for that backend.
  **Chai-1 needs SMILES** for ligands.
- **Backend flags marked `TODO verify`** in `foldbridge/backends/*.py` (Chai-1 MSA `.pqt` handling,
  ColabFold complex-a3m format) were written against specific releases. Only Boltz-2 (2.2.1) has
  been run end to end. Predictor CLIs change often: run with `--dry-run` first and read `command.txt`.
- The built-in PDB/mmCIF reader keeps the first model and altloc A only.
- ipTM/pTM are taken from backend output, not recomputed. ipSAE/iPAE need the full PAE matrix
  (e.g. Boltz `--write_full_pae`, which foldbridge sets).

## Tests

```bash
python -m pytest -q
ruff check foldbridge tests
```

`tests/data/` holds small RCSB entries (1BRS, 2PTC, 1A2K, 4HHB, 1CRN) in PDB and mmCIF format.
See [CHANGELOG.md](CHANGELOG.md) for release notes.
