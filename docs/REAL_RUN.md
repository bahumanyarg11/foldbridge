# Real prediction: Boltz-2 barnase-barstar through foldbridge

Run on 2026-10-09 on an NVIDIA RTX A6000 with Boltz 2.2.1, driven entirely by `foldbridge.backends.boltz` (spec -> YAML -> `boltz predict` -> collect). MSA from the ColabFold server. 5 diffusion samples, seed 42, 3 recycles. Prediction took about 66 s on the GPU after first-run downloads.

Sequences are the 1BRS SEQRES (barnase chain A, barstar chain D, which carries the C40A/C82A mutations). Reference is crystal 1BRS chains A+D.

| model | DockQ | Fnat | iRMSD (Å) | LRMSD (Å) | CA RMSD (Å) | ipSAE | ipAE | pDockQ | pLDDT | BSA (Å²) | clashes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.945 | 0.873 | 0.28 | 0.57 | 0.31 | 0.935 | 0.78 | 0.50 | 96.7 | 1565 | 0 |
| 1 | 0.935 | 0.855 | 0.31 | 0.74 | 0.35 | 0.938 | 0.75 | 0.50 | 96.6 | 1568 | 1 |
| 2 | 0.945 | 0.891 | 0.33 | 0.88 | 0.36 | 0.934 | 0.79 | 0.50 | 96.7 | 1578 | 0 |
| 3 | 0.942 | 0.873 | 0.31 | 0.65 | 0.33 | 0.929 | 0.83 | 0.48 | 96.7 | 1579 | 1 |
| 4 | 0.939 | 0.855 | 0.29 | 0.53 | 0.31 | 0.933 | 0.79 | 0.50 | 96.6 | 1533 | 0 |

All 5 models are CAPRI high quality (DockQ >= 0.80). Crystal buried surface area computed by foldbridge is 1562 Å², and the models give 1533-1579 Å².

Raw files: `validation/boltz_barnase_barstar/`. Scores: `foldbridge_scores_vs_1BRS.json`.

## Agreement with the reference ipsae.py (Dunbrack lab, Jan 2026 release)

Same five models, PAE cutoff 10 Å, distance cutoff 10 Å, `max` row of the reference output.

| model | ipSAE reference | ipSAE foldbridge | pDockQ reference | pDockQ foldbridge |
|---|---|---|---|---|
| 0 | 0.934977 | 0.934977 | 0.4986 | 0.4986 |
| 1 | 0.937884 | 0.937884 | 0.5041 | 0.5041 |
| 2 | 0.934125 | 0.934125 | 0.5036 | 0.5036 |
| 3 | 0.929347 | 0.929347 | 0.4824 | 0.4824 |
| 4 | 0.933048 | 0.933048 | 0.4978 | 0.4978 |

Maximum absolute difference: ipSAE 0 at 6 decimals, pDockQ 4e-5 (reference prints 4 decimals). Data: `validation/boltz_barnase_barstar/ipsae_reference_comparison.json`.

## Real prediction: Chai-1 barnase-barstar through foldbridge

Chai-1 0.6.1 on the same A6000, driven by `foldbridge.backends.chai`, same spec (ColabFold MSA server, 5 diffusion samples, seed 42, 3 trunk recycles). Chai writes chains as A/B, so the 1BRS reference chain D was renamed to B for comparison.

| model | DockQ | Fnat | iRMSD (Å) | LRMSD (Å) | CA RMSD (Å) | pLDDT | BSA (Å²) | clashes |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.938 | 0.873 | 0.36 | 0.59 | 0.39 | 98.2 | 1558 | 1 |
| 1 | 0.969 | 0.945 | 0.28 | 0.50 | 0.35 | 98.2 | 1578 | 1 |
| 2 | 0.953 | 0.909 | 0.32 | 0.62 | 0.37 | 98.2 | 1569 | 0 |
| 3 | 0.953 | 0.909 | 0.33 | 0.65 | 0.41 | 98.2 | 1565 | 0 |
| 4 | 0.950 | 0.909 | 0.36 | 0.69 | 0.46 | 98.2 | 1565 | 0 |

All 5 models are CAPRI high quality. Two environment fixes were needed on the box, both outside foldbridge: Chai's MSA path needs `pyarrow`, and Chai pins numpy 1.26, so `pyarrow<17` is required.
