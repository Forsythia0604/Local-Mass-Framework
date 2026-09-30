# Conservative consolidation — 2026-09-30

Created as a new folder, `Code for Git`, next to the two original source trees.
The original trees and their Git history remain untouched. This copy contains no
Git metadata and has not been initialized, committed, pushed or published.

Source Exp1–3 HEAD: `e49589734f2a73d85afd1fbc37582069893564f6`.

## Included

- All 34 original Python files, including older helpers and drivers.
- All 18 supplied PDF/PNG figures, 3 LaTeX tables and 3 Exp2 CSV/JSON data files.
- All 3 ImageNet marginal inputs, 12 JSON run records, 12 stdout logs and 10 PBS records.
- Both manuscript fragments, both PBS scripts and the small CIFAR base checkpoint.
- Original READMEs, original provenance and original Exp1–3 ignore rules archived
  under `docs/originals/`.

## Excluded from this copy

- The delivery ZIP and 12 large ImageNet `.pt` checkpoints.
- `.git`, `.DS_Store`, dependency directories and Python/plot caches.

Excluded items remain in the original source directories. No public checkpoint
download endpoint has been created.

## Source changes

At the initial consolidation, only three Python files differed from the sources:

1. `experiments/local_mass_experiments.py`: default output becomes
   `results/generated/exp01_03/`.
2. `experiments/neural_networks/experiment_scripts/local_mass_nn.py`: root-path
   resolution is adjusted for the new location; a `--output-dir` option defaults
   to `results/generated/exp04_05/figures/`.
3. `experiments/neural_networks/experiment_scripts/imagenet_multiseed.py`: the same
   output option is added, using the root imported from `local_mass_nn.py`.

The subsequent minimum-reproduction review added output-directory fixes,
portable defaults, input errors, run metadata and a four-family ImageNet table
export. The cumulative patch and manifest linked below include those later changes.

Scientific functions, hyperparameters, random seeds, plotted labels, reference
figures, tables, data and weights are unchanged. The root `requirements.txt` is
unchanged. `requirements-training.txt` only adds the two training dependencies.

See [the exact source diff](source_adjustments.patch) and
[the source-to-destination SHA-256 manifest](file_manifest.csv).

## Validation

The initial consolidation used static checks only. The later review located
an existing compatible local environment and completed actual plotting, original
ImageNet checkpoint extraction and synthetic-batch training checks. Current
evidence and limits are in `VALIDATION.json` and `REPRODUCIBILITY.md`. No full
neural-network training was performed, and original scientific inputs remain unchanged.
