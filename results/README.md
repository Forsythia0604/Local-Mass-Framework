# Reference results and new outputs

`figures/`, `tables/` and `data/` contain the supplied results, preserved
byte-for-byte. Filenames are unchanged so the manuscript fragments still refer
to `results/figures/...` when included from a root-level manuscript project.

- `figure1_*`, `figure2_*`, `figure3_*`: Experiments 1–3.
- `fig_nn_*` with no dataset suffix: supplied MNIST / Experiment 4 figures.
- `fig_nn_*_cifar`: supplied CIFAR-10 / Experiment 5 figures.
- `fig_nn_*_imagenet`: supplied ImageNet / Experiment 5 figures.
- `tables/table_nn_mi*.tex`: original tables, including the incomplete two-family
  ImageNet table. See `docs/KNOWN_ISSUES.md` before treating it as the final table.

The three primary plotting entry points now default to `generated/`, which is
ignored by Git. They can take an explicit `--output-dir` for a different location.
Single-run NN plotting writes its generated `.tex` table beside its generated
figures, preserving its original output behavior. The multi-seed ImageNet script
also exports a complete four-family `.tex` table and per-seed CSV beside the new
figures. Those exports do not replace the original two-family reference table.
