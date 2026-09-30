# Remaining limits after the minimum-reproduction review

1. **Missing original MNIST/CIFAR inputs:** their original marginals were not
   supplied. The actual training/extraction entry points passed synthetic-batch
   checks, but full retraining and the paper's trained numerical values have not
   been reproduced here. New runs should be labeled separately.
2. **Historical ImageNet table:** the supplied two-family `.tex` remains unchanged
   for provenance. The current multi-seed script now generates a complete separate
   four-family table and per-seed CSV from the included inputs.
3. **Student-t naming:** plot labels and internal keys still say `Horseshoe` for
   the supplied mean-field Student-t representative. This review preserves paper
   figures and computation; it documents the alias rather than changing it.
4. **Method scope:** Exp1–3 are finite-radius illustrations; the NN plotting code
   pools coordinate/channel marginals and approximates the hard-concrete active
   component by a Gaussian. Base parameters are frozen in the gate-training paper
   paths while the supplied code's BatchNorm training mode is retained. These
   scientific choices have not been altered or adjudicated in this review.
5. **Logarithmic index scope:** the historical README's blanket statement that it
   is not implemented is inaccurate: `mass_index.py` contains `mi2_hat` estimation,
   separate from the main paper-figure pipeline.
6. **Runtime limits:** CPU plotting, synthetic-batch training and ImageNet marginal
   re-extraction passed in the recorded environment. Full dataset downloads,
   full NN training, CUDA/MPS and a fresh dependency installation were not checked.
   The verified requirement files pin direct dependencies, not transitive ones.
   Torch emitted an AMP deprecation warning during CPU smoke tests; it did not
   prevent execution. No AMP API migration was made solely to suppress a warning.
7. **Older drivers and PBS:** original site-specific templates remain historical.
   Use the active guides' commands; old docstrings may refer to previous locations.
   Auxiliary modules are retained because the training import chain uses them.
8. **Paper and publication metadata:** only two manuscript fragments are supplied;
   the old Exp1–3 requirements document is absent. The authors' license choice and
   precise upstream source/redistribution terms for the small CIFAR base weight
   still need to be documented before a public release.
