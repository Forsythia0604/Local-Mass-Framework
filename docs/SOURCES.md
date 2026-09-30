# Sources and attribution

This folder consolidates two supplied source trees:

| Source | Contribution | Current location |
|---|---|---|
| `Local-Mass-Framework` | User's Exp1–3 code and results | `experiments/local_mass_experiments.py`, `results/` |
| `Experiments4and5` | Supervisor-provided Exp4–5 code, results and records | `experiments/neural_networks/`, `data/`, `results/`, `paper/` |

The original paper citation names Hanli Xu, Fengxiang He and Sarat Moka, in that
order. Source comments that identify contributors remain intact. No author or
commit identity was fabricated during consolidation.

## Third-party material

`experiments/neural_networks/experiment_scripts/flopp/resnet56_flopp.py` describes
compatibility with DepGraph's CIFAR ResNet-56 and the Torch-Pruning v1.1.4 release
checkpoint. The accompanying `cifar10_resnet56.pth` is preserved in place. The
supplied package does not include a separate license notice or verified download
URL for that weight. Its precise upstream source and redistribution terms remain
to be recorded before a public release.

The ImageNet base uses Torchvision's ResNet-50 implementation. Third-party
dependencies are installed separately through the requirements files.

Neither source tree included a LICENSE file. This merge introduces no new license.
