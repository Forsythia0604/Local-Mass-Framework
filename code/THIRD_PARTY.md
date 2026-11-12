# Source and third-party notes

This code accompanies the paper by Hanli Xu, Fengxiang He, and Sarat Moka.

`experiments/neural_networks/experiment_scripts/flopp/resnet56_flopp.py` describes
compatibility with DepGraph's CIFAR ResNet-56 and the Torch-Pruning v1.1.4 release
checkpoint. The accompanying `cifar10_resnet56.pth` is preserved in place.
The supplied archive contains no separate license notice or verified download
URL for that weight; its precise upstream source and redistribution terms
are not established by this package.

The ImageNet base uses Torchvision's ResNet-50 implementation. Third-party
dependencies are installed separately through the requirements files.

No LICENSE file was present in the supplied archive. This package assigns no
new license.
