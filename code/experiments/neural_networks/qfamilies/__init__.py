"""Registry of the four E1 variational families."""

from .gaussian_meanfield import GaussianMeanField
from .spike_slab_q import SpikeSlabQ
from .hardconcrete_l0 import HardConcreteL0
from .horseshoe_q import HorseshoeQ

_FAMILIES = {
    "gaussian": GaussianMeanField,
    "spikeslab": SpikeSlabQ,
    "hardconcrete": HardConcreteL0,
    "horseshoe": HorseshoeQ,
}

ATOM_FAMILIES = {"spikeslab", "hardconcrete"}
CONTINUOUS_FAMILIES = {"gaussian", "horseshoe"}


def make_qfamily(name, shape, **kw):
    if name not in _FAMILIES:
        raise ValueError(f"unknown q-family '{name}'; choose from {list(_FAMILIES)}")
    return _FAMILIES[name](shape, **kw)


def predicted_class(name):
    return "atom-capable" if name in ATOM_FAMILIES else "continuous"
