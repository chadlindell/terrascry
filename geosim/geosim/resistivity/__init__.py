"""Electrical resistivity forward models.

Provides analytical and numerical DC resistivity solutions for
the HIRT ERT instrument simulation.

Analytical models (Phase 2a):
- geometric: Geometric factors for electrode arrays
- electrodes: HIRT electrode geometry definitions
- ert: Homogeneous half-space and 1D layered earth solutions

Numerical backends (Phase 2d):
- pyGIMLi integration via ert_forward(backend="pygimli")
"""

from geosim.resistivity.electrodes import Electrode, ElectrodeArray, hirt_default_electrodes
from geosim.resistivity.ert import (
    apparent_resistivity_halfspace,
    apparent_resistivity_layered,
    ert_forward,
)
from geosim.resistivity.geometric import (
    geometric_factor,
    geometric_factor_dipole_dipole,
    geometric_factor_hirt_crosshole,
    geometric_factor_schlumberger,
    geometric_factor_wenner,
)

__all__ = [
    "geometric_factor",
    "geometric_factor_wenner",
    "geometric_factor_schlumberger",
    "geometric_factor_dipole_dipole",
    "geometric_factor_hirt_crosshole",
    "Electrode",
    "ElectrodeArray",
    "hirt_default_electrodes",
    "apparent_resistivity_halfspace",
    "apparent_resistivity_layered",
    "ert_forward",
]
