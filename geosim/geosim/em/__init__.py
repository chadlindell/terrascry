"""Electromagnetic induction forward models.

Provides analytical and numerical FDEM solutions for the HIRT
MIT-3D instrument simulation.

Analytical models (Phase 2a):
- skin_depth: Electromagnetic skin depth calculations
- coil: HIRT coil geometry definitions
- fdem: Conductive sphere and 1D layered earth responses

Numerical backends (Phase 2c):
- SimPEG integration via fdem_forward(backend="simpeg")
"""

from geosim.em.coil import CoilConfig, ProbeCoilSet, hirt_default_coils
from geosim.em.fdem import fdem_forward, fdem_response_1d, secondary_field_conductive_sphere
from geosim.em.skin_depth import skin_depth, skin_depth_practical

__all__ = [
    "skin_depth",
    "skin_depth_practical",
    "CoilConfig",
    "ProbeCoilSet",
    "hirt_default_coils",
    "secondary_field_conductive_sphere",
    "fdem_response_1d",
    "fdem_forward",
]
