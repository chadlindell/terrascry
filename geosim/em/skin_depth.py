"""Electromagnetic skin depth calculations.

The skin depth is the characteristic depth at which EM fields attenuate
to 1/e (~37%) of their surface value in a conductive medium. It is the
fundamental parameter controlling HIRT's depth of investigation.

Physics:
    δ = 1 / √(π·f·μ·σ)
    δ ≈ 503 · √(ρ/f)   [practical formula, meters]

where:
    f = frequency (Hz)
    μ = magnetic permeability (H/m)
    σ = electrical conductivity (S/m)
    ρ = electrical resistivity (Ω·m) = 1/σ
"""

from __future__ import annotations

import numpy as np

# Vacuum permeability (H/m)
MU_0 = 4.0 * np.pi * 1e-7


def skin_depth(
    frequency: float | np.ndarray,
    conductivity: float | np.ndarray,
    mu_r: float = 1.0,
) -> float | np.ndarray:
    """Compute electromagnetic skin depth.

    Parameters
    ----------
    frequency : float or ndarray
        Frequency in Hz. Must be positive.
    conductivity : float or ndarray
        Electrical conductivity in S/m. Must be positive.
    mu_r : float
        Relative magnetic permeability (dimensionless).
        Default 1.0 for non-magnetic materials.

    Returns
    -------
    delta : float or ndarray
        Skin depth in meters.
    """
    frequency = np.asarray(frequency, dtype=np.float64)
    conductivity = np.asarray(conductivity, dtype=np.float64)
    mu = mu_r * MU_0
    return 1.0 / np.sqrt(np.pi * frequency * mu * conductivity)


def skin_depth_practical(
    resistivity: float | np.ndarray,
    frequency: float | np.ndarray,
) -> float | np.ndarray:
    """Compute skin depth using the practical approximation.

    δ ≈ 503.29 · √(ρ/f)

    This is the formula commonly used in geophysical field guides.

    Parameters
    ----------
    resistivity : float or ndarray
        Electrical resistivity in Ω·m.
    frequency : float or ndarray
        Frequency in Hz.

    Returns
    -------
    delta : float or ndarray
        Skin depth in meters.
    """
    resistivity = np.asarray(resistivity, dtype=np.float64)
    frequency = np.asarray(frequency, dtype=np.float64)
    # The exact coefficient is 1/√(π·μ₀) ≈ 503.292
    coeff = 1.0 / np.sqrt(np.pi * MU_0)
    return coeff * np.sqrt(resistivity / frequency)
