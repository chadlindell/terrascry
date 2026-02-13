"""Geometric factors for electrical resistivity arrays.

The geometric factor K relates measured resistance (V/I) to apparent
resistivity:
    ρ_a = K · (V/I)

For a homogeneous half-space, ρ_a equals the true resistivity.

Common arrays:
- Wenner: K = 2πa
- Schlumberger: K = πn(n+1)a
- Dipole-dipole: K = πn(n+1)(n+2)a
- HIRT cross-hole: specialized geometry
"""

from __future__ import annotations

import numpy as np


def geometric_factor(
    c1: np.ndarray,
    c2: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
) -> float:
    """Compute geometric factor for arbitrary 4-electrode configuration.

    K = 2π / (1/r_C1P1 - 1/r_C2P1 - 1/r_C1P2 + 1/r_C2P2)

    Parameters
    ----------
    c1, c2 : ndarray, shape (2,) or (3,)
        Current electrode positions (source, sink).
    p1, p2 : ndarray, shape (2,) or (3,)
        Potential electrode positions.

    Returns
    -------
    K : float
        Geometric factor in meters.
    """
    c1 = np.asarray(c1, dtype=np.float64)
    c2 = np.asarray(c2, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)

    r_c1p1 = np.linalg.norm(c1 - p1)
    r_c2p1 = np.linalg.norm(c2 - p1)
    r_c1p2 = np.linalg.norm(c1 - p2)
    r_c2p2 = np.linalg.norm(c2 - p2)

    denom = 1.0 / r_c1p1 - 1.0 / r_c2p1 - 1.0 / r_c1p2 + 1.0 / r_c2p2
    return 2.0 * np.pi / denom


def geometric_factor_wenner(a: float) -> float:
    """Geometric factor for Wenner array.

    K = 2πa

    Parameters
    ----------
    a : float
        Electrode spacing in meters.

    Returns
    -------
    K : float
        Geometric factor in meters.
    """
    return 2.0 * np.pi * a


def geometric_factor_schlumberger(a: float, n: int) -> float:
    """Geometric factor for Schlumberger array.

    K = πn(n+1)a

    Parameters
    ----------
    a : float
        Inner electrode spacing in meters.
    n : int
        Expansion factor (AB/2 = (n+0.5)a).

    Returns
    -------
    K : float
        Geometric factor in meters.
    """
    return np.pi * n * (n + 1) * a


def geometric_factor_dipole_dipole(a: float, n: int) -> float:
    """Geometric factor for dipole-dipole array.

    K = πn(n+1)(n+2)a

    Parameters
    ----------
    a : float
        Dipole length in meters.
    n : int
        Separation factor (gap = n·a between dipoles).

    Returns
    -------
    K : float
        Geometric factor in meters.
    """
    return np.pi * n * (n + 1) * (n + 2) * a


def geometric_factor_hirt_crosshole(
    electrode_spacing: float,
    borehole_separation: float,
) -> float:
    """Geometric factor for HIRT cross-hole configuration.

    Two boreholes with ring electrodes at the same depth,
    using a bipole-bipole arrangement.

    Parameters
    ----------
    electrode_spacing : float
        Vertical spacing between electrodes in the same borehole (meters).
    borehole_separation : float
        Horizontal distance between boreholes (meters).

    Returns
    -------
    K : float
        Geometric factor in meters.
    """
    L = borehole_separation
    dz = electrode_spacing

    # C1 at (0, 0), C2 at (0, dz), P1 at (L, 0), P2 at (L, dz)
    c1 = np.array([0.0, 0.0])
    c2 = np.array([0.0, dz])
    p1 = np.array([L, 0.0])
    p2 = np.array([L, dz])

    return geometric_factor(c1, c2, p1, p2)
