"""Electrical Resistivity Tomography (ERT) forward models.

Implements analytical solutions for DC resistivity, starting with
the homogeneous half-space and 1D layered earth.

These are the ERT equivalents of the magnetic dipole field in
geosim/magnetics/dipole.py — analytical solutions that serve as
validation cases for numerical backends.

Physics:
    For a point current source I on the surface of a homogeneous
    half-space with resistivity ρ:

        V(r) = ρ·I / (2π·r)

    Apparent resistivity:
        ρ_a = K · V/I

    where K is the geometric factor of the electrode array.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from geosim.resistivity.geometric import geometric_factor


def apparent_resistivity_halfspace(
    resistivity: float,
    c1: np.ndarray,
    c2: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
) -> float:
    """Compute apparent resistivity for a homogeneous half-space.

    For a uniform half-space, the apparent resistivity equals the
    true resistivity regardless of electrode configuration.

    Parameters
    ----------
    resistivity : float
        True resistivity in Ω·m.
    c1, c2 : ndarray
        Current electrode positions.
    p1, p2 : ndarray
        Potential electrode positions.

    Returns
    -------
    rho_a : float
        Apparent resistivity in Ω·m (should equal input resistivity).
    """
    # For a homogeneous half-space, ρ_a = ρ always
    # This function computes it explicitly via V = ρI/(2πr) to serve
    # as a validation case.

    c1 = np.asarray(c1, dtype=np.float64)
    c2 = np.asarray(c2, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)

    # Potential at P1 due to both current electrodes (unit current I=1)
    r_c1p1 = np.linalg.norm(c1 - p1)
    r_c2p1 = np.linalg.norm(c2 - p1)
    V_p1 = resistivity / (2.0 * np.pi) * (1.0 / r_c1p1 - 1.0 / r_c2p1)

    # Potential at P2
    r_c1p2 = np.linalg.norm(c1 - p2)
    r_c2p2 = np.linalg.norm(c2 - p2)
    V_p2 = resistivity / (2.0 * np.pi) * (1.0 / r_c1p2 - 1.0 / r_c2p2)

    # Potential difference
    delta_V = V_p1 - V_p2

    # Apparent resistivity
    K = geometric_factor(c1, c2, p1, p2)
    rho_a = K * delta_V  # I = 1

    return rho_a


def apparent_resistivity_layered(
    thicknesses: list[float],
    resistivities: list[float],
    electrode_spacing: float,
    array_type: str = "wenner",
    n_terms: int = 50,
) -> float:
    """Compute apparent resistivity for a 1D layered earth.

    Uses the linear filter method for the Hankel transform to compute
    the potential from a point source over a layered half-space.

    Parameters
    ----------
    thicknesses : list[float]
        Layer thicknesses in meters (top to bottom).
        Length N-1 for N layers (bottom is half-space).
    resistivities : list[float]
        Layer resistivities in Ω·m (top to bottom). Length N.
    electrode_spacing : float
        Electrode spacing 'a' in meters.
    array_type : str
        'wenner' or 'schlumberger'.
    n_terms : int
        Number of terms in the recurrence calculation.

    Returns
    -------
    rho_a : float
        Apparent resistivity in Ω·m.
    """
    n_layers = len(resistivities)

    if n_layers == 1:
        return resistivities[0]

    # Compute apparent resistivity using the recurrence relation
    # for the potential kernel (Koefoed, 1979)
    #
    # Start from bottom layer and work up:
    #   T_N = ρ_N (half-space)
    #   T_i = ρ_i · (T_{i+1} + ρ_i·tanh(λ·h_i)) / (ρ_i + T_{i+1}·tanh(λ·h_i))

    # Integration via discrete summation over lambda values
    # Use log-spaced lambda for stable numerical integration
    if array_type == "wenner":
        a = electrode_spacing
    else:
        a = electrode_spacing

    lambda_min = 1e-3 / a
    lambda_max = 50.0 / a
    lambdas = np.logspace(np.log10(lambda_min), np.log10(lambda_max), n_terms)

    # Compute kernel for each lambda
    rho_a_sum = 0.0
    weight_sum = 0.0

    for lam in lambdas:
        # Recurrence from bottom
        T = resistivities[-1]

        for i in range(n_layers - 2, -1, -1):
            rho_i = resistivities[i]
            h_i = thicknesses[i]
            tanh_val = np.tanh(lam * h_i)

            T = rho_i * (T + rho_i * tanh_val) / (rho_i + T * tanh_val)

        # For Wenner array: apparent resistivity kernel
        # The kernel relates the surface potential to the 1D model
        # Weight by the Wenner sensitivity kernel
        if array_type == "wenner":
            # Simplified: integrate T(λ) · J₀(λr) · λ
            # For Wenner: r = a (spacing)
            from scipy.special import j0

            kernel = T * j0(lam * a) * lam
        else:
            from scipy.special import j0

            kernel = T * j0(lam * a) * lam

        rho_a_sum += kernel
        weight_sum += resistivities[0] * np.exp(-lam * a) * lam  # normalization

    if weight_sum > 0:
        return resistivities[0] * rho_a_sum / weight_sum
    return resistivities[0]


def ert_forward(
    electrode_positions: np.ndarray,
    measurements: list[tuple[int, int, int, int]],
    resistivities: list[float],
    thicknesses: list[float] | None = None,
    backend: str = "analytical",
) -> dict[str, Any]:
    """Compute ERT forward response with selectable backend.

    Parameters
    ----------
    electrode_positions : ndarray, shape (N, 2) or (N, 3)
        Electrode positions.
    measurements : list of tuple
        Each tuple is (C1_idx, C2_idx, P1_idx, P2_idx) — electrode indices.
    resistivities : list[float]
        Layer resistivities in Ω·m.
    thicknesses : list[float], optional
        Layer thicknesses (N-1 for N layers). None = half-space.
    backend : str
        'analytical' (default) or 'pygimli'.

    Returns
    -------
    result : dict
        Keys: 'apparent_resistivity', 'geometric_factors', 'backend'.
    """
    if backend == "pygimli":
        return _ert_forward_pygimli(
            electrode_positions, measurements, resistivities, thicknesses
        )

    # Analytical backend
    positions = np.asarray(electrode_positions, dtype=np.float64)
    rho_a_list = []
    K_list = []

    for c1_idx, c2_idx, p1_idx, p2_idx in measurements:
        c1 = positions[c1_idx]
        c2 = positions[c2_idx]
        p1 = positions[p1_idx]
        p2 = positions[p2_idx]

        K = geometric_factor(c1, c2, p1, p2)
        K_list.append(float(K))

        if thicknesses is None or len(thicknesses) == 0:
            # Half-space: ρ_a = ρ
            rho_a_list.append(resistivities[0])
        else:
            # 1D layered: use apparent_resistivity_layered
            # Approximate spacing from C1-P1 distance
            spacing = float(np.linalg.norm(c1 - p1))
            rho_a = apparent_resistivity_layered(
                thicknesses, resistivities, spacing
            )
            rho_a_list.append(float(rho_a))

    return {
        'apparent_resistivity': rho_a_list,
        'geometric_factors': K_list,
        'backend': 'analytical',
    }


def _build_pygimli_scheme(
    electrode_positions: np.ndarray,
    measurements: list[tuple[int, int, int, int]],
) -> Any:
    """Build a pyGIMLi measurement scheme."""
    import pygimli as pg

    scheme = pg.DataContainerERT()

    # Add electrodes
    for pos in electrode_positions:
        if len(pos) == 2:
            scheme.createSensor(pg.Pos(pos[0], pos[1]))
        else:
            scheme.createSensor(pg.Pos(pos[0], pos[1], pos[2]))

    # Add measurements
    for c1, c2, p1, p2 in measurements:
        scheme.createFourPointData(scheme.size(), c1, c2, p1, p2)

    return scheme


def _build_pygimli_mesh(
    electrode_positions: np.ndarray,
) -> Any:
    """Build a pyGIMLi mesh for forward modeling."""
    import pygimli.meshtools as mt

    positions = np.asarray(electrode_positions)
    x_min = positions[:, 0].min() - 5.0
    x_max = positions[:, 0].max() + 5.0

    world = mt.createWorld(
        start=[x_min, -20.0],
        end=[x_max, 0.0],
        worldMarker=True,
    )

    mesh = mt.createMesh(world, quality=33, area=0.5)
    return mesh


def _run_pygimli_forward(
    mesh: Any,
    scheme: Any,
    resistivities: list[float],
    thicknesses: list[float] | None,
) -> np.ndarray:
    """Run pyGIMLi ERT forward simulation."""
    from pygimli.physics import ert

    # Create resistivity model
    rho = np.full(mesh.cellCount(), resistivities[0])

    if thicknesses and len(resistivities) > 1:
        depth = 0.0
        for i, thickness in enumerate(thicknesses):
            depth += thickness
            if i + 1 < len(resistivities):
                for cell in mesh.cells():
                    if cell.center().y() < -depth:
                        rho[cell.id()] = resistivities[i + 1]

    fwd = ert.ERTModelling()
    fwd.setMesh(mesh)
    fwd.setData(scheme)

    return fwd.response(rho)


def _ert_forward_pygimli(
    electrode_positions: np.ndarray,
    measurements: list[tuple[int, int, int, int]],
    resistivities: list[float],
    thicknesses: list[float] | None,
) -> dict[str, Any]:
    """ERT forward using pyGIMLi backend.

    All pyGIMLi imports are inside this function for lazy loading.
    Raises ImportError if pyGIMLi is not installed.
    """
    import pygimli  # noqa: F401 — validate availability

    scheme = _build_pygimli_scheme(electrode_positions, measurements)
    mesh = _build_pygimli_mesh(electrode_positions)
    data = _run_pygimli_forward(mesh, scheme, resistivities, thicknesses)

    # Compute geometric factors analytically
    positions = np.asarray(electrode_positions, dtype=np.float64)
    K_list = []
    rho_a_list = []
    for i, (c1_idx, c2_idx, p1_idx, p2_idx) in enumerate(measurements):
        K = geometric_factor(
            positions[c1_idx], positions[c2_idx],
            positions[p1_idx], positions[p2_idx]
        )
        K_list.append(float(K))
        rho_a_list.append(float(K * data[i]))

    return {
        'apparent_resistivity': rho_a_list,
        'geometric_factors': K_list,
        'backend': 'pygimli',
    }
