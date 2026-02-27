"""Frequency-domain electromagnetic (FDEM) forward models.

Implements analytical solutions for FDEM responses, starting with
the conductive sphere (Wait, 1951) and 1D layered earth.

These are the EM equivalents of the magnetic dipole field in
geosim/magnetics/dipole.py — analytical solutions that serve as
validation cases for numerical backends.

Physics:
    The secondary field from a conductive sphere in a uniform
    primary field depends on the induction number:
        α = r / δ  (sphere radius / skin depth)

    At low induction numbers (α << 1): response ∝ σ·ω·r⁵
    At high induction numbers (α >> 1): response → constant (saturation)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from geosim.em.skin_depth import skin_depth


def secondary_field_conductive_sphere(
    radius: float,
    conductivity: float,
    frequency: float,
    r_obs: float,
    mu_r: float = 1.0,
) -> complex:
    """Analytical secondary field of a conductive sphere (Wait, 1951).

    Computes the complex secondary magnetic field response of a
    conductive, permeable sphere in a uniform oscillating primary field.

    The response is normalized to the primary field (Hs/Hp).

    Parameters
    ----------
    radius : float
        Sphere radius in meters.
    conductivity : float
        Sphere conductivity in S/m.
    frequency : float
        Frequency in Hz.
    r_obs : float
        Distance from sphere center to observation point in meters.
        Must be > radius.
    mu_r : float
        Relative magnetic permeability of the sphere.

    Returns
    -------
    response : complex
        Complex secondary/primary field ratio (dimensionless).
        Real part = in-phase, Imaginary part = quadrature.
    """
    delta = skin_depth(frequency, conductivity, mu_r)
    k = (1 + 1j) / delta  # complex wavenumber

    kr = k * radius

    # Sphere response function (Wait, 1951)
    # Q = (2/9) · [1 - 3/(kr) · coth(kr) + 3/(kr)²]
    # For numerical stability at small kr, use series expansion
    coth_kr = np.cosh(kr) / np.sinh(kr) if abs(kr) > 1e-6 else 1.0 / kr + kr / 3.0
    q = (2.0 / 9.0) * (1.0 - 3.0 * coth_kr / kr + 3.0 / kr**2)

    # Dipolar decay: secondary field falls off as (radius/r_obs)³
    return q * (radius / r_obs) ** 3


def fdem_response_1d(
    thicknesses: list[float],
    conductivities: list[float],
    frequency: float,
    coil_separation: float,
    height: float = 0.0,
) -> complex:
    """1D layered-earth FDEM response for horizontal coplanar coils.

    Uses the analytical cumulative sensitivity approach for a
    layered half-space.

    Parameters
    ----------
    thicknesses : list[float]
        Layer thicknesses in meters (top to bottom).
        Length N-1 for N layers (bottom layer is infinite half-space).
    conductivities : list[float]
        Layer conductivities in S/m (top to bottom). Length N.
    frequency : float
        Operating frequency in Hz.
    coil_separation : float
        Distance between transmitter and receiver coils in meters.
    height : float
        Height of coils above ground surface in meters.

    Returns
    -------
    response : complex
        Complex apparent conductivity ratio (σ_a / σ_ref).
    """
    n_layers = len(conductivities)

    # Cumulative depth sensitivity (McNeill, 1980 approximation)
    # For horizontal coplanar coils at height h above layered earth
    s = coil_separation

    # Compute cumulative depths
    depths = np.zeros(n_layers)
    for i in range(n_layers - 1):
        depths[i + 1] = depths[i] + thicknesses[i]

    # Normalized depths (z/s)
    z_norm = (depths + height) / s

    # Cumulative sensitivity function for HCP coils:
    # R_HCP(z) = 1 / √(4z² + 1)
    def sensitivity_cum(z):
        return 1.0 / np.sqrt(4.0 * z**2 + 1.0)

    # Layer sensitivities
    sigma_apparent = 0.0 + 0.0j
    for i in range(n_layers):
        z_top = z_norm[i]
        if i < n_layers - 1:
            z_bot = z_norm[i + 1]
            weight = sensitivity_cum(z_top) - sensitivity_cum(z_bot)
        else:
            weight = sensitivity_cum(z_top)

        sigma_apparent += conductivities[i] * weight

    return sigma_apparent


def fdem_forward(
    thicknesses: list[float],
    conductivities: list[float],
    frequencies: list[float],
    coil_separation: float,
    height: float = 0.0,
    backend: str = "analytical",
) -> dict[str, Any]:
    """Compute FDEM forward response with selectable backend.

    Parameters
    ----------
    thicknesses : list[float]
        Layer thicknesses in meters (N-1 for N layers).
    conductivities : list[float]
        Layer conductivities in S/m (length N).
    frequencies : list[float]
        Operating frequencies in Hz.
    coil_separation : float
        TX-RX separation in meters.
    height : float
        Coil height above surface in meters.
    backend : str
        'analytical' (default) or 'simpeg'.

    Returns
    -------
    result : dict
        Keys: 'frequencies', 'real', 'imag', 'backend'.
    """
    if backend == "simpeg":
        return _fdem_forward_simpeg(
            thicknesses, conductivities, frequencies, coil_separation, height
        )

    # Analytical backend
    real_parts = []
    imag_parts = []
    for freq in frequencies:
        resp = fdem_response_1d(
            thicknesses, conductivities, freq, coil_separation, height
        )
        real_parts.append(float(np.real(resp)))
        imag_parts.append(float(np.imag(resp)))

    return {
        'frequencies': frequencies,
        'real': real_parts,
        'imag': imag_parts,
        'backend': 'analytical',
    }


def _build_simpeg_survey(
    frequencies: list[float],
    coil_separation: float,
    height: float,
) -> Any:
    """Build a SimPEG FDEM survey for horizontal coplanar coils."""
    from SimPEG.electromagnetics import frequency_domain as fdem

    source_list = []
    for freq in frequencies:
        # Receiver
        rx_loc = np.array([[coil_separation, 0.0, height]])
        rx = fdem.receivers.PointMagneticFluxDensitySecondary(
            rx_loc, orientation="z", component="both"
        )
        # Source (vertical magnetic dipole)
        src_loc = np.array([0.0, 0.0, height])
        src = fdem.sources.MagDipole(
            receiver_list=[rx], frequency=freq, location=src_loc
        )
        source_list.append(src)

    return fdem.Survey(source_list)


def _fdem_forward_simpeg(
    thicknesses: list[float],
    conductivities: list[float],
    frequencies: list[float],
    coil_separation: float,
    height: float,
) -> dict[str, Any]:
    """FDEM forward using SimPEG backend.

    All SimPEG imports are inside this function for lazy loading.
    Raises ImportError if SimPEG is not installed.
    """
    from SimPEG import maps
    from SimPEG.electromagnetics import frequency_domain as fdem

    survey = _build_simpeg_survey(frequencies, coil_separation, height)

    sigma = np.array(conductivities)
    sigma_map = maps.IdentityMap(nP=len(sigma))
    simulation = fdem.Simulation1DLayered(
        survey=survey,
        sigmaMap=sigma_map,
        thicknesses=np.array(thicknesses),
    )
    data = simulation.dpred(sigma)

    # SimPEG returns interleaved real/imag for each frequency
    n_freq = len(frequencies)
    real_parts = data[0::2].tolist() if len(data) >= 2 * n_freq else data.tolist()
    imag_parts = data[1::2].tolist() if len(data) >= 2 * n_freq else [0.0] * n_freq

    return {
        'frequencies': frequencies,
        'real': real_parts,
        'imag': imag_parts,
        'backend': 'simpeg',
    }
