"""Magnetic dipole forward model.

Implements the analytical magnetic dipole field calculation for buried
ferrous objects. This is the core physics for Pathfinder gradiometer
simulation.

Physics:
    A magnetic dipole with moment m at position r_src produces a field
    at observation point r_obs:

        B(r) = μ₀/4π · [3(m·r̂)r̂ - m] / |r|³

    where r = r_obs - r_src, r̂ = r/|r|, and μ₀ = 4π×10⁻⁷ T·m/A.

    For a gradiometer with two vertically-separated sensors:
        gradient = B(lower) - B(upper)

Coordinate convention:
    Right-handed: X=East, Y=North, Z=Up
    All positions in meters, moments in A·m², fields in Tesla.
"""

import numpy as np

# Vacuum permeability (T·m/A)
MU_0 = 4.0 * np.pi * 1e-7

# Prefactor: μ₀ / (4π)
_PREFACTOR = MU_0 / (4.0 * np.pi)  # = 1e-7 T·m/A


def dipole_field(r_obs: np.ndarray, r_src: np.ndarray, moment: np.ndarray) -> np.ndarray:
    """Compute magnetic field of a single dipole at observation points.

    Parameters
    ----------
    r_obs : ndarray, shape (3,) or (N, 3)
        Observation point(s) in meters [x, y, z].
    r_src : ndarray, shape (3,)
        Dipole source position in meters [x, y, z].
    moment : ndarray, shape (3,)
        Magnetic dipole moment in A·m² [mx, my, mz].

    Returns
    -------
    B : ndarray, shape (3,) or (N, 3)
        Magnetic field in Tesla at each observation point.
    """
    r_obs = np.asarray(r_obs, dtype=np.float64)
    r_src = np.asarray(r_src, dtype=np.float64)
    moment = np.asarray(moment, dtype=np.float64)

    single = r_obs.ndim == 1
    if single:
        r_obs = r_obs[np.newaxis, :]

    # Displacement vectors: observation - source
    dr = r_obs - r_src  # (N, 3)
    r_mag = np.linalg.norm(dr, axis=1, keepdims=True)  # (N, 1)

    # Guard against evaluation at source location
    too_close = r_mag.ravel() < 1e-12
    r_mag_safe = np.where(r_mag < 1e-12, 1.0, r_mag)

    r_hat = dr / r_mag_safe  # (N, 3)

    # m · r̂ for each observation point
    m_dot_rhat = np.sum(moment * r_hat, axis=1, keepdims=True)  # (N, 1)

    # B = μ₀/(4π) · [3(m·r̂)r̂ - m] / r³
    r3 = r_mag_safe ** 3
    B = _PREFACTOR * (3.0 * m_dot_rhat * r_hat - moment) / r3  # (N, 3)

    # Zero out field at source location (physical singularity)
    B[too_close] = 0.0

    if single:
        return B[0]
    return B


def superposition_field(
    r_obs: np.ndarray,
    sources: list[dict],
) -> np.ndarray:
    """Compute total field from multiple dipole sources by superposition.

    Parameters
    ----------
    r_obs : ndarray, shape (3,) or (N, 3)
        Observation point(s) in meters.
    sources : list of dict
        Each dict has keys:
        - 'position': array-like, shape (3,) — source position [x, y, z] in meters
        - 'moment': array-like, shape (3,) — dipole moment [mx, my, mz] in A·m²

    Returns
    -------
    B_total : ndarray, shape (3,) or (N, 3)
        Total magnetic field in Tesla.
    """
    r_obs = np.asarray(r_obs, dtype=np.float64)
    single = r_obs.ndim == 1
    if single:
        r_obs = r_obs[np.newaxis, :]

    B_total = np.zeros_like(r_obs)
    for src in sources:
        r_src = np.asarray(src['position'], dtype=np.float64)
        moment = np.asarray(src['moment'], dtype=np.float64)
        B_total += dipole_field(r_obs, r_src, moment)

    if single:
        return B_total[0]
    return B_total


def dipole_field_gradient(
    r_obs: np.ndarray,
    r_src: np.ndarray,
    moment: np.ndarray,
    sensor_separation: float,
    component: int = 2,
) -> np.ndarray:
    """Compute vertical gradient of a dipole field component.

    Simulates a gradiometer by computing the difference between field
    values at two vertically-separated sensor positions.

    Parameters
    ----------
    r_obs : ndarray, shape (3,) or (N, 3)
        Position of the LOWER sensor in meters.
    r_src : ndarray, shape (3,)
        Dipole source position in meters.
    moment : ndarray, shape (3,)
        Dipole moment in A·m².
    sensor_separation : float
        Vertical distance between lower and upper sensors in meters.
        Upper sensor is at r_obs + [0, 0, sensor_separation].
    component : int, default 2
        Which field component to difference (0=Bx, 1=By, 2=Bz).

    Returns
    -------
    gradient : float or ndarray, shape (N,)
        Field gradient in T/m (lower minus upper, divided by separation).
        Positive gradient means field is stronger at the lower sensor.
    """
    r_obs = np.asarray(r_obs, dtype=np.float64)
    single = r_obs.ndim == 1
    if single:
        r_obs = r_obs[np.newaxis, :]

    # Upper sensor position
    r_upper = r_obs.copy()
    r_upper[:, 2] += sensor_separation

    B_lower = dipole_field(r_obs, r_src, moment)
    B_upper = dipole_field(r_upper, r_src, moment)

    # Gradient: (B_lower - B_upper) / separation
    gradient = (B_lower[:, component] - B_upper[:, component]) / sensor_separation

    if single:
        return gradient[0]
    return gradient


def gradiometer_reading(
    r_obs: np.ndarray,
    sources: list[dict],
    sensor_separation: float,
    component: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a gradiometer measurement over multiple dipole sources.

    Returns the field at both sensor positions and the gradient difference,
    matching the Pathfinder firmware output (bottom - top).

    Parameters
    ----------
    r_obs : ndarray, shape (3,) or (N, 3)
        Position of the LOWER (bottom) sensor in meters.
    sources : list of dict
        Dipole sources (see superposition_field).
    sensor_separation : float
        Vertical distance between bottom and top sensors in meters.
    component : int, default 2
        Field component (0=Bx, 1=By, 2=Bz).

    Returns
    -------
    B_bottom : float or ndarray
        Field at bottom sensor (component), in Tesla.
    B_top : float or ndarray
        Field at top sensor (component), in Tesla.
    gradient : float or ndarray
        B_bottom - B_top, in Tesla (not normalized by distance).
    """
    r_obs = np.asarray(r_obs, dtype=np.float64)
    single = r_obs.ndim == 1
    if single:
        r_obs = r_obs[np.newaxis, :]

    r_upper = r_obs.copy()
    r_upper[:, 2] += sensor_separation

    B_bot = superposition_field(r_obs, sources)
    B_top = superposition_field(r_upper, sources)

    b_bot = B_bot[:, component]
    b_top = B_top[:, component]
    grad = b_bot - b_top

    if single:
        return float(b_bot[0]), float(b_top[0]), float(grad[0])
    return b_bot, b_top, grad


def remanent_moment(
    volume: float,
    remanence_direction: np.ndarray,
    remanence_magnitude: float,
) -> np.ndarray:
    """Compute remanent dipole moment for a permanently magnetized object.

    Natural remanent magnetization (NRM) is retained from manufacturing,
    transport, or geological history. Unlike induced magnetization, it is
    independent of the current ambient field.

    Parameters
    ----------
    volume : float
        Object volume in m³.
    remanence_direction : ndarray, shape (3,)
        Unit vector giving the direction of remanent magnetization.
    remanence_magnitude : float
        Remanent magnetization intensity in A/m (NRM intensity).
        Typical values: mild steel 1-10 A/m, hardened steel 10-100 A/m.

    Returns
    -------
    moment : ndarray, shape (3,)
        Remanent dipole moment in A·m²: ``m_r = V · M_r · direction``.
    """
    direction = np.asarray(remanence_direction, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm < 1e-15:
        return np.zeros(3)
    direction = direction / norm
    return volume * remanence_magnitude * direction


def combined_moment(
    induced: np.ndarray,
    remanent: np.ndarray,
) -> np.ndarray:
    """Combine induced and remanent dipole moments by superposition.

    Parameters
    ----------
    induced : ndarray, shape (3,)
        Induced dipole moment in A·m² (aligned with Earth field).
    remanent : ndarray, shape (3,)
        Remanent dipole moment in A·m² (arbitrary direction).

    Returns
    -------
    total : ndarray, shape (3,)
        Combined moment: ``m_total = m_induced + m_remanent``.
    """
    return np.asarray(induced, dtype=np.float64) + np.asarray(remanent, dtype=np.float64)


def dipole_moment_from_sphere(
    radius: float,
    susceptibility: float,
    B_earth: float = 50e-6,
) -> np.ndarray:
    """Estimate effective dipole moment of a magnetized sphere.

    A uniformly magnetized sphere in an external field behaves as
    a magnetic dipole with moment:
        m = (4π/3) · r³ · (3χ)/(χ+3) · B_earth/μ₀

    For high susceptibility (ferrous objects, χ >> 3):
        m ≈ 4π r³ · B_earth/μ₀

    Parameters
    ----------
    radius : float
        Sphere radius in meters.
    susceptibility : float
        Magnetic susceptibility (dimensionless, SI).
        Typical: mild steel ~100-1000, iron ~5000.
    B_earth : float
        Ambient Earth field magnitude in Tesla.
        Default 50 μT (mid-latitude typical).

    Returns
    -------
    moment : ndarray, shape (3,)
        Dipole moment in A·m², oriented along Z (vertical).
    """
    volume = (4.0 / 3.0) * np.pi * radius ** 3
    # Demagnetization factor for sphere is 1/3
    effective_chi = 3.0 * susceptibility / (susceptibility + 3.0)
    M = effective_chi * B_earth / MU_0  # Magnetization in A/m
    m_magnitude = volume * M
    return np.array([0.0, 0.0, m_magnitude])


def detection_depth_estimate(
    moment_magnitude: float,
    noise_floor: float,
    sensor_separation: float = 0.35,
) -> float:
    """Estimate maximum detection depth for a dipole target.

    At distance r along the axis of a dipole, the gradient scales as:
        |dBz/dz| ~ μ₀ · 2m / (4π · r⁴)

    The gradiometer difference (over separation Δz) at distance r:
        ΔB ~ μ₀ · 2m · Δz / (4π · r⁴)  (for r >> Δz)

    Detection requires ΔB > noise_floor.

    Parameters
    ----------
    moment_magnitude : float
        Dipole moment magnitude in A·m².
    noise_floor : float
        Minimum detectable gradient difference in Tesla.
    sensor_separation : float
        Gradiometer baseline in meters.

    Returns
    -------
    depth : float
        Estimated maximum detection depth in meters.
    """
    # ΔB ≈ μ₀/(4π) · 2m · 4Δz / r⁵  (derivative of 1/r³ gives 3/r⁴,
    # but for axial dipole the z-gradient scales as ~1/r⁴)
    # Solve: noise = μ₀/(4π) · 8·m·Δz / r⁵
    numerator = _PREFACTOR * 8.0 * moment_magnitude * sensor_separation
    depth = (numerator / noise_floor) ** 0.2
    return depth
