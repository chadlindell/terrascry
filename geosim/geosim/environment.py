"""Environmental soil model with Archie's Law conductivity.

Provides runtime-adjustable soil conditions (temperature, saturation, frost)
that modify effective electrical conductivity through Archie's Law with
clay correction. Used by the physics server to make EM/ERT responses
sensitive to ground conditions.

Physics reference:
    Archie (1942): σ_b = σ_w(T) × φ^m × S_w^n / a  +  σ_surf
    Temperature correction: σ_w(T) = σ_w(T0) × [1 + 0.021 × (T - 25)]
    Frost: σ_b *= 0.05

Coordinate convention: same as rest of GeoSim (X=East, Y=North, Z=Up, SI units).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SoilEnvironment:
    """Runtime-adjustable ground conditions.

    Parameters
    ----------
    temperature_c : float
        Ground temperature in degrees Celsius.
    saturation : float
        Water saturation S_w (0-1).
    pore_water_sigma : float
        Pore water conductivity at 25°C in S/m.
    frozen : bool
        Whether the ground is frozen.
    """

    temperature_c: float = 15.0
    saturation: float = 0.6
    pore_water_sigma: float = 0.05
    frozen: bool = False

    def to_dict(self) -> dict:
        """Serialize for JSON/ZMQ transport."""
        return {
            "temperature_c": self.temperature_c,
            "saturation": self.saturation,
            "pore_water_sigma": self.pore_water_sigma,
            "frozen": self.frozen,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SoilEnvironment:
        """Deserialize from dict."""
        return cls(
            temperature_c=float(data.get("temperature_c", 15.0)),
            saturation=float(data.get("saturation", 0.6)),
            pore_water_sigma=float(data.get("pore_water_sigma", 0.05)),
            frozen=bool(data.get("frozen", False)),
        )


def archie_conductivity(
    porosity: float,
    archie_a: float,
    archie_m: float,
    archie_n: float,
    surface_conductivity: float,
    env: SoilEnvironment,
) -> float:
    """Compute effective bulk conductivity via Archie's Law with clay correction.

    Parameters
    ----------
    porosity : float
        Porosity φ (0-1).
    archie_a : float
        Tortuosity factor a.
    archie_m : float
        Cementation exponent m.
    archie_n : float
        Saturation exponent n.
    surface_conductivity : float
        Clay surface conductivity σ_surf in S/m.
    env : SoilEnvironment
        Current environmental conditions.

    Returns
    -------
    float
        Effective bulk conductivity σ_b in S/m.
    """
    # Temperature-corrected pore water conductivity
    sigma_w = env.pore_water_sigma * (1.0 + 0.021 * (env.temperature_c - 25.0))
    sigma_w = max(sigma_w, 0.0)

    # Archie's Law: σ_b = σ_w × φ^m × S_w^n / a  +  σ_surf
    s_w = max(min(env.saturation, 1.0), 0.0)
    phi = max(min(porosity, 1.0), 1e-6)

    if s_w < 1e-10:
        sigma_b = surface_conductivity
    else:
        sigma_b = sigma_w * (phi ** archie_m) * (s_w ** archie_n) / max(archie_a, 1e-6)
        sigma_b += surface_conductivity

    # Frost: conductivity drops dramatically
    if env.frozen:
        sigma_b *= 0.05

    return max(sigma_b, 1e-10)


# ---------------------------------------------------------------------------
# Soil type presets
# ---------------------------------------------------------------------------

SOIL_PRESETS: dict[str, dict] = {
    "sandy": {
        "porosity": 0.38, "archie_a": 1.0, "archie_m": 1.6,
        "archie_n": 2.0, "surface_conductivity": 0.005, "susceptibility": 5e-5,
    },
    "loam": {
        "porosity": 0.45, "archie_a": 1.2, "archie_m": 1.8,
        "archie_n": 2.0, "surface_conductivity": 0.03, "susceptibility": 2e-4,
    },
    "clay": {
        "porosity": 0.50, "archie_a": 1.5, "archie_m": 1.5,
        "archie_n": 2.5, "surface_conductivity": 0.15, "susceptibility": 3e-4,
    },
    "saturated_clay": {
        "porosity": 0.55, "archie_a": 1.5, "archie_m": 1.4,
        "archie_n": 2.0, "surface_conductivity": 0.5, "susceptibility": 5e-4,
    },
    "peat": {
        "porosity": 0.85, "archie_a": 2.0, "archie_m": 1.3,
        "archie_n": 2.0, "surface_conductivity": 0.08, "susceptibility": 1e-5,
    },
    "laterite": {
        "porosity": 0.40, "archie_a": 1.5, "archie_m": 1.8,
        "archie_n": 2.0, "surface_conductivity": 0.05, "susceptibility": 5e-3,
    },
    "volcanic": {
        "porosity": 0.30, "archie_a": 1.2, "archie_m": 2.0,
        "archie_n": 2.0, "surface_conductivity": 0.02, "susceptibility": 1e-2,
    },
    "burnt_soil": {
        "porosity": 0.50, "archie_a": 1.3, "archie_m": 1.7,
        "archie_n": 2.0, "surface_conductivity": 0.04, "susceptibility": 2e-3,
    },
}


def get_soil_preset(name: str) -> dict:
    """Return Archie parameters for a named soil type.

    Parameters
    ----------
    name : str
        One of: sandy, loam, clay, saturated_clay, peat, laterite, volcanic, burnt_soil.

    Returns
    -------
    dict
        Keys: porosity, archie_a, archie_m, archie_n, surface_conductivity, susceptibility.
    """
    if name not in SOIL_PRESETS:
        raise ValueError(f"Unknown soil preset: {name!r}. Available: {list(SOIL_PRESETS)}")
    return dict(SOIL_PRESETS[name])
