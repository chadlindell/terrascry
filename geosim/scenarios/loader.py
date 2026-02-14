"""Scenario loading and ground truth management.

Scenarios are JSON files defining the ground truth: terrain, buried objects,
soil properties, and instrument configuration. They are the single source
of truth for all physics calculations and visualizations.

Coordinate convention:
    Right-handed: X=East, Y=North, Z=Up
    Origin: scenario-defined (typically SW corner at surface)
    All positions in meters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class BuriedObject:
    """A buried object (target) in the scenario."""

    name: str
    position: np.ndarray  # [x, y, z] in meters (z negative = below surface)
    object_type: str  # 'ferrous_sphere', 'ferrous_cylinder', 'void', 'fill', etc.
    moment: np.ndarray | None = None  # magnetic dipole moment [mx, my, mz] in A·m²
    radius: float = 0.0  # effective radius in meters
    dimensions: dict[str, float] = field(default_factory=dict)  # for non-sphere shapes
    susceptibility: float = 0.0  # magnetic susceptibility (SI)
    conductivity: float = 0.0  # electrical conductivity in S/m
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dipole_source(self) -> dict:
        """Convert to dict for dipole field calculation."""
        if self.moment is not None:
            return {'position': self.position, 'moment': self.moment}
        raise ValueError(
            f"Object '{self.name}' has no magnetic moment defined. "
            "Set moment directly or use Scenario.compute_induced_moments()."
        )


@dataclass
class SoilLayer:
    """A horizontal soil layer."""

    name: str
    z_top: float  # top surface z-coordinate (meters, negative = below ground)
    z_bottom: float  # bottom surface z-coordinate
    conductivity: float = 0.01  # S/m
    relative_permittivity: float = 10.0
    susceptibility: float = 0.0
    color: str = "#8B7355"  # for visualization


@dataclass
class AnomalyZone:
    """A volumetric anomaly zone (e.g., crater fill, grave shaft).

    Represents a region where soil properties differ from the surrounding
    layers. Used by HIRT EM and ERT models.

    Parameters
    ----------
    name : str
        Human-readable name.
    center : list[float]
        Center position [x, y, z] in meters.
    dimensions : dict
        Shape dimensions. Keys depend on shape:
        - 'sphere': {'radius': float}
        - 'cylinder': {'radius': float, 'height': float}
        - 'box': {'length': float, 'width': float, 'depth': float}
    shape : str
        Geometry type: 'sphere', 'cylinder', 'box'.
    conductivity : float
        Electrical conductivity in S/m.
    resistivity : float
        Electrical resistivity in Ω·m (inverse of conductivity).
    relative_permittivity : float
        Relative dielectric permittivity.
    susceptibility : float
        Magnetic susceptibility (SI).
    """

    name: str = ""
    center: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    dimensions: dict[str, float] = field(default_factory=dict)
    shape: str = "box"
    conductivity: float = 0.0
    resistivity: float = 0.0
    relative_permittivity: float = 10.0
    susceptibility: float = 0.0


@dataclass
class ProbeConfig:
    """HIRT probe position and geometry.

    Parameters
    ----------
    position : list[float]
        Probe insertion point [x, y, z] in meters.
    length : float
        Probe length in meters.
    orientation : str
        'vertical' or 'angled'.
    ring_depths : list[float]
        Depths of ring electrodes relative to probe top.
    coil_depths : list[float]
        Depths of EM coils relative to probe top.
    """

    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    length: float = 1.0
    orientation: str = "vertical"
    ring_depths: list[float] = field(default_factory=list)
    coil_depths: list[float] = field(default_factory=list)


@dataclass
class HIRTConfig:
    """HIRT instrument configuration.

    Parameters
    ----------
    probes : list[ProbeConfig]
        Probe positions and configurations.
    frequencies : list[float]
        FDEM operating frequencies in Hz.
    injection_current : float
        ERT injection current in Amps.
    array_type : str
        Electrode array type: 'crosshole', 'wenner', 'dipole-dipole'.
    """

    probes: list[ProbeConfig] = field(default_factory=list)
    frequencies: list[float] = field(default_factory=lambda: [1000.0, 5000.0, 25000.0])
    injection_current: float = 0.01  # 10 mA
    array_type: str = "crosshole"


@dataclass
class Terrain:
    """Terrain definition."""

    x_extent: tuple[float, float] = (0.0, 20.0)  # (min, max) in meters
    y_extent: tuple[float, float] = (0.0, 20.0)
    surface_elevation: float = 0.0  # flat terrain default
    layers: list[SoilLayer] = field(default_factory=list)


@dataclass
class Scenario:
    """Complete scenario: terrain + buried objects + metadata."""

    name: str
    description: str
    terrain: Terrain
    objects: list[BuriedObject]
    earth_field: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 20e-6, 45e-6])
    )  # [Bx, By, Bz] in Tesla (mid-latitude default: ~50 μT, 65° inclination)
    metadata: dict[str, Any] = field(default_factory=dict)
    hirt_config: HIRTConfig | None = None
    anomaly_zones: list[AnomalyZone] = field(default_factory=list)

    @property
    def magnetic_sources(self) -> list[dict]:
        """Return all objects with magnetic moments as dipole source dicts."""
        sources = []
        for obj in self.objects:
            if obj.moment is not None:
                sources.append(obj.as_dipole_source())
        return sources

    @property
    def em_sources(self) -> list[dict]:
        """Return objects with EM-relevant properties (conductivity > 0).

        These are objects that would produce a secondary field response
        in FDEM measurements.
        """
        sources = []
        for obj in self.objects:
            if obj.conductivity > 0 and obj.radius > 0:
                sources.append({
                    'position': obj.position,
                    'radius': obj.radius,
                    'conductivity': obj.conductivity,
                    'susceptibility': obj.susceptibility,
                    'name': obj.name,
                })
        return sources

    @property
    def resistivity_model(self) -> dict:
        """Return a 1D resistivity model from terrain layers.

        Returns a dict with 'thicknesses' and 'resistivities' suitable
        for analytical 1D forward models.
        """
        layers = self.terrain.layers
        if not layers:
            return {'thicknesses': [], 'resistivities': [100.0]}

        thicknesses = []
        resistivities = []
        for i, layer in enumerate(layers):
            rho = 1.0 / layer.conductivity if layer.conductivity > 0 else 1e6
            resistivities.append(rho)
            if i < len(layers) - 1:
                thicknesses.append(abs(layer.z_top - layer.z_bottom))

        return {'thicknesses': thicknesses, 'resistivities': resistivities}

    def build_conductivity_model(self) -> dict:
        """Build a conductivity model including anomaly zones.

        Returns a dict with:
        - 'background': layer conductivities from terrain
        - 'anomalies': list of anomaly zone dicts with geometry and properties
        """
        background = []
        for layer in self.terrain.layers:
            background.append({
                'z_top': layer.z_top,
                'z_bottom': layer.z_bottom,
                'conductivity': layer.conductivity,
            })

        anomalies = []
        for zone in self.anomaly_zones:
            anomalies.append({
                'name': zone.name,
                'center': zone.center,
                'dimensions': zone.dimensions,
                'shape': zone.shape,
                'conductivity': zone.conductivity,
            })

        return {'background': background, 'anomalies': anomalies}

    def compute_induced_moments(self) -> None:
        """Compute induced dipole moments and combine with remanence.

        Uses the effective sphere approximation with the scenario's Earth field.
        If an object has remanence data, the remanent moment is added to the
        induced moment via superposition. Objects with explicit ``moment``
        already set (and no remanence) are not modified.
        """
        from geosim.magnetics.dipole import MU_0, combined_moment, remanent_moment

        B_earth_mag = np.linalg.norm(self.earth_field)

        for obj in self.objects:
            # Compute remanent moment if remanence data is present
            m_remanent = None
            rem = obj.metadata.get("remanence") if obj.metadata else None
            rem_direct = obj.metadata.get("remanent_moment") if obj.metadata else None

            if rem_direct is not None:
                # Direct remanent moment vector [mx, my, mz] in A·m²
                m_remanent = np.array(rem_direct, dtype=np.float64)
            elif rem is not None:
                # Computed from direction + intensity
                direction = np.array(rem["direction"], dtype=np.float64)
                intensity = rem["intensity"]  # A/m
                volume = (4.0 / 3.0) * np.pi * obj.radius ** 3
                m_remanent = remanent_moment(volume, direction, intensity)

            # If object already has an explicit moment, add remanence if present
            if obj.moment is not None:
                if m_remanent is not None:
                    obj.moment = combined_moment(obj.moment, m_remanent)
                continue

            # Compute induced moment for susceptible objects
            if obj.susceptibility <= 0 or obj.radius <= 0:
                if m_remanent is not None:
                    obj.moment = m_remanent
                continue

            volume = (4.0 / 3.0) * np.pi * obj.radius ** 3
            chi = obj.susceptibility
            effective_chi = 3.0 * chi / (chi + 3.0)
            M = effective_chi * B_earth_mag / MU_0
            m_magnitude = volume * M

            # Moment aligned with Earth field direction
            B_hat = self.earth_field / B_earth_mag
            m_induced = m_magnitude * B_hat

            # Combine with remanence if present
            if m_remanent is not None:
                obj.moment = combined_moment(m_induced, m_remanent)
            else:
                obj.moment = m_induced


def load_scenario(path: str | Path) -> Scenario:
    """Load a scenario from a JSON file.

    Parameters
    ----------
    path : str or Path
        Path to scenario JSON file.

    Returns
    -------
    scenario : Scenario
        Loaded and validated scenario.
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    # Parse terrain
    terrain_data = data.get('terrain', {})
    layers = []
    for layer_data in terrain_data.get('layers', []):
        layers.append(SoilLayer(
            name=layer_data['name'],
            z_top=layer_data['z_top'],
            z_bottom=layer_data['z_bottom'],
            conductivity=layer_data.get('conductivity', 0.01),
            relative_permittivity=layer_data.get('relative_permittivity', 10.0),
            susceptibility=layer_data.get('susceptibility', 0.0),
            color=layer_data.get('color', '#8B7355'),
        ))

    terrain = Terrain(
        x_extent=tuple(terrain_data.get('x_extent', [0.0, 20.0])),
        y_extent=tuple(terrain_data.get('y_extent', [0.0, 20.0])),
        surface_elevation=terrain_data.get('surface_elevation', 0.0),
        layers=layers,
    )

    # Parse buried objects
    objects = []
    for obj_data in data.get('objects', []):
        position = np.array(obj_data['position'], dtype=np.float64)
        moment = None
        if 'moment' in obj_data:
            moment = np.array(obj_data['moment'], dtype=np.float64)

        objects.append(BuriedObject(
            name=obj_data['name'],
            position=position,
            object_type=obj_data.get('type', 'ferrous_sphere'),
            moment=moment,
            radius=obj_data.get('radius', 0.0),
            dimensions=obj_data.get('dimensions', {}),
            susceptibility=obj_data.get('susceptibility', 0.0),
            conductivity=obj_data.get('conductivity', 0.0),
            metadata=obj_data.get('metadata', {}),
        ))

    # Parse Earth field
    earth_field_data = data.get('earth_field', [0.0, 20e-6, 45e-6])
    earth_field = np.array(earth_field_data, dtype=np.float64)

    # Parse anomaly zones (optional)
    anomaly_zones = []
    for az_data in data.get('anomaly_zones', []):
        anomaly_zones.append(AnomalyZone(
            name=az_data.get('name', ''),
            center=az_data.get('center', [0.0, 0.0, 0.0]),
            dimensions=az_data.get('dimensions', {}),
            shape=az_data.get('shape', 'box'),
            conductivity=az_data.get('conductivity', 0.0),
            resistivity=az_data.get('resistivity', 0.0),
            relative_permittivity=az_data.get('relative_permittivity', 10.0),
            susceptibility=az_data.get('susceptibility', 0.0),
        ))

    # Parse HIRT config (optional)
    hirt_config = None
    hirt_data = data.get('hirt_config')
    if hirt_data:
        probes = []
        for p_data in hirt_data.get('probes', []):
            probes.append(ProbeConfig(
                position=p_data.get('position', [0.0, 0.0, 0.0]),
                length=p_data.get('length', 1.0),
                orientation=p_data.get('orientation', 'vertical'),
                ring_depths=p_data.get('ring_depths', []),
                coil_depths=p_data.get('coil_depths', []),
            ))
        hirt_config = HIRTConfig(
            probes=probes,
            frequencies=hirt_data.get('frequencies', [1000.0, 5000.0, 25000.0]),
            injection_current=hirt_data.get('injection_current', 0.01),
            array_type=hirt_data.get('array_type', 'crosshole'),
        )

    scenario = Scenario(
        name=data.get('name', path.stem),
        description=data.get('description', ''),
        terrain=terrain,
        objects=objects,
        earth_field=earth_field,
        metadata=data.get('metadata', {}),
        hirt_config=hirt_config,
        anomaly_zones=anomaly_zones,
    )

    # Auto-compute induced moments for objects that don't have explicit ones
    scenario.compute_induced_moments()

    return scenario


def save_scenario(scenario: Scenario, path: str | Path) -> None:
    """Save a scenario to JSON.

    Parameters
    ----------
    scenario : Scenario
        Scenario to save.
    path : str or Path
        Output path for JSON file.
    """
    data = {
        'name': scenario.name,
        'description': scenario.description,
        'earth_field': scenario.earth_field.tolist(),
        'terrain': {
            'x_extent': list(scenario.terrain.x_extent),
            'y_extent': list(scenario.terrain.y_extent),
            'surface_elevation': scenario.terrain.surface_elevation,
            'layers': [
                {
                    'name': layer.name,
                    'z_top': layer.z_top,
                    'z_bottom': layer.z_bottom,
                    'conductivity': layer.conductivity,
                    'relative_permittivity': layer.relative_permittivity,
                    'susceptibility': layer.susceptibility,
                    'color': layer.color,
                }
                for layer in scenario.terrain.layers
            ],
        },
        'objects': [
            {
                'name': obj.name,
                'position': obj.position.tolist(),
                'type': obj.object_type,
                'radius': obj.radius,
                'susceptibility': obj.susceptibility,
                'conductivity': obj.conductivity,
                **(
                    {'moment': obj.moment.tolist()} if obj.moment is not None else {}
                ),
                **(
                    {'dimensions': obj.dimensions} if obj.dimensions else {}
                ),
                **(
                    {'metadata': obj.metadata} if obj.metadata else {}
                ),
            }
            for obj in scenario.objects
        ],
        'metadata': scenario.metadata,
    }

    # Serialize anomaly zones if present
    if scenario.anomaly_zones:
        data['anomaly_zones'] = [
            {
                'name': az.name,
                'center': az.center,
                'dimensions': az.dimensions,
                'shape': az.shape,
                'conductivity': az.conductivity,
                'resistivity': az.resistivity,
                'relative_permittivity': az.relative_permittivity,
                'susceptibility': az.susceptibility,
            }
            for az in scenario.anomaly_zones
        ]

    # Serialize HIRT config if present
    if scenario.hirt_config:
        hc = scenario.hirt_config
        data['hirt_config'] = {
            'frequencies': hc.frequencies,
            'injection_current': hc.injection_current,
            'array_type': hc.array_type,
            'probes': [
                {
                    'position': p.position,
                    'length': p.length,
                    'orientation': p.orientation,
                    'ring_depths': p.ring_depths,
                    'coil_depths': p.coil_depths,
                }
                for p in hc.probes
            ],
        }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
