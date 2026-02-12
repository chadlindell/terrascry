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

    @property
    def magnetic_sources(self) -> list[dict]:
        """Return all objects with magnetic moments as dipole source dicts."""
        sources = []
        for obj in self.objects:
            if obj.moment is not None:
                sources.append(obj.as_dipole_source())
        return sources

    def compute_induced_moments(self) -> None:
        """Compute induced dipole moments for objects without explicit moments.

        Uses the effective sphere approximation with the scenario's Earth field.
        Objects with existing moments are not modified.
        """
        from geosim.magnetics.dipole import MU_0

        B_earth_mag = np.linalg.norm(self.earth_field)

        for obj in self.objects:
            if obj.moment is not None:
                continue
            if obj.susceptibility <= 0 or obj.radius <= 0:
                continue

            volume = (4.0 / 3.0) * np.pi * obj.radius ** 3
            chi = obj.susceptibility
            effective_chi = 3.0 * chi / (chi + 3.0)
            M = effective_chi * B_earth_mag / MU_0
            m_magnitude = volume * M

            # Moment aligned with Earth field direction
            B_hat = self.earth_field / B_earth_mag
            obj.moment = m_magnitude * B_hat


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

    scenario = Scenario(
        name=data.get('name', path.stem),
        description=data.get('description', ''),
        terrain=terrain,
        objects=objects,
        earth_field=earth_field,
        metadata=data.get('metadata', {}),
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

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
