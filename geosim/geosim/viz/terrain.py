"""Terrain and subsurface mesh generation for PyVista."""

from __future__ import annotations

import numpy as np
import pyvista as pv

from geosim.scenarios.loader import Scenario


def create_terrain_mesh(
    scenario: Scenario,
    resolution: int = 50,
    opacity: float = 0.6,
) -> pv.StructuredGrid:
    """Create a terrain surface mesh from a scenario.

    Parameters
    ----------
    scenario : Scenario
        Loaded scenario with terrain extents.
    resolution : int
        Grid points per axis.
    opacity : float
        Surface opacity (0-1).

    Returns
    -------
    mesh : pv.StructuredGrid
        Terrain surface mesh with color attributes.
    """
    x_min, x_max = scenario.terrain.x_extent
    y_min, y_max = scenario.terrain.y_extent
    z_surface = scenario.terrain.surface_elevation

    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.full_like(X, z_surface)

    grid = pv.StructuredGrid(X, Y, Z)
    grid["elevation"] = Z.ravel()
    return grid


def create_subsurface_slice(
    scenario: Scenario,
    y_position: float | None = None,
    resolution: int = 100,
    depth: float = 5.0,
) -> pv.StructuredGrid:
    """Create a vertical cross-section slice showing soil layers.

    Parameters
    ----------
    scenario : Scenario
        Loaded scenario with terrain and soil layers.
    y_position : float, optional
        Y-coordinate for the slice. Defaults to center.
    resolution : int
        Grid points per axis.
    depth : float
        Maximum depth below surface to show (positive value).

    Returns
    -------
    mesh : pv.StructuredGrid
        Vertical slice with conductivity coloring.
    """
    x_min, x_max = scenario.terrain.x_extent
    z_surface = scenario.terrain.surface_elevation

    if y_position is None:
        y_min, y_max = scenario.terrain.y_extent
        y_position = (y_min + y_max) / 2

    x = np.linspace(x_min, x_max, resolution)
    z = np.linspace(z_surface - depth, z_surface, resolution)
    X, Z = np.meshgrid(x, z)
    Y = np.full_like(X, y_position)

    grid = pv.StructuredGrid(X, Y, Z)

    # Assign conductivity based on soil layers
    conductivity = np.full(X.size, 0.01)  # default background
    z_flat = Z.ravel()

    for layer in scenario.terrain.layers:
        mask = (z_flat >= layer.z_bottom) & (z_flat <= layer.z_top)
        conductivity[mask] = layer.conductivity

    grid["conductivity"] = conductivity
    return grid


def create_soil_layers(
    scenario: Scenario,
    opacity: float = 0.3,
) -> list[pv.PolyData]:
    """Create transparent box meshes for each soil layer.

    Parameters
    ----------
    scenario : Scenario
        Loaded scenario.
    opacity : float
        Layer opacity.

    Returns
    -------
    layers : list of pv.PolyData
        Box meshes for each soil layer.
    """
    x_min, x_max = scenario.terrain.x_extent
    y_min, y_max = scenario.terrain.y_extent

    layers = []
    for soil_layer in scenario.terrain.layers:
        box = pv.Box(bounds=(
            x_min, x_max,
            y_min, y_max,
            soil_layer.z_bottom, soil_layer.z_top,
        ))
        box.cell_data["layer_name"] = [soil_layer.name] * box.n_cells
        box.cell_data["conductivity"] = [soil_layer.conductivity] * box.n_cells
        layers.append(box)

    return layers
