"""High-level scene composition for PyVista.

Assembles terrain, objects, fields, and sensor paths into complete
3D scenes. Supports both interactive display and static image export.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv

from geosim.scenarios.loader import Scenario
from geosim.viz.terrain import create_terrain_mesh, create_soil_layers
from geosim.viz.objects import (
    create_buried_objects,
    create_sensor_path,
    create_sensor_positions,
    create_moment_arrows,
)
from geosim.viz.fields import create_gradient_surface


def create_survey_scene(
    scenario: Scenario,
    positions: np.ndarray | None = None,
    gradient_data: np.ndarray | None = None,
    show_objects: bool = True,
    show_terrain: bool = True,
    show_gradient_map: bool = True,
    show_soil_layers: bool = False,
    show_moment_arrows: bool = True,
    object_scale: float = 3.0,
    off_screen: bool = False,
) -> pv.Plotter:
    """Create a complete survey visualization scene.

    Parameters
    ----------
    scenario : Scenario
        Loaded scenario.
    positions : ndarray, shape (N, 2), optional
        Walk path positions for sensor track.
    gradient_data : ndarray, shape (N,), optional
        Gradient readings along the walk path (ADC counts).
    show_objects : bool
        Render buried objects as spheres.
    show_terrain : bool
        Render terrain surface.
    show_gradient_map : bool
        Render gradient anomaly map at sensor height.
    show_soil_layers : bool
        Render transparent soil layer boxes.
    show_moment_arrows : bool
        Render dipole moment direction arrows.
    object_scale : float
        Scale factor for buried object visualization size.
    off_screen : bool
        If True, render off-screen for image export.

    Returns
    -------
    plotter : pv.Plotter
        Configured PyVista plotter ready for .show() or .screenshot().
    """
    plotter = pv.Plotter(off_screen=off_screen)
    plotter.set_background("white")

    sources = scenario.magnetic_sources

    # Terrain surface
    if show_terrain:
        terrain = create_terrain_mesh(scenario)
        plotter.add_mesh(
            terrain, color="#8B7355", opacity=0.4,
            label="Ground surface",
        )

    # Soil layers
    if show_soil_layers:
        layers = create_soil_layers(scenario, opacity=0.15)
        for layer_mesh in layers:
            plotter.add_mesh(layer_mesh, opacity=0.15, color="#6B5B4E")

    # Gradient anomaly map
    if show_gradient_map and sources:
        x_range = scenario.terrain.x_extent
        y_range = scenario.terrain.y_extent
        grad_surface = create_gradient_surface(
            sources, x_range, y_range, resolution=80,
        )
        vmax = np.percentile(np.abs(grad_surface["gradient_nT"]), 98)
        if vmax > 0:
            plotter.add_mesh(
                grad_surface,
                scalars="gradient_nT",
                cmap="RdBu_r",
                clim=[-vmax, vmax],
                opacity=0.8,
                scalar_bar_args={"title": "Gradient (nT)", "n_labels": 5},
            )

    # Buried objects
    if show_objects:
        objects = create_buried_objects(scenario, scale_factor=object_scale)
        for mesh, props in objects:
            plotter.add_mesh(
                mesh,
                color="red",
                opacity=0.7,
                label=f"{props['name']} ({props['depth']:.1f}m)",
            )

    # Moment arrows
    if show_moment_arrows:
        arrows = create_moment_arrows(scenario, scale=0.5)
        if arrows.n_points > 0:
            plotter.add_mesh(arrows, color="yellow", label="Dipole moments")

    # Sensor walk path
    if positions is not None:
        if gradient_data is not None:
            points_mesh = create_sensor_positions(
                positions, gradient_data.astype(float), height=0.25
            )
            plotter.add_mesh(
                points_mesh,
                scalars="gradient",
                cmap="RdBu_r",
                point_size=3,
                render_points_as_spheres=True,
                label="Sensor readings",
            )
        else:
            path_tube = create_sensor_path(positions, height=0.25, tube_radius=0.03)
            plotter.add_mesh(path_tube, color="blue", opacity=0.6, label="Walk path")

    # Camera and labels
    plotter.add_axes()

    return plotter


def create_field_scene(
    sources: list[dict],
    bounds: tuple[float, float, float, float, float, float] | None = None,
    show_isosurfaces: bool = True,
    show_arrows: bool = False,
    n_isosurfaces: int = 5,
    off_screen: bool = False,
) -> pv.Plotter:
    """Create a 3D magnetic field visualization scene.

    Parameters
    ----------
    sources : list of dict
        Dipole sources.
    bounds : tuple, optional
        (x_min, x_max, y_min, y_max, z_min, z_max).
        Defaults to ±3m around origin.
    show_isosurfaces : bool
        Render field magnitude isosurfaces.
    show_arrows : bool
        Render field direction arrows.
    n_isosurfaces : int
        Number of isosurface levels.
    off_screen : bool
        If True, render off-screen.

    Returns
    -------
    plotter : pv.Plotter
    """
    from geosim.viz.fields import create_field_volume, create_isosurfaces, create_field_arrows

    if bounds is None:
        bounds = (-3, 3, -3, 3, -3, 3)

    plotter = pv.Plotter(off_screen=off_screen)
    plotter.set_background("white")

    # Create field volume
    volume = create_field_volume(sources, bounds, resolution=30)

    if show_isosurfaces:
        surfaces = create_isosurfaces(volume, n_levels=n_isosurfaces)
        opacities = np.linspace(0.5, 0.1, len(surfaces))
        for surf, opacity in zip(surfaces, opacities):
            plotter.add_mesh(
                surf, scalars="field", cmap="hot",
                opacity=float(opacity), smooth_shading=True,
            )

    if show_arrows:
        arrows = create_field_arrows(sources, bounds, resolution=6, scale=0.05)
        if arrows.n_points > 0:
            plotter.add_mesh(
                arrows, scalars="B_magnitude", cmap="coolwarm",
            )

    # Mark source positions
    for src in sources:
        pos = np.array(src["position"])
        sphere = pv.Sphere(radius=0.1, center=pos.tolist())
        plotter.add_mesh(sphere, color="red", opacity=0.9)

    plotter.add_axes()
    return plotter


def render_to_image(
    plotter: pv.Plotter,
    output_path: str | Path,
    window_size: tuple[int, int] = (1920, 1080),
) -> Path:
    """Render a plotter scene to a static image file.

    Parameters
    ----------
    plotter : pv.Plotter
        Configured plotter (should be created with off_screen=True).
    output_path : str or Path
        Output image path (.png, .jpg, .pdf).
    window_size : tuple
        Image dimensions in pixels.

    Returns
    -------
    path : Path
        Path to the saved image.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plotter.window_size = window_size
    plotter.screenshot(str(output_path))
    return output_path
