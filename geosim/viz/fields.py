"""Magnetic field visualization for PyVista.

Creates volumetric field data, isosurfaces, and gradient maps
from the dipole physics engine.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv

from geosim.magnetics.dipole import dipole_field, superposition_field, gradiometer_reading


def create_field_volume(
    sources: list[dict],
    bounds: tuple[float, float, float, float, float, float],
    resolution: int = 30,
    component: str = "magnitude",
) -> pv.ImageData:
    """Create a 3D volume of magnetic field values.

    Parameters
    ----------
    sources : list of dict
        Dipole sources (position + moment).
    bounds : tuple
        (x_min, x_max, y_min, y_max, z_min, z_max) in meters.
    resolution : int
        Grid points per axis.
    component : str
        "magnitude", "Bx", "By", "Bz", or "gradient_z".

    Returns
    -------
    volume : pv.ImageData
        Uniform grid with field values.
    """
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    z = np.linspace(z_min, z_max, resolution)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    B = superposition_field(points, sources)

    # Create uniform grid
    grid = pv.ImageData(
        dimensions=(resolution, resolution, resolution),
        spacing=(
            (x_max - x_min) / (resolution - 1),
            (y_max - y_min) / (resolution - 1),
            (z_max - z_min) / (resolution - 1),
        ),
        origin=(x_min, y_min, z_min),
    )

    if component == "magnitude":
        grid["field"] = np.linalg.norm(B, axis=1) * 1e9  # nT
    elif component == "Bx":
        grid["field"] = B[:, 0] * 1e9
    elif component == "By":
        grid["field"] = B[:, 1] * 1e9
    elif component == "Bz":
        grid["field"] = B[:, 2] * 1e9
    else:
        grid["field"] = np.linalg.norm(B, axis=1) * 1e9

    # Also store the vector field for arrow glyphs
    grid["B_vector"] = B * 1e9  # nT

    return grid


def create_gradient_surface(
    sources: list[dict],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    sensor_height: float = 0.175,
    sensor_separation: float = 0.35,
    resolution: int = 80,
) -> pv.StructuredGrid:
    """Create a 2D surface colored by vertical gradient values.

    Simulates the gradient map a Pathfinder survey would produce,
    rendered as a colored surface at sensor height.

    Parameters
    ----------
    sources : list of dict
        Dipole sources.
    x_range, y_range : tuple
        (min, max) extents in meters.
    sensor_height : float
        Bottom sensor height above z=0 (meters).
    sensor_separation : float
        Vertical separation between sensors (meters).
    resolution : int
        Grid points per axis.

    Returns
    -------
    surface : pv.StructuredGrid
        Surface mesh with gradient coloring.
    """
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)

    points_bottom = np.column_stack([
        X.ravel(), Y.ravel(),
        np.full(X.size, sensor_height),
    ])

    B_bot, B_top, grad = gradiometer_reading(
        points_bottom, sources, sensor_separation, component=2
    )

    # Create surface mesh slightly above ground
    Z = np.full_like(X, sensor_height)
    grid = pv.StructuredGrid(X, Y, Z)
    grid["gradient_nT"] = grad * 1e9
    grid["gradient_abs"] = np.abs(grad) * 1e9

    return grid


def create_field_arrows(
    sources: list[dict],
    bounds: tuple[float, float, float, float, float, float],
    resolution: int = 8,
    scale: float = 1.0,
) -> pv.PolyData:
    """Create arrow glyphs showing magnetic field direction.

    Parameters
    ----------
    sources : list of dict
        Dipole sources.
    bounds : tuple
        (x_min, x_max, y_min, y_max, z_min, z_max).
    resolution : int
        Grid points per axis (keep low for readability).
    scale : float
        Arrow scale factor.

    Returns
    -------
    arrows : pv.PolyData
        Arrow glyph mesh.
    """
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    z = np.linspace(z_min, z_max, resolution)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    B = superposition_field(points, sources)
    B_mag = np.linalg.norm(B, axis=1, keepdims=True)

    # Normalize for direction, scale by log magnitude for visibility
    B_safe = np.where(B_mag > 0, B_mag, 1.0)
    B_hat = B / B_safe

    cloud = pv.PolyData(points)
    cloud["B_direction"] = B_hat
    cloud["B_magnitude"] = B_mag.ravel() * 1e9  # nT

    arrows = cloud.glyph(
        orient="B_direction",
        scale="B_magnitude",
        factor=scale,
    )
    return arrows


def create_isosurfaces(
    volume: pv.ImageData,
    levels: list[float] | None = None,
    n_levels: int = 5,
) -> list[pv.PolyData]:
    """Extract isosurfaces from a field volume.

    Parameters
    ----------
    volume : pv.ImageData
        Volume with "field" data array.
    levels : list of float, optional
        Specific isosurface values (in nT).
    n_levels : int
        Number of auto-generated levels if levels is None.

    Returns
    -------
    surfaces : list of pv.PolyData
        Isosurface meshes.
    """
    field = volume["field"]
    field_range = field[np.isfinite(field)]
    if len(field_range) == 0:
        return []

    if levels is None:
        f_min = np.percentile(field_range[field_range > 0], 10)
        f_max = np.percentile(field_range, 95)
        if f_min >= f_max:
            return []
        levels = np.logspace(np.log10(f_min), np.log10(f_max), n_levels).tolist()

    surfaces = []
    for level in levels:
        try:
            iso = volume.contour([level], scalars="field")
            if iso.n_points > 0:
                surfaces.append(iso)
        except Exception:
            continue

    return surfaces
