"""Buried object and sensor path visualization for PyVista."""

from __future__ import annotations

import numpy as np
import pyvista as pv

from geosim.scenarios.loader import BuriedObject, Scenario


def create_buried_objects(
    scenario: Scenario,
    scale_factor: float = 1.0,
) -> list[tuple[pv.PolyData, dict]]:
    """Create sphere meshes for buried objects.

    Parameters
    ----------
    scenario : Scenario
        Loaded scenario with objects.
    scale_factor : float
        Visual scale multiplier for object radii (for visibility).

    Returns
    -------
    objects : list of (mesh, properties)
        Each entry is (pv.PolyData sphere, dict with name/moment/depth).
    """
    result = []
    for obj in scenario.objects:
        radius = max(obj.radius * scale_factor, 0.1)  # minimum visible size
        sphere = pv.Sphere(
            radius=radius,
            center=obj.position.tolist(),
            theta_resolution=20,
            phi_resolution=20,
        )

        moment_mag = float(np.linalg.norm(obj.moment)) if obj.moment is not None else 0.0
        sphere["moment"] = np.full(sphere.n_points, moment_mag)

        props = {
            "name": obj.name,
            "moment": moment_mag,
            "depth": abs(obj.position[2]),
            "radius": obj.radius,
            "type": obj.object_type,
        }
        result.append((sphere, props))

    return result


def create_sensor_path(
    positions: np.ndarray,
    values: np.ndarray | None = None,
    height: float = 0.175,
    tube_radius: float = 0.05,
) -> pv.PolyData:
    """Create a tube mesh along the sensor walk path.

    Parameters
    ----------
    positions : ndarray, shape (N, 2) or (N, 3)
        Sensor positions (x, y) or (x, y, z).
    values : ndarray, shape (N,), optional
        Scalar values to map as colors (e.g., gradient readings).
    height : float
        Z-height for the path if positions are 2D.
    tube_radius : float
        Tube radius for the path visualization.

    Returns
    -------
    path : pv.PolyData
        Tube mesh along the walk path.
    """
    n = len(positions)
    if positions.shape[1] == 2:
        points = np.column_stack([
            positions[:, 0],
            positions[:, 1],
            np.full(n, height),
        ])
    else:
        points = positions

    # Create polyline
    lines = np.zeros(n + 1, dtype=int)
    lines[0] = n
    lines[1:] = np.arange(n)
    path = pv.PolyData(points, lines=lines)

    if values is not None:
        path["readings"] = values

    # Create tube for visibility
    tube = path.tube(radius=tube_radius)
    if values is not None and "readings" in path.point_data:
        # Interpolate values to tube points
        tube = tube.interpolate(path, radius=tube_radius * 5)

    return tube


def create_sensor_positions(
    positions: np.ndarray,
    values: np.ndarray | None = None,
    height: float = 0.175,
    point_size: float = 0.08,
) -> pv.PolyData:
    """Create point cloud of sensor measurement positions.

    Parameters
    ----------
    positions : ndarray, shape (N, 2)
        Sensor (x, y) positions.
    values : ndarray, shape (N,), optional
        Scalar values (e.g., gradient ADC counts).
    height : float
        Z-height for points.
    point_size : float
        Glyph size.

    Returns
    -------
    points : pv.PolyData
        Point cloud with optional scalar values.
    """
    n = len(positions)
    points_3d = np.column_stack([
        positions[:, 0],
        positions[:, 1],
        np.full(n, height),
    ])

    cloud = pv.PolyData(points_3d)
    if values is not None:
        cloud["gradient"] = values

    return cloud


def create_moment_arrows(
    scenario: Scenario,
    scale: float = 2.0,
) -> pv.PolyData:
    """Create arrow glyphs showing dipole moment directions.

    Parameters
    ----------
    scenario : Scenario
        Loaded scenario.
    scale : float
        Arrow length scale factor.

    Returns
    -------
    arrows : pv.PolyData
        Arrow glyphs at each object location.
    """
    positions = []
    directions = []
    magnitudes = []

    for obj in scenario.objects:
        if obj.moment is not None and np.linalg.norm(obj.moment) > 0:
            positions.append(obj.position)
            m_hat = obj.moment / np.linalg.norm(obj.moment)
            directions.append(m_hat)
            magnitudes.append(np.linalg.norm(obj.moment))

    if not positions:
        return pv.PolyData()

    positions = np.array(positions)
    directions = np.array(directions)
    magnitudes = np.array(magnitudes)

    cloud = pv.PolyData(positions)
    cloud["vectors"] = directions * scale
    cloud["magnitude"] = magnitudes
    arrows = cloud.glyph(orient="vectors", scale=False, factor=scale)
    return arrows
