"""Coordinate transformation utilities for HIRT/Pathfinder integration.

Transforms between WGS84 (GPS lat/lon) and local survey grid (X/Y meters).
Used in the two-stage workflow: Pathfinder screens with GPS -> HIRT images
with local grid coordinates.

The local grid uses a tangent-plane approximation (valid for survey areas
< 10 km across, which covers all practical HIRT deployments).

Coordinate convention:
    Grid X = East, Grid Y = North (matching GeoSim convention).
    Rotation: bearing_deg is clockwise from true north to grid north.
    WGS84 coordinates in decimal degrees.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# WGS84 ellipsoid parameters
WGS84_A = 6378137.0  # Semi-major axis (m)
WGS84_F = 1 / 298.257223563  # Flattening
WGS84_B = WGS84_A * (1 - WGS84_F)  # Semi-minor axis
WGS84_E2 = 2 * WGS84_F - WGS84_F**2  # Eccentricity squared


@dataclass
class GridOrigin:
    """Local grid origin in WGS84 coordinates.

    Parameters
    ----------
    lat_deg : float
        Origin latitude in decimal degrees.
    lon_deg : float
        Origin longitude in decimal degrees.
    bearing_deg : float
        Grid north rotation from true north, in degrees clockwise.
        0.0 means grid north = true north.
    """

    lat_deg: float
    lon_deg: float
    bearing_deg: float = 0.0

    @property
    def lat_rad(self) -> float:
        """Origin latitude in radians."""
        return np.radians(self.lat_deg)

    @property
    def lon_rad(self) -> float:
        """Origin longitude in radians."""
        return np.radians(self.lon_deg)

    @property
    def bearing_rad(self) -> float:
        """Grid rotation in radians (clockwise from true north)."""
        return np.radians(self.bearing_deg)

    @property
    def meters_per_deg_lat(self) -> float:
        """Meters per degree latitude at origin (accounts for WGS84 ellipsoid).

        Uses the meridional radius of curvature M at the origin latitude:
            M = a(1-e^2) / (1-e^2 sin^2 phi)^(3/2)
        Then meters_per_degree = M * pi/180.
        """
        sin_lat = np.sin(self.lat_rad)
        return np.pi / 180 * WGS84_A * (1 - WGS84_E2) / (1 - WGS84_E2 * sin_lat**2) ** 1.5

    @property
    def meters_per_deg_lon(self) -> float:
        """Meters per degree longitude at origin.

        Uses the prime-vertical radius of curvature N at the origin latitude:
            N = a / sqrt(1 - e^2 sin^2 phi)
        Then meters_per_degree = N * cos(phi) * pi/180.
        """
        sin_lat = np.sin(self.lat_rad)
        return (
            np.pi / 180 * WGS84_A * np.cos(self.lat_rad) / np.sqrt(1 - WGS84_E2 * sin_lat**2)
        )


def gps_to_grid(
    lat: float | np.ndarray,
    lon: float | np.ndarray,
    origin: GridOrigin,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert WGS84 GPS coordinates to local grid meters.

    Uses a tangent-plane approximation centered on the grid origin,
    with optional rotation for grids not aligned with true north.

    Parameters
    ----------
    lat : float or ndarray
        Latitude(s) in decimal degrees.
    lon : float or ndarray
        Longitude(s) in decimal degrees.
    origin : GridOrigin
        Grid origin and rotation parameters.

    Returns
    -------
    x : ndarray
        Easting in meters (grid-east direction).
    y : ndarray
        Northing in meters (grid-north direction).

    Notes
    -----
    Accuracy is better than 1 mm for areas up to ~10 km across,
    which is well within the operating range of HIRT deployments.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    # Differences in degrees from origin
    dlat = lat - origin.lat_deg
    dlon = lon - origin.lon_deg

    # Convert to meters using local scale factors
    north = dlat * origin.meters_per_deg_lat
    east = dlon * origin.meters_per_deg_lon

    # Apply rotation if grid is not aligned with true north.
    # bearing_deg is clockwise from true north to grid north.
    # Rotation matrix for clockwise angle theta transforms
    # (east_true, north_true) -> (east_grid, north_grid):
    #   x_grid =  cos(theta) * east  + sin(theta) * north
    #   y_grid = -sin(theta) * east  + cos(theta) * north
    if origin.bearing_deg != 0.0:
        cos_b = np.cos(origin.bearing_rad)
        sin_b = np.sin(origin.bearing_rad)
        x = cos_b * east + sin_b * north
        y = -sin_b * east + cos_b * north
    else:
        x = east
        y = north

    return x, y


def grid_to_gps(
    x: float | np.ndarray,
    y: float | np.ndarray,
    origin: GridOrigin,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert local grid meters back to WGS84 GPS coordinates.

    Inverse of ``gps_to_grid``.

    Parameters
    ----------
    x : float or ndarray
        Easting in meters (grid-east direction).
    y : float or ndarray
        Northing in meters (grid-north direction).
    origin : GridOrigin
        Grid origin and rotation parameters.

    Returns
    -------
    lat : ndarray
        Latitude(s) in decimal degrees.
    lon : ndarray
        Longitude(s) in decimal degrees.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Undo rotation: inverse of clockwise rotation by theta is
    # clockwise rotation by -theta.
    #   east_true  =  cos(theta) * x_grid - sin(theta) * y_grid
    #   north_true =  sin(theta) * x_grid + cos(theta) * y_grid
    if origin.bearing_deg != 0.0:
        cos_b = np.cos(origin.bearing_rad)
        sin_b = np.sin(origin.bearing_rad)
        east = cos_b * x - sin_b * y
        north = sin_b * x + cos_b * y
    else:
        east = x
        north = y

    # Convert meters back to degree offsets
    dlat = north / origin.meters_per_deg_lat
    dlon = east / origin.meters_per_deg_lon

    lat = origin.lat_deg + dlat
    lon = origin.lon_deg + dlon

    return lat, lon


def suggest_grid_origin(
    lats: float | np.ndarray,
    lons: float | np.ndarray,
) -> GridOrigin:
    """Suggest a grid origin from a set of GPS points.

    Uses the centroid of the provided points as the origin, with
    no grid rotation (grid north = true north).

    Parameters
    ----------
    lats : float or ndarray
        Latitude(s) in decimal degrees.
    lons : float or ndarray
        Longitude(s) in decimal degrees.

    Returns
    -------
    origin : GridOrigin
        Grid origin at the centroid, with bearing_deg = 0.

    Raises
    ------
    ValueError
        If no points are provided.
    """
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)

    if lats.size == 0:
        raise ValueError("At least one point is required to suggest a grid origin.")

    return GridOrigin(
        lat_deg=float(np.mean(lats)),
        lon_deg=float(np.mean(lons)),
        bearing_deg=0.0,
    )


def gps_distance(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float | np.ndarray,
    lon2: float | np.ndarray,
) -> np.ndarray:
    """Haversine distance between GPS points in meters.

    Computes great-circle distance on the WGS84 sphere (using the
    semi-major axis as the radius). Accurate to ~0.3% for most
    practical distances.

    Parameters
    ----------
    lat1, lon1 : float or ndarray
        First point(s) in decimal degrees.
    lat2, lon2 : float or ndarray
        Second point(s) in decimal degrees.

    Returns
    -------
    distance : ndarray
        Distance(s) in meters.

    Notes
    -----
    Uses the Haversine formula with WGS84 semi-major axis (6,378,137 m).
    For geodetic-grade accuracy (< 0.01%), use Vincenty or Karney's method
    instead, but Haversine is sufficient for survey-scale work.
    """
    lat1 = np.asarray(lat1, dtype=np.float64)
    lon1 = np.asarray(lon1, dtype=np.float64)
    lat2 = np.asarray(lat2, dtype=np.float64)
    lon2 = np.asarray(lon2, dtype=np.float64)

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return WGS84_A * c
