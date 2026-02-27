"""Tests for coordinate transformation utilities.

Validates WGS84 <-> local grid conversions against known geodetic values
and ensures round-trip accuracy within sub-millimeter tolerance for
survey-scale areas.
"""

import numpy as np
import pytest

from geosim.coordinates import (
    WGS84_A,
    WGS84_E2,
    GridOrigin,
    gps_distance,
    gps_to_grid,
    grid_to_gps,
    suggest_grid_origin,
)


class TestGridOrigin:
    """Tests for GridOrigin properties."""

    def test_radian_conversions(self):
        """Properties return correct radian values."""
        origin = GridOrigin(lat_deg=45.0, lon_deg=-90.0, bearing_deg=30.0)
        assert origin.lat_rad == pytest.approx(np.radians(45.0))
        assert origin.lon_rad == pytest.approx(np.radians(-90.0))
        assert origin.bearing_rad == pytest.approx(np.radians(30.0))

    def test_meters_per_deg_lat_equator(self):
        """At equator, meters/deg latitude ~ 110,574 m (WGS84 meridional)."""
        origin = GridOrigin(lat_deg=0.0, lon_deg=0.0)
        # At equator: M = a(1-e^2) / (1)^1.5 = a(1-e^2)
        expected = np.pi / 180 * WGS84_A * (1 - WGS84_E2)
        assert origin.meters_per_deg_lat == pytest.approx(expected, rel=1e-12)
        # Sanity check: ~110,574 m
        assert 110_500 < origin.meters_per_deg_lat < 110_600

    def test_meters_per_deg_lon_equator(self):
        """At equator, meters/deg longitude ~ 111,320 m (WGS84 prime-vertical)."""
        origin = GridOrigin(lat_deg=0.0, lon_deg=0.0)
        # At equator: N = a, cos(0)=1
        expected = np.pi / 180 * WGS84_A
        assert origin.meters_per_deg_lon == pytest.approx(expected, rel=1e-12)
        # Sanity check: ~111,320 m
        assert 111_300 < origin.meters_per_deg_lon < 111_340

    def test_meters_per_deg_lon_shrinks_with_latitude(self):
        """Meters/deg longitude decreases toward poles."""
        origin_eq = GridOrigin(lat_deg=0.0, lon_deg=0.0)
        origin_45 = GridOrigin(lat_deg=45.0, lon_deg=0.0)
        origin_80 = GridOrigin(lat_deg=80.0, lon_deg=0.0)

        assert origin_eq.meters_per_deg_lon > origin_45.meters_per_deg_lon
        assert origin_45.meters_per_deg_lon > origin_80.meters_per_deg_lon

    def test_meters_per_deg_lat_increases_toward_poles(self):
        """Meters/deg latitude increases slightly toward poles (WGS84 flattening)."""
        origin_eq = GridOrigin(lat_deg=0.0, lon_deg=0.0)
        origin_90 = GridOrigin(lat_deg=89.0, lon_deg=0.0)

        # At poles the meridional radius of curvature is larger
        assert origin_90.meters_per_deg_lat > origin_eq.meters_per_deg_lat

    def test_default_bearing_zero(self):
        """Default bearing is 0 (grid north = true north)."""
        origin = GridOrigin(lat_deg=40.0, lon_deg=-74.0)
        assert origin.bearing_deg == 0.0


class TestGpsToGrid:
    """Tests for GPS to local grid conversion."""

    def test_origin_maps_to_zero(self):
        """Origin point should map to (0, 0) in grid coordinates."""
        origin = GridOrigin(lat_deg=40.0, lon_deg=-74.0)
        x, y = gps_to_grid(40.0, -74.0, origin)
        assert float(x) == pytest.approx(0.0, abs=1e-12)
        assert float(y) == pytest.approx(0.0, abs=1e-12)

    def test_one_degree_lat_at_equator(self):
        """One degree latitude at equator should be ~110,574 m northing."""
        origin = GridOrigin(lat_deg=0.0, lon_deg=0.0)
        x, y = gps_to_grid(1.0, 0.0, origin)
        assert float(x) == pytest.approx(0.0, abs=1e-6)
        assert float(y) == pytest.approx(origin.meters_per_deg_lat, rel=1e-10)

    def test_one_degree_lon_at_equator(self):
        """One degree longitude at equator should be ~111,320 m easting."""
        origin = GridOrigin(lat_deg=0.0, lon_deg=0.0)
        x, y = gps_to_grid(0.0, 1.0, origin)
        assert float(x) == pytest.approx(origin.meters_per_deg_lon, rel=1e-10)
        assert float(y) == pytest.approx(0.0, abs=1e-6)

    def test_known_distance_one_degree_lat(self):
        """1 degree of latitude at equator is approximately 111,139 m (textbook Haversine)."""
        # The tangent-plane approximation uses the meridional radius, which gives
        # ~110,574 m. The 111,139 m value is the mean radius approximation.
        # Our value should match the WGS84 meridional calculation precisely.
        origin = GridOrigin(lat_deg=0.0, lon_deg=0.0)
        _, y = gps_to_grid(1.0, 0.0, origin)
        # WGS84 meridional at equator: a(1-e^2) * pi/180
        expected = np.pi / 180 * WGS84_A * (1 - WGS84_E2)
        assert float(y) == pytest.approx(expected, rel=1e-10)

    def test_scalar_input(self):
        """Scalar inputs produce scalar-like outputs."""
        origin = GridOrigin(lat_deg=40.0, lon_deg=-74.0)
        x, y = gps_to_grid(40.001, -73.999, origin)
        assert x.ndim == 0 or x.shape == ()
        assert y.ndim == 0 or y.shape == ()

    def test_array_input(self):
        """Array inputs produce array outputs of matching shape."""
        origin = GridOrigin(lat_deg=40.0, lon_deg=-74.0)
        lats = np.array([40.001, 40.002, 40.003])
        lons = np.array([-73.999, -73.998, -73.997])
        x, y = gps_to_grid(lats, lons, origin)
        assert x.shape == (3,)
        assert y.shape == (3,)

    def test_north_is_positive_y(self):
        """Moving north increases Y."""
        origin = GridOrigin(lat_deg=40.0, lon_deg=-74.0)
        _, y = gps_to_grid(40.001, -74.0, origin)
        assert float(y) > 0

    def test_east_is_positive_x(self):
        """Moving east increases X."""
        origin = GridOrigin(lat_deg=40.0, lon_deg=-74.0)
        x, _ = gps_to_grid(40.0, -73.999, origin)
        assert float(x) > 0


class TestGridToGps:
    """Tests for local grid to GPS conversion."""

    def test_zero_maps_to_origin(self):
        """Grid (0, 0) should map to the origin lat/lon."""
        origin = GridOrigin(lat_deg=40.0, lon_deg=-74.0)
        lat, lon = grid_to_gps(0.0, 0.0, origin)
        assert float(lat) == pytest.approx(40.0, abs=1e-12)
        assert float(lon) == pytest.approx(-74.0, abs=1e-12)

    def test_array_input(self):
        """Array inputs produce array outputs."""
        origin = GridOrigin(lat_deg=40.0, lon_deg=-74.0)
        x = np.array([100.0, 200.0, 300.0])
        y = np.array([50.0, 100.0, 150.0])
        lat, lon = grid_to_gps(x, y, origin)
        assert lat.shape == (3,)
        assert lon.shape == (3,)


class TestRoundTrip:
    """Tests for GPS -> grid -> GPS round-trip accuracy."""

    @pytest.mark.parametrize("lat_origin,lon_origin", [
        (0.0, 0.0),        # Equator
        (45.0, -90.0),     # Mid-latitude
        (70.0, 25.0),      # High latitude
        (-33.8, 151.2),    # Sydney
        (35.68, 139.69),   # Tokyo
    ])
    def test_round_trip_at_various_latitudes(self, lat_origin, lon_origin):
        """Round-trip GPS -> grid -> GPS error < 1 mm for survey-scale offsets."""
        origin = GridOrigin(lat_deg=lat_origin, lon_deg=lon_origin)

        # Points within ~1 km of origin
        offsets_deg = np.array([0.001, 0.005, 0.01])
        lats = lat_origin + offsets_deg
        lons = lon_origin + offsets_deg

        x, y = gps_to_grid(lats, lons, origin)
        lat_rt, lon_rt = grid_to_gps(x, y, origin)

        np.testing.assert_allclose(lat_rt, lats, atol=1e-12)
        np.testing.assert_allclose(lon_rt, lons, atol=1e-12)

    def test_round_trip_sub_millimeter_accuracy(self):
        """Verify sub-millimeter positional accuracy in round-trip."""
        origin = GridOrigin(lat_deg=40.0, lon_deg=-74.0)

        # Create a grid of points ~500m from origin
        lats = np.array([40.0045, 40.005, 40.0055])
        lons = np.array([-73.9945, -73.994, -73.9935])

        x, y = gps_to_grid(lats, lons, origin)
        lat_rt, lon_rt = grid_to_gps(x, y, origin)

        # Convert degree errors to meters
        lat_err_m = np.abs(lat_rt - lats) * origin.meters_per_deg_lat
        lon_err_m = np.abs(lon_rt - lons) * origin.meters_per_deg_lon

        # Sub-millimeter accuracy
        assert np.all(lat_err_m < 1e-3), f"Lat error: {lat_err_m} m"
        assert np.all(lon_err_m < 1e-3), f"Lon error: {lon_err_m} m"

    def test_round_trip_with_rotation(self):
        """Round-trip with non-zero bearing preserves sub-mm accuracy."""
        origin = GridOrigin(lat_deg=40.0, lon_deg=-74.0, bearing_deg=30.0)

        lats = np.array([40.002, 40.003, 40.004])
        lons = np.array([-73.998, -73.997, -73.996])

        x, y = gps_to_grid(lats, lons, origin)
        lat_rt, lon_rt = grid_to_gps(x, y, origin)

        lat_err_m = np.abs(lat_rt - lats) * origin.meters_per_deg_lat
        lon_err_m = np.abs(lon_rt - lons) * origin.meters_per_deg_lon

        assert np.all(lat_err_m < 1e-3)
        assert np.all(lon_err_m < 1e-3)


class TestGridRotation:
    """Tests for grid rotation (bearing_deg != 0)."""

    def test_90_degree_rotation(self):
        """90-degree rotation swaps east/north with sign change."""
        origin_no_rot = GridOrigin(lat_deg=40.0, lon_deg=-74.0, bearing_deg=0.0)
        origin_rotated = GridOrigin(lat_deg=40.0, lon_deg=-74.0, bearing_deg=90.0)

        lat, lon = 40.001, -74.0  # Point due north

        x0, y0 = gps_to_grid(lat, lon, origin_no_rot)
        x90, y90 = gps_to_grid(lat, lon, origin_rotated)

        # Due north (y0 > 0, x0 = 0) should become due east with 90 CW rotation
        # x_grid =  cos(90)*east + sin(90)*north = north
        # y_grid = -sin(90)*east + cos(90)*north = -east
        assert float(x90) == pytest.approx(float(y0), rel=1e-10)
        assert float(y90) == pytest.approx(-float(x0), abs=1e-6)

    def test_180_degree_rotation(self):
        """180-degree rotation negates both coordinates."""
        origin_no_rot = GridOrigin(lat_deg=40.0, lon_deg=-74.0, bearing_deg=0.0)
        origin_rotated = GridOrigin(lat_deg=40.0, lon_deg=-74.0, bearing_deg=180.0)

        lat, lon = 40.001, -73.999

        x0, y0 = gps_to_grid(lat, lon, origin_no_rot)
        x180, y180 = gps_to_grid(lat, lon, origin_rotated)

        assert float(x180) == pytest.approx(-float(x0), rel=1e-10)
        assert float(y180) == pytest.approx(-float(y0), rel=1e-10)

    def test_360_degree_rotation_identity(self):
        """360-degree rotation is the same as no rotation."""
        origin_no_rot = GridOrigin(lat_deg=40.0, lon_deg=-74.0, bearing_deg=0.0)
        origin_360 = GridOrigin(lat_deg=40.0, lon_deg=-74.0, bearing_deg=360.0)

        lats = np.array([40.001, 40.002])
        lons = np.array([-73.999, -73.998])

        x0, y0 = gps_to_grid(lats, lons, origin_no_rot)
        x360, y360 = gps_to_grid(lats, lons, origin_360)

        np.testing.assert_allclose(x360, x0, atol=1e-10)
        np.testing.assert_allclose(y360, y0, atol=1e-10)

    def test_rotation_preserves_distance(self):
        """Rotation should not change the distance from origin."""
        origin_no_rot = GridOrigin(lat_deg=40.0, lon_deg=-74.0, bearing_deg=0.0)
        origin_rotated = GridOrigin(lat_deg=40.0, lon_deg=-74.0, bearing_deg=45.0)

        lat, lon = 40.001, -73.999

        x0, y0 = gps_to_grid(lat, lon, origin_no_rot)
        xr, yr = gps_to_grid(lat, lon, origin_rotated)

        dist_0 = np.sqrt(float(x0) ** 2 + float(y0) ** 2)
        dist_r = np.sqrt(float(xr) ** 2 + float(yr) ** 2)

        assert dist_r == pytest.approx(dist_0, rel=1e-10)


class TestSuggestGridOrigin:
    """Tests for suggest_grid_origin."""

    def test_single_point(self):
        """Single point becomes the origin."""
        origin = suggest_grid_origin(40.0, -74.0)
        assert origin.lat_deg == pytest.approx(40.0)
        assert origin.lon_deg == pytest.approx(-74.0)
        assert origin.bearing_deg == 0.0

    def test_centroid_of_multiple_points(self):
        """Origin is the centroid of multiple points."""
        lats = np.array([40.0, 41.0, 42.0])
        lons = np.array([-74.0, -75.0, -76.0])

        origin = suggest_grid_origin(lats, lons)
        assert origin.lat_deg == pytest.approx(41.0)
        assert origin.lon_deg == pytest.approx(-75.0)

    def test_empty_raises(self):
        """Empty arrays raise ValueError."""
        with pytest.raises(ValueError, match="At least one point"):
            suggest_grid_origin(np.array([]), np.array([]))

    def test_bearing_always_zero(self):
        """Suggested origin always has zero bearing."""
        origin = suggest_grid_origin(
            np.array([40.0, 41.0]),
            np.array([-74.0, -75.0]),
        )
        assert origin.bearing_deg == 0.0


class TestGpsDistance:
    """Tests for Haversine distance calculation."""

    def test_zero_distance(self):
        """Same point gives zero distance."""
        d = gps_distance(40.0, -74.0, 40.0, -74.0)
        assert float(d) == pytest.approx(0.0, abs=1e-10)

    def test_one_degree_lat_at_equator(self):
        """One degree of latitude at equator ~ 111,195 m (Haversine)."""
        d = gps_distance(0.0, 0.0, 1.0, 0.0)
        # Haversine with WGS84_A gives ~ 111,195 m
        # (the great-circle arc length for 1 deg on a sphere of radius WGS84_A)
        expected = WGS84_A * np.radians(1.0)
        assert float(d) == pytest.approx(expected, rel=1e-10)

    def test_one_degree_lon_at_equator(self):
        """One degree of longitude at equator ~ 111,195 m (same as lat at equator)."""
        d = gps_distance(0.0, 0.0, 0.0, 1.0)
        expected = WGS84_A * np.radians(1.0)
        assert float(d) == pytest.approx(expected, rel=1e-10)

    def test_known_distance_new_york_to_los_angeles(self):
        """NYC to LA great-circle distance ~ 3,944 km."""
        # NYC: 40.7128 N, 74.0060 W; LA: 34.0522 N, 118.2437 W
        d = gps_distance(40.7128, -74.0060, 34.0522, -118.2437)
        # Known great-circle distance (sphere) ~ 3,944 km
        d_km = float(d) / 1000
        assert 3_930 < d_km < 3_960

    def test_antipodal_points(self):
        """Antipodal points are half the circumference apart."""
        d = gps_distance(0.0, 0.0, 0.0, 180.0)
        expected = np.pi * WGS84_A
        assert float(d) == pytest.approx(expected, rel=1e-10)

    def test_symmetry(self):
        """Distance is symmetric: d(A,B) == d(B,A)."""
        d1 = gps_distance(40.0, -74.0, 35.0, -118.0)
        d2 = gps_distance(35.0, -118.0, 40.0, -74.0)
        assert float(d1) == pytest.approx(float(d2), rel=1e-14)

    def test_array_input(self):
        """Array inputs produce array outputs."""
        lat1 = np.array([0.0, 0.0, 0.0])
        lon1 = np.array([0.0, 0.0, 0.0])
        lat2 = np.array([1.0, 2.0, 3.0])
        lon2 = np.array([0.0, 0.0, 0.0])

        d = gps_distance(lat1, lon1, lat2, lon2)
        assert d.shape == (3,)

        # Each should be approximately N * 111_195 m
        for i in range(3):
            expected = WGS84_A * np.radians(float(i + 1))
            assert d[i] == pytest.approx(expected, rel=1e-10)

    def test_short_distance_matches_tangent_plane(self):
        """For short distances, Haversine should agree with tangent-plane."""
        origin = GridOrigin(lat_deg=40.0, lon_deg=-74.0)
        lat2 = 40.001
        lon2 = -73.999

        d_haversine = float(gps_distance(40.0, -74.0, lat2, lon2))
        x, y = gps_to_grid(lat2, lon2, origin)
        d_tangent = float(np.sqrt(x**2 + y**2))

        # Haversine uses a sphere (WGS84_A) while tangent-plane uses the
        # full ellipsoid, so they differ by ~0.1% at mid-latitudes.
        # For ~140 m distances, this means agreement within ~0.2 m.
        assert d_haversine == pytest.approx(d_tangent, rel=2e-3)

    def test_broadcasting(self):
        """Scalar and array inputs broadcast correctly."""
        d = gps_distance(0.0, 0.0, np.array([1.0, 2.0]), np.array([0.0, 0.0]))
        assert d.shape == (2,)
