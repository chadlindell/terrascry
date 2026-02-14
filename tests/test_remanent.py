"""Tests for remanent magnetization support.

Validates remanent_moment(), combined_moment(), scenario loader integration,
and asymmetric anomaly signatures from remanence.
"""

import numpy as np
import pytest

from geosim.magnetics.dipole import (
    _PREFACTOR,
    combined_moment,
    dipole_field,
    dipole_moment_from_sphere,
    remanent_moment,
)
from geosim.scenarios.loader import BuriedObject, Scenario, Terrain, load_scenario


class TestRemanentMoment:
    """Tests for the remanent_moment() function."""

    def test_basic_computation(self):
        """m_r = V * M_r * direction for a known case."""
        volume = 1.0  # 1 m³
        direction = np.array([1.0, 0.0, 0.0])
        intensity = 10.0  # A/m

        m = remanent_moment(volume, direction, intensity)
        expected = np.array([10.0, 0.0, 0.0])
        np.testing.assert_allclose(m, expected, atol=1e-15)

    def test_direction_normalization(self):
        """Non-unit direction vectors are normalized."""
        volume = 1.0
        direction = np.array([3.0, 4.0, 0.0])  # magnitude 5
        intensity = 10.0

        m = remanent_moment(volume, direction, intensity)
        expected = np.array([6.0, 8.0, 0.0])  # 10 * [3/5, 4/5, 0]
        np.testing.assert_allclose(m, expected, atol=1e-14)

    def test_zero_direction_returns_zero(self):
        """Zero direction vector returns zero moment."""
        m = remanent_moment(1.0, np.array([0.0, 0.0, 0.0]), 10.0)
        np.testing.assert_array_equal(m, [0.0, 0.0, 0.0])

    def test_scales_with_volume(self):
        """Moment magnitude scales linearly with volume."""
        direction = np.array([0.0, 0.0, 1.0])
        m1 = remanent_moment(1.0, direction, 10.0)
        m2 = remanent_moment(2.0, direction, 10.0)
        np.testing.assert_allclose(m2, 2.0 * m1, atol=1e-15)

    def test_scales_with_intensity(self):
        """Moment magnitude scales linearly with remanence intensity."""
        direction = np.array([0.0, 0.0, 1.0])
        m1 = remanent_moment(1.0, direction, 5.0)
        m2 = remanent_moment(1.0, direction, 10.0)
        np.testing.assert_allclose(m2, 2.0 * m1, atol=1e-15)

    def test_sphere_volume(self):
        """Correct remanent moment for a sphere of known radius."""
        radius = 0.1  # 10 cm
        volume = (4.0 / 3.0) * np.pi * radius ** 3
        direction = np.array([1.0, 0.0, 0.0])
        intensity = 20.0  # A/m

        m = remanent_moment(volume, direction, intensity)
        expected_mag = volume * intensity
        assert np.linalg.norm(m) == pytest.approx(expected_mag, rel=1e-14)
        # Should point along x
        assert m[0] == pytest.approx(expected_mag, rel=1e-14)
        assert m[1] == pytest.approx(0.0, abs=1e-20)
        assert m[2] == pytest.approx(0.0, abs=1e-20)


class TestCombinedMoment:
    """Tests for the combined_moment() function."""

    def test_superposition(self):
        """Combined moment is vector sum of induced + remanent."""
        induced = np.array([0.0, 0.0, 1.0])
        rem = np.array([0.5, 0.0, 0.0])
        total = combined_moment(induced, rem)
        expected = np.array([0.5, 0.0, 1.0])
        np.testing.assert_allclose(total, expected, atol=1e-15)

    def test_zero_remanent(self):
        """Zero remanence returns induced moment unchanged."""
        induced = np.array([0.0, 0.0, 1.0])
        total = combined_moment(induced, np.zeros(3))
        np.testing.assert_allclose(total, induced, atol=1e-15)

    def test_zero_induced(self):
        """Zero induced returns remanent moment."""
        rem = np.array([0.3, 0.4, 0.5])
        total = combined_moment(np.zeros(3), rem)
        np.testing.assert_allclose(total, rem, atol=1e-15)

    def test_antiparallel(self):
        """Antiparallel moments partially cancel."""
        induced = np.array([0.0, 0.0, 1.0])
        rem = np.array([0.0, 0.0, -0.3])
        total = combined_moment(induced, rem)
        np.testing.assert_allclose(total, [0.0, 0.0, 0.7], atol=1e-15)


class TestRemanentFieldSignature:
    """Test that remanence produces asymmetric anomaly signatures."""

    def test_remanent_only_arbitrary_direction(self):
        """A pure remanent source (x-directed) produces non-zero Bx at equatorial point."""
        # Remanent moment pointing East (x)
        m_rem = np.array([1.0, 0.0, 0.0])
        r_src = np.array([0.0, 0.0, -1.0])

        # Observation on z-axis above source
        r_obs = np.array([0.0, 0.0, 0.0])
        B = dipole_field(r_obs, r_src, m_rem)

        # For an x-directed dipole on the z-axis:
        # B = μ₀/(4π) * [3(m·r̂)r̂ - m] / r³
        # r̂ = [0,0,1], m·r̂ = 0
        # B = -μ₀/(4π) * m / r³ = -μ₀/(4π) * [1,0,0]
        expected_Bx = -_PREFACTOR * 1.0 / 1.0**3
        assert B[0] == pytest.approx(expected_Bx, rel=1e-10)
        assert B[2] == pytest.approx(0.0, abs=1e-20)

    def test_asymmetric_profile(self):
        """Combined moment ≠ induced direction produces asymmetric anomaly."""
        r_src = np.array([0.0, 0.0, -1.0])

        # Pure induced (z-directed)
        m_induced = np.array([0.0, 0.0, 1.0])

        # Combined with x-remanence
        m_combined = combined_moment(m_induced, np.array([0.5, 0.0, 0.0]))

        # Profile along x-axis
        x_pts = np.linspace(-3, 3, 61)
        r_obs = np.column_stack([x_pts, np.zeros_like(x_pts), np.zeros_like(x_pts)])

        Bz_induced = dipole_field(r_obs, r_src, m_induced)[:, 2]
        Bz_combined = dipole_field(r_obs, r_src, m_combined)[:, 2]

        # Induced-only profile should be symmetric about x=0
        assert Bz_induced[0] == pytest.approx(Bz_induced[-1], rel=1e-10)

        # Combined profile should be asymmetric (different at ±x)
        assert Bz_combined[0] != pytest.approx(Bz_combined[-1], rel=0.01)

    def test_combined_differs_from_induced(self):
        """Combined moment field ≠ induced-only field when remanence present."""
        r_src = np.array([5.0, 5.0, -1.0])
        m_induced = dipole_moment_from_sphere(0.1, 500.0, 50e-6)
        m_rem = np.array([0.01, 0.0, 0.0])  # Small horizontal remanence
        m_total = combined_moment(m_induced, m_rem)

        r_obs = np.array([5.0, 5.0, 0.0])
        B_induced = dipole_field(r_obs, r_src, m_induced)
        B_combined = dipole_field(r_obs, r_src, m_total)

        # Fields should differ
        assert not np.allclose(B_induced, B_combined, atol=1e-20)


class TestLoaderRemanenceIntegration:
    """Test scenario loader handles remanence data correctly."""

    def test_scattered_debris_loads(self):
        """scattered-debris.json loads without error and has remanent objects."""
        scenario = load_scenario("scenarios/scattered-debris.json")

        # Fastener cluster has remanence — its moment should differ from
        # pure induced (which would be z-aligned)
        fastener = [o for o in scenario.objects if "fastener" in o.name.lower()][0]
        assert fastener.moment is not None
        # Remanence is x-directed, so moment should have non-zero x-component
        assert abs(fastener.moment[0]) > 0

    def test_bomb_crater_loads(self):
        """bomb-crater-heterogeneous.json loads with remanent UXB."""
        scenario = load_scenario("scenarios/bomb-crater-heterogeneous.json")

        uxb = [o for o in scenario.objects if "UXB" in o.name][0]
        assert uxb.moment is not None
        # UXB has direct remanent_moment [0.5, 0.0, 0.2] added to induced
        # So moment should have x-component from remanence
        assert abs(uxb.moment[0]) > 0.4

    def test_remanence_dict_parsing(self):
        """Object with remanence dict gets computed remanent moment."""
        scenario = Scenario(
            name="test",
            description="test",
            terrain=Terrain(),
            objects=[
                BuriedObject(
                    name="test_obj",
                    position=np.array([0.0, 0.0, -1.0]),
                    object_type="ferrous_sphere",
                    radius=0.1,
                    susceptibility=500.0,
                    metadata={
                        "remanence": {
                            "direction": [1.0, 0.0, 0.0],
                            "intensity": 20.0,
                        }
                    },
                ),
            ],
        )
        scenario.compute_induced_moments()

        obj = scenario.objects[0]
        assert obj.moment is not None
        # Should have both induced (z-dominant) and remanent (x-directed) components
        assert abs(obj.moment[0]) > 0  # remanent x-component
        assert abs(obj.moment[2]) > 0  # induced z-component

    def test_direct_remanent_moment_parsing(self):
        """Object with direct remanent_moment vector in metadata."""
        scenario = Scenario(
            name="test",
            description="test",
            terrain=Terrain(),
            objects=[
                BuriedObject(
                    name="test_obj",
                    position=np.array([0.0, 0.0, -1.0]),
                    object_type="ferrous_sphere",
                    radius=0.1,
                    susceptibility=500.0,
                    metadata={
                        "remanent_moment": [0.5, 0.0, 0.2],
                    },
                ),
            ],
        )
        scenario.compute_induced_moments()

        obj = scenario.objects[0]
        # x-component should include the 0.5 A·m² remanent contribution
        assert abs(obj.moment[0]) >= 0.49

    def test_no_remanence_unchanged(self):
        """Objects without remanence data behave exactly as before."""
        scenario = Scenario(
            name="test",
            description="test",
            terrain=Terrain(),
            objects=[
                BuriedObject(
                    name="test_obj",
                    position=np.array([0.0, 0.0, -1.0]),
                    object_type="ferrous_sphere",
                    radius=0.1,
                    susceptibility=500.0,
                ),
            ],
        )
        scenario.compute_induced_moments()

        obj = scenario.objects[0]
        # Induced-only: moment aligned with Earth field
        assert obj.moment is not None
        assert abs(obj.moment[0]) < 1e-10  # No x-component (earth field has Bx=0)
