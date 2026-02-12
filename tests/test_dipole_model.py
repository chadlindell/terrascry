"""Tests for magnetic dipole forward model.

Validates against textbook values and known analytical results.
"""

import numpy as np
import pytest

from geosim.magnetics.dipole import (
    MU_0,
    _PREFACTOR,
    detection_depth_estimate,
    dipole_field,
    dipole_field_gradient,
    dipole_moment_from_sphere,
    gradiometer_reading,
    superposition_field,
)


class TestDipoleField:
    """Tests for single dipole field calculation."""

    def test_axial_field_z_dipole(self):
        """On-axis field of a z-directed dipole: B_z = μ₀·2m/(4π·r³)."""
        moment = np.array([0.0, 0.0, 1.0])  # 1 A·m² along z
        r_src = np.array([0.0, 0.0, 0.0])

        # Point on z-axis, 1m above dipole
        r_obs = np.array([0.0, 0.0, 1.0])
        B = dipole_field(r_obs, r_src, moment)

        # On axis: B_z = μ₀/(4π) · 2m/r³
        expected_Bz = _PREFACTOR * 2.0 * 1.0 / 1.0**3
        assert B[0] == pytest.approx(0.0, abs=1e-20)
        assert B[1] == pytest.approx(0.0, abs=1e-20)
        assert B[2] == pytest.approx(expected_Bz, rel=1e-10)

    def test_equatorial_field_z_dipole(self):
        """Equatorial field of a z-directed dipole: B_z = -μ₀·m/(4π·r³)."""
        moment = np.array([0.0, 0.0, 1.0])
        r_src = np.array([0.0, 0.0, 0.0])

        # Point on x-axis, 1m from dipole
        r_obs = np.array([1.0, 0.0, 0.0])
        B = dipole_field(r_obs, r_src, moment)

        # Equatorial: B_z = -μ₀/(4π) · m/r³
        expected_Bz = -_PREFACTOR * 1.0 / 1.0**3
        assert B[0] == pytest.approx(0.0, abs=1e-20)
        assert B[1] == pytest.approx(0.0, abs=1e-20)
        assert B[2] == pytest.approx(expected_Bz, rel=1e-10)

    def test_inverse_cube_distance_scaling(self):
        """Field magnitude scales as 1/r³."""
        moment = np.array([0.0, 0.0, 1.0])
        r_src = np.array([0.0, 0.0, 0.0])

        B1 = dipole_field(np.array([0.0, 0.0, 1.0]), r_src, moment)
        B2 = dipole_field(np.array([0.0, 0.0, 2.0]), r_src, moment)

        # At 2x distance, field should be 1/8 as strong
        ratio = np.linalg.norm(B1) / np.linalg.norm(B2)
        assert ratio == pytest.approx(8.0, rel=1e-10)

    def test_field_symmetry(self):
        """Field is symmetric about dipole axis."""
        moment = np.array([0.0, 0.0, 1.0])
        r_src = np.array([0.0, 0.0, 0.0])

        B_pos_x = dipole_field(np.array([1.0, 0.0, 0.0]), r_src, moment)
        B_neg_x = dipole_field(np.array([-1.0, 0.0, 0.0]), r_src, moment)
        B_pos_y = dipole_field(np.array([0.0, 1.0, 0.0]), r_src, moment)

        # Bz should be same at all equatorial points
        assert B_pos_x[2] == pytest.approx(B_neg_x[2], rel=1e-10)
        assert B_pos_x[2] == pytest.approx(B_pos_y[2], rel=1e-10)

    def test_batch_computation(self):
        """Multiple observation points give same results as individual calls."""
        moment = np.array([0.0, 0.0, 1.0])
        r_src = np.array([0.0, 0.0, 0.0])

        points = np.array([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])

        B_batch = dipole_field(points, r_src, moment)
        for i in range(3):
            B_single = dipole_field(points[i], r_src, moment)
            np.testing.assert_allclose(B_batch[i], B_single, rtol=1e-14)

    def test_at_source_returns_zero(self):
        """Field at the source location returns zero (not infinity)."""
        moment = np.array([0.0, 0.0, 1.0])
        r_src = np.array([1.0, 2.0, 3.0])
        B = dipole_field(r_src, r_src, moment)
        np.testing.assert_array_equal(B, [0.0, 0.0, 0.0])

    def test_displaced_source(self):
        """Displaced source produces correct field."""
        moment = np.array([0.0, 0.0, 1.0])
        r_src = np.array([5.0, 5.0, -1.0])

        # 1m above the source
        r_obs = np.array([5.0, 5.0, 0.0])
        B = dipole_field(r_obs, r_src, moment)

        expected_Bz = _PREFACTOR * 2.0 / 1.0**3
        assert B[2] == pytest.approx(expected_Bz, rel=1e-10)

    def test_x_directed_moment(self):
        """X-directed moment produces correct on-axis field."""
        moment = np.array([1.0, 0.0, 0.0])
        r_src = np.array([0.0, 0.0, 0.0])

        # Point along x-axis
        r_obs = np.array([1.0, 0.0, 0.0])
        B = dipole_field(r_obs, r_src, moment)

        expected_Bx = _PREFACTOR * 2.0 / 1.0**3
        assert B[0] == pytest.approx(expected_Bx, rel=1e-10)
        assert B[1] == pytest.approx(0.0, abs=1e-20)
        assert B[2] == pytest.approx(0.0, abs=1e-20)


class TestSuperposition:
    """Tests for multiple source superposition."""

    def test_single_source_matches_dipole(self):
        """Superposition with one source matches direct dipole calculation."""
        sources = [{'position': [0, 0, 0], 'moment': [0, 0, 1.0]}]
        r_obs = np.array([0.0, 0.0, 1.0])

        B_super = superposition_field(r_obs, sources)
        B_direct = dipole_field(r_obs, np.array([0, 0, 0]), np.array([0, 0, 1.0]))

        np.testing.assert_allclose(B_super, B_direct, rtol=1e-14)

    def test_two_identical_sources_double_field(self):
        """Two identical co-located sources produce double the field."""
        moment = [0.0, 0.0, 1.0]
        sources_1 = [{'position': [0, 0, 0], 'moment': moment}]
        sources_2 = [
            {'position': [0, 0, 0], 'moment': moment},
            {'position': [0, 0, 0], 'moment': moment},
        ]
        r_obs = np.array([0.0, 0.0, 1.0])

        B1 = superposition_field(r_obs, sources_1)
        B2 = superposition_field(r_obs, sources_2)

        np.testing.assert_allclose(B2, 2.0 * B1, rtol=1e-14)

    def test_opposing_moments_cancel(self):
        """Two equal and opposite co-located sources cancel."""
        sources = [
            {'position': [0, 0, 0], 'moment': [0, 0, 1.0]},
            {'position': [0, 0, 0], 'moment': [0, 0, -1.0]},
        ]
        r_obs = np.array([0.0, 0.0, 1.0])
        B = superposition_field(r_obs, sources)
        np.testing.assert_allclose(B, [0, 0, 0], atol=1e-20)


class TestGradient:
    """Tests for gradiometer gradient calculation."""

    def test_gradient_sign_convention(self):
        """Gradient is positive when lower sensor sees stronger field."""
        # Source below surface → bottom sensor is closer → gradient positive
        r_src = np.array([0.0, 0.0, -1.0])
        moment = np.array([0.0, 0.0, 1.0])
        r_obs = np.array([0.0, 0.0, 0.175])  # bottom sensor at 17.5 cm

        gradient = dipole_field_gradient(
            r_obs, r_src, moment, sensor_separation=0.35, component=2
        )
        assert gradient > 0, "Gradient should be positive for source below"

    def test_gradient_zero_at_large_distance(self):
        """Gradient approaches zero far from source."""
        r_src = np.array([0.0, 0.0, -1.0])
        moment = np.array([0.0, 0.0, 1.0])

        # Very far from source
        r_obs = np.array([1000.0, 0.0, 0.175])
        gradient = dipole_field_gradient(
            r_obs, r_src, moment, sensor_separation=0.35, component=2
        )
        assert abs(gradient) < 1e-20

    def test_gradient_scales_with_moment(self):
        """Doubling the moment doubles the gradient."""
        r_src = np.array([0.0, 0.0, -1.0])
        r_obs = np.array([0.0, 0.0, 0.175])

        g1 = dipole_field_gradient(
            r_obs, r_src, np.array([0, 0, 1.0]), 0.35, 2
        )
        g2 = dipole_field_gradient(
            r_obs, r_src, np.array([0, 0, 2.0]), 0.35, 2
        )
        assert g2 == pytest.approx(2.0 * g1, rel=1e-10)


class TestGradiometerReading:
    """Tests for complete gradiometer reading simulation."""

    def test_returns_three_components(self):
        """gradiometer_reading returns B_bottom, B_top, gradient."""
        sources = [{'position': [0, 0, -1], 'moment': [0, 0, 1.0]}]
        r_obs = np.array([0.0, 0.0, 0.175])

        B_bot, B_top, grad = gradiometer_reading(r_obs, sources, 0.35)
        assert isinstance(B_bot, float)
        assert isinstance(B_top, float)
        assert isinstance(grad, float)
        assert grad == pytest.approx(B_bot - B_top)

    def test_batch_gradiometer_reading(self):
        """Batch computation produces consistent results."""
        sources = [{'position': [5, 5, -1], 'moment': [0, 0, 1.0]}]
        positions = np.array([
            [3.0, 5.0, 0.175],
            [5.0, 5.0, 0.175],
            [7.0, 5.0, 0.175],
        ])

        B_bot, B_top, grad = gradiometer_reading(positions, sources, 0.35)
        assert B_bot.shape == (3,)
        assert B_top.shape == (3,)
        assert grad.shape == (3,)

        # Center point (directly above) should have strongest gradient
        assert abs(grad[1]) > abs(grad[0])
        assert abs(grad[1]) > abs(grad[2])


class TestDipoleMoment:
    """Tests for dipole moment estimation."""

    def test_high_susceptibility_limit(self):
        """For χ >> 3, effective_chi → 3 and m → 4πr³·B/μ₀."""
        r = 0.05  # 5 cm radius
        chi = 10000.0  # very high susceptibility

        moment = dipole_moment_from_sphere(r, chi)

        # In the high-χ limit, effective_chi → 3
        volume = (4.0 / 3.0) * np.pi * r**3
        expected = volume * 3.0 * 50e-6 / MU_0
        assert moment[2] == pytest.approx(expected, rel=0.01)

    def test_moment_scales_with_volume(self):
        """Doubling radius increases moment by 8x (volume scaling)."""
        m1 = dipole_moment_from_sphere(0.05, 1000)
        m2 = dipole_moment_from_sphere(0.10, 1000)

        ratio = np.linalg.norm(m2) / np.linalg.norm(m1)
        assert ratio == pytest.approx(8.0, rel=0.01)

    def test_moment_is_vertical(self):
        """Default moment is along z (vertical Earth field assumption)."""
        moment = dipole_moment_from_sphere(0.05, 1000)
        assert moment[0] == 0.0
        assert moment[1] == 0.0
        assert moment[2] > 0.0


class TestDetectionDepth:
    """Tests for detection depth estimation."""

    def test_larger_target_deeper_detection(self):
        """Larger moment → deeper detection depth."""
        d1 = detection_depth_estimate(1.0, 1e-12)
        d2 = detection_depth_estimate(10.0, 1e-12)
        assert d2 > d1

    def test_lower_noise_deeper_detection(self):
        """Lower noise floor → deeper detection depth."""
        d1 = detection_depth_estimate(1.0, 1e-11)
        d2 = detection_depth_estimate(1.0, 1e-12)
        assert d2 > d1

    def test_detection_depth_is_positive(self):
        """Detection depth is always positive."""
        d = detection_depth_estimate(0.01, 1e-10)
        assert d > 0
