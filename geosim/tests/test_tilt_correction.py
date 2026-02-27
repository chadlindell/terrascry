"""Tests for real-time tilt correction pipeline.

Validates tilt correction against hand-calculated values using known
Earth field parameters and tilt angles.
"""

import numpy as np
import pytest

from geosim.streaming.tilt_correction import (
    EarthField,
    batch_correct,
    compute_tilt_correction,
    correct_gradient,
)


class TestEarthField:
    """Tests for EarthField component decomposition."""

    def test_vertical_component(self):
        """Vertical component = total * sin(inclination)."""
        ef = EarthField(total_nt=50000.0, inclination_deg=65.0)
        expected = 50000.0 * np.sin(np.radians(65.0))
        assert ef.vertical_nt == pytest.approx(expected, rel=1e-10)

    def test_horizontal_component(self):
        """Horizontal component = total * cos(inclination)."""
        ef = EarthField(total_nt=50000.0, inclination_deg=65.0)
        expected = 50000.0 * np.cos(np.radians(65.0))
        assert ef.horizontal_nt == pytest.approx(expected, rel=1e-10)

    def test_pythagorean_identity(self):
        """vertical^2 + horizontal^2 = total^2."""
        ef = EarthField(total_nt=50000.0, inclination_deg=65.0)
        reconstructed = np.sqrt(ef.vertical_nt**2 + ef.horizontal_nt**2)
        assert reconstructed == pytest.approx(ef.total_nt, rel=1e-10)

    def test_vertical_inclination_gives_all_vertical(self):
        """At 90 deg inclination, all field is vertical."""
        ef = EarthField(total_nt=50000.0, inclination_deg=90.0)
        assert ef.vertical_nt == pytest.approx(50000.0, rel=1e-10)
        assert ef.horizontal_nt == pytest.approx(0.0, abs=1e-10)

    def test_horizontal_inclination_gives_all_horizontal(self):
        """At 0 deg inclination, all field is horizontal."""
        ef = EarthField(total_nt=50000.0, inclination_deg=0.0)
        assert ef.vertical_nt == pytest.approx(0.0, abs=1e-10)
        assert ef.horizontal_nt == pytest.approx(50000.0, rel=1e-10)


class TestComputeTiltCorrection:
    """Tests for tilt correction calculation."""

    def test_zero_tilt_gives_zero_correction(self):
        """Zero pitch and zero roll produces zero correction."""
        ef = EarthField()
        correction = compute_tilt_correction(0.0, 0.0, ef)
        assert correction == pytest.approx(0.0, abs=1e-15)

    def test_zero_pitch_zero_roll_separately(self):
        """Each axis independently gives zero when flat."""
        ef = EarthField()
        assert compute_tilt_correction(0.0, 5.0, ef) != 0.0
        assert compute_tilt_correction(5.0, 0.0, ef) != 0.0

    def test_two_degree_tilt_magnitude(self):
        """At 2 deg tilt with default Earth field, correction is ~1,750 nT.

        This is the critical sanity check: 2 deg tilt on a 50,000 nT field
        at 65 deg inclination produces ~1,580 nT from pitch alone.
        The exact value: 50000 * sin(65 deg) * sin(2 deg) = ~1,580 nT.
        """
        ef = EarthField(total_nt=50000.0, inclination_deg=65.0)
        # Pure pitch tilt of 2 degrees
        correction_pitch = compute_tilt_correction(2.0, 0.0, ef)
        expected_pitch = ef.vertical_nt * np.sin(np.radians(2.0))
        assert correction_pitch == pytest.approx(expected_pitch, rel=1e-10)
        # Should be around 1580 nT
        assert 1500.0 < abs(correction_pitch) < 1700.0

    def test_combined_pitch_and_roll(self):
        """Combined correction is sum of pitch and roll contributions."""
        ef = EarthField()
        pitch_deg, roll_deg = 1.5, 2.5
        correction = compute_tilt_correction(pitch_deg, roll_deg, ef)
        expected = (
            ef.vertical_nt * np.sin(np.radians(pitch_deg))
            + ef.horizontal_nt * np.sin(np.radians(roll_deg))
        )
        assert correction == pytest.approx(expected, rel=1e-10)

    def test_negative_tilt_gives_negative_correction(self):
        """Negative tilt angles produce negative corrections."""
        ef = EarthField()
        correction = compute_tilt_correction(-2.0, 0.0, ef)
        assert correction < 0

    def test_symmetry(self):
        """Equal positive and negative tilt give equal magnitude corrections."""
        ef = EarthField()
        c_pos = compute_tilt_correction(3.0, 0.0, ef)
        c_neg = compute_tilt_correction(-3.0, 0.0, ef)
        assert c_pos == pytest.approx(-c_neg, rel=1e-10)


class TestCorrectGradient:
    """Tests for the correct_gradient convenience function."""

    def test_returns_corrected_and_correction(self):
        """Returns a 2-tuple of (corrected, correction)."""
        ef = EarthField()
        corrected, correction = correct_gradient(100.0, 1.0, 0.5, ef)
        assert isinstance(corrected, float)
        assert isinstance(correction, float)

    def test_corrected_equals_raw_minus_correction(self):
        """corrected = raw - correction."""
        ef = EarthField()
        raw = 150.0
        corrected, correction = correct_gradient(raw, 2.0, 1.0, ef)
        assert corrected == pytest.approx(raw - correction, rel=1e-14)

    def test_zero_tilt_preserves_gradient(self):
        """At zero tilt, corrected gradient equals raw gradient."""
        ef = EarthField()
        raw = 42.0
        corrected, correction = correct_gradient(raw, 0.0, 0.0, ef)
        assert corrected == pytest.approx(raw, abs=1e-15)
        assert correction == pytest.approx(0.0, abs=1e-15)


class TestBatchCorrect:
    """Tests for vectorized batch correction."""

    def test_batch_matches_single_sample(self):
        """Batch correction matches element-wise single-sample results."""
        ef = EarthField()
        raw = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        pitches = np.array([0.5, 1.0, 1.5, 2.0, -1.0])
        rolls = np.array([0.0, 0.5, -0.5, 1.0, 2.0])

        corrected_batch, corrections_batch = batch_correct(raw, pitches, rolls, ef)

        for i in range(len(raw)):
            corrected_single, correction_single = correct_gradient(
                raw[i], pitches[i], rolls[i], ef
            )
            assert corrected_batch[i] == pytest.approx(corrected_single, rel=1e-14)
            assert corrections_batch[i] == pytest.approx(correction_single, rel=1e-14)

    def test_batch_output_shapes(self):
        """Output arrays have same shape as input."""
        ef = EarthField()
        n = 100
        raw = np.random.default_rng(42).normal(0, 10, n)
        pitches = np.random.default_rng(43).normal(0, 1, n)
        rolls = np.random.default_rng(44).normal(0, 1, n)

        corrected, corrections = batch_correct(raw, pitches, rolls, ef)
        assert corrected.shape == (n,)
        assert corrections.shape == (n,)

    def test_batch_zero_tilt(self):
        """Batch with zero tilt returns raw values unchanged."""
        ef = EarthField()
        raw = np.array([10.0, 20.0, 30.0])
        zeros = np.zeros(3)

        corrected, corrections = batch_correct(raw, zeros, zeros, ef)
        np.testing.assert_allclose(corrected, raw, atol=1e-15)
        np.testing.assert_allclose(corrections, 0.0, atol=1e-15)
