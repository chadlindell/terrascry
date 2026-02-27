"""Tests for noise models.

Validates SensorNoise, DiurnalDrift, HeadingError, NoiseModel,
and the pathfinder_noise_model() factory.
"""

import numpy as np
import pytest

from geosim.noise.models import (
    DiurnalDrift,
    HeadingError,
    NoiseModel,
    SensorNoise,
    pathfinder_noise_model,
)


class TestSensorNoise:
    """Tests for white + 1/f sensor noise model."""

    def test_output_shape(self):
        """Output has the requested number of samples."""
        noise = SensorNoise()
        result = noise.generate(1000)
        assert result.shape == (1000,)

    def test_output_shape_single(self):
        """Single sample returns 1-element array."""
        noise = SensorNoise()
        result = noise.generate(1)
        assert result.shape == (1,)

    def test_zero_mean(self):
        """Noise has approximately zero mean over many samples."""
        noise = SensorNoise(noise_floor=50e-12, pink_corner=0.5)
        rng = np.random.default_rng(42)
        result = noise.generate(100_000, rng=rng)
        assert abs(np.mean(result)) < 5e-12  # within 10% of noise_floor

    def test_rms_matches_noise_floor(self):
        """RMS of generated noise matches the specified noise_floor."""
        floor = 50e-12
        noise = SensorNoise(noise_floor=floor, pink_corner=0.5)
        rng = np.random.default_rng(42)
        result = noise.generate(100_000, rng=rng)
        rms = np.sqrt(np.mean(result**2))
        assert rms == pytest.approx(floor, rel=0.15)

    def test_white_only_when_pink_corner_zero(self):
        """With pink_corner=0, output is pure white noise."""
        floor = 50e-12
        noise = SensorNoise(noise_floor=floor, pink_corner=0.0)
        rng = np.random.default_rng(42)
        result = noise.generate(10_000, rng=rng)
        rms = np.sqrt(np.mean(result**2))
        assert rms == pytest.approx(floor, rel=0.05)

    def test_spectral_shape_pink_noise(self):
        """Low-frequency power exceeds high-frequency power for pink noise."""
        noise = SensorNoise(noise_floor=50e-12, pink_corner=2.0)
        rng = np.random.default_rng(42)
        result = noise.generate(10_000, sample_rate=100.0, rng=rng)

        fft = np.abs(np.fft.rfft(result))
        freqs = np.fft.rfftfreq(len(result), d=1.0 / 100.0)

        # Compare power in low (0.1-1 Hz) vs high (10-50 Hz) bands
        low_mask = (freqs >= 0.1) & (freqs <= 1.0)
        high_mask = (freqs >= 10.0) & (freqs <= 50.0)
        low_power = np.mean(fft[low_mask] ** 2)
        high_power = np.mean(fft[high_mask] ** 2)
        assert low_power > high_power

    def test_dc_removal(self):
        """Pink filter removes DC component (pink_filter[0] = 0)."""
        noise = SensorNoise(noise_floor=50e-12, pink_corner=0.5)
        rng = np.random.default_rng(42)
        result = noise.generate(10_000, rng=rng)
        # DC should be near zero (mean close to zero)
        assert abs(np.mean(result)) < 10e-12

    def test_seed_reproducibility(self):
        """Same seed produces identical noise."""
        noise = SensorNoise(noise_floor=50e-12, pink_corner=0.5)
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        result1 = noise.generate(1000, rng=rng1)
        result2 = noise.generate(1000, rng=rng2)
        np.testing.assert_array_equal(result1, result2)

    def test_different_seeds_differ(self):
        """Different seeds produce different noise."""
        noise = SensorNoise(noise_floor=50e-12, pink_corner=0.5)
        rng1 = np.random.default_rng(1)
        rng2 = np.random.default_rng(2)
        result1 = noise.generate(1000, rng=rng1)
        result2 = noise.generate(1000, rng=rng2)
        assert not np.array_equal(result1, result2)

    def test_noise_floor_scaling(self):
        """Doubling noise_floor doubles the RMS."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        n1 = SensorNoise(noise_floor=25e-12, pink_corner=0.0)
        n2 = SensorNoise(noise_floor=50e-12, pink_corner=0.0)
        rms1 = np.std(n1.generate(100_000, rng=rng1))
        rms2 = np.std(n2.generate(100_000, rng=rng2))
        assert rms2 / rms1 == pytest.approx(2.0, rel=0.05)

    def test_default_rng_creates_generator(self):
        """Calling without rng still works (uses default)."""
        noise = SensorNoise()
        result = noise.generate(100)
        assert result.shape == (100,)


class TestDiurnalDrift:
    """Tests for diurnal magnetic field variation model."""

    def test_sinusoidal_at_zero(self):
        """At t=0, sin(0)=0 so drift is zero."""
        drift = DiurnalDrift(amplitude=50e-9, period=86400.0, rejection_ratio=0.01)
        result = drift.evaluate(np.array([0.0]))
        assert result[0] == pytest.approx(0.0, abs=1e-20)

    def test_sinusoidal_at_quarter_period(self):
        """At t=period/4, sin(π/2)=1 so drift = amplitude * rejection_ratio."""
        amp = 50e-9
        ratio = 0.01
        drift = DiurnalDrift(amplitude=amp, period=86400.0, rejection_ratio=ratio)
        t = np.array([86400.0 / 4.0])
        result = drift.evaluate(t)
        assert result[0] == pytest.approx(amp * ratio, rel=1e-10)

    def test_sinusoidal_at_half_period(self):
        """At t=period/2, sin(π)=0 so drift is zero."""
        drift = DiurnalDrift(amplitude=50e-9, period=86400.0, rejection_ratio=0.01)
        t = np.array([86400.0 / 2.0])
        result = drift.evaluate(t)
        assert result[0] == pytest.approx(0.0, abs=1e-20)

    def test_sinusoidal_at_full_period(self):
        """At t=period, sin(2π)=0 so drift is zero."""
        drift = DiurnalDrift(amplitude=50e-9, period=86400.0, rejection_ratio=0.01)
        t = np.array([86400.0])
        result = drift.evaluate(t)
        assert result[0] == pytest.approx(0.0, abs=1e-18)

    def test_peak_amplitude(self):
        """Peak drift equals amplitude * rejection_ratio."""
        amp = 100e-9
        ratio = 0.05
        drift = DiurnalDrift(amplitude=amp, period=1000.0, rejection_ratio=ratio)
        t = np.linspace(0, 1000, 10_000)
        result = drift.evaluate(t)
        assert np.max(np.abs(result)) == pytest.approx(amp * ratio, rel=1e-3)

    def test_amplitude_scaling(self):
        """Doubling amplitude doubles the drift."""
        d1 = DiurnalDrift(amplitude=50e-9, rejection_ratio=0.01)
        d2 = DiurnalDrift(amplitude=100e-9, rejection_ratio=0.01)
        t = np.array([86400.0 / 4.0])
        r1 = d1.evaluate(t)
        r2 = d2.evaluate(t)
        assert r2[0] / r1[0] == pytest.approx(2.0, rel=1e-14)

    def test_rejection_ratio_scaling(self):
        """Doubling rejection_ratio doubles the drift."""
        d1 = DiurnalDrift(amplitude=50e-9, rejection_ratio=0.01)
        d2 = DiurnalDrift(amplitude=50e-9, rejection_ratio=0.02)
        t = np.array([86400.0 / 4.0])
        r1 = d1.evaluate(t)
        r2 = d2.evaluate(t)
        assert r2[0] / r1[0] == pytest.approx(2.0, rel=1e-14)

    def test_batch_computation(self):
        """Multiple timestamps computed at once."""
        drift = DiurnalDrift()
        t = np.array([0.0, 100.0, 200.0, 300.0])
        result = drift.evaluate(t)
        assert result.shape == (4,)


class TestHeadingError:
    """Tests for heading-dependent systematic error."""

    def test_zero_at_north(self):
        """sin(2*0) = 0: no error when heading north."""
        he = HeadingError(amplitude=2e-9)
        result = he.evaluate(np.array([0.0]))
        assert result[0] == pytest.approx(0.0, abs=1e-20)

    def test_zero_at_east(self):
        """sin(2*π/2) = sin(π) = 0: no error when heading east."""
        he = HeadingError(amplitude=2e-9)
        result = he.evaluate(np.array([np.pi / 2]))
        assert result[0] == pytest.approx(0.0, abs=1e-18)

    def test_zero_at_south(self):
        """sin(2*π) = 0: no error when heading south."""
        he = HeadingError(amplitude=2e-9)
        result = he.evaluate(np.array([np.pi]))
        assert result[0] == pytest.approx(0.0, abs=1e-18)

    def test_peak_at_pi_over_4(self):
        """sin(2*π/4) = sin(π/2) = 1: max error at 45 degrees."""
        amp = 2e-9
        he = HeadingError(amplitude=amp)
        result = he.evaluate(np.array([np.pi / 4]))
        assert result[0] == pytest.approx(amp, rel=1e-14)

    def test_period_is_pi(self):
        """Error has period π (repeats twice per revolution)."""
        he = HeadingError(amplitude=2e-9)
        h = np.array([0.3])
        r1 = he.evaluate(h)
        r2 = he.evaluate(h + np.pi)
        np.testing.assert_allclose(r1, r2, atol=1e-20)

    def test_amplitude_scaling(self):
        """Doubling amplitude doubles the error."""
        h1 = HeadingError(amplitude=1e-9)
        h2 = HeadingError(amplitude=2e-9)
        heading = np.array([np.pi / 4])
        r1 = h1.evaluate(heading)
        r2 = h2.evaluate(heading)
        assert r2[0] / r1[0] == pytest.approx(2.0, rel=1e-14)

    def test_batch_computation(self):
        """Multiple headings computed at once."""
        he = HeadingError(amplitude=2e-9)
        headings = np.linspace(0, 2 * np.pi, 100)
        result = he.evaluate(headings)
        assert result.shape == (100,)

    def test_negative_values(self):
        """Error is negative for headings in (π/2, π)."""
        he = HeadingError(amplitude=2e-9)
        result = he.evaluate(np.array([3 * np.pi / 4]))  # sin(3π/2) = -1
        assert result[0] == pytest.approx(-2e-9, rel=1e-14)


class TestNoiseModel:
    """Tests for combined noise model."""

    def test_default_initialization(self):
        """Default NoiseModel creates all three sub-models."""
        model = NoiseModel()
        assert isinstance(model.sensor, SensorNoise)
        assert isinstance(model.diurnal, DiurnalDrift)
        assert isinstance(model.heading, HeadingError)

    def test_custom_sub_models(self):
        """Custom sub-models are preserved."""
        sensor = SensorNoise(noise_floor=100e-12)
        model = NoiseModel(sensor=sensor)
        assert model.sensor.noise_floor == 100e-12

    def test_apply_is_additive(self):
        """Result = clean + sensor + diurnal + heading contributions."""
        clean = np.full(100, 1e-6)  # 1 µT baseline
        timestamps = np.linspace(0, 10, 100)
        headings = np.full(100, np.pi / 4)

        model = NoiseModel()
        rng = np.random.default_rng(42)
        result = model.apply(clean, timestamps, headings, rng=rng)

        # Result should differ from clean (noise added)
        assert not np.array_equal(result, clean)
        # Result should be close to clean (noise is tiny compared to 1 µT)
        assert np.allclose(result, clean, atol=1e-8)

    def test_shape_preservation(self):
        """Output shape matches input shape."""
        clean = np.zeros(500)
        timestamps = np.linspace(0, 50, 500)
        model = NoiseModel()
        result = model.apply(clean, timestamps, rng=np.random.default_rng(42))
        assert result.shape == (500,)

    def test_without_headings(self):
        """Without headings, heading error is omitted."""
        clean = np.zeros(100)
        timestamps = np.linspace(0, 10, 100)

        model = NoiseModel()
        rng = np.random.default_rng(42)
        result = model.apply(clean, timestamps, headings=None, rng=rng)
        assert result.shape == (100,)

    def test_with_headings(self):
        """With headings, result includes heading error component."""
        clean = np.zeros(100)
        timestamps = np.zeros(100)  # zero timestamps → no diurnal
        headings_zero = np.zeros(100)  # zero heading → no heading error
        headings_45 = np.full(100, np.pi / 4)  # max heading error

        model = NoiseModel(
            sensor=SensorNoise(noise_floor=0, pink_corner=0),
            diurnal=DiurnalDrift(amplitude=0),
        )
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        r_zero = model.apply(clean, timestamps, headings_zero, rng=rng1)
        r_45 = model.apply(clean, timestamps, headings_45, rng=rng2)

        # Heading at 45° should add HeadingError.amplitude to every sample
        diff = r_45 - r_zero
        assert np.allclose(diff, model.heading.amplitude, atol=1e-20)

    def test_nonzero_baseline_preserved(self):
        """Noise is added to the signal, not replacing it."""
        baseline = 5e-6
        clean = np.full(1000, baseline)
        timestamps = np.linspace(0, 100, 1000)
        model = NoiseModel()
        result = model.apply(clean, timestamps, rng=np.random.default_rng(42))
        # Mean should be very close to baseline
        assert np.mean(result) == pytest.approx(baseline, rel=0.01)


class TestPathfinderNoiseModel:
    """Tests for Pathfinder-specific noise model factory."""

    def test_returns_noise_model(self):
        """Factory returns a NoiseModel instance."""
        model = pathfinder_noise_model()
        assert isinstance(model, NoiseModel)

    def test_sensor_noise_floor(self):
        """Sensor noise floor is 50 pT."""
        model = pathfinder_noise_model()
        assert model.sensor.noise_floor == 50e-12

    def test_sensor_pink_corner(self):
        """Pink corner frequency is 0.5 Hz."""
        model = pathfinder_noise_model()
        assert model.sensor.pink_corner == 0.5

    def test_diurnal_amplitude(self):
        """Diurnal amplitude is 50 nT."""
        model = pathfinder_noise_model()
        assert model.diurnal.amplitude == 50e-9

    def test_diurnal_rejection_ratio(self):
        """Diurnal rejection ratio is 0.01 (99% rejection)."""
        model = pathfinder_noise_model()
        assert model.diurnal.rejection_ratio == 0.01

    def test_heading_error_amplitude(self):
        """Heading error amplitude is 2 nT."""
        model = pathfinder_noise_model()
        assert model.heading.amplitude == 2e-9
