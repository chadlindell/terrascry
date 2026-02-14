"""Tests for motion-induced noise model.

Validates MotionNoise behavior: speed scaling, zero-speed silence,
spectral characteristics, and backward compatibility of NoiseModel.apply().
"""

import numpy as np
import pytest

from geosim.noise.models import MotionNoise, NoiseModel, SensorNoise


class TestMotionNoise:
    """Tests for the MotionNoise dataclass."""

    def test_zero_speed_produces_zero_noise(self):
        """At zero speed, motion noise should be zero."""
        mn = MotionNoise()
        n = 1000
        timestamps = np.linspace(0, 10, n)
        speeds = np.zeros(n)
        field_gradient = np.full(n, 1e-6)  # 1 µT/m gradient

        noise = mn.evaluate(timestamps, speeds, field_gradient)
        np.testing.assert_array_equal(noise, 0.0)

    def test_amplitude_scales_with_speed(self):
        """Motion noise amplitude should scale linearly with speed."""
        mn = MotionNoise(pendulum_amplitude=0.01, reference_speed=1.5)
        n = 10000
        timestamps = np.linspace(0, 100, n)
        field_gradient = np.full(n, 1e-6)

        speeds_slow = np.full(n, 0.5)
        speeds_fast = np.full(n, 1.5)

        noise_slow = mn.evaluate(timestamps, speeds_slow, field_gradient)
        noise_fast = mn.evaluate(timestamps, speeds_fast, field_gradient)

        rms_slow = np.sqrt(np.mean(noise_slow**2))
        rms_fast = np.sqrt(np.mean(noise_fast**2))

        # RMS should scale with speed ratio (3x speed = 3x noise)
        assert rms_fast / rms_slow == pytest.approx(3.0, rel=0.1)

    def test_amplitude_scales_with_gradient(self):
        """Motion noise should be proportional to field gradient."""
        mn = MotionNoise()
        n = 10000
        timestamps = np.linspace(0, 100, n)
        speeds = np.full(n, 1.5)

        grad_weak = np.full(n, 1e-7)
        grad_strong = np.full(n, 1e-6)

        noise_weak = mn.evaluate(timestamps, speeds, grad_weak)
        noise_strong = mn.evaluate(timestamps, speeds, grad_strong)

        rms_weak = np.sqrt(np.mean(noise_weak**2))
        rms_strong = np.sqrt(np.mean(noise_strong**2))

        assert rms_strong / rms_weak == pytest.approx(10.0, rel=0.1)

    def test_spectral_peak_at_step_frequency(self):
        """Motion noise should show a spectral peak near step_frequency."""
        step_freq = 2.0
        mn = MotionNoise(
            pendulum_amplitude=0.01,
            step_frequency=step_freq,
        )
        sample_rate = 100.0
        n = 10000
        timestamps = np.linspace(0, n / sample_rate, n)
        speeds = np.full(n, 1.5)
        field_gradient = np.full(n, 1e-6)

        noise = mn.evaluate(timestamps, speeds, field_gradient)

        # FFT analysis
        fft_mag = np.abs(np.fft.rfft(noise))
        freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

        # Find the peak frequency (excluding DC)
        fft_mag[0] = 0
        peak_idx = np.argmax(fft_mag)
        peak_freq = freqs[peak_idx]

        assert peak_freq == pytest.approx(step_freq, abs=0.2)

    def test_phase_offset_shifts_waveform(self):
        """Different phase offsets produce different noise waveforms."""
        mn = MotionNoise()
        n = 1000
        timestamps = np.linspace(0, 10, n)
        speeds = np.full(n, 1.5)
        field_gradient = np.full(n, 1e-6)

        noise_a = mn.evaluate(timestamps, speeds, field_gradient, phase_offset=0.0)
        noise_b = mn.evaluate(timestamps, speeds, field_gradient, phase_offset=np.pi / 2)

        # Should be different but same RMS
        assert not np.array_equal(noise_a, noise_b)
        rms_a = np.sqrt(np.mean(noise_a**2))
        rms_b = np.sqrt(np.mean(noise_b**2))
        assert rms_a == pytest.approx(rms_b, rel=0.1)

    def test_default_parameters(self):
        """Default MotionNoise parameters are physically reasonable."""
        mn = MotionNoise()
        assert mn.pendulum_amplitude == 0.005
        assert mn.step_frequency == 2.0
        assert mn.speed_noise_scale == 1.5
        assert mn.reference_speed == 1.5


class TestNoiseModelMotionIntegration:
    """Tests for motion noise integration into NoiseModel.apply()."""

    def test_backward_compatible_without_speeds(self):
        """NoiseModel.apply() works without speeds parameter (backward compat)."""
        model = NoiseModel()
        clean = np.zeros(100)
        timestamps = np.linspace(0, 10, 100)
        rng = np.random.default_rng(42)

        # Should not raise
        result = model.apply(clean, timestamps, rng=rng)
        assert result.shape == (100,)

    def test_backward_compatible_without_field_gradient(self):
        """Providing speeds without field_gradient omits motion noise."""
        model = NoiseModel(
            sensor=SensorNoise(noise_floor=0, pink_corner=0),
        )
        clean = np.zeros(100)
        timestamps = np.linspace(0, 10, 100)
        speeds = np.full(100, 1.5)

        rng = np.random.default_rng(42)
        result = model.apply(clean, timestamps, rng=rng, speeds=speeds)
        assert result.shape == (100,)

    def test_motion_noise_adds_to_signal(self):
        """With speeds and gradient, motion noise contributes to output."""
        model = NoiseModel(
            sensor=SensorNoise(noise_floor=0, pink_corner=0),
            motion=MotionNoise(pendulum_amplitude=0.01, speed_noise_scale=0),
        )
        clean = np.zeros(1000)
        timestamps = np.linspace(0, 10, 1000)
        speeds = np.full(1000, 1.5)
        field_gradient = np.full(1000, 1e-5)

        rng = np.random.default_rng(42)
        result_no_motion = model.apply(clean, timestamps, rng=rng)
        rng2 = np.random.default_rng(42)
        result_with_motion = model.apply(
            clean, timestamps, rng=rng2,
            speeds=speeds, field_gradient=field_gradient,
        )

        rms_no = np.sqrt(np.mean(result_no_motion**2))
        rms_with = np.sqrt(np.mean(result_with_motion**2))

        # Motion noise should increase total noise
        assert rms_with > rms_no

    def test_speed_dependent_sensor_noise_scaling(self):
        """Sensor noise floor scales up with operator speed."""
        model = NoiseModel(
            sensor=SensorNoise(noise_floor=50e-12, pink_corner=0),
            motion=MotionNoise(speed_noise_scale=2.0, pendulum_amplitude=0),
        )
        n = 100_000
        clean = np.zeros(n)
        timestamps = np.linspace(0, 1000, n)
        field_gradient = np.zeros(n)

        # Stationary
        rng1 = np.random.default_rng(42)
        result_still = model.apply(
            clean, timestamps, rng=rng1,
            speeds=np.zeros(n), field_gradient=field_gradient,
        )

        # Walking at reference speed
        rng2 = np.random.default_rng(43)
        result_walk = model.apply(
            clean, timestamps, rng=rng2,
            speeds=np.full(n, 1.5), field_gradient=field_gradient,
        )

        rms_still = np.std(result_still)
        rms_walk = np.std(result_walk)

        # At reference speed, scale factor = 1 + 2.0 = 3.0
        assert rms_walk / rms_still == pytest.approx(3.0, rel=0.15)

    def test_noise_model_has_motion_attribute(self):
        """NoiseModel includes MotionNoise by default."""
        model = NoiseModel()
        assert isinstance(model.motion, MotionNoise)
