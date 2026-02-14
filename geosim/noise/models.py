"""Realistic noise models for geophysical sensors.

Noise sources modeled:
- Sensor noise floor (white noise / 1/f noise)
- Heading error (orientation-dependent systematic error)
- Diurnal drift (slow temporal variation from Earth field changes)
- Thermal drift (temperature-dependent sensor offset)

All noise values are in Tesla unless otherwise specified.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SensorNoise:
    """White + 1/f sensor noise model.

    Parameters
    ----------
    noise_floor : float
        RMS noise in Tesla. Typical fluxgate: 10-100 pT.
    pink_corner : float
        1/f corner frequency in Hz. Below this, noise power rises as 1/f.
        Typical fluxgate: 0.1-1.0 Hz.
    """

    noise_floor: float = 50e-12  # 50 pT RMS (typical fluxgate)
    pink_corner: float = 0.5  # Hz

    def generate(
        self,
        n_samples: int,
        sample_rate: float = 10.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Generate sensor noise time series.

        Parameters
        ----------
        n_samples : int
            Number of samples.
        sample_rate : float
            Sample rate in Hz.
        rng : numpy Generator, optional
            Random number generator for reproducibility.

        Returns
        -------
        noise : ndarray, shape (n_samples,)
            Noise values in Tesla.
        """
        if rng is None:
            rng = np.random.default_rng()

        # White noise component
        white = rng.normal(0, self.noise_floor, n_samples)

        if self.pink_corner <= 0 or n_samples < 2:
            return white

        # 1/f noise via spectral shaping
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
        white_fft = np.fft.rfft(white)

        # Shape: amplify low frequencies below corner
        with np.errstate(divide='ignore', invalid='ignore'):
            pink_filter = np.where(
                freqs > 0,
                np.sqrt(np.minimum(self.pink_corner / freqs, 10.0)),
                1.0,
            )
        pink_filter[0] = 0  # Remove DC

        shaped = np.fft.irfft(white_fft * pink_filter, n=n_samples)

        # Normalize to maintain target RMS
        if np.std(shaped) > 0:
            shaped *= self.noise_floor / np.std(shaped)

        return shaped


@dataclass
class DiurnalDrift:
    """Diurnal magnetic field variation model.

    The Earth's field varies over the course of a day due to ionospheric
    currents. Typical amplitude: 20-100 nT peak-to-peak.

    This is largely cancelled by the gradiometer configuration, but residual
    effects remain due to imperfect sensor matching.

    Parameters
    ----------
    amplitude : float
        Peak amplitude of diurnal variation in Tesla.
    period : float
        Period in seconds (default: 24 hours).
    rejection_ratio : float
        How much the gradiometer rejects common-mode drift.
        1.0 = no rejection, 0.001 = 60 dB rejection.
    """

    amplitude: float = 50e-9  # 50 nT typical mid-latitude
    period: float = 86400.0  # 24 hours
    rejection_ratio: float = 0.01  # gradiometer rejects ~99%

    def evaluate(self, timestamps: np.ndarray) -> np.ndarray:
        """Compute diurnal drift at given timestamps.

        Parameters
        ----------
        timestamps : ndarray
            Time values in seconds from survey start.

        Returns
        -------
        drift : ndarray
            Drift contribution in Tesla (after gradiometer rejection).
        """
        phase = 2.0 * np.pi * timestamps / self.period
        # Simplified sinusoidal model
        raw_drift = self.amplitude * np.sin(phase)
        return raw_drift * self.rejection_ratio


@dataclass
class HeadingError:
    """Heading-dependent systematic error.

    Fluxgate sensors have a small sensitivity to orientation relative
    to the Earth's field. This produces a systematic error that varies
    with walking direction.

    Parameters
    ----------
    amplitude : float
        Peak heading error amplitude in Tesla.
    """

    amplitude: float = 2e-9  # 2 nT typical for well-calibrated fluxgate

    def evaluate(self, headings: np.ndarray) -> np.ndarray:
        """Compute heading error.

        Parameters
        ----------
        headings : ndarray
            Heading angles in radians (0=North, π/2=East).

        Returns
        -------
        error : ndarray
            Heading error in Tesla.
        """
        # sin(2θ) pattern typical of fluxgate heading error
        return self.amplitude * np.sin(2.0 * headings)


@dataclass
class MotionNoise:
    """Motion-induced noise from boom pendulation during walking.

    Real gradiometers are highly sensitive to operator motion — the boom
    oscillates with each step, modulating sensor height and introducing
    speed-correlated noise. This is often the dominant noise source in
    field surveys.

    Parameters
    ----------
    pendulum_amplitude : float
        Peak boom oscillation amplitude in meters at reference walk speed.
    step_frequency : float
        Gait frequency in Hz (typical walking: ~2 Hz).
    speed_noise_scale : float
        Multiplier for sensor noise floor at reference walk speed.
        Total sensor noise is scaled by ``1 + scale * (v / v_ref)``.
    reference_speed : float
        Reference walk speed in m/s for scaling.
    """

    pendulum_amplitude: float = 0.005  # 5 mm boom oscillation at walk speed
    step_frequency: float = 2.0  # Hz (gait frequency)
    speed_noise_scale: float = 1.5  # noise floor multiplier at walk speed
    reference_speed: float = 1.5  # m/s

    def evaluate(
        self,
        timestamps: np.ndarray,
        speeds: np.ndarray,
        field_gradient: np.ndarray,
        phase_offset: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Compute motion-induced noise.

        Parameters
        ----------
        timestamps : ndarray, shape (N,)
            Time values in seconds.
        speeds : ndarray, shape (N,)
            Operator speed at each sample in m/s.
        field_gradient : ndarray, shape (N,)
            Local vertical field gradient (dBz/dz) in T/m at each sample.
        phase_offset : float
            Phase offset in radians (different per sensor pair).
        rng : Generator, optional
            Random number generator (unused, reserved for future jitter).

        Returns
        -------
        noise : ndarray, shape (N,)
            Additive motion noise in Tesla.
        """
        v_ratio = speeds / self.reference_speed

        # Pendulum height oscillation: amplitude scales with speed
        dz = (
            self.pendulum_amplitude
            * v_ratio
            * np.sin(2.0 * np.pi * self.step_frequency * timestamps + phase_offset)
        )

        # Gradient modulation: boom height change × local gradient
        gradient_noise = field_gradient * dz

        return gradient_noise


@dataclass
class NoiseModel:
    """Combined noise model for a sensor channel.

    Aggregates all noise sources for easy application to clean sensor data.
    """

    sensor: SensorNoise = None
    diurnal: DiurnalDrift = None
    heading: HeadingError = None
    motion: MotionNoise = None

    def __post_init__(self):
        if self.sensor is None:
            self.sensor = SensorNoise()
        if self.diurnal is None:
            self.diurnal = DiurnalDrift()
        if self.heading is None:
            self.heading = HeadingError()
        if self.motion is None:
            self.motion = MotionNoise()

    def apply(
        self,
        clean_signal: np.ndarray,
        timestamps: np.ndarray,
        headings: np.ndarray | None = None,
        sample_rate: float = 10.0,
        rng: np.random.Generator | None = None,
        speeds: np.ndarray | None = None,
        field_gradient: np.ndarray | None = None,
        phase_offset: float = 0.0,
    ) -> np.ndarray:
        """Apply all noise sources to a clean signal.

        Parameters
        ----------
        clean_signal : ndarray
            Clean sensor readings in Tesla.
        timestamps : ndarray
            Time values in seconds.
        headings : ndarray, optional
            Heading angles in radians. If None, heading error is omitted.
        sample_rate : float
            Sample rate in Hz.
        rng : Generator, optional
            Random number generator.
        speeds : ndarray, optional
            Operator speed at each sample in m/s. If None, motion noise is
            omitted (backward compatible).
        field_gradient : ndarray, optional
            Local vertical field gradient in T/m. Required with speeds.
        phase_offset : float
            Phase offset for motion noise (radians).

        Returns
        -------
        noisy_signal : ndarray
            Signal with all noise sources added.
        """
        n = len(clean_signal)
        result = clean_signal.copy()

        # Speed-dependent sensor noise scaling
        if speeds is not None:
            v_ratio = speeds / self.motion.reference_speed
            scale_factors = 1.0 + self.motion.speed_noise_scale * v_ratio
            result += self.sensor.generate(n, sample_rate, rng) * scale_factors
        else:
            result += self.sensor.generate(n, sample_rate, rng)

        # Diurnal drift
        result += self.diurnal.evaluate(timestamps)

        # Heading error
        if headings is not None:
            result += self.heading.evaluate(headings)

        # Motion-induced noise (pendulum + gradient modulation)
        if speeds is not None and field_gradient is not None:
            result += self.motion.evaluate(
                timestamps, speeds, field_gradient, phase_offset, rng
            )

        return result


def hirt_fdem_noise_model() -> NoiseModel:
    """Create a noise model with HIRT FDEM-typical parameters.

    HIRT FDEM noise is electronics-dominated with no heading error
    (stationary borehole probes) and minimal diurnal drift
    (rejected by differential measurement).

    Returns
    -------
    model : NoiseModel
        HIRT FDEM noise model.
    """
    return NoiseModel(
        sensor=SensorNoise(noise_floor=1e-4, pink_corner=0.1),
        diurnal=DiurnalDrift(amplitude=50e-9, rejection_ratio=0.001),
        heading=HeadingError(amplitude=0.0),
    )


def pathfinder_noise_model() -> NoiseModel:
    """Create a noise model with Pathfinder-typical parameters.

    Based on Pathfinder design-concept.md:
    - Fluxgate noise floor: ~50 pT RMS
    - Gradiometer rejects >99% of common-mode drift
    - Target gradient noise: <20 ADC counts RMS
    - Motion noise: 5mm boom pendulation at walk speed
    """
    return NoiseModel(
        sensor=SensorNoise(noise_floor=50e-12, pink_corner=0.5),
        diurnal=DiurnalDrift(amplitude=50e-9, rejection_ratio=0.01),
        heading=HeadingError(amplitude=2e-9),
        motion=MotionNoise(
            pendulum_amplitude=0.005,
            step_frequency=2.0,
            speed_noise_scale=1.5,
        ),
    )
