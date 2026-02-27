"""
Real-time tilt correction for Pathfinder gradiometer.

The gradiometer must be level. Tilt introduces error proportional to
Earth's field: error = earth_field * sin(tilt_angle).

At 2 deg tilt with 50,000 nT Earth field: ~1,750 nT error -- far larger
than most anomaly signals (10-500 nT).

With BNO055 IMU providing real-time pitch/roll, we correct each reading:
    corrected_grad = raw_grad - earth_field_vertical * sin(pitch)
                              - earth_field_horizontal * sin(roll)
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class EarthField:
    """Local Earth's magnetic field components.

    Parameters
    ----------
    total_nt : float
        Total field magnitude in nanotesla.
    inclination_deg : float
        Dip angle in degrees (positive = downward into ground).
    declination_deg : float
        Magnetic vs true north in degrees.
    """

    total_nt: float = 50000.0       # Total field magnitude
    inclination_deg: float = 65.0   # Dip angle (positive = downward)
    declination_deg: float = 5.0    # Magnetic vs true north

    @property
    def vertical_nt(self) -> float:
        """Vertical component (into ground)."""
        return self.total_nt * np.sin(np.radians(self.inclination_deg))

    @property
    def horizontal_nt(self) -> float:
        """Horizontal component."""
        return self.total_nt * np.cos(np.radians(self.inclination_deg))


def compute_tilt_correction(
    pitch_deg: float,
    roll_deg: float,
    earth_field: EarthField,
) -> float:
    """Compute gradient error from tilt.

    Parameters
    ----------
    pitch_deg : float
        Forward/backward tilt in degrees.
    roll_deg : float
        Left/right tilt in degrees.
    earth_field : EarthField
        Local Earth field parameters.

    Returns
    -------
    float
        Estimated gradient error in nT to subtract from raw reading.
    """
    pitch_rad = np.radians(pitch_deg)
    roll_rad = np.radians(roll_deg)

    # Tilt error: projection of Earth's field onto tilted sensor axis
    # For small angles (< 10 deg), sin(theta) ~ theta is accurate to < 0.5%
    error_pitch = earth_field.vertical_nt * np.sin(pitch_rad)
    error_roll = earth_field.horizontal_nt * np.sin(roll_rad)

    return error_pitch + error_roll


def correct_gradient(
    raw_gradient_nt: float,
    pitch_deg: float,
    roll_deg: float,
    earth_field: EarthField,
) -> tuple[float, float]:
    """Apply tilt correction to a gradiometer reading.

    Parameters
    ----------
    raw_gradient_nt : float
        Uncorrected gradiometer reading in nT.
    pitch_deg : float
        Forward/backward tilt in degrees.
    roll_deg : float
        Left/right tilt in degrees.
    earth_field : EarthField
        Local Earth field parameters.

    Returns
    -------
    tuple[float, float]
        (corrected_gradient_nt, correction_applied_nt)
    """
    correction = compute_tilt_correction(pitch_deg, roll_deg, earth_field)
    return raw_gradient_nt - correction, correction


def batch_correct(
    raw_gradients: np.ndarray,
    pitches_deg: np.ndarray,
    rolls_deg: np.ndarray,
    earth_field: EarthField,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized tilt correction for batch processing.

    Parameters
    ----------
    raw_gradients : np.ndarray
        Array of raw gradient readings in nT.
    pitches_deg : np.ndarray
        Array of pitch angles in degrees.
    rolls_deg : np.ndarray
        Array of roll angles in degrees.
    earth_field : EarthField
        Local Earth field parameters.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (corrected_gradients, corrections_applied) both in nT.
    """
    corrections = (
        earth_field.vertical_nt * np.sin(np.radians(pitches_deg))
        + earth_field.horizontal_nt * np.sin(np.radians(rolls_deg))
    )
    return raw_gradients - corrections, corrections
