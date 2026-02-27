"""
Real-time anomaly detection for Pathfinder gradiometer data.

Uses running statistics to track the background field and flag residual
spikes as anomalies. Designed to run on NVIDIA Jetson edge compute for
<1ms per sample latency.

A more sophisticated EKF/LSTM version is planned for Phase 2.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class AnomalyDetectorConfig:
    """Configuration for real-time anomaly detection.

    Parameters
    ----------
    background_window : int
        Number of samples for rolling background estimation.
    detection_threshold_sigma : float
        Number of standard deviations above background to flag as anomaly.
    min_anomaly_duration : int
        Minimum consecutive anomaly samples required to confirm detection.
    max_anomaly_gap : int
        Maximum gap between anomaly samples to merge into one event.
    noise_floor_nt : float
        Instrument noise floor in nT; lower bound for background std.
    """

    background_window: int = 50
    detection_threshold_sigma: float = 3.0
    min_anomaly_duration: int = 3
    max_anomaly_gap: int = 2
    noise_floor_nt: float = 0.5


class SimpleAnomalyDetector:
    """Running-statistics anomaly detector for real-time use.

    Maintains a rolling estimate of background gradient and its variance.
    Samples exceeding threshold * sigma from background are flagged.

    This is the simple version suitable for immediate deployment.
    A more sophisticated EKF/LSTM version is planned for Phase 2.

    Parameters
    ----------
    config : AnomalyDetectorConfig, optional
        Detection configuration. Uses defaults if not provided.
    """

    def __init__(self, config: AnomalyDetectorConfig | None = None):
        self.config = config or AnomalyDetectorConfig()
        self.buffer: list[float] = []
        self.anomaly_count: int = 0
        self.consecutive_anomaly: int = 0

    @property
    def background_mean(self) -> float:
        """Current estimate of background gradient mean."""
        if not self.buffer:
            return 0.0
        return float(np.mean(self.buffer))

    @property
    def background_std(self) -> float:
        """Current estimate of background gradient standard deviation.

        Returns at least ``config.noise_floor_nt`` to avoid division by zero
        and to reflect the instrument's intrinsic noise.
        """
        if len(self.buffer) < 3:
            return self.config.noise_floor_nt
        return max(float(np.std(self.buffer)), self.config.noise_floor_nt)

    def process_sample(self, gradient_nt: float) -> dict:
        """Process a single gradient sample and return detection result.

        Parameters
        ----------
        gradient_nt : float
            Tilt-corrected gradient reading in nT.

        Returns
        -------
        dict
            Detection result with keys:
            - ``is_anomaly`` : bool -- True if confirmed anomaly
            - ``gradient_nt`` : float -- Input gradient value
            - ``background_nt`` : float -- Current background estimate
            - ``residual_nt`` : float -- Deviation from background
            - ``sigma`` : float -- Residual expressed in std units
            - ``threshold_nt`` : float -- Current detection threshold in nT
        """
        bg_mean = self.background_mean
        bg_std = self.background_std
        residual = gradient_nt - bg_mean
        threshold = self.config.detection_threshold_sigma * bg_std
        is_anomaly = abs(residual) > threshold

        if is_anomaly:
            self.consecutive_anomaly += 1
            confirmed = self.consecutive_anomaly >= self.config.min_anomaly_duration
            if confirmed:
                self.anomaly_count += 1
        else:
            self.consecutive_anomaly = 0
            # Only update background with non-anomaly samples
            self.buffer.append(gradient_nt)
            if len(self.buffer) > self.config.background_window:
                self.buffer.pop(0)

        return {
            "is_anomaly": (
                is_anomaly and self.consecutive_anomaly >= self.config.min_anomaly_duration
            ),
            "gradient_nt": gradient_nt,
            "background_nt": bg_mean,
            "residual_nt": residual,
            "sigma": abs(residual) / bg_std if bg_std > 0 else 0,
            "threshold_nt": threshold,
        }

    def reset(self) -> None:
        """Reset detector state, clearing all history."""
        self.buffer.clear()
        self.anomaly_count = 0
        self.consecutive_anomaly = 0
