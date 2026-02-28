"""Anomaly detection service for stored datasets.

Scans grid values through SimpleAnomalyDetector and returns cells
that exceed the detection threshold.
"""

from __future__ import annotations

from geosim.streaming.anomaly_detector import AnomalyDetectorConfig, SimpleAnomalyDetector

from app.dataset import GridData


class AnomalyCell:
    """A single grid cell flagged as anomalous."""

    __slots__ = ("x", "y", "gradient_nt", "residual_nt", "sigma")

    def __init__(self, x: float, y: float, gradient_nt: float, residual_nt: float, sigma: float):
        self.x = x
        self.y = y
        self.gradient_nt = gradient_nt
        self.residual_nt = residual_nt
        self.sigma = sigma

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "gradient_nt": self.gradient_nt,
            "residual_nt": self.residual_nt,
            "sigma": self.sigma,
        }


class AnomalyService:
    """Detect anomalous cells in a grid dataset."""

    def detect(self, grid: GridData, threshold_sigma: float = 3.0) -> list[dict]:
        """Scan grid values and return cells exceeding detection threshold.

        Parameters
        ----------
        grid : GridData
            Grid of gradient values to analyze.
        threshold_sigma : float
            Detection threshold in standard deviations.

        Returns
        -------
        list[dict]
            Anomaly cells with position, gradient, residual, and sigma.
        """
        config = AnomalyDetectorConfig(detection_threshold_sigma=threshold_sigma)
        detector = SimpleAnomalyDetector(config)

        anomalies: list[dict] = []

        for row in range(grid.rows):
            for col in range(grid.cols):
                val = grid.values[row * grid.cols + col]
                result = detector.process_sample(val)

                if result["is_anomaly"]:
                    x = grid.x_min + col * grid.dx
                    y = grid.y_min + row * grid.dy
                    cell = AnomalyCell(
                        x=x,
                        y=y,
                        gradient_nt=result["gradient_nt"],
                        residual_nt=result["residual_nt"],
                        sigma=result["sigma"],
                    )
                    anomalies.append(cell.to_dict())

        return anomalies
