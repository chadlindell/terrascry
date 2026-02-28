"""PhysicsEngine service — TERRASCRY survey simulation.

Uses the geosim physics engine internally for dipole field calculations.

Design decisions (consensus-validated):
- Direct Python import of the physics engine (no ZMQ) — same process, lower complexity.
- Single centered gradiometer for both grid and path (v1 MVP).
- Snake/zigzag path generation with configurable spacing.
"""

from __future__ import annotations

import numpy as np
from geosim.magnetics.dipole import gradiometer_reading
from geosim.scenarios.loader import Scenario, load_scenario

from app.config import settings
from app.dataset import Dataset, DatasetMetadata, GridData, SurveyPoint

# Pathfinder defaults
SENSOR_SEPARATION = 0.35  # meters, vertical baseline
SENSOR_HEIGHT = 0.175  # meters, bottom sensor above ground


class PhysicsEngine:
    """TERRASCRY physics engine interface for web platform simulation."""

    def load_scenario(self, name: str) -> Scenario:
        """Load a scenario by name from the configured scenarios directory."""
        path = settings.scenarios_dir / f"{name}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Scenario '{name}' not found at {path}")
        return load_scenario(path)

    def generate_path(
        self,
        scenario: Scenario,
        line_spacing: float = 1.0,
        sample_spacing: float = 0.5,
    ) -> np.ndarray:
        """Generate a snake/zigzag survey path within scenario extents.

        Parameters
        ----------
        scenario : Scenario
            Loaded scenario with terrain extents.
        line_spacing : float
            Cross-track distance between survey lines (meters).
        sample_spacing : float
            Along-track distance between readings (meters).

        Returns
        -------
        path : ndarray, shape (N, 3)
            Survey positions [x, y, z] where z = SENSOR_HEIGHT.
        """
        x_min, x_max = scenario.terrain.x_extent
        y_min, y_max = scenario.terrain.y_extent

        # Build survey lines along Y, spaced in X
        x_lines = np.arange(x_min, x_max + line_spacing / 2, line_spacing)
        points = []

        for i, x in enumerate(x_lines):
            y_vals = np.arange(y_min, y_max + sample_spacing / 2, sample_spacing)
            # Reverse every other line for snake pattern
            if i % 2 == 1:
                y_vals = y_vals[::-1]
            for y in y_vals:
                points.append([x, y, SENSOR_HEIGHT])

        return np.array(points, dtype=np.float64)

    def simulate_survey(
        self,
        scenario: Scenario,
        path: np.ndarray,
    ) -> list[SurveyPoint]:
        """Simulate gradiometer readings along a survey path.

        Parameters
        ----------
        scenario : Scenario
            Loaded scenario (must have magnetic_sources computed).
        path : ndarray, shape (N, 3)
            Survey positions.

        Returns
        -------
        points : list[SurveyPoint]
            Survey readings with gradient in nT.
        """
        sources = scenario.magnetic_sources
        if not sources:
            # No magnetic objects — return zero readings
            return [
                SurveyPoint(x=float(p[0]), y=float(p[1]), gradient_nt=0.0) for p in path
            ]

        _, _, gradient = gradiometer_reading(
            path, sources, SENSOR_SEPARATION
        )
        # Convert Tesla → nanoTesla
        gradient_nt = gradient * 1e9

        return [
            SurveyPoint(x=float(path[i, 0]), y=float(path[i, 1]), gradient_nt=float(gradient_nt[i]))
            for i in range(len(path))
        ]

    def compute_grid(
        self,
        scenario: Scenario,
        resolution: float = 0.5,
    ) -> GridData:
        """Compute a regular grid of gradient values over the scenario area.

        Parameters
        ----------
        scenario : Scenario
            Loaded scenario.
        resolution : float
            Grid cell size in meters.

        Returns
        -------
        grid : GridData
            Flat row-major grid of gradient values in nT.
        """
        x_min, x_max = scenario.terrain.x_extent
        y_min, y_max = scenario.terrain.y_extent

        x_vals = np.arange(x_min, x_max + resolution / 2, resolution)
        y_vals = np.arange(y_min, y_max + resolution / 2, resolution)
        cols = len(x_vals)
        rows = len(y_vals)

        # Build grid positions: row-major (y varies slowest)
        xx, yy = np.meshgrid(x_vals, y_vals)
        positions = np.column_stack([
            xx.ravel(),
            yy.ravel(),
            np.full(rows * cols, SENSOR_HEIGHT),
        ])

        sources = scenario.magnetic_sources
        if not sources:
            values = [0.0] * (rows * cols)
        else:
            _, _, gradient = gradiometer_reading(
                positions, sources, SENSOR_SEPARATION
            )
            values = (gradient * 1e9).tolist()  # Tesla → nT

        return GridData(
            rows=rows,
            cols=cols,
            x_min=float(x_min),
            y_min=float(y_min),
            dx=resolution,
            dy=resolution,
            values=values,
        )

    def simulate(
        self,
        scenario_name: str,
        line_spacing: float = 1.0,
        sample_spacing: float = 0.5,
        resolution: float = 0.5,
    ) -> Dataset:
        """Full simulation pipeline: load → path → survey → grid → Dataset.

        Parameters
        ----------
        scenario_name : str
            Name of the scenario JSON file (without extension).
        line_spacing : float
            Survey line spacing in meters.
        sample_spacing : float
            Along-line sample spacing in meters.
        resolution : float
            Grid resolution in meters.

        Returns
        -------
        dataset : Dataset
            Complete simulation results.
        """
        scenario = self.load_scenario(scenario_name)
        path = self.generate_path(scenario, line_spacing, sample_spacing)
        survey_points = self.simulate_survey(scenario, path)
        grid_data = self.compute_grid(scenario, resolution)

        return Dataset(
            metadata=DatasetMetadata(
                scenario_name=scenario_name,
                params={
                    "line_spacing": line_spacing,
                    "sample_spacing": sample_spacing,
                    "resolution": resolution,
                },
            ),
            grid_data=grid_data,
            survey_points=survey_points,
        )
