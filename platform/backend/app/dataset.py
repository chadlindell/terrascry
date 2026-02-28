"""Unified Dataset model — the core data contract between backend and frontend.

Design decisions (consensus-validated):
- GridData uses flat row-major list[float] with geometry metadata for direct
  mapping to WebGL TypedArrays (deck.gl BitmapLayer).
- PointData stores gradient + position only (B_bottom/B_top deferred to
  future tooltip endpoint).
- Gradient values stored in nanoTesla (nT) matching Pathfinder conventions.
"""

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class GridData(BaseModel):
    """Regular grid of gradient values for 2D heatmap rendering.

    Values are stored as a flat row-major array. To reconstruct the 2D grid:
        grid[row][col] = values[row * cols + col]
    where row 0 is y_min, col 0 is x_min.
    """

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "rows": 3,
                    "cols": 3,
                    "x_min": 0.0,
                    "y_min": 0.0,
                    "dx": 1.0,
                    "dy": 1.0,
                    "values": [0.1, 0.3, 0.2, 0.5, 12.4, 0.6, 0.2, 0.4, 0.1],
                    "unit": "nT",
                }
            ]
        }
    }

    rows: int
    cols: int
    x_min: float
    y_min: float
    dx: float  # cell spacing in x (meters)
    dy: float  # cell spacing in y (meters)
    values: list[float]  # flat row-major, gradient in nT
    unit: str = "nT"


class SurveyPoint(BaseModel):
    """A single survey reading along the walk path."""

    model_config = {
        "json_schema_extra": {
            "examples": [{"x": 5.0, "y": 5.0, "gradient_nt": 12.4}]
        }
    }

    x: float  # meters
    y: float  # meters
    gradient_nt: float  # B_bottom - B_top, in nanoTesla


class DatasetMetadata(BaseModel):
    """Metadata for a simulation dataset."""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                    "scenario_name": "single-ferrous-target",
                    "created_at": "2026-01-15T12:00:00Z",
                    "params": {"resolution": 0.5, "line_spacing": 1.0},
                }
            ]
        }
    }

    id: UUID = Field(default_factory=uuid4)
    scenario_name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    params: dict[str, Any] = Field(default_factory=dict)


class Dataset(BaseModel):
    """Complete simulation output: metadata + grid + survey points."""

    metadata: DatasetMetadata
    grid_data: GridData
    survey_points: list[SurveyPoint]
