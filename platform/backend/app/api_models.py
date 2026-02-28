"""Pydantic response models for the scenarios API."""

from typing import Any

from pydantic import BaseModel


class ObjectSummary(BaseModel):
    """Summary of a buried object in a scenario."""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "Steel sphere",
                    "object_type": "ferrous_sphere",
                    "position": [5.0, 5.0, -1.0],
                    "radius": 0.05,
                }
            ]
        }
    }

    name: str
    object_type: str
    position: list[float]
    radius: float


class TerrainSummary(BaseModel):
    """Summary of terrain extents."""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "x_extent": [0.0, 20.0],
                    "y_extent": [0.0, 20.0],
                    "surface_elevation": 0.0,
                }
            ]
        }
    }

    x_extent: list[float]
    y_extent: list[float]
    surface_elevation: float


class ScenarioSummary(BaseModel):
    """Compact representation for scenario list endpoint."""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "Single Ferrous Target",
                    "file_name": "single-ferrous-target",
                    "description": "A single buried steel sphere.",
                    "object_count": 1,
                    "terrain": {
                        "x_extent": [0.0, 20.0],
                        "y_extent": [0.0, 20.0],
                        "surface_elevation": 0.0,
                    },
                }
            ]
        }
    }

    name: str
    file_name: str
    description: str
    object_count: int
    terrain: TerrainSummary


class ScenarioDetail(BaseModel):
    """Full scenario representation for detail endpoint."""

    name: str
    file_name: str
    description: str
    object_count: int
    terrain: TerrainSummary
    objects: list[ObjectSummary]
    earth_field: list[float]
    metadata: dict[str, Any]
    raw: dict[str, Any]
