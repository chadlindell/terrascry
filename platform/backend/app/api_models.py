"""Pydantic response models for the scenarios API."""

from typing import Any

from pydantic import BaseModel


class ObjectSummary(BaseModel):
    """Summary of a buried object in a scenario."""

    name: str
    object_type: str
    position: list[float]
    radius: float


class TerrainSummary(BaseModel):
    """Summary of terrain extents."""

    x_extent: list[float]
    y_extent: list[float]
    surface_elevation: float


class ScenarioSummary(BaseModel):
    """Compact representation for scenario list endpoint."""

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
