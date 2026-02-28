"""Scenarios router — list, retrieve, and upload TERRASCRY scenario files."""

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.api_models import (
    ObjectSummary,
    ScenarioDetail,
    ScenarioSummary,
    TerrainSummary,
)
from app.config import settings

router = APIRouter(prefix="/api/scenarios", tags=["scenarios"])


def _read_scenario(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _terrain_summary(data: dict) -> TerrainSummary:
    terrain = data.get("terrain", {})
    return TerrainSummary(
        x_extent=terrain.get("x_extent", [0.0, 20.0]),
        y_extent=terrain.get("y_extent", [0.0, 20.0]),
        surface_elevation=terrain.get("surface_elevation", 0.0),
    )


def _to_summary(file_name: str, data: dict) -> ScenarioSummary:
    return ScenarioSummary(
        name=data.get("name", file_name),
        file_name=file_name,
        description=data.get("description", ""),
        object_count=len(data.get("objects", [])),
        terrain=_terrain_summary(data),
    )


def _to_detail(file_name: str, data: dict) -> ScenarioDetail:
    objects = [
        ObjectSummary(
            name=obj["name"],
            object_type=obj.get("type", "unknown"),
            position=obj["position"],
            radius=obj.get("radius", 0.0),
        )
        for obj in data.get("objects", [])
    ]
    return ScenarioDetail(
        name=data.get("name", file_name),
        file_name=file_name,
        description=data.get("description", ""),
        object_count=len(objects),
        terrain=_terrain_summary(data),
        objects=objects,
        earth_field=data.get("earth_field", [0.0, 20e-6, 45e-6]),
        metadata=data.get("metadata", {}),
        raw=data,
    )


@router.get("", response_model=list[ScenarioSummary])
async def list_scenarios() -> list[ScenarioSummary]:
    scenarios_dir = settings.scenarios_dir
    if not scenarios_dir.is_dir():
        return []
    results = []
    for path in sorted(scenarios_dir.glob("*.json")):
        if path.stem.endswith("_meta") or path.stem.endswith("_center"):
            continue
        data = _read_scenario(path)
        results.append(_to_summary(path.stem, data))
    return results


@router.get("/{name}", response_model=ScenarioDetail)
async def get_scenario(name: str) -> ScenarioDetail:
    path = settings.scenarios_dir / f"{name}.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Scenario '{name}' not found")
    data = _read_scenario(path)
    return _to_detail(name, data)


class ScenarioUpload(BaseModel):
    """Request body for uploading a new scenario."""

    name: str
    description: str = ""
    terrain: dict[str, Any] = {}
    objects: list[dict[str, Any]] = []
    earth_field: list[float] = [0.0, 20e-6, 45e-6]
    metadata: dict[str, Any] = {}


@router.post("", response_model=ScenarioDetail, status_code=201)
async def upload_scenario(body: ScenarioUpload) -> ScenarioDetail:
    # Derive file name from scenario name
    file_name = body.name.lower().replace(" ", "-")
    path = settings.scenarios_dir / f"{file_name}.json"
    if path.is_file():
        raise HTTPException(status_code=409, detail=f"Scenario '{file_name}' already exists")

    data: dict[str, Any] = body.model_dump()
    settings.scenarios_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    return _to_detail(file_name, data)
