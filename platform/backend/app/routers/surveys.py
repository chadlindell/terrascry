"""Surveys router — trigger simulation and return results."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.dataset import Dataset, DatasetMetadata
from app.services.connection_manager import manager
from app.services.dataset_store import DatasetStore
from app.services.physics import PhysicsEngine

router = APIRouter(prefix="/api/surveys", tags=["surveys"])

engine = PhysicsEngine()
store = DatasetStore()


class SimulateRequest(BaseModel):
    """Request body for survey simulation."""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "scenario_name": "single-ferrous-target",
                    "line_spacing": 1.0,
                    "sample_spacing": 0.5,
                    "resolution": 0.5,
                }
            ]
        }
    }

    scenario_name: str
    line_spacing: float = Field(default=1.0, gt=0, le=10.0)
    sample_spacing: float = Field(default=0.5, gt=0, le=5.0)
    resolution: float = Field(default=0.5, gt=0, le=5.0)


def _run_simulation(request: SimulateRequest) -> Dataset:
    """Run a single simulation and persist the result."""
    try:
        dataset = engine.simulate(
            scenario_name=request.scenario_name,
            line_spacing=request.line_spacing,
            sample_spacing=request.sample_spacing,
            resolution=request.resolution,
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Scenario '{request.scenario_name}' not found",
        )
    store.save(dataset)
    return dataset


@router.post("/simulate", response_model=Dataset)
async def simulate_survey(request: SimulateRequest) -> Dataset:
    dataset = _run_simulation(request)
    await manager.broadcast(
        "datasets",
        "dataset_created",
        {"id": str(dataset.metadata.id), "scenario_name": dataset.metadata.scenario_name},
    )
    return dataset


@router.post("/batch", response_model=list[DatasetMetadata])
async def batch_simulate(requests: list[SimulateRequest]) -> list[DatasetMetadata]:
    if len(requests) > 10:
        raise HTTPException(status_code=422, detail="Maximum 10 simulations per batch")
    results: list[DatasetMetadata] = []
    for req in requests:
        dataset = _run_simulation(req)
        await manager.broadcast(
            "datasets",
            "dataset_created",
            {"id": str(dataset.metadata.id), "scenario_name": dataset.metadata.scenario_name},
        )
        results.append(dataset.metadata)
    return results
