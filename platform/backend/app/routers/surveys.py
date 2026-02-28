"""Surveys router — trigger simulation and return results."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.dataset import Dataset
from app.services.physics import PhysicsEngine

router = APIRouter(prefix="/api/surveys", tags=["surveys"])

engine = PhysicsEngine()


class SimulateRequest(BaseModel):
    """Request body for survey simulation."""

    scenario_name: str
    line_spacing: float = Field(default=1.0, gt=0, le=10.0)
    sample_spacing: float = Field(default=0.5, gt=0, le=5.0)
    resolution: float = Field(default=0.5, gt=0, le=5.0)


@router.post("/simulate", response_model=Dataset)
async def simulate_survey(request: SimulateRequest) -> Dataset:
    try:
        return engine.simulate(
            scenario_name=request.scenario_name,
            line_spacing=request.line_spacing,
            sample_spacing=request.sample_spacing,
            resolution=request.resolution,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Scenario '{request.scenario_name}' not found")
