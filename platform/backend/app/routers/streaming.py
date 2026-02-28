"""Streaming control endpoints — start/stop simulated or MQTT streaming."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.streaming import streaming_service

router = APIRouter(prefix="/api/streaming", tags=["streaming"])


class StartRequest(BaseModel):
    scenario_name: str = ""


@router.post("/start")
async def start_streaming(request: StartRequest) -> dict:
    """Start streaming data to WebSocket clients."""
    if streaming_service.running:
        raise HTTPException(status_code=409, detail="Streaming already running")
    try:
        await streaming_service.start(scenario_name=request.scenario_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"status": "started", "mode": streaming_service.mode}


@router.post("/stop")
async def stop_streaming() -> dict:
    """Stop streaming."""
    await streaming_service.stop()
    return {"status": "stopped"}


@router.get("/status")
async def streaming_status() -> dict:
    """Return current streaming status."""
    return streaming_service.status()
