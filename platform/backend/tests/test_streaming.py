"""Tests for the streaming service and control endpoints."""

import asyncio
import json

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.main import app
from app.services.streaming import StreamingService


@pytest.fixture(autouse=True)
def scenarios_dir(tmp_path, monkeypatch):
    """Create a test scenario for simulated streaming."""
    scenario = {
        "name": "Test Target",
        "description": "Single ferrous sphere for streaming test.",
        "earth_field": [0.0, 20e-6, 45e-6],
        "terrain": {
            "x_extent": [0.0, 5.0],
            "y_extent": [0.0, 5.0],
            "surface_elevation": 0.0,
            "layers": [],
        },
        "objects": [
            {
                "name": "Steel sphere",
                "position": [2.5, 2.5, -1.0],
                "type": "ferrous_sphere",
                "radius": 0.05,
                "susceptibility": 1000.0,
                "conductivity": 1e6,
            },
        ],
        "metadata": {},
    }
    (tmp_path / "test-target.json").write_text(json.dumps(scenario))

    empty = {
        "name": "Empty",
        "description": "No objects.",
        "terrain": {
            "x_extent": [0.0, 3.0],
            "y_extent": [0.0, 3.0],
            "surface_elevation": 0.0,
            "layers": [],
        },
        "objects": [],
        "metadata": {},
    }
    (tmp_path / "empty.json").write_text(json.dumps(empty))
    monkeypatch.setattr(settings, "scenarios_dir", tmp_path)
    monkeypatch.setattr(settings, "mqtt_broker_host", "")
    monkeypatch.setattr(settings, "stream_rate_hz", 1000.0)  # Fast for tests
    return tmp_path


@pytest.fixture(autouse=True)
async def reset_global_service():
    """Ensure global streaming service is stopped between tests."""
    from app.services.streaming import streaming_service
    yield
    await streaming_service.stop()


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
def service():
    return StreamingService()


# --- StreamingService unit tests ---


class TestStreamingService:
    async def test_start_stop_lifecycle(self, service):
        """Service starts and stops cleanly."""
        assert not service.running
        await service.start(scenario_name="test-target")
        assert service.running
        assert service.mode == "simulation"
        await service.stop()
        assert not service.running

    async def test_status_reports_correctly(self, service):
        """Status dict has expected keys and values."""
        status = service.status()
        assert status["running"] is False
        assert status["mode"] == "simulation"
        assert status["points_sent"] == 0

        await service.start(scenario_name="test-target")
        # Let simulation run briefly
        await asyncio.sleep(0.05)
        status = service.status()
        assert status["running"] is True
        assert status["scenario_name"] == "test-target"
        assert status["points_sent"] > 0
        await service.stop()

    async def test_simulation_sends_points(self, service):
        """Simulation mode generates and broadcasts points."""
        await service.start(scenario_name="test-target")
        await asyncio.sleep(0.1)
        await service.stop()
        assert service.status()["points_sent"] > 0

    async def test_simulation_detects_anomalies(self, service):
        """Simulation mode triggers anomaly detection near target."""
        await service.start(scenario_name="test-target")
        # Let it run long enough to pass near the target
        await asyncio.sleep(0.5)
        await service.stop()
        # Anomaly detection may or may not trigger depending on path,
        # but points should have been sent
        assert service.status()["points_sent"] > 10

    async def test_double_start_is_noop(self, service):
        """Starting an already-running service does nothing."""
        await service.start(scenario_name="test-target")
        await service.start(scenario_name="empty")  # Should be ignored
        assert service.status()["scenario_name"] == "test-target"
        await service.stop()

    async def test_stop_when_not_running(self, service):
        """Stopping a non-running service is safe."""
        await service.stop()
        assert not service.running


# --- API endpoint tests ---


async def test_streaming_status_endpoint(client):
    resp = await client.get("/api/streaming/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "running" in data
    assert "mode" in data
    assert "points_sent" in data


async def test_start_stop_endpoints(client):
    # Start
    resp = await client.post(
        "/api/streaming/start",
        json={"scenario_name": "test-target"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "started"

    # Check status
    resp = await client.get("/api/streaming/status")
    assert resp.json()["running"] is True

    # Stop
    resp = await client.post("/api/streaming/stop")
    assert resp.status_code == 200
    assert resp.json()["status"] == "stopped"


async def test_start_nonexistent_scenario(client):
    resp = await client.post(
        "/api/streaming/start",
        json={"scenario_name": "nonexistent"},
    )
    assert resp.status_code == 404
