"""Tests for the physics service and survey simulation endpoints."""

import json

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.main import app
from app.services.physics import PhysicsEngine


@pytest.fixture(autouse=True)
def scenarios_dir(tmp_path, monkeypatch):
    """Create a test scenario with a known ferrous target."""
    scenario = {
        "name": "Test Target",
        "description": "Single ferrous sphere for testing.",
        "earth_field": [0.0, 20e-6, 45e-6],
        "terrain": {
            "x_extent": [0.0, 10.0],
            "y_extent": [0.0, 10.0],
            "surface_elevation": 0.0,
            "layers": [],
        },
        "objects": [
            {
                "name": "Steel sphere",
                "position": [5.0, 5.0, -1.0],
                "type": "ferrous_sphere",
                "radius": 0.05,
                "susceptibility": 1000.0,
                "conductivity": 1e6,
            },
        ],
        "metadata": {},
    }
    empty_scenario = {
        "name": "Empty",
        "description": "No objects.",
        "terrain": {
            "x_extent": [0.0, 5.0],
            "y_extent": [0.0, 5.0],
            "surface_elevation": 0.0,
            "layers": [],
        },
        "objects": [],
        "metadata": {},
    }
    (tmp_path / "test-target.json").write_text(json.dumps(scenario))
    (tmp_path / "empty.json").write_text(json.dumps(empty_scenario))
    monkeypatch.setattr(settings, "scenarios_dir", tmp_path)
    return tmp_path


@pytest.fixture
def engine():
    return PhysicsEngine()


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


# --- PhysicsEngine unit tests ---


class TestPhysicsEngine:
    def test_load_scenario(self, engine):
        scenario = engine.load_scenario("test-target")
        assert scenario.name == "Test Target"
        assert len(scenario.objects) == 1

    def test_load_scenario_not_found(self, engine):
        with pytest.raises(FileNotFoundError):
            engine.load_scenario("nonexistent")

    def test_generate_path_shape(self, engine):
        scenario = engine.load_scenario("test-target")
        path = engine.generate_path(scenario, line_spacing=2.0, sample_spacing=1.0)
        assert path.ndim == 2
        assert path.shape[1] == 3
        # 10m extent / 2m spacing = 6 lines (0,2,4,6,8,10), 11 samples each
        assert path.shape[0] == 6 * 11

    def test_generate_path_snake_pattern(self, engine):
        scenario = engine.load_scenario("test-target")
        path = engine.generate_path(scenario, line_spacing=5.0, sample_spacing=2.0)
        # First line goes y ascending, second line goes y descending
        line0 = path[:6]  # x=0, y=0,2,4,6,8,10
        line1 = path[6:12]  # x=5, y=10,8,6,4,2,0
        assert line0[0, 1] < line0[-1, 1]  # ascending
        assert line1[0, 1] > line1[-1, 1]  # descending

    def test_simulate_survey_returns_points(self, engine):
        scenario = engine.load_scenario("test-target")
        path = engine.generate_path(scenario, line_spacing=2.0, sample_spacing=2.0)
        points = engine.simulate_survey(scenario, path)
        assert len(points) == len(path)
        assert all(isinstance(p.gradient_nt, float) for p in points)

    def test_simulate_survey_detects_target(self, engine):
        scenario = engine.load_scenario("test-target")
        path = engine.generate_path(scenario, line_spacing=1.0, sample_spacing=1.0)
        points = engine.simulate_survey(scenario, path)
        # Point near target (5,5) should have non-zero gradient
        near_target = [p for p in points if abs(p.x - 5.0) < 0.1 and abs(p.y - 5.0) < 0.1]
        assert len(near_target) > 0
        assert any(abs(p.gradient_nt) > 0.1 for p in near_target)

    def test_simulate_survey_empty_scenario(self, engine):
        scenario = engine.load_scenario("empty")
        path = engine.generate_path(scenario)
        points = engine.simulate_survey(scenario, path)
        assert all(p.gradient_nt == 0.0 for p in points)

    def test_compute_grid_shape(self, engine):
        scenario = engine.load_scenario("test-target")
        grid = engine.compute_grid(scenario, resolution=1.0)
        # 10m / 1m = 11 points per axis
        assert grid.rows == 11
        assert grid.cols == 11
        assert len(grid.values) == grid.rows * grid.cols
        assert grid.dx == 1.0
        assert grid.dy == 1.0
        assert grid.x_min == 0.0
        assert grid.y_min == 0.0

    def test_compute_grid_has_anomaly_near_target(self, engine):
        scenario = engine.load_scenario("test-target")
        grid = engine.compute_grid(scenario, resolution=1.0)
        # Find max absolute gradient — should be near (5,5)
        max_val = max(abs(v) for v in grid.values)
        max_idx = next(i for i, v in enumerate(grid.values) if abs(v) == max_val)
        row = max_idx // grid.cols
        col = max_idx % grid.cols
        x = grid.x_min + col * grid.dx
        y = grid.y_min + row * grid.dy
        assert abs(x - 5.0) <= 1.0, f"Max gradient at x={x}, expected near 5.0"
        assert abs(y - 5.0) <= 1.0, f"Max gradient at y={y}, expected near 5.0"

    def test_full_simulate_pipeline(self, engine):
        dataset = engine.simulate("test-target", line_spacing=2.0, resolution=2.0)
        assert dataset.metadata.scenario_name == "test-target"
        assert dataset.grid_data.rows > 0
        assert len(dataset.survey_points) > 0
        assert dataset.metadata.params["line_spacing"] == 2.0


# --- API endpoint tests ---


async def test_simulate_endpoint(client):
    resp = await client.post(
        "/api/surveys/simulate",
        json={"scenario_name": "test-target", "line_spacing": 2.0, "resolution": 2.0},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "metadata" in data
    assert "grid_data" in data
    assert "survey_points" in data
    assert data["metadata"]["scenario_name"] == "test-target"
    assert data["grid_data"]["rows"] > 0
    assert len(data["grid_data"]["values"]) == data["grid_data"]["rows"] * data["grid_data"]["cols"]
    assert len(data["survey_points"]) > 0


async def test_simulate_endpoint_not_found(client):
    resp = await client.post(
        "/api/surveys/simulate",
        json={"scenario_name": "nonexistent"},
    )
    assert resp.status_code == 404


async def test_simulate_endpoint_invalid_params(client):
    resp = await client.post(
        "/api/surveys/simulate",
        json={"scenario_name": "test-target", "line_spacing": -1},
    )
    assert resp.status_code == 422


# --- Batch simulation tests ---


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "datasets")
    return tmp_path / "datasets"


async def test_batch_simulate(client, data_dir):
    resp = await client.post(
        "/api/surveys/batch",
        json=[
            {"scenario_name": "test-target", "resolution": 2.0, "line_spacing": 2.0},
            {"scenario_name": "empty", "resolution": 2.0, "line_spacing": 2.0},
        ],
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    assert data[0]["scenario_name"] == "test-target"
    assert data[1]["scenario_name"] == "empty"
    # Each result should have an id
    assert "id" in data[0]
    assert "id" in data[1]


async def test_batch_simulate_max_limit(client):
    requests = [{"scenario_name": "test-target"}] * 11
    resp = await client.post("/api/surveys/batch", json=requests)
    assert resp.status_code == 422
    assert "Maximum 10" in resp.json()["detail"]


async def test_batch_simulate_empty(client):
    resp = await client.post("/api/surveys/batch", json=[])
    assert resp.status_code == 200
    assert resp.json() == []


async def test_batch_simulate_not_found(client):
    resp = await client.post(
        "/api/surveys/batch",
        json=[{"scenario_name": "nonexistent"}],
    )
    assert resp.status_code == 404
