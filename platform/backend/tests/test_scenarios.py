"""Tests for the scenarios API endpoints."""

import json

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.main import app


@pytest.fixture(autouse=True)
def scenarios_dir(tmp_path, monkeypatch):
    """Create test scenarios in a temp directory."""
    scenario_a = {
        "name": "Test Scenario A",
        "description": "A test scenario with two objects.",
        "earth_field": [0.0, 20e-6, 45e-6],
        "terrain": {
            "x_extent": [0.0, 10.0],
            "y_extent": [0.0, 10.0],
            "surface_elevation": 0.0,
            "layers": [],
        },
        "objects": [
            {
                "name": "Iron sphere",
                "position": [5.0, 5.0, -1.0],
                "type": "ferrous_sphere",
                "radius": 0.05,
                "susceptibility": 1000.0,
                "conductivity": 1e6,
            },
            {
                "name": "Copper pipe",
                "position": [3.0, 7.0, -0.5],
                "type": "ferrous_cylinder",
                "radius": 0.02,
                "susceptibility": 0.0,
                "conductivity": 5.8e7,
            },
        ],
        "metadata": {"author": "test", "category": "unit-test"},
    }
    scenario_b = {
        "name": "Test Scenario B",
        "description": "Empty scenario.",
        "terrain": {
            "x_extent": [0.0, 5.0],
            "y_extent": [0.0, 5.0],
            "surface_elevation": 0.0,
            "layers": [],
        },
        "objects": [],
        "metadata": {},
    }

    (tmp_path / "test-scenario-a.json").write_text(json.dumps(scenario_a))
    (tmp_path / "test-scenario-b.json").write_text(json.dumps(scenario_b))
    # Should be excluded (metadata sidecar)
    (tmp_path / "test-scenario-a_meta.json").write_text("{}")

    monkeypatch.setattr(settings, "scenarios_dir", tmp_path)
    return tmp_path


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


async def test_list_scenarios_returns_all(client):
    resp = await client.get("/api/scenarios")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    names = {s["file_name"] for s in data}
    assert names == {"test-scenario-a", "test-scenario-b"}


async def test_list_scenarios_has_summary_fields(client):
    resp = await client.get("/api/scenarios")
    data = resp.json()
    scenario_a = next(s for s in data if s["file_name"] == "test-scenario-a")
    assert scenario_a["name"] == "Test Scenario A"
    assert scenario_a["description"] == "A test scenario with two objects."
    assert scenario_a["object_count"] == 2
    assert scenario_a["terrain"]["x_extent"] == [0.0, 10.0]


async def test_get_scenario_detail(client):
    resp = await client.get("/api/scenarios/test-scenario-a")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Test Scenario A"
    assert data["object_count"] == 2
    assert len(data["objects"]) == 2
    assert data["objects"][0]["name"] == "Iron sphere"
    assert data["objects"][0]["object_type"] == "ferrous_sphere"
    assert data["objects"][0]["position"] == [5.0, 5.0, -1.0]
    assert data["earth_field"] == [0.0, 20e-6, 45e-6]
    assert data["metadata"]["author"] == "test"
    assert "raw" in data


async def test_get_scenario_empty(client):
    resp = await client.get("/api/scenarios/test-scenario-b")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object_count"] == 0
    assert data["objects"] == []


async def test_get_scenario_not_found(client):
    resp = await client.get("/api/scenarios/nonexistent")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


async def test_meta_files_excluded(client):
    resp = await client.get("/api/scenarios")
    file_names = [s["file_name"] for s in resp.json()]
    assert "test-scenario-a_meta" not in file_names


# --- Upload tests ---


async def test_upload_scenario(client):
    body = {
        "name": "New Scenario",
        "description": "Uploaded via API.",
        "terrain": {
            "x_extent": [0.0, 10.0],
            "y_extent": [0.0, 10.0],
            "surface_elevation": 0.0,
        },
        "objects": [
            {
                "name": "Target",
                "position": [5.0, 5.0, -1.0],
                "type": "ferrous_sphere",
                "radius": 0.05,
            }
        ],
    }
    resp = await client.post("/api/scenarios", json=body)
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "New Scenario"
    assert data["file_name"] == "new-scenario"
    assert data["object_count"] == 1


async def test_upload_scenario_appears_in_list(client):
    body = {"name": "Listed Scenario"}
    resp = await client.post("/api/scenarios", json=body)
    assert resp.status_code == 201
    list_resp = await client.get("/api/scenarios")
    file_names = [s["file_name"] for s in list_resp.json()]
    assert "listed-scenario" in file_names


async def test_upload_scenario_duplicate(client):
    body = {"name": "Duplicate"}
    resp1 = await client.post("/api/scenarios", json=body)
    assert resp1.status_code == 201
    resp2 = await client.post("/api/scenarios", json=body)
    assert resp2.status_code == 409
    assert "already exists" in resp2.json()["detail"]
