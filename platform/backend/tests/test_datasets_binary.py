"""Tests for binary grid data transfer endpoint."""

import json
import struct

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.main import app
from app.services.dataset_store import DatasetStore
from app.services.physics import PhysicsEngine


@pytest.fixture(autouse=True)
def scenarios_dir(tmp_path, monkeypatch):
    scenario = {
        "name": "Test Target",
        "description": "Single ferrous sphere.",
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
    monkeypatch.setattr(settings, "scenarios_dir", tmp_path)
    return tmp_path


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    d = tmp_path / "datasets"
    monkeypatch.setattr(settings, "data_dir", d)
    return d


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
def stored_dataset(data_dir):
    engine = PhysicsEngine()
    dataset = engine.simulate("test-target", resolution=1.0, line_spacing=1.0)
    store = DatasetStore()
    store.save(dataset)
    return dataset


async def test_binary_correct_byte_count(client, stored_dataset):
    resp = await client.get(f"/api/datasets/{stored_dataset.metadata.id}/binary")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/octet-stream"

    rows = int(resp.headers["X-Grid-Rows"])
    cols = int(resp.headers["X-Grid-Cols"])
    expected_bytes = rows * cols * 4
    assert len(resp.content) == expected_bytes


async def test_binary_headers_match_grid(client, stored_dataset):
    resp = await client.get(f"/api/datasets/{stored_dataset.metadata.id}/binary")
    grid = stored_dataset.grid_data

    assert int(resp.headers["X-Grid-Rows"]) == grid.rows
    assert int(resp.headers["X-Grid-Cols"]) == grid.cols
    assert float(resp.headers["X-Grid-Xmin"]) == grid.x_min
    assert float(resp.headers["X-Grid-Ymin"]) == grid.y_min
    assert float(resp.headers["X-Grid-Dx"]) == grid.dx
    assert float(resp.headers["X-Grid-Dy"]) == grid.dy
    assert resp.headers["X-Grid-Unit"] == grid.unit


async def test_binary_values_roundtrip(client, stored_dataset):
    resp = await client.get(f"/api/datasets/{stored_dataset.metadata.id}/binary")
    grid = stored_dataset.grid_data

    n = grid.rows * grid.cols
    values = list(struct.unpack(f"<{n}f", resp.content))

    for i in range(n):
        # Float32 has ~7 significant digits
        assert abs(values[i] - grid.values[i]) < abs(grid.values[i]) * 1e-6 + 1e-10


async def test_binary_404_for_missing(client, data_dir):
    resp = await client.get("/api/datasets/00000000-0000-0000-0000-000000000000/binary")
    assert resp.status_code == 404
