"""Tests for anomaly detection service and endpoint."""

import json

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.dataset import Dataset, DatasetMetadata, GridData
from app.main import app
from app.services.anomaly import AnomalyService
from app.services.dataset_store import DatasetStore
from app.services.physics import PhysicsEngine


@pytest.fixture(autouse=True)
def scenarios_dir(tmp_path, monkeypatch):
    """Create a test scenario."""
    scenario = {
        "name": "Test Target",
        "description": "Single ferrous sphere.",
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
    """Simulate and store a dataset for endpoint tests."""
    engine = PhysicsEngine()
    dataset = engine.simulate("test-target", resolution=0.5, line_spacing=1.0)
    store = DatasetStore()
    store.save(dataset)
    return dataset


# --- AnomalyService unit tests ---


class TestAnomalyService:
    def test_detects_anomalies_near_target(self):
        """Anomaly cells should appear near the buried target."""
        engine = PhysicsEngine()
        dataset = engine.simulate("test-target", resolution=0.5, line_spacing=1.0)
        service = AnomalyService()
        anomalies = service.detect(dataset.grid_data, threshold_sigma=3.0)
        assert len(anomalies) > 0
        # At least one anomaly should be near (5, 5)
        near = [a for a in anomalies if abs(a["x"] - 5.0) <= 2.0 and abs(a["y"] - 5.0) <= 2.0]
        assert len(near) > 0

    def test_no_anomalies_for_flat_data(self):
        """Uniform data at zero should produce no anomalies."""
        grid = GridData(
            rows=10, cols=10,
            x_min=0.0, y_min=0.0, dx=1.0, dy=1.0,
            values=[0.0] * 100,
        )
        service = AnomalyService()
        anomalies = service.detect(grid, threshold_sigma=3.0)
        assert len(anomalies) == 0

    def test_threshold_parameter_affects_count(self):
        """Higher threshold should produce fewer anomalies."""
        engine = PhysicsEngine()
        dataset = engine.simulate("test-target", resolution=0.5, line_spacing=1.0)
        service = AnomalyService()
        low = service.detect(dataset.grid_data, threshold_sigma=2.0)
        high = service.detect(dataset.grid_data, threshold_sigma=5.0)
        assert len(low) >= len(high)


# --- Endpoint tests ---


async def test_anomalies_endpoint(client, stored_dataset):
    resp = await client.get(f"/api/datasets/{stored_dataset.metadata.id}/anomalies")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    # Should have anomalies near the target
    assert len(data) > 0
    assert "x" in data[0]
    assert "sigma" in data[0]


async def test_anomalies_not_found(client, data_dir):
    resp = await client.get("/api/datasets/00000000-0000-0000-0000-000000000000/anomalies")
    assert resp.status_code == 404


async def test_anomalies_threshold_param(client, stored_dataset):
    resp = await client.get(
        f"/api/datasets/{stored_dataset.metadata.id}/anomalies?threshold_sigma=8.0"
    )
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
