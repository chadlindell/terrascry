"""Tests for dataset storage service and endpoints."""

import json

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.dataset import Dataset, DatasetMetadata, GridData, SurveyPoint
from app.main import app
from app.services.dataset_store import DatasetStore


def _make_dataset(scenario_name: str = "test-scenario") -> Dataset:
    """Create a minimal test dataset."""
    return Dataset(
        metadata=DatasetMetadata(
            scenario_name=scenario_name,
            params={"resolution": 1.0},
        ),
        grid_data=GridData(
            rows=2,
            cols=2,
            x_min=0.0,
            y_min=0.0,
            dx=1.0,
            dy=1.0,
            values=[1.0, 2.0, 3.0, 4.0],
        ),
        survey_points=[
            SurveyPoint(x=0.0, y=0.0, gradient_nt=1.0),
            SurveyPoint(x=1.0, y=0.0, gradient_nt=2.0),
        ],
    )


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    """Use a temp directory for dataset storage."""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    return tmp_path


@pytest.fixture
def store():
    return DatasetStore()


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


# --- DatasetStore unit tests ---


class TestDatasetStore:
    def test_save_creates_file(self, store, data_dir):
        dataset = _make_dataset()
        dataset_id = store.save(dataset)
        path = data_dir / f"{dataset_id}.json"
        assert path.is_file()

    def test_save_returns_id(self, store):
        dataset = _make_dataset()
        dataset_id = store.save(dataset)
        assert dataset_id == dataset.metadata.id

    def test_list_empty(self, store):
        assert store.list() == []

    def test_list_returns_saved(self, store):
        d1 = _make_dataset("scenario-a")
        d2 = _make_dataset("scenario-b")
        store.save(d1)
        store.save(d2)
        metadata_list = store.list()
        assert len(metadata_list) == 2
        names = {m.scenario_name for m in metadata_list}
        assert names == {"scenario-a", "scenario-b"}

    def test_list_newest_first(self, store):
        d1 = _make_dataset("first")
        store.save(d1)
        d2 = _make_dataset("second")
        store.save(d2)
        metadata_list = store.list()
        assert metadata_list[0].created_at >= metadata_list[1].created_at

    def test_get_by_id(self, store):
        dataset = _make_dataset()
        store.save(dataset)
        loaded = store.get(dataset.metadata.id)
        assert loaded is not None
        assert loaded.metadata.id == dataset.metadata.id
        assert loaded.metadata.scenario_name == "test-scenario"
        assert loaded.grid_data.values == [1.0, 2.0, 3.0, 4.0]
        assert len(loaded.survey_points) == 2

    def test_get_not_found(self, store):
        from uuid import uuid4

        assert store.get(uuid4()) is None

    def test_persistence_round_trip(self, store, data_dir):
        dataset = _make_dataset()
        store.save(dataset)
        raw = json.loads((data_dir / f"{dataset.metadata.id}.json").read_text())
        assert raw["metadata"]["scenario_name"] == "test-scenario"
        assert raw["grid_data"]["rows"] == 2
        assert len(raw["survey_points"]) == 2


# --- API endpoint tests ---


async def test_list_datasets_empty(client):
    resp = await client.get("/api/datasets")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_list_datasets_after_save(client, store):
    dataset = _make_dataset()
    store.save(dataset)
    resp = await client.get("/api/datasets")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["scenario_name"] == "test-scenario"


async def test_get_dataset_by_id(client, store):
    dataset = _make_dataset()
    store.save(dataset)
    resp = await client.get(f"/api/datasets/{dataset.metadata.id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["metadata"]["scenario_name"] == "test-scenario"
    assert data["grid_data"]["values"] == [1.0, 2.0, 3.0, 4.0]


async def test_get_dataset_not_found(client):
    from uuid import uuid4

    resp = await client.get(f"/api/datasets/{uuid4()}")
    assert resp.status_code == 404


# --- Auto-save integration test ---


@pytest.fixture
def scenarios_dir(tmp_path, monkeypatch):
    """Also set up a scenario for the simulate endpoint."""
    scenario = {
        "name": "Auto-save Test",
        "description": "For testing auto-save.",
        "earth_field": [0.0, 20e-6, 45e-6],
        "terrain": {
            "x_extent": [0.0, 5.0],
            "y_extent": [0.0, 5.0],
            "surface_elevation": 0.0,
            "layers": [],
        },
        "objects": [
            {
                "name": "Target",
                "position": [2.5, 2.5, -1.0],
                "type": "ferrous_sphere",
                "radius": 0.05,
                "susceptibility": 1000.0,
                "conductivity": 1e6,
            }
        ],
        "metadata": {},
    }
    scenarios_path = tmp_path / "scenarios"
    scenarios_path.mkdir()
    (scenarios_path / "autosave-test.json").write_text(json.dumps(scenario))
    monkeypatch.setattr(settings, "scenarios_dir", scenarios_path)
    return scenarios_path


async def test_simulate_auto_saves_dataset(client, scenarios_dir, store):
    resp = await client.post(
        "/api/surveys/simulate",
        json={"scenario_name": "autosave-test", "resolution": 2.0, "line_spacing": 2.0},
    )
    assert resp.status_code == 200
    dataset_id = resp.json()["metadata"]["id"]

    # Verify it was persisted
    stored = store.list()
    assert len(stored) == 1
    assert str(stored[0].id) == dataset_id

    # Verify retrievable via API
    resp2 = await client.get(f"/api/datasets/{dataset_id}")
    assert resp2.status_code == 200
    assert resp2.json()["metadata"]["scenario_name"] == "autosave-test"
