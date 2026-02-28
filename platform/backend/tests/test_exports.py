"""Tests for the dataset export endpoint."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.dataset import Dataset, DatasetMetadata, GridData, SurveyPoint
from app.main import app
from app.services.dataset_store import DatasetStore


def _make_dataset() -> Dataset:
    return Dataset(
        metadata=DatasetMetadata(
            scenario_name="export-test",
            params={"resolution": 1.0},
        ),
        grid_data=GridData(
            rows=2,
            cols=3,
            x_min=0.0,
            y_min=0.0,
            dx=1.0,
            dy=1.0,
            values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ),
        survey_points=[
            SurveyPoint(x=0.0, y=0.0, gradient_nt=1.5),
            SurveyPoint(x=1.0, y=1.0, gradient_nt=3.2),
        ],
    )


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    return tmp_path


@pytest.fixture
def store():
    return DatasetStore()


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


async def test_export_csv(client, store):
    dataset = _make_dataset()
    store.save(dataset)
    resp = await client.get(f"/api/datasets/{dataset.metadata.id}/export?format=csv")
    assert resp.status_code == 200
    assert "text/csv" in resp.headers["content-type"]
    assert "attachment" in resp.headers["content-disposition"]
    body = resp.text
    assert "x,y,gradient_nt" in body
    assert "0.0,0.0,1.5" in body
    assert "1.0,1.0,3.2" in body
    assert "# scenario: export-test" in body


async def test_export_grid_csv(client, store):
    dataset = _make_dataset()
    store.save(dataset)
    resp = await client.get(f"/api/datasets/{dataset.metadata.id}/export?format=grid_csv")
    assert resp.status_code == 200
    body = resp.text
    assert "# rows: 2, cols: 3" in body
    # Row 0 values
    assert "1.0,2.0,3.0" in body
    # Row 1 values
    assert "4.0,5.0,6.0" in body


async def test_export_asc(client, store):
    dataset = _make_dataset()
    store.save(dataset)
    resp = await client.get(f"/api/datasets/{dataset.metadata.id}/export?format=asc")
    assert resp.status_code == 200
    body = resp.text
    assert "ncols         3" in body
    assert "nrows         2" in body
    assert "xllcorner     0.0" in body
    assert "yllcorner     0.0" in body
    assert "cellsize      1.0" in body
    # ASC format: top row first (row 1 in grid = [4.0, 5.0, 6.0])
    lines = body.strip().split("\n")
    data_lines = [l for l in lines if not l.startswith(("ncols", "nrows", "xll", "yll", "cell", "NODATA"))]
    assert data_lines[0] == "4.0 5.0 6.0"
    assert data_lines[1] == "1.0 2.0 3.0"


async def test_export_default_format(client, store):
    dataset = _make_dataset()
    store.save(dataset)
    resp = await client.get(f"/api/datasets/{dataset.metadata.id}/export")
    assert resp.status_code == 200
    assert "x,y,gradient_nt" in resp.text


async def test_export_not_found(client):
    from uuid import uuid4

    resp = await client.get(f"/api/datasets/{uuid4()}/export?format=csv")
    assert resp.status_code == 404


async def test_export_invalid_format(client, store):
    dataset = _make_dataset()
    store.save(dataset)
    resp = await client.get(f"/api/datasets/{dataset.metadata.id}/export?format=invalid")
    assert resp.status_code == 422
