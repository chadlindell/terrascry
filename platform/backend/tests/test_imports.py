"""Tests for CSV import upload and validation endpoints."""

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.main import app

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    """Use a temp directory for dataset storage."""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    return tmp_path


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


async def test_upload_valid_csv(client):
    content = (FIXTURES / "valid_survey.csv").read_bytes()
    resp = await client.post(
        "/api/imports/upload",
        files={"file": ("survey.csv", content, "text/csv")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["metadata"]["scenario_name"] == "survey"
    assert data["grid_data"]["rows"] >= 2
    assert data["grid_data"]["cols"] >= 2
    assert len(data["survey_points"]) == 20


async def test_upload_invalid_csv(client):
    content = (FIXTURES / "invalid_missing_cols.csv").read_bytes()
    resp = await client.post(
        "/api/imports/upload",
        files={"file": ("bad.csv", content, "text/csv")},
    )
    assert resp.status_code == 422


async def test_validate_only(client):
    content = (FIXTURES / "valid_survey.csv").read_bytes()
    resp = await client.post(
        "/api/imports/validate",
        files={"file": ("survey.csv", content, "text/csv")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is True
    assert data["row_count"] == 20


async def test_upload_large_file_rejected(client):
    # Generate content > 10 MB
    header = "x,y,gradient_nt\n"
    row = "0.0,0.0,1.0\n"
    content = (header + row * (11 * 1024 * 1024 // len(row))).encode()
    resp = await client.post(
        "/api/imports/upload",
        files={"file": ("huge.csv", content, "text/csv")},
    )
    assert resp.status_code == 422
    assert "too large" in resp.json()["detail"].lower()
