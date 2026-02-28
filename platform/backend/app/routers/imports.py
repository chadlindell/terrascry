"""Imports router — CSV file upload and validation."""

from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

from app.dataset import Dataset, DatasetMetadata, GridData, SurveyPoint
from app.services.connection_manager import manager
from app.services.dataset_store import DatasetStore
from app.services.import_validator import ImportValidator

router = APIRouter(prefix="/api/imports", tags=["imports"])

store = DatasetStore()
validator = ImportValidator()

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


class ValidationResponse(BaseModel):
    """Validation result returned to the client."""

    valid: bool
    errors: list[str]
    row_count: int
    columns: list[str]


def _parse_csv_to_points(content: str) -> list[SurveyPoint]:
    """Parse validated CSV content into SurveyPoint list."""
    import csv
    import io

    lines = content.strip().splitlines()
    data_lines = [line for line in lines if not line.startswith("#")]
    reader = csv.DictReader(io.StringIO("\n".join(data_lines)))

    points: list[SurveyPoint] = []
    for row in reader:
        x = float(row["x"].strip())
        y = float(row["y"].strip())
        # Accept either gradient_nt or value column
        grad_str = row.get("gradient_nt", row.get("value", "0")).strip()
        gradient = float(grad_str)
        points.append(SurveyPoint(x=x, y=y, gradient_nt=gradient))

    return points


def _points_to_grid(points: list[SurveyPoint], resolution: float = 0.5) -> GridData:
    """Interpolate irregular survey points onto a regular grid."""
    from scipy.interpolate import griddata

    xs = np.array([p.x for p in points])
    ys = np.array([p.y for p in points])
    vals = np.array([p.gradient_nt for p in points])

    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())

    # Ensure at least 2 cells in each dimension
    cols = max(2, int(np.ceil((x_max - x_min) / resolution)) + 1)
    rows = max(2, int(np.ceil((y_max - y_min) / resolution)) + 1)

    dx = (x_max - x_min) / max(cols - 1, 1)
    dy = (y_max - y_min) / max(rows - 1, 1)

    grid_x = np.linspace(x_min, x_max, cols)
    grid_y = np.linspace(y_min, y_max, rows)
    gx, gy = np.meshgrid(grid_x, grid_y)

    grid_vals = griddata(
        np.column_stack([xs, ys]),
        vals,
        (gx, gy),
        method="cubic",
        fill_value=0.0,
    )

    # Fall back to linear if cubic produces too many NaNs
    nan_mask = np.isnan(grid_vals)
    if nan_mask.any():
        linear_vals = griddata(
            np.column_stack([xs, ys]),
            vals,
            (gx, gy),
            method="linear",
            fill_value=0.0,
        )
        grid_vals[nan_mask] = linear_vals[nan_mask]
        # Fill any remaining NaNs with nearest
        still_nan = np.isnan(grid_vals)
        if still_nan.any():
            nearest_vals = griddata(
                np.column_stack([xs, ys]),
                vals,
                (gx, gy),
                method="nearest",
            )
            grid_vals[still_nan] = nearest_vals[still_nan]

    return GridData(
        rows=rows,
        cols=cols,
        x_min=x_min,
        y_min=y_min,
        dx=dx if dx > 0 else resolution,
        dy=dy if dy > 0 else resolution,
        values=grid_vals.flatten().tolist(),
        unit="nT",
    )


@router.post("/validate", response_model=ValidationResponse)
async def validate_upload(file: UploadFile) -> ValidationResponse:
    """Validate a CSV file without persisting it."""
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=422, detail="File too large (max 10 MB)")

    result = validator.validate_csv(content.decode("utf-8", errors="replace"))
    return ValidationResponse(
        valid=result.valid,
        errors=result.errors,
        row_count=result.row_count,
        columns=result.columns,
    )


@router.post("/upload", response_model=Dataset)
async def upload_dataset(file: UploadFile) -> Dataset:
    """Upload and import a CSV file as a new dataset."""
    content_bytes = await file.read()
    if len(content_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=422, detail="File too large (max 10 MB)")

    content = content_bytes.decode("utf-8", errors="replace")

    result = validator.validate_csv(content)
    if not result.valid:
        raise HTTPException(status_code=422, detail="; ".join(result.errors))

    points = _parse_csv_to_points(content)
    grid = _points_to_grid(points)

    # Derive scenario name from filename
    scenario_name = file.filename or "imported"
    if scenario_name.endswith(".csv"):
        scenario_name = scenario_name[:-4]

    dataset = Dataset(
        metadata=DatasetMetadata(
            id=uuid4(),
            scenario_name=scenario_name,
            created_at=datetime.now(timezone.utc),
            params={"source": "import", "original_filename": file.filename or "unknown"},
        ),
        grid_data=grid,
        survey_points=points,
    )

    store.save(dataset)

    await manager.broadcast(
        "datasets",
        "dataset_created",
        {"id": str(dataset.metadata.id), "scenario_name": dataset.metadata.scenario_name},
    )

    return dataset
