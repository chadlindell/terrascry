"""Exports router — download dataset data in various formats."""

import io
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.dataset import Dataset
from app.services.dataset_store import DatasetStore

router = APIRouter(prefix="/api/datasets", tags=["exports"])

store = DatasetStore()


def _dataset_or_404(dataset_id: UUID) -> Dataset:
    dataset = store.get(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    return dataset


def _export_csv(dataset: Dataset) -> str:
    """Export survey points as CSV with metadata header."""
    lines = []
    lines.append("# TERRASCRY Survey Export")
    lines.append(f"# scenario: {dataset.metadata.scenario_name}")
    lines.append(f"# created: {dataset.metadata.created_at.isoformat()}")
    lines.append(f"# dataset_id: {dataset.metadata.id}")
    lines.append("x,y,gradient_nt")
    for pt in dataset.survey_points:
        lines.append(f"{pt.x},{pt.y},{pt.gradient_nt}")
    return "\n".join(lines) + "\n"


def _export_grid_csv(dataset: Dataset) -> str:
    """Export grid data as CSV with metadata header."""
    grid = dataset.grid_data
    lines = []
    lines.append("# TERRASCRY Grid Export")
    lines.append(f"# scenario: {dataset.metadata.scenario_name}")
    lines.append(f"# rows: {grid.rows}, cols: {grid.cols}")
    lines.append(f"# x_min: {grid.x_min}, y_min: {grid.y_min}")
    lines.append(f"# dx: {grid.dx}, dy: {grid.dy}")
    lines.append(f"# unit: {grid.unit}")
    # Write grid values row by row (row 0 = y_min)
    for row in range(grid.rows):
        vals = grid.values[row * grid.cols : (row + 1) * grid.cols]
        lines.append(",".join(str(v) for v in vals))
    return "\n".join(lines) + "\n"


def _export_asc(dataset: Dataset) -> str:
    """Export grid data as ESRI ASCII Grid format."""
    grid = dataset.grid_data
    lines = []
    lines.append(f"ncols         {grid.cols}")
    lines.append(f"nrows         {grid.rows}")
    lines.append(f"xllcorner     {grid.x_min}")
    lines.append(f"yllcorner     {grid.y_min}")
    lines.append(f"cellsize      {grid.dx}")
    lines.append("NODATA_value  -9999")
    # ESRI ASCII Grid: first row is the top (highest y), so we reverse row order
    for row in range(grid.rows - 1, -1, -1):
        vals = grid.values[row * grid.cols : (row + 1) * grid.cols]
        lines.append(" ".join(str(v) for v in vals))
    return "\n".join(lines) + "\n"


_FORMATS = {
    "csv": ("text/csv", ".csv", _export_csv),
    "grid_csv": ("text/csv", "_grid.csv", _export_grid_csv),
    "asc": ("text/plain", ".asc", _export_asc),
}


@router.get("/{dataset_id}/export")
async def export_dataset(
    dataset_id: UUID,
    format: str = Query(default="csv", pattern="^(csv|grid_csv|asc)$"),
) -> StreamingResponse:
    dataset = _dataset_or_404(dataset_id)

    media_type, suffix, formatter = _FORMATS[format]
    content = formatter(dataset)
    filename = f"{dataset.metadata.scenario_name}_{dataset_id}{suffix}"

    return StreamingResponse(
        io.BytesIO(content.encode("utf-8")),
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
