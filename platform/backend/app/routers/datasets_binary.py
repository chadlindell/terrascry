"""Binary grid data transfer endpoint for fast Float32Array delivery."""

import struct

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app.services.dataset_store import DatasetStore

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.get("/{dataset_id}/binary")
async def get_dataset_binary(dataset_id: str) -> Response:
    """Return grid data as a binary Float32Array with metadata in headers.

    Response body: little-endian Float32Array (rows * cols * 4 bytes).
    Grid geometry is in response headers: X-Grid-Rows, X-Grid-Cols,
    X-Grid-Xmin, X-Grid-Ymin, X-Grid-Dx, X-Grid-Dy, X-Grid-Unit.
    """
    store = DatasetStore()
    dataset = store.get(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    grid = dataset.grid_data
    body = struct.pack(f"<{len(grid.values)}f", *grid.values)

    return Response(
        content=body,
        media_type="application/octet-stream",
        headers={
            "X-Grid-Rows": str(grid.rows),
            "X-Grid-Cols": str(grid.cols),
            "X-Grid-Xmin": str(grid.x_min),
            "X-Grid-Ymin": str(grid.y_min),
            "X-Grid-Dx": str(grid.dx),
            "X-Grid-Dy": str(grid.dy),
            "X-Grid-Unit": grid.unit,
        },
    )
