"""Anomaly detection endpoint for stored datasets."""

from fastapi import APIRouter, HTTPException, Query

from app.services.anomaly import AnomalyService
from app.services.dataset_store import DatasetStore

router = APIRouter(prefix="/api/datasets", tags=["anomalies"])


@router.get("/{dataset_id}/anomalies")
async def get_anomalies(
    dataset_id: str,
    threshold_sigma: float = Query(default=3.0, ge=1.0, le=10.0),
) -> list[dict]:
    """Detect anomalous cells in a dataset's grid data."""
    store = DatasetStore()
    dataset = store.get(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    service = AnomalyService()
    return service.detect(dataset.grid_data, threshold_sigma=threshold_sigma)
