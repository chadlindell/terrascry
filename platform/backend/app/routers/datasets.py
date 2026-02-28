"""Datasets router — list, retrieve, and delete stored simulation datasets."""

from uuid import UUID

from fastapi import APIRouter, HTTPException, Response

from app.dataset import Dataset, DatasetMetadata
from app.services.connection_manager import manager
from app.services.dataset_store import DatasetStore

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

store = DatasetStore()


@router.get("", response_model=list[DatasetMetadata])
async def list_datasets() -> list[DatasetMetadata]:
    return store.list()


@router.get("/{dataset_id}", response_model=Dataset)
async def get_dataset(dataset_id: UUID) -> Dataset:
    dataset = store.get(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    return dataset


@router.delete("/{dataset_id}", status_code=204)
async def delete_dataset(dataset_id: UUID) -> Response:
    deleted = store.delete(dataset_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    await manager.broadcast("datasets", "dataset_deleted", {"id": str(dataset_id)})
    return Response(status_code=204)
