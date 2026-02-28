"""Datasets router — list and retrieve stored simulation datasets."""

from uuid import UUID

from fastapi import APIRouter, HTTPException

from app.dataset import Dataset, DatasetMetadata
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
