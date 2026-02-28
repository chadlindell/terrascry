"""DatasetStore service — filesystem JSON CRUD for simulation datasets."""

import json
from pathlib import Path
from uuid import UUID

from app.config import settings
from app.dataset import Dataset, DatasetMetadata


class DatasetStore:
    """Persist and retrieve Dataset objects as JSON files."""

    @property
    def _dir(self) -> Path:
        d = settings.data_dir
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save(self, dataset: Dataset) -> UUID:
        """Write a dataset to disk. Returns the dataset ID."""
        path = self._dir / f"{dataset.metadata.id}.json"
        path.write_text(dataset.model_dump_json(indent=2))
        return dataset.metadata.id

    def list(self) -> list[DatasetMetadata]:
        """Return metadata for all stored datasets, newest first."""
        results = []
        for path in self._dir.glob("*.json"):
            data = json.loads(path.read_text())
            results.append(DatasetMetadata(**data["metadata"]))
        results.sort(key=lambda m: m.created_at, reverse=True)
        return results

    def get(self, dataset_id: UUID) -> Dataset | None:
        """Load a dataset by ID, or None if not found."""
        path = self._dir / f"{dataset_id}.json"
        if not path.is_file():
            return None
        return Dataset.model_validate_json(path.read_text())

    def delete(self, dataset_id: UUID) -> bool:
        """Delete a dataset by ID. Returns True if deleted, False if not found."""
        path = self._dir / f"{dataset_id}.json"
        if not path.is_file():
            return False
        path.unlink()
        return True
