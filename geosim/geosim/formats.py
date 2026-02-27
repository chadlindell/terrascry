"""Data format converters for instrument interoperability.

Reads native instrument CSV files and converts to the common GeoSim
survey data format for cross-project analysis.

Supported formats:
    - Pathfinder CSV: GPS-tagged gradiometer data (firmware or GeoSim-simulated)
    - HIRT CSV: Local-grid ERT/EM data (placeholder, firmware not yet implemented)
    - Common JSON: Unified record format for cross-instrument workflows

The common format uses a list of dicts with standardized keys:
    timestamp_s, lat_deg, lon_deg, grid_x_m, grid_y_m, sensor_readings, ...
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from geosim.coordinates import GridOrigin, gps_to_grid


def read_pathfinder_csv(
    filepath: str | Path,
    origin: GridOrigin | None = None,
) -> list[dict[str, Any]]:
    """Read a Pathfinder CSV and convert to common survey format.

    Handles both relative millisecond timestamps (firmware native) and
    ISO 8601 timestamps. If a grid origin is provided, computes local
    grid coordinates (grid_x_m, grid_y_m) from the GPS lat/lon.

    Parameters
    ----------
    filepath : str or Path
        Path to the Pathfinder CSV file.
    origin : GridOrigin, optional
        If provided, GPS coordinates are transformed to local grid meters
        and added as ``grid_x_m`` and ``grid_y_m`` fields.

    Returns
    -------
    records : list of dict
        Each dict contains:
        - ``timestamp_s`` : float — timestamp in seconds
        - ``timestamp_ms`` : int — original millisecond timestamp (if numeric)
        - ``timestamp_iso`` : str — ISO 8601 string (if parsed from ISO format)
        - ``lat_deg`` : float — latitude in decimal degrees
        - ``lon_deg`` : float — longitude in decimal degrees
        - ``grid_x_m`` : float — local grid easting (only if origin provided)
        - ``grid_y_m`` : float — local grid northing (only if origin provided)
        - ``gradients`` : dict — per-pair gradient readings, e.g.
          ``{'g1': {'top': int, 'bot': int, 'grad': int}, ...}``
        - All other columns are preserved with their original names.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    ValueError
        If the CSV is missing required columns (timestamp, lat, lon).
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Pathfinder CSV not found: {filepath}")

    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty or malformed CSV: {filepath}")

        columns = set(reader.fieldnames)
        for required in ("timestamp", "lat", "lon"):
            if required not in columns:
                raise ValueError(
                    f"Missing required column '{required}' in {filepath}. "
                    f"Found columns: {sorted(columns)}"
                )

        # Detect sensor pairs from column names
        pair_numbers = []
        for col in reader.fieldnames:
            if col.startswith("g") and col.endswith("_grad"):
                try:
                    pair_numbers.append(int(col[1:-5]))
                except ValueError:
                    pass
        pair_numbers.sort()

        rows = list(reader)

    # Parse timestamps and build records
    records: list[dict[str, Any]] = []
    lats = np.empty(len(rows), dtype=np.float64)
    lons = np.empty(len(rows), dtype=np.float64)

    for i, row in enumerate(rows):
        record: dict[str, Any] = {}

        # Parse timestamp
        ts_raw = row["timestamp"].strip()
        record["timestamp_s"] = _parse_timestamp(ts_raw, record)

        # Parse GPS coordinates
        lat = float(row["lat"])
        lon = float(row["lon"])
        record["lat_deg"] = lat
        record["lon_deg"] = lon
        lats[i] = lat
        lons[i] = lon

        # Parse gradient data
        gradients: dict[str, dict[str, int]] = {}
        for pn in pair_numbers:
            top_col = f"g{pn}_top"
            bot_col = f"g{pn}_bot"
            grad_col = f"g{pn}_grad"
            gradients[f"g{pn}"] = {
                "top": int(float(row.get(top_col, 0))),
                "bot": int(float(row.get(bot_col, 0))),
                "grad": int(float(row.get(grad_col, 0))),
            }
        record["gradients"] = gradients

        # Preserve additional columns
        skip_cols = {"timestamp", "lat", "lon"}
        for pn in pair_numbers:
            skip_cols.update({f"g{pn}_top", f"g{pn}_bot", f"g{pn}_grad"})
        for col, val in row.items():
            if col not in skip_cols:
                record[col] = _try_numeric(val)

        records.append(record)

    # Add grid coordinates if origin is provided
    if origin is not None and len(records) > 0:
        x, y = gps_to_grid(lats, lons, origin)
        for i, rec in enumerate(records):
            rec["grid_x_m"] = float(x[i])
            rec["grid_y_m"] = float(y[i])

    return records


def read_hirt_csv(filepath: str | Path) -> list[dict[str, Any]]:
    """Read HIRT CSV format.

    Placeholder for future HIRT firmware CSV reading. HIRT firmware is
    not yet implemented, so this function reads a generic CSV with
    expected HIRT columns (x, y, measurement values) and returns
    records in the common format.

    Parameters
    ----------
    filepath : str or Path
        Path to the HIRT CSV file.

    Returns
    -------
    records : list of dict
        Each dict contains available fields mapped to common names:
        - ``grid_x_m`` : float — local grid X (meters)
        - ``grid_y_m`` : float — local grid Y (meters)
        - ``timestamp_s`` : float — timestamp if available
        - All other columns preserved with original names.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"HIRT CSV not found: {filepath}")

    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty or malformed CSV: {filepath}")
        rows = list(reader)

    records: list[dict[str, Any]] = []
    for row in rows:
        record: dict[str, Any] = {}

        # Map expected HIRT columns to common names
        if "x" in row:
            record["grid_x_m"] = float(row["x"])
        if "y" in row:
            record["grid_y_m"] = float(row["y"])
        if "timestamp" in row:
            ts_raw = row["timestamp"].strip()
            record["timestamp_s"] = _parse_timestamp(ts_raw, record)

        # Preserve all columns with their original names
        for col, val in row.items():
            if col not in record:
                record[col] = _try_numeric(val)

        records.append(record)

    return records


def write_common_format(
    records: list[dict[str, Any]],
    filepath: str | Path,
) -> None:
    """Write records in common JSON survey format.

    Outputs a JSON file with a top-level structure containing metadata
    and the record list, suitable for cross-instrument analysis.

    Parameters
    ----------
    records : list of dict
        Records to write (from ``read_pathfinder_csv``, ``read_hirt_csv``,
        or manually constructed).
    filepath : str or Path
        Output JSON file path.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert any numpy types to Python native for JSON serialization
    clean_records = [_sanitize_for_json(rec) for rec in records]

    output = {
        "format": "geosim-common-v1",
        "record_count": len(clean_records),
        "records": clean_records,
    }

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_timestamp(ts_raw: str, record: dict[str, Any]) -> float:
    """Parse a timestamp string as either numeric ms or ISO 8601.

    Parameters
    ----------
    ts_raw : str
        Raw timestamp value from CSV.
    record : dict
        Record dict to update with ``timestamp_ms`` or ``timestamp_iso``.

    Returns
    -------
    timestamp_s : float
        Timestamp in seconds.
    """
    try:
        ts_ms = int(float(ts_raw))
        record["timestamp_ms"] = ts_ms
        return ts_ms / 1000.0
    except (ValueError, OverflowError):
        pass

    # Try ISO 8601 parsing
    try:
        dt = datetime.fromisoformat(ts_raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        record["timestamp_iso"] = dt.isoformat()
        return dt.timestamp()
    except (ValueError, TypeError):
        pass

    # Fallback: store raw and return 0
    record["timestamp_raw"] = ts_raw
    return 0.0


def _try_numeric(val: str) -> int | float | str:
    """Try to convert a string to int or float; return as-is if neither."""
    try:
        f = float(val)
        i = int(f)
        if f == i and "." not in val and "e" not in val.lower():
            return i
        return f
    except (ValueError, TypeError):
        return val


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert numpy types to Python native for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(item) for item in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
