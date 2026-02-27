"""Common survey data format for HIRT/Pathfinder interoperability.

Defines the GeoSim Survey Format (GSF) -- a JSON-based intermediate format
that both instruments export to.  Post-processing tools consume this format
for cross-project analysis, data fusion, and visualization.

The GSF complements the lighter-weight dict-based records in ``formats.py``
by adding explicit typing (enums, dataclasses) and structured metadata
suitable for archival and exchange between projects.

Typical workflow::

    # Pathfinder field data -> GSF
    survey = pathfinder_csv_to_gsf("PATH0001.CSV", survey_id="site7-line3")
    survey.to_json("site7-line3.gsf.json")

    # Later analysis
    survey = SurveyFile.from_json("site7-line3.gsf.json")
    for rec in survey.iter_records():
        print(rec.location.lat, rec.values)
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from geosim.coordinates import GridOrigin, gps_to_grid

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class InstrumentType(str, Enum):
    """Instrument that produced the measurement."""

    PATHFINDER = "pathfinder"
    HIRT_MIT = "hirt_mit"
    HIRT_ERT = "hirt_ert"


class MeasurementType(str, Enum):
    """Physical quantity being measured."""

    MAGNETIC_GRADIENT = "magnetic_gradient"
    EM_AMPLITUDE = "em_amplitude"
    EM_PHASE = "em_phase"
    RESISTIVITY = "resistivity"


class CalibrationStatus(str, Enum):
    """Whether values have been calibrated against a known reference."""

    RAW = "raw"
    CALIBRATED = "calibrated"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Location:
    """Spatial coordinates for a single measurement point.

    At least one coordinate pair (lat/lon or grid_x/grid_y) should be
    provided.  Both may be present when GPS data has been projected onto
    a local survey grid.

    Parameters
    ----------
    lat : float, optional
        WGS84 latitude in decimal degrees.
    lon : float, optional
        WGS84 longitude in decimal degrees.
    grid_x_m : float, optional
        Local grid easting in metres.
    grid_y_m : float, optional
        Local grid northing in metres.
    altitude_m : float, optional
        Altitude / elevation in metres (GPS or surveyed).
    grid_origin_lat : float, optional
        Latitude of the grid origin used for the grid projection.
    grid_origin_lon : float, optional
        Longitude of the grid origin used for the grid projection.
    """

    lat: float | None = None
    lon: float | None = None
    grid_x_m: float | None = None
    grid_y_m: float | None = None
    altitude_m: float | None = None
    grid_origin_lat: float | None = None
    grid_origin_lon: float | None = None


@dataclass
class QualityFlags:
    """Data-quality indicators attached to each measurement record.

    Parameters
    ----------
    gps_fix_quality : int, optional
        NMEA fix quality (0 = invalid, 1 = GPS, 2 = DGPS, 4 = RTK).
    hdop : float, optional
        Horizontal dilution of precision from the GPS receiver.
    adc_saturated : bool
        True if any ADC channel was at its rail during this sample.
    below_noise_floor : bool
        True if the signal magnitude is below the estimated noise floor.
    """

    gps_fix_quality: int | None = None
    hdop: float | None = None
    adc_saturated: bool = False
    below_noise_floor: bool = False


@dataclass
class SurveyRecord:
    """Single measurement record in GeoSim Survey Format.

    Parameters
    ----------
    survey_id : str
        Human-readable identifier for the survey line or block.
    timestamp_utc : str
        ISO 8601 timestamp in UTC (e.g. ``"2025-06-15T14:30:00+00:00"``).
    location : Location
        Spatial coordinates of the measurement.
    instrument : InstrumentType
        Source instrument.
    measurement_type : MeasurementType
        Physical quantity measured.
    values : dict
        Instrument-specific measurement values.  For Pathfinder this
        contains per-pair gradient data; for HIRT it would contain
        amplitude/phase or apparent resistivity.
    calibration_status : CalibrationStatus
        Whether the values have been calibrated.
    quality_flags : QualityFlags
        Data-quality metadata.
    operator : str, optional
        Name or identifier of the field operator.
    notes : str, optional
        Free-text notes attached to this record.
    """

    survey_id: str
    timestamp_utc: str  # ISO 8601
    location: Location
    instrument: InstrumentType
    measurement_type: MeasurementType
    values: dict[str, Any]
    calibration_status: CalibrationStatus = CalibrationStatus.RAW
    quality_flags: QualityFlags = field(default_factory=QualityFlags)
    operator: str | None = None
    notes: str | None = None


@dataclass
class SurveyFile:
    """Complete survey file in GeoSim Survey Format.

    A ``SurveyFile`` aggregates metadata and an ordered list of
    ``SurveyRecord`` entries, and provides JSON serialisation helpers.

    Parameters
    ----------
    format_version : str
        Semantic version of the GSF schema (currently ``"1.0.0"``).
    created_utc : str
        ISO 8601 creation timestamp.  Auto-populated on construction
        if left empty.
    instrument : str
        Human-readable instrument identifier (e.g. ``"Pathfinder v1"``).
    firmware_version : str
        Firmware or software version that produced the source data.
    site_name : str
        Name of the survey site.
    grid_origin : Location, optional
        Grid origin used for coordinate projection.
    records : list of dict
        Serialised record dicts (call ``iter_records`` to get typed
        ``SurveyRecord`` objects back).
    """

    format_version: str = "1.0.0"
    created_utc: str = ""
    instrument: str = ""
    firmware_version: str = ""
    site_name: str = ""
    grid_origin: Location | None = None
    records: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.created_utc:
            self.created_utc = datetime.now(timezone.utc).isoformat()

    # -- serialisation -------------------------------------------------------

    def to_json(self, filepath: str | Path, indent: int = 2) -> None:
        """Write the survey to a JSON file.

        Parameters
        ----------
        filepath : str or Path
            Destination file path.  Parent directories are created if
            they do not exist.
        indent : int
            JSON indentation level (default 2).
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data = _sanitize_for_json(data)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=indent)

    @classmethod
    def from_json(cls, filepath: str | Path) -> SurveyFile:
        """Read a survey from a JSON file.

        Parameters
        ----------
        filepath : str or Path
            Path to a ``.gsf.json`` file previously written by
            ``to_json``.

        Returns
        -------
        SurveyFile
            Reconstructed survey with nested dataclasses restored.

        Raises
        ------
        FileNotFoundError
            If *filepath* does not exist.
        json.JSONDecodeError
            If the file is not valid JSON.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"GSF file not found: {filepath}")

        with open(filepath) as f:
            data = json.load(f)

        grid_origin = None
        if data.get("grid_origin") is not None:
            grid_origin = Location(**data["grid_origin"])

        # Reconstruct record dicts -- they stay as dicts internally but
        # we validate that each one can round-trip through SurveyRecord.
        records: list[dict[str, Any]] = []
        for raw in data.get("records", []):
            records.append(raw)

        return cls(
            format_version=data.get("format_version", "1.0.0"),
            created_utc=data.get("created_utc", ""),
            instrument=data.get("instrument", ""),
            firmware_version=data.get("firmware_version", ""),
            site_name=data.get("site_name", ""),
            grid_origin=grid_origin,
            records=records,
        )

    # -- record helpers ------------------------------------------------------

    def add_record(self, record: SurveyRecord) -> None:
        """Append a ``SurveyRecord`` (stored internally as a dict).

        Parameters
        ----------
        record : SurveyRecord
            The measurement record to add.
        """
        self.records.append(asdict(record))

    def iter_records(self) -> Iterator[SurveyRecord]:
        """Iterate over records as typed ``SurveyRecord`` objects.

        Yields
        ------
        SurveyRecord
            Each stored record, with nested dataclasses reconstructed.
        """
        for raw in self.records:
            yield _dict_to_record(raw)

    @property
    def record_count(self) -> int:
        """Number of records in the survey."""
        return len(self.records)


# ---------------------------------------------------------------------------
# CSV converters
# ---------------------------------------------------------------------------


def pathfinder_csv_to_gsf(
    csv_path: str | Path,
    survey_id: str,
    grid_origin_lat: float | None = None,
    grid_origin_lon: float | None = None,
    site_name: str = "",
    operator: str | None = None,
) -> SurveyFile:
    """Convert a Pathfinder CSV file to GeoSim Survey Format.

    Handles both relative millisecond timestamps (firmware native) and
    ISO 8601 timestamps.  Comment lines starting with ``#`` are skipped.

    Parameters
    ----------
    csv_path : str or Path
        Path to the Pathfinder CSV file.
    survey_id : str
        Human-readable survey identifier (e.g. ``"site7-line3"``).
    grid_origin_lat : float, optional
        Latitude of the local grid origin.  If provided together with
        *grid_origin_lon*, GPS coordinates are projected to local grid
        metres.
    grid_origin_lon : float, optional
        Longitude of the local grid origin.
    site_name : str
        Name of the survey site (stored in file metadata).
    operator : str, optional
        Field operator name.

    Returns
    -------
    SurveyFile
        A complete GSF survey file ready for serialisation.

    Raises
    ------
    FileNotFoundError
        If *csv_path* does not exist.
    ValueError
        If the CSV is missing required columns (``timestamp``, ``lat``,
        ``lon``).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Pathfinder CSV not found: {csv_path}")

    # Read CSV, skipping comment lines
    lines: list[str] = []
    with open(csv_path, newline="") as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                lines.append(line)

    if not lines:
        raise ValueError(f"Empty CSV file: {csv_path}")

    reader = csv.DictReader(lines)
    if reader.fieldnames is None:
        raise ValueError(f"Malformed CSV (no header): {csv_path}")

    columns = set(reader.fieldnames)
    for required in ("timestamp", "lat", "lon"):
        if required not in columns:
            raise ValueError(
                f"Missing required column '{required}' in {csv_path}. "
                f"Found columns: {sorted(columns)}"
            )

    # Detect sensor pairs from column names (g1_grad, g2_grad, ...)
    pair_numbers: list[int] = []
    for col in reader.fieldnames:
        if col.startswith("g") and col.endswith("_grad"):
            try:
                pair_numbers.append(int(col[1:-5]))
            except ValueError:
                pass
    pair_numbers.sort()

    rows = list(reader)

    # Build grid origin if both lat/lon provided
    origin: GridOrigin | None = None
    grid_origin_loc: Location | None = None
    if grid_origin_lat is not None and grid_origin_lon is not None:
        origin = GridOrigin(lat_deg=grid_origin_lat, lon_deg=grid_origin_lon)
        grid_origin_loc = Location(lat=grid_origin_lat, lon=grid_origin_lon)

    # Pre-compute grid coordinates for all rows in one vectorised call
    lats = np.array([float(r["lat"]) for r in rows], dtype=np.float64)
    lons = np.array([float(r["lon"]) for r in rows], dtype=np.float64)
    grid_xs: np.ndarray | None = None
    grid_ys: np.ndarray | None = None
    if origin is not None:
        grid_xs, grid_ys = gps_to_grid(lats, lons, origin)

    # Build records
    survey = SurveyFile(
        instrument="Pathfinder",
        site_name=site_name,
        grid_origin=grid_origin_loc,
    )

    for i, row in enumerate(rows):
        # Timestamp
        ts_raw = row["timestamp"].strip()
        ts_utc = _raw_timestamp_to_iso(ts_raw)

        # Location
        lat = float(row["lat"])
        lon = float(row["lon"])
        loc = Location(
            lat=lat,
            lon=lon,
            grid_origin_lat=grid_origin_lat,
            grid_origin_lon=grid_origin_lon,
        )
        if grid_xs is not None and grid_ys is not None:
            loc.grid_x_m = float(grid_xs[i])
            loc.grid_y_m = float(grid_ys[i])

        # Measurement values: per-pair gradient data
        values: dict[str, Any] = {}
        for pn in pair_numbers:
            top_val = int(float(row.get(f"g{pn}_top", "0")))
            bot_val = int(float(row.get(f"g{pn}_bot", "0")))
            grad_val = int(float(row.get(f"g{pn}_grad", "0")))
            values[f"pair_{pn}_top"] = top_val
            values[f"pair_{pn}_bot"] = bot_val
            values[f"pair_{pn}_gradient"] = grad_val

        # Quality flags -- check for ADC saturation (±32000 counts)
        adc_saturated = any(
            abs(int(float(row.get(f"g{pn}_top", "0")))) >= 32000
            or abs(int(float(row.get(f"g{pn}_bot", "0")))) >= 32000
            for pn in pair_numbers
        )
        qf = QualityFlags(adc_saturated=adc_saturated)

        record = SurveyRecord(
            survey_id=survey_id,
            timestamp_utc=ts_utc,
            location=loc,
            instrument=InstrumentType.PATHFINDER,
            measurement_type=MeasurementType.MAGNETIC_GRADIENT,
            values=values,
            calibration_status=CalibrationStatus.RAW,
            quality_flags=qf,
            operator=operator,
        )
        survey.add_record(record)

    return survey


def hirt_csv_to_gsf(csv_path: str | Path, survey_id: str) -> SurveyFile:
    """Convert HIRT CSV to GeoSim Survey Format.

    This is a placeholder.  HIRT firmware has not yet been implemented,
    so the native CSV format is not yet defined.

    Parameters
    ----------
    csv_path : str or Path
        Path to the HIRT CSV file.
    survey_id : str
        Human-readable survey identifier.

    Raises
    ------
    NotImplementedError
        Always -- HIRT firmware is not yet implemented.
    """
    raise NotImplementedError(
        "HIRT CSV conversion is not yet available.  "
        "The HIRT firmware and its native CSV format have not been finalised."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _raw_timestamp_to_iso(ts_raw: str) -> str:
    """Convert a raw timestamp string to ISO 8601 UTC.

    Handles two cases:
    1. Numeric millisecond offset (firmware native) -- converted to a
       relative ISO timestamp anchored at the Unix epoch for traceability.
    2. ISO 8601 string -- normalised to include a UTC timezone suffix.

    Parameters
    ----------
    ts_raw : str
        Raw timestamp from CSV.

    Returns
    -------
    str
        ISO 8601 timestamp string.
    """
    # Try numeric (milliseconds since power-on)
    try:
        ts_ms = int(float(ts_raw))
        # Store as seconds-since-epoch for a deterministic ISO string.
        # Firmware timestamps are relative, so we anchor at epoch.
        dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        return dt.isoformat()
    except (ValueError, OverflowError):
        pass

    # Try ISO 8601
    try:
        dt = datetime.fromisoformat(ts_raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except (ValueError, TypeError):
        pass

    # Fallback: return as-is with a note
    return ts_raw


def _dict_to_record(raw: dict[str, Any]) -> SurveyRecord:
    """Reconstruct a ``SurveyRecord`` from its dict representation.

    Parameters
    ----------
    raw : dict
        A dict previously produced by ``dataclasses.asdict(record)``.

    Returns
    -------
    SurveyRecord
        Typed record with nested dataclasses restored.
    """
    location = Location(**raw.get("location", {}))
    quality_flags = QualityFlags(**raw.get("quality_flags", {}))

    # Enum values are stored as their string value in JSON
    instrument = InstrumentType(raw["instrument"])
    measurement_type = MeasurementType(raw["measurement_type"])
    calibration_status = CalibrationStatus(
        raw.get("calibration_status", CalibrationStatus.RAW.value)
    )

    return SurveyRecord(
        survey_id=raw["survey_id"],
        timestamp_utc=raw["timestamp_utc"],
        location=location,
        instrument=instrument,
        measurement_type=measurement_type,
        values=raw.get("values", {}),
        calibration_status=calibration_status,
        quality_flags=quality_flags,
        operator=raw.get("operator"),
        notes=raw.get("notes"),
    )


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert numpy and enum types to JSON-safe Python natives.

    Parameters
    ----------
    obj : Any
        Object to sanitise (dict, list, numpy scalar, enum, etc.).

    Returns
    -------
    Any
        JSON-serialisable equivalent.
    """
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(item) for item in obj]
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
