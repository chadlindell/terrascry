"""Tests for GeoSim Survey Format (GSF) serialisation and CSV conversion.

Covers:
    - Creating and serialising SurveyFile / SurveyRecord
    - Round-trip JSON write / read
    - Pathfinder CSV -> GSF conversion using real example data
    - Edge cases and error handling
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from geosim.survey_format import (
    CalibrationStatus,
    InstrumentType,
    Location,
    MeasurementType,
    QualityFlags,
    SurveyFile,
    SurveyRecord,
    hirt_csv_to_gsf,
    pathfinder_csv_to_gsf,
)

# Path to Pathfinder example data (relative to repo root)
EXAMPLE_CSV = Path(__file__).resolve().parent.parent.parent / (
    "Pathfinder/firmware/tools/example_data.csv"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_record() -> SurveyRecord:
    """A minimal valid SurveyRecord."""
    return SurveyRecord(
        survey_id="test-001",
        timestamp_utc="2025-06-15T14:30:00+00:00",
        location=Location(lat=51.234567, lon=18.345678, grid_x_m=0.0, grid_y_m=0.0),
        instrument=InstrumentType.PATHFINDER,
        measurement_type=MeasurementType.MAGNETIC_GRADIENT,
        values={"pair_1_gradient": 228, "pair_1_top": 12450, "pair_1_bot": 12678},
        calibration_status=CalibrationStatus.RAW,
        quality_flags=QualityFlags(gps_fix_quality=1, hdop=1.2),
        operator="test-operator",
        notes="unit test record",
    )


@pytest.fixture
def sample_survey(sample_record: SurveyRecord) -> SurveyFile:
    """A SurveyFile with a few records."""
    survey = SurveyFile(
        instrument="Pathfinder v1",
        firmware_version="0.1.0",
        site_name="Test Site",
        grid_origin=Location(lat=51.234567, lon=18.345678),
    )
    survey.add_record(sample_record)
    # Add a second record with slightly different data
    rec2 = SurveyRecord(
        survey_id="test-001",
        timestamp_utc="2025-06-15T14:30:01+00:00",
        location=Location(lat=51.234568, lon=18.345679),
        instrument=InstrumentType.PATHFINDER,
        measurement_type=MeasurementType.MAGNETIC_GRADIENT,
        values={"pair_1_gradient": 222},
    )
    survey.add_record(rec2)
    return survey


# ---------------------------------------------------------------------------
# SurveyRecord creation
# ---------------------------------------------------------------------------


class TestSurveyRecordCreation:
    """Verify SurveyRecord construction and defaults."""

    def test_required_fields(self, sample_record: SurveyRecord) -> None:
        assert sample_record.survey_id == "test-001"
        assert sample_record.instrument == InstrumentType.PATHFINDER
        assert sample_record.measurement_type == MeasurementType.MAGNETIC_GRADIENT

    def test_default_calibration_status(self) -> None:
        rec = SurveyRecord(
            survey_id="x",
            timestamp_utc="2025-01-01T00:00:00+00:00",
            location=Location(),
            instrument=InstrumentType.PATHFINDER,
            measurement_type=MeasurementType.MAGNETIC_GRADIENT,
            values={},
        )
        assert rec.calibration_status == CalibrationStatus.RAW

    def test_default_quality_flags(self) -> None:
        rec = SurveyRecord(
            survey_id="x",
            timestamp_utc="2025-01-01T00:00:00+00:00",
            location=Location(),
            instrument=InstrumentType.PATHFINDER,
            measurement_type=MeasurementType.MAGNETIC_GRADIENT,
            values={},
        )
        assert rec.quality_flags.adc_saturated is False
        assert rec.quality_flags.below_noise_floor is False
        assert rec.quality_flags.gps_fix_quality is None

    def test_enum_values_are_strings(self) -> None:
        assert InstrumentType.PATHFINDER.value == "pathfinder"
        assert MeasurementType.MAGNETIC_GRADIENT.value == "magnetic_gradient"
        assert CalibrationStatus.RAW.value == "raw"

    def test_location_optional_fields(self) -> None:
        loc = Location(lat=51.0, lon=18.0)
        assert loc.grid_x_m is None
        assert loc.altitude_m is None


# ---------------------------------------------------------------------------
# SurveyFile creation and serialisation
# ---------------------------------------------------------------------------


class TestSurveyFileSerialization:
    """Verify SurveyFile JSON write and structure."""

    def test_created_utc_auto_populated(self) -> None:
        sf = SurveyFile()
        assert sf.created_utc != ""
        assert "T" in sf.created_utc  # ISO 8601 contains 'T'

    def test_add_record_increments_count(self, sample_survey: SurveyFile) -> None:
        assert sample_survey.record_count == 2

    def test_to_json_creates_file(self, sample_survey: SurveyFile) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.gsf.json")
            sample_survey.to_json(path)
            assert os.path.exists(path)

    def test_to_json_valid_json(self, sample_survey: SurveyFile) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.gsf.json")
            sample_survey.to_json(path)
            with open(path) as f:
                data = json.load(f)
            assert data["format_version"] == "1.0.0"
            assert data["instrument"] == "Pathfinder v1"
            assert len(data["records"]) == 2

    def test_to_json_creates_parent_dirs(self, sample_survey: SurveyFile) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sub", "dir", "test.gsf.json")
            sample_survey.to_json(path)
            assert os.path.exists(path)

    def test_enum_values_serialised_as_strings(
        self, sample_survey: SurveyFile
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.gsf.json")
            sample_survey.to_json(path)
            with open(path) as f:
                data = json.load(f)
            rec = data["records"][0]
            assert rec["instrument"] == "pathfinder"
            assert rec["measurement_type"] == "magnetic_gradient"
            assert rec["calibration_status"] == "raw"


# ---------------------------------------------------------------------------
# Round-trip JSON
# ---------------------------------------------------------------------------


class TestJsonRoundTrip:
    """Verify that write -> read preserves all data."""

    def test_round_trip_metadata(self, sample_survey: SurveyFile) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "rt.gsf.json")
            sample_survey.to_json(path)
            loaded = SurveyFile.from_json(path)

            assert loaded.format_version == sample_survey.format_version
            assert loaded.instrument == sample_survey.instrument
            assert loaded.firmware_version == sample_survey.firmware_version
            assert loaded.site_name == sample_survey.site_name
            assert loaded.record_count == sample_survey.record_count

    def test_round_trip_grid_origin(self, sample_survey: SurveyFile) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "rt.gsf.json")
            sample_survey.to_json(path)
            loaded = SurveyFile.from_json(path)

            assert loaded.grid_origin is not None
            assert loaded.grid_origin.lat == sample_survey.grid_origin.lat
            assert loaded.grid_origin.lon == sample_survey.grid_origin.lon

    def test_round_trip_records(self, sample_survey: SurveyFile) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "rt.gsf.json")
            sample_survey.to_json(path)
            loaded = SurveyFile.from_json(path)

            original_recs = list(sample_survey.iter_records())
            loaded_recs = list(loaded.iter_records())

            assert len(loaded_recs) == len(original_recs)
            for orig, loaded_rec in zip(original_recs, loaded_recs):
                assert loaded_rec.survey_id == orig.survey_id
                assert loaded_rec.timestamp_utc == orig.timestamp_utc
                assert loaded_rec.instrument == orig.instrument
                assert loaded_rec.measurement_type == orig.measurement_type
                assert loaded_rec.values == orig.values

    def test_round_trip_location(self, sample_survey: SurveyFile) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "rt.gsf.json")
            sample_survey.to_json(path)
            loaded = SurveyFile.from_json(path)

            orig = list(sample_survey.iter_records())[0]
            loaded_rec = list(loaded.iter_records())[0]
            assert loaded_rec.location.lat == orig.location.lat
            assert loaded_rec.location.lon == orig.location.lon
            assert loaded_rec.location.grid_x_m == orig.location.grid_x_m

    def test_round_trip_quality_flags(self, sample_survey: SurveyFile) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "rt.gsf.json")
            sample_survey.to_json(path)
            loaded = SurveyFile.from_json(path)

            orig = list(sample_survey.iter_records())[0]
            loaded_rec = list(loaded.iter_records())[0]
            assert loaded_rec.quality_flags.gps_fix_quality == orig.quality_flags.gps_fix_quality
            assert loaded_rec.quality_flags.hdop == orig.quality_flags.hdop
            assert loaded_rec.quality_flags.adc_saturated == orig.quality_flags.adc_saturated

    def test_round_trip_optional_fields(self, sample_survey: SurveyFile) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "rt.gsf.json")
            sample_survey.to_json(path)
            loaded = SurveyFile.from_json(path)

            orig = list(sample_survey.iter_records())[0]
            loaded_rec = list(loaded.iter_records())[0]
            assert loaded_rec.operator == orig.operator
            assert loaded_rec.notes == orig.notes

    def test_from_json_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            SurveyFile.from_json("/nonexistent/path/file.json")

    def test_round_trip_no_grid_origin(self) -> None:
        survey = SurveyFile(instrument="test", site_name="no-origin")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "rt.gsf.json")
            survey.to_json(path)
            loaded = SurveyFile.from_json(path)
            assert loaded.grid_origin is None


# ---------------------------------------------------------------------------
# Pathfinder CSV conversion
# ---------------------------------------------------------------------------


class TestPathfinderCsvConversion:
    """Convert Pathfinder example CSV to GSF and validate the result."""

    @pytest.fixture
    def example_csv_path(self) -> Path:
        """Resolve and verify the example CSV exists."""
        if not EXAMPLE_CSV.exists():
            pytest.skip(f"Example CSV not found at {EXAMPLE_CSV}")
        return EXAMPLE_CSV

    def test_basic_conversion(self, example_csv_path: Path) -> None:
        survey = pathfinder_csv_to_gsf(str(example_csv_path), survey_id="example")
        assert survey.record_count > 0
        assert survey.instrument == "Pathfinder"

    def test_record_count_matches_csv_rows(self, example_csv_path: Path) -> None:
        survey = pathfinder_csv_to_gsf(str(example_csv_path), survey_id="example")
        # example_data.csv has 10 data rows
        assert survey.record_count == 10

    def test_records_have_correct_instrument(self, example_csv_path: Path) -> None:
        survey = pathfinder_csv_to_gsf(str(example_csv_path), survey_id="example")
        for rec in survey.iter_records():
            assert rec.instrument == InstrumentType.PATHFINDER
            assert rec.measurement_type == MeasurementType.MAGNETIC_GRADIENT

    def test_records_have_gps_coordinates(self, example_csv_path: Path) -> None:
        survey = pathfinder_csv_to_gsf(str(example_csv_path), survey_id="example")
        for rec in survey.iter_records():
            assert rec.location.lat is not None
            assert rec.location.lon is not None
            assert 50.0 < rec.location.lat < 53.0  # Plausible European latitude
            assert 17.0 < rec.location.lon < 20.0

    def test_records_have_gradient_values(self, example_csv_path: Path) -> None:
        survey = pathfinder_csv_to_gsf(str(example_csv_path), survey_id="example")
        rec = list(survey.iter_records())[0]
        # Example CSV has 4 pairs
        assert "pair_1_gradient" in rec.values
        assert "pair_4_gradient" in rec.values
        assert "pair_1_top" in rec.values
        assert "pair_1_bot" in rec.values

    def test_gradient_values_match_csv(self, example_csv_path: Path) -> None:
        """First row: g1_top=12450, g1_bot=12678, g1_grad=228."""
        survey = pathfinder_csv_to_gsf(str(example_csv_path), survey_id="example")
        rec = list(survey.iter_records())[0]
        assert rec.values["pair_1_top"] == 12450
        assert rec.values["pair_1_bot"] == 12678
        assert rec.values["pair_1_gradient"] == 228

    def test_grid_coordinates_computed(self, example_csv_path: Path) -> None:
        survey = pathfinder_csv_to_gsf(
            str(example_csv_path),
            survey_id="example",
            grid_origin_lat=51.234567,
            grid_origin_lon=18.345678,
        )
        rec = list(survey.iter_records())[0]
        assert rec.location.grid_x_m is not None
        assert rec.location.grid_y_m is not None
        # First row should be at or very near the origin
        assert abs(rec.location.grid_x_m) < 1.0
        assert abs(rec.location.grid_y_m) < 1.0

    def test_conversion_round_trips_through_json(self, example_csv_path: Path) -> None:
        survey = pathfinder_csv_to_gsf(str(example_csv_path), survey_id="rt-test")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "rt.gsf.json")
            survey.to_json(path)
            loaded = SurveyFile.from_json(path)
            assert loaded.record_count == survey.record_count
            # Spot-check first record values
            orig = list(survey.iter_records())[0]
            rt = list(loaded.iter_records())[0]
            assert rt.values == orig.values
            assert rt.location.lat == orig.location.lat

    def test_survey_id_propagated(self, example_csv_path: Path) -> None:
        survey = pathfinder_csv_to_gsf(str(example_csv_path), survey_id="my-survey")
        for rec in survey.iter_records():
            assert rec.survey_id == "my-survey"

    def test_operator_propagated(self, example_csv_path: Path) -> None:
        survey = pathfinder_csv_to_gsf(
            str(example_csv_path), survey_id="x", operator="J. Smith"
        )
        for rec in survey.iter_records():
            assert rec.operator == "J. Smith"

    def test_timestamps_are_iso(self, example_csv_path: Path) -> None:
        survey = pathfinder_csv_to_gsf(str(example_csv_path), survey_id="x")
        for rec in survey.iter_records():
            assert "T" in rec.timestamp_utc  # ISO 8601 contains 'T'

    def test_file_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            pathfinder_csv_to_gsf("/nonexistent/path.csv", survey_id="x")


# ---------------------------------------------------------------------------
# HIRT CSV placeholder
# ---------------------------------------------------------------------------


class TestHirtCsvPlaceholder:
    """Verify HIRT conversion raises NotImplementedError."""

    def test_hirt_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError, match="HIRT"):
            hirt_csv_to_gsf("/any/path.csv", survey_id="x")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Boundary conditions and error paths."""

    def test_empty_survey_serialises(self) -> None:
        survey = SurveyFile(instrument="empty", site_name="nowhere")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "empty.gsf.json")
            survey.to_json(path)
            loaded = SurveyFile.from_json(path)
            assert loaded.record_count == 0

    def test_missing_column_raises(self) -> None:
        """A CSV without 'lat' should raise ValueError."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write("timestamp,lon,value\n")
            f.write("1000,18.0,42\n")
            path = f.name

        try:
            with pytest.raises(ValueError, match="Missing required column"):
                pathfinder_csv_to_gsf(path, survey_id="x")
        finally:
            os.unlink(path)

    def test_comment_lines_skipped(self) -> None:
        """Lines starting with # should be ignored."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write("# This is a comment\n")
            f.write("# Another comment\n")
            f.write("timestamp,lat,lon,g1_top,g1_bot,g1_grad\n")
            f.write("1000,51.0,18.0,100,200,100\n")
            path = f.name

        try:
            survey = pathfinder_csv_to_gsf(path, survey_id="comments")
            assert survey.record_count == 1
        finally:
            os.unlink(path)

    def test_iter_records_yields_typed_objects(
        self, sample_survey: SurveyFile
    ) -> None:
        for rec in sample_survey.iter_records():
            assert isinstance(rec, SurveyRecord)
            assert isinstance(rec.location, Location)
            assert isinstance(rec.quality_flags, QualityFlags)
            assert isinstance(rec.instrument, InstrumentType)
