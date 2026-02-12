"""Tests for CSV compatibility with Pathfinder's visualize_data.py.

These tests verify that GeoSim's exported CSV files can be loaded
and processed by the existing Pathfinder visualization tools.
"""

import tempfile

import numpy as np
import pytest

from geosim.sensors.pathfinder import (
    PathfinderConfig,
    export_csv,
    generate_walk_path,
    simulate_survey,
)


@pytest.fixture
def survey_csv():
    """Generate a survey CSV file for testing."""
    sources = [
        {'position': [10, 10, -1], 'moment': [0, 0, 5.0]},
        {'position': [15, 10, -0.5], 'moment': [0, 0, 2.0]},
    ]
    positions, timestamps, headings = generate_walk_path(
        start=(0, 10), end=(20, 10), speed=1.0, sample_rate=10.0
    )

    data = simulate_survey(
        positions, timestamps, headings, sources,
        add_noise=False,
    )

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        path = f.name

    export_csv(data, path, num_pairs=4)
    return path


class TestVisualizDataCompatibility:
    """Verify CSV format matches visualize_data.py expectations."""

    def test_detect_pairs(self, survey_csv):
        """detect_pairs() from visualize_data.py should find 4 pairs."""
        pd = pytest.importorskip("pandas")
        df = pd.read_csv(survey_csv)

        # Replicate detect_pairs logic from visualize_data.py
        n = 0
        for i in range(1, 5):
            if f'g{i}_grad' in df.columns:
                n = i
        assert n == 4

    def test_timestamp_column_numeric(self, survey_csv):
        """Timestamp column must be numeric (milliseconds)."""
        pd = pytest.importorskip("pandas")
        df = pd.read_csv(survey_csv)

        assert df['timestamp'].dtype in [np.int64, np.float64, np.int32, np.uint32]
        assert df['timestamp'].iloc[0] >= 0

    def test_gps_columns_present(self, survey_csv):
        """lat and lon columns are present."""
        pd = pytest.importorskip("pandas")
        df = pd.read_csv(survey_csv)

        assert 'lat' in df.columns
        assert 'lon' in df.columns

    def test_gradient_columns_are_integers(self, survey_csv):
        """Gradient values should be integer ADC counts."""
        pd = pytest.importorskip("pandas")
        df = pd.read_csv(survey_csv)

        for i in range(1, 5):
            col = f'g{i}_grad'
            # Should be numeric and integer-valued
            assert df[col].dtype in [np.int64, np.float64]
            if df[col].dtype == np.float64:
                np.testing.assert_array_equal(df[col], df[col].astype(int))

    def test_gradient_is_bot_minus_top(self, survey_csv):
        """Gradient = bottom - top (firmware convention)."""
        pd = pytest.importorskip("pandas")
        df = pd.read_csv(survey_csv)

        for i in range(1, 5):
            computed = df[f'g{i}_bot'] - df[f'g{i}_top']
            np.testing.assert_array_equal(df[f'g{i}_grad'].values, computed.values)

    def test_no_nan_values(self, survey_csv):
        """CSV should have no NaN or missing values."""
        pd = pytest.importorskip("pandas")
        df = pd.read_csv(survey_csv)

        assert not df.isnull().any().any(), "CSV contains NaN values"

    def test_sample_count_reasonable(self, survey_csv):
        """Walking 20m at 1m/s at 10Hz should give ~200 samples."""
        pd = pytest.importorskip("pandas")
        df = pd.read_csv(survey_csv)

        assert 150 < len(df) < 250, f"Got {len(df)} samples, expected ~200"
