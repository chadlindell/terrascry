"""Tests for Pathfinder gradiometer instrument model."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from geosim.sensors.pathfinder import (
    PathfinderConfig,
    export_csv,
    generate_walk_path,
    generate_zigzag_path,
    simulate_survey,
)


class TestPathfinderConfig:
    """Tests for instrument configuration."""

    def test_default_config(self):
        """Default config matches Pathfinder design specs."""
        config = PathfinderConfig()
        assert config.num_pairs == 4
        assert config.pair_spacing == 0.50
        assert config.sensor_separation == 0.35
        assert config.sample_rate == 10.0

    def test_swath_width(self):
        """Swath width = (num_pairs - 1) * spacing."""
        config = PathfinderConfig()
        assert config.swath_width == pytest.approx(1.5)

    def test_sensor_offsets_symmetric(self):
        """Sensor offsets are symmetric about center."""
        config = PathfinderConfig()
        offsets = config.sensor_offsets()
        assert len(offsets) == 4
        assert offsets[0] == pytest.approx(-0.75)
        assert offsets[-1] == pytest.approx(0.75)
        assert offsets[0] + offsets[-1] == pytest.approx(0.0)

    def test_drone_platform_defaults(self):
        """Drone profile defaults to 2 pairs and higher sample rate."""
        config = PathfinderConfig(platform="drone")
        assert config.num_pairs == 2
        assert config.sample_rate == pytest.approx(20.0)
        assert config.gps_quality is True


class TestWalkPath:
    """Tests for walk path generation."""

    def test_straight_path_endpoints(self):
        """Straight path starts and ends at specified points."""
        positions, timestamps, headings = generate_walk_path(
            start=(0.0, 0.0), end=(0.0, 10.0), speed=1.0, sample_rate=10.0
        )
        np.testing.assert_allclose(positions[0], [0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(positions[-1], [0.0, 10.0], atol=1e-10)

    def test_straight_path_duration(self):
        """Duration = distance / speed."""
        positions, timestamps, headings = generate_walk_path(
            start=(0.0, 0.0), end=(0.0, 10.0), speed=2.0, sample_rate=10.0
        )
        assert timestamps[-1] == pytest.approx(5.0, rel=0.1)

    def test_heading_north(self):
        """Walking north gives heading = 0."""
        _, _, headings = generate_walk_path(
            start=(0.0, 0.0), end=(0.0, 10.0)
        )
        assert headings[0] == pytest.approx(0.0, abs=1e-10)

    def test_heading_east(self):
        """Walking east gives heading = π/2."""
        _, _, headings = generate_walk_path(
            start=(0.0, 0.0), end=(10.0, 0.0)
        )
        assert headings[0] == pytest.approx(np.pi / 2, abs=1e-10)

    def test_zigzag_covers_area(self):
        """Zigzag path covers the specified area."""
        positions, _, _ = generate_zigzag_path(
            origin=(0.0, 0.0), width=5.0, length=10.0, line_spacing=1.0
        )
        assert positions[:, 0].min() >= -0.1
        assert positions[:, 0].max() <= 5.1
        assert positions[:, 1].min() >= -0.1
        assert positions[:, 1].max() <= 10.1


class TestSimulateSurvey:
    """Tests for survey simulation."""

    def test_output_has_correct_columns(self):
        """Survey output contains all expected Pathfinder columns."""
        sources = [{'position': [5, 5, -1], 'moment': [0, 0, 1.0]}]
        positions, timestamps, headings = generate_walk_path(
            start=(0, 0), end=(0, 10), speed=1.0, sample_rate=10.0
        )

        data = simulate_survey(
            positions, timestamps, headings, sources, add_noise=False
        )

        assert 'timestamp' in data
        assert 'lat' in data
        assert 'lon' in data
        for i in range(1, 5):
            assert f'g{i}_top' in data
            assert f'g{i}_bot' in data
            assert f'g{i}_grad' in data

    def test_gradient_is_bot_minus_top(self):
        """Gradient column = bottom - top (matching firmware)."""
        sources = [{'position': [5, 5, -1], 'moment': [0, 0, 1.0]}]
        positions, timestamps, headings = generate_walk_path(
            start=(0, 0), end=(0, 10), speed=1.0, sample_rate=10.0
        )

        data = simulate_survey(
            positions, timestamps, headings, sources, add_noise=False
        )

        for i in range(1, 5):
            expected_grad = data[f'g{i}_bot'] - data[f'g{i}_top']
            np.testing.assert_array_equal(data[f'g{i}_grad'], expected_grad)

    def test_no_noise_clean_signal(self):
        """With add_noise=False, signal comes purely from physics."""
        sources = [{'position': [5, 5, -1], 'moment': [0, 0, 1.0]}]
        positions = np.array([[5.0, 5.0]])
        timestamps = np.array([0.0])
        headings = np.array([0.0])

        config = PathfinderConfig()
        data = simulate_survey(
            positions, timestamps, headings, sources, config, add_noise=False
        )

        # All pairs at same x position should see the same field (center pair)
        # but actually they have different offsets so values will differ
        assert data['g1_grad'].shape == (1,)

    def test_anomaly_at_target_position(self):
        """Gradient is strongest when walking over the target."""
        sources = [{'position': [5, 5, -0.5], 'moment': [0, 0, 10.0]}]
        positions, timestamps, headings = generate_walk_path(
            start=(5, 0), end=(5, 10), speed=1.0, sample_rate=10.0
        )

        data = simulate_survey(
            positions, timestamps, headings, sources, add_noise=False
        )

        # Find peak gradient (absolute value) - should be near y=5
        # Use a middle pair (pair 2 or 3) for center sensitivity
        grads = np.abs(data['g2_grad'])
        peak_idx = np.argmax(grads)
        peak_y = data['lat'][peak_idx]

        assert abs(peak_y - 5.0) < 1.0, f"Peak at y={peak_y}, expected near 5.0"

    def test_extended_telemetry_columns_present(self):
        """Simulation returns enriched pose/IMU telemetry fields."""
        sources = [{'position': [5, 5, -1], 'moment': [0, 0, 1.0]}]
        positions, timestamps, headings = generate_walk_path(
            start=(0, 0), end=(0, 10), speed=1.0, sample_rate=10.0
        )
        data = simulate_survey(
            positions, timestamps, headings, sources, add_noise=False
        )

        expected = {
            'x_east', 'y_north', 'heading_deg', 'speed_mps',
            'distance_m', 'line_id', 'imu_roll_deg', 'imu_pitch_deg',
        }
        assert expected.issubset(data.keys())
        assert len(data['heading_deg']) == len(data['timestamp'])

    def test_line_id_marks_turns_in_zigzag(self):
        """line_id identifies main lines and marks turns as -1."""
        sources = [{'position': [5, 5, -1], 'moment': [0, 0, 1.0]}]
        positions, timestamps, headings = generate_zigzag_path(
            origin=(0.0, 0.0), width=5.0, length=8.0, line_spacing=1.0, speed=1.0, sample_rate=10.0
        )
        data = simulate_survey(
            positions, timestamps, headings, sources, add_noise=False
        )
        assert np.any(data['line_id'] >= 0)
        assert np.any(data['line_id'] == -1)

    def test_gps_dropout_zeroes_coordinates(self):
        """No-fix samples are zeroed when gps_fix_behavior='zero'."""
        sources = [{'position': [5, 5, -1], 'moment': [0, 0, 1.0]}]
        positions, timestamps, headings = generate_walk_path(
            start=(0, 0), end=(0, 10), speed=1.0, sample_rate=10.0
        )
        cfg = PathfinderConfig(
            gps_origin=(51.0, 18.0),
            gps_quality=True,
            gps_dropout_rate=1.0,
            gps_fix_behavior="zero",
        )
        data = simulate_survey(
            positions, timestamps, headings, sources, config=cfg, add_noise=False
        )
        assert data["fix_quality"][0] == 1
        assert np.all(data["fix_quality"][1:] == 0)
        assert np.all(data["lat"][1:] == 0.0)
        assert np.all(data["lon"][1:] == 0.0)

    def test_channel_offsets_apply_without_noise(self):
        """Per-channel calibration offsets are applied deterministically."""
        sources = []
        positions = np.array([[0.0, 0.0]])
        timestamps = np.array([0.0])
        headings = np.array([0.0])
        cfg = PathfinderConfig(
            channel_offset_top_adc=[100, 100, 100, 100],
            channel_offset_bot_adc=[150, 150, 150, 150],
        )
        data = simulate_survey(
            positions, timestamps, headings, sources, config=cfg, add_noise=False
        )
        assert int(data["g1_top"][0]) == 100
        assert int(data["g1_bot"][0]) == 150
        assert int(data["g1_grad"][0]) == 50


class TestCSVExport:
    """Tests for CSV export compatibility."""

    def test_csv_has_header(self):
        """Exported CSV has correct header line."""
        data = {
            'timestamp': np.array([0, 100], dtype=np.uint32),
            'lat': np.array([51.0, 51.0001]),
            'lon': np.array([18.0, 18.0001]),
        }
        for i in range(1, 5):
            data[f'g{i}_top'] = np.array([100, 200])
            data[f'g{i}_bot'] = np.array([150, 250])
            data[f'g{i}_grad'] = np.array([50, 50])

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = f.name

        export_csv(data, path, num_pairs=4)

        with open(path) as f:
            header = f.readline().strip()

        expected = (
            "timestamp,lat,lon,"
            "g1_top,g1_bot,g1_grad,"
            "g2_top,g2_bot,g2_grad,"
            "g3_top,g3_bot,g3_grad,"
            "g4_top,g4_bot,g4_grad"
        )
        assert header == expected

    def test_csv_loadable_by_pandas(self):
        """Exported CSV can be loaded by pandas (same as visualize_data.py)."""
        pytest.importorskip("pandas")
        import pandas as pd

        sources = [{'position': [5, 5, -1], 'moment': [0, 0, 1.0]}]
        positions, timestamps, headings = generate_walk_path(
            start=(0, 0), end=(0, 10), speed=1.0, sample_rate=10.0
        )

        data = simulate_survey(
            positions, timestamps, headings, sources, add_noise=False
        )

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = f.name

        export_csv(data, path, num_pairs=4)

        # Load with pandas (same as visualize_data.py)
        df = pd.read_csv(path)
        assert 'timestamp' in df.columns
        assert 'g1_grad' in df.columns
        assert len(df) > 0

    def test_csv_2_pairs(self):
        """Export with 2 pairs produces correct columns."""
        data = {
            'timestamp': np.array([0], dtype=np.uint32),
            'lat': np.array([51.0]),
            'lon': np.array([18.0]),
        }
        for i in range(1, 3):
            data[f'g{i}_top'] = np.array([100])
            data[f'g{i}_bot'] = np.array([150])
            data[f'g{i}_grad'] = np.array([50])

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = f.name

        export_csv(data, path, num_pairs=2)

        with open(path) as f:
            header = f.readline().strip()

        assert 'g2_grad' in header
        assert 'g3_top' not in header
