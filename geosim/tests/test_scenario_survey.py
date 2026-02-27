"""Tests for scenario-based survey simulation.

Validates run_scenario_survey() and all scenario JSON files.
"""

from pathlib import Path

import numpy as np
import pytest

from geosim.scenarios.loader import load_scenario
from geosim.sensors.pathfinder import run_scenario_survey

SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"

SCENARIO_FILES = [
    "single-ferrous-target.json",
    "scattered-debris.json",
    "clandestine-burial.json",
    "bomb-crater-heterogeneous.json",
    "swamp-crash-site.json",
]


class TestRunScenarioSurvey:
    """Tests for the high-level run_scenario_survey() function."""

    def test_zigzag_produces_valid_csv(self, tmp_path):
        """Zigzag survey produces a CSV file with data."""
        csv_path = tmp_path / "survey.csv"
        data = run_scenario_survey(
            SCENARIOS_DIR / "single-ferrous-target.json",
            csv_path,
            walk_type="zigzag",
            seed=42,
        )
        assert csv_path.exists()
        assert csv_path.stat().st_size > 0
        assert "timestamp" in data
        assert "g1_grad" in data
        assert len(data["timestamp"]) > 0

    def test_straight_produces_valid_csv(self, tmp_path):
        """Straight survey produces a CSV file with data."""
        csv_path = tmp_path / "survey.csv"
        data = run_scenario_survey(
            SCENARIOS_DIR / "single-ferrous-target.json",
            csv_path,
            walk_type="straight",
            seed=42,
        )
        assert csv_path.exists()
        assert "timestamp" in data
        assert len(data["timestamp"]) > 0

    def test_return_dict_has_expected_keys(self, tmp_path):
        """Returned dict has all expected Pathfinder column keys."""
        csv_path = tmp_path / "survey.csv"
        data = run_scenario_survey(
            SCENARIOS_DIR / "single-ferrous-target.json",
            csv_path,
            seed=42,
        )
        expected_keys = {"timestamp", "lat", "lon"}
        for i in range(1, 5):
            expected_keys.update({f"g{i}_top", f"g{i}_bot", f"g{i}_grad"})
        assert expected_keys.issubset(data.keys())

    def test_csv_loadable_by_pandas(self, tmp_path):
        """Output CSV is loadable by pandas."""
        pd = pytest.importorskip("pandas")
        csv_path = tmp_path / "survey.csv"
        run_scenario_survey(
            SCENARIOS_DIR / "single-ferrous-target.json",
            csv_path,
            seed=42,
        )
        df = pd.read_csv(csv_path)
        assert len(df) > 0
        assert "g1_grad" in df.columns

    def test_seed_reproducibility(self, tmp_path):
        """Same seed produces identical results."""
        csv1 = tmp_path / "s1.csv"
        csv2 = tmp_path / "s2.csv"
        data1 = run_scenario_survey(
            SCENARIOS_DIR / "single-ferrous-target.json",
            csv1,
            seed=99,
        )
        data2 = run_scenario_survey(
            SCENARIOS_DIR / "single-ferrous-target.json",
            csv2,
            seed=99,
        )
        np.testing.assert_array_equal(data1["g1_grad"], data2["g1_grad"])

    def test_different_seeds_differ(self, tmp_path):
        """Different seeds produce different results (with noise)."""
        csv1 = tmp_path / "s1.csv"
        csv2 = tmp_path / "s2.csv"
        data1 = run_scenario_survey(
            SCENARIOS_DIR / "single-ferrous-target.json",
            csv1,
            seed=1,
        )
        data2 = run_scenario_survey(
            SCENARIOS_DIR / "single-ferrous-target.json",
            csv2,
            seed=2,
        )
        assert not np.array_equal(data1["g1_grad"], data2["g1_grad"])

    def test_no_noise_mode(self, tmp_path):
        """add_noise=False produces clean physics-only signal."""
        csv1 = tmp_path / "noisy.csv"
        csv2 = tmp_path / "clean.csv"
        data_noisy = run_scenario_survey(
            SCENARIOS_DIR / "single-ferrous-target.json",
            csv1,
            add_noise=True,
            seed=42,
        )
        data_clean = run_scenario_survey(
            SCENARIOS_DIR / "single-ferrous-target.json",
            csv2,
            add_noise=False,
            seed=42,
        )
        # Clean signal should be deterministic regardless of seed
        csv3 = tmp_path / "clean2.csv"
        data_clean2 = run_scenario_survey(
            SCENARIOS_DIR / "single-ferrous-target.json",
            csv3,
            add_noise=False,
            seed=999,
        )
        np.testing.assert_array_equal(data_clean["g1_grad"], data_clean2["g1_grad"])

    def test_zigzag_has_more_samples_than_straight(self, tmp_path):
        """Zigzag survey covers more ground → more samples."""
        csv_zz = tmp_path / "zz.csv"
        csv_st = tmp_path / "st.csv"
        data_zz = run_scenario_survey(
            SCENARIOS_DIR / "single-ferrous-target.json",
            csv_zz,
            walk_type="zigzag",
            seed=42,
        )
        data_st = run_scenario_survey(
            SCENARIOS_DIR / "single-ferrous-target.json",
            csv_st,
            walk_type="straight",
            seed=42,
        )
        assert len(data_zz["timestamp"]) > len(data_st["timestamp"])

    def test_extended_artifacts_written(self, tmp_path):
        """Telemetry CSV and survey summary JSON are exported by default."""
        csv_path = tmp_path / "survey.csv"
        data = run_scenario_survey(
            SCENARIOS_DIR / "single-ferrous-target.json",
            csv_path,
            seed=42,
        )
        assert "telemetry_csv" in data
        assert "summary_json" in data
        assert "labels_csv" in data
        assert "session_json" in data
        assert Path(data["telemetry_csv"]).exists()
        assert Path(data["summary_json"]).exists()
        assert Path(data["labels_csv"]).exists()
        assert Path(data["session_json"]).exists()
        assert "anomaly_candidates" in data


class TestScenarioSpecificSurveys:
    """Parametrized tests over all 4 scenario files."""

    @pytest.fixture(params=SCENARIO_FILES)
    def scenario_path(self, request):
        return SCENARIOS_DIR / request.param

    def test_scenario_loads(self, scenario_path):
        """Scenario JSON loads without error."""
        scenario = load_scenario(scenario_path)
        assert scenario.name

    def test_scenario_has_objects(self, scenario_path):
        """Each scenario has at least one buried object."""
        scenario = load_scenario(scenario_path)
        assert len(scenario.objects) > 0

    def test_scenario_has_magnetic_sources(self, scenario_path):
        """Each scenario has at least one magnetic source."""
        scenario = load_scenario(scenario_path)
        sources = scenario.magnetic_sources
        assert len(sources) > 0

    def test_full_survey_completes(self, scenario_path, tmp_path):
        """Full survey pipeline completes for each scenario."""
        csv_path = tmp_path / "survey.csv"
        data = run_scenario_survey(
            scenario_path,
            csv_path,
            walk_type="zigzag",
            line_spacing=2.0,  # coarser for speed
            seed=42,
        )
        assert len(data["timestamp"]) > 0

    def test_output_has_rows(self, scenario_path, tmp_path):
        """Output CSV has at least one data row."""
        csv_path = tmp_path / "survey.csv"
        data = run_scenario_survey(
            scenario_path,
            csv_path,
            walk_type="straight",
            seed=42,
        )
        assert len(data["timestamp"]) >= 2

    def test_object_positions_within_terrain(self, scenario_path):
        """All object positions are within the terrain extents (x, y)."""
        scenario = load_scenario(scenario_path)
        x_min, x_max = scenario.terrain.x_extent
        y_min, y_max = scenario.terrain.y_extent
        for obj in scenario.objects:
            assert x_min <= obj.position[0] <= x_max, (
                f"{obj.name}: x={obj.position[0]} outside [{x_min}, {x_max}]"
            )
            assert y_min <= obj.position[1] <= y_max, (
                f"{obj.name}: y={obj.position[1]} outside [{y_min}, {y_max}]"
            )

    def test_scenario_has_hirt_config(self, scenario_path):
        """All scenarios have a HIRT configuration."""
        scenario = load_scenario(scenario_path)
        assert scenario.hirt_config is not None, (
            f"{scenario.name}: missing hirt_config"
        )

    def test_hirt_has_at_least_two_probes(self, scenario_path):
        """HIRT config has at least 2 probes for crosshole measurement."""
        scenario = load_scenario(scenario_path)
        assert scenario.hirt_config is not None
        assert len(scenario.hirt_config.probes) >= 2, (
            f"{scenario.name}: only {len(scenario.hirt_config.probes)} probes"
        )

    def test_hirt_probe_positions_within_terrain(self, scenario_path):
        """HIRT probe positions are within terrain bounds."""
        scenario = load_scenario(scenario_path)
        assert scenario.hirt_config is not None
        x_min, x_max = scenario.terrain.x_extent
        y_min, y_max = scenario.terrain.y_extent
        for i, probe in enumerate(scenario.hirt_config.probes):
            px, py = probe.position[0], probe.position[1]
            assert x_min <= px <= x_max, (
                f"{scenario.name} probe {i}: x={px} outside [{x_min}, {x_max}]"
            )
            assert y_min <= py <= y_max, (
                f"{scenario.name} probe {i}: y={py} outside [{y_min}, {y_max}]"
            )

    def test_hirt_ring_depths_within_probe_length(self, scenario_path):
        """Ring electrode depths don't exceed probe length."""
        scenario = load_scenario(scenario_path)
        assert scenario.hirt_config is not None
        for i, probe in enumerate(scenario.hirt_config.probes):
            for depth in probe.ring_depths:
                assert depth <= probe.length, (
                    f"{scenario.name} probe {i}: ring depth {depth} > "
                    f"probe length {probe.length}"
                )

    def test_hirt_coil_depths_within_probe_length(self, scenario_path):
        """EM coil depths don't exceed probe length."""
        scenario = load_scenario(scenario_path)
        assert scenario.hirt_config is not None
        for i, probe in enumerate(scenario.hirt_config.probes):
            for depth in probe.coil_depths:
                assert depth <= probe.length, (
                    f"{scenario.name} probe {i}: coil depth {depth} > "
                    f"probe length {probe.length}"
                )

    def test_hirt_has_frequencies(self, scenario_path):
        """HIRT config has at least one operating frequency."""
        scenario = load_scenario(scenario_path)
        assert scenario.hirt_config is not None
        assert len(scenario.hirt_config.frequencies) >= 1, (
            f"{scenario.name}: no HIRT frequencies defined"
        )

    def test_scenario_has_anomaly_zones(self, scenario_path):
        """All scenarios have at least one anomaly zone."""
        scenario = load_scenario(scenario_path)
        assert len(scenario.anomaly_zones) >= 1, (
            f"{scenario.name}: no anomaly zones defined"
        )
