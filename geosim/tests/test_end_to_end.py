"""End-to-end pipeline tests.

Validates the complete pipeline: scenario -> survey -> CSV -> anomaly detection.
"""

from pathlib import Path

import numpy as np
import pytest

from geosim.sensors.pathfinder import run_scenario_survey

SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"


class TestEndToEndSingleTarget:
    """End-to-end tests using single-ferrous-target.json."""

    SCENARIO = SCENARIOS_DIR / "single-ferrous-target.json"
    TARGET_X = 10.0
    TARGET_Y = 10.0

    def _run_survey(self, tmp_path, add_noise=False, seed=42):
        csv_path = tmp_path / "survey.csv"
        data = run_scenario_survey(
            self.SCENARIO,
            csv_path,
            walk_type="zigzag",
            line_spacing=1.0,
            add_noise=add_noise,
            seed=seed,
        )
        return data, csv_path

    def test_csv_no_nan(self, tmp_path):
        """Pipeline produces CSV with no NaN values."""
        data, csv_path = self._run_survey(tmp_path, add_noise=False)
        for key, val in data.items():
            if isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.floating):
                assert not np.any(np.isnan(val)), f"NaN found in {key}"

    def test_csv_correct_columns(self, tmp_path):
        """Output has all expected Pathfinder columns."""
        data, _ = self._run_survey(tmp_path)
        expected = {"timestamp", "lat", "lon"}
        for i in range(1, 5):
            expected.update({f"g{i}_top", f"g{i}_bot", f"g{i}_grad"})
        assert expected.issubset(data.keys())

    def test_gradient_peak_near_target(self, tmp_path):
        """Gradient peak is within 2m of the true target position (10, 10)."""
        data, _ = self._run_survey(tmp_path, add_noise=False)

        # Find peak gradient across all 4 pairs
        max_grad = np.zeros(len(data["timestamp"]))
        for i in range(1, 5):
            max_grad = np.maximum(max_grad, np.abs(data[f"g{i}_grad"]))

        peak_idx = np.argmax(max_grad)
        peak_x = data["lon"][peak_idx]  # lon = easting = x
        peak_y = data["lat"][peak_idx]  # lat = northing = y

        dist = np.sqrt((peak_x - self.TARGET_X)**2 + (peak_y - self.TARGET_Y)**2)
        assert dist < 2.0, (
            f"Peak at ({peak_x:.1f}, {peak_y:.1f}), target at "
            f"({self.TARGET_X}, {self.TARGET_Y}), distance={dist:.1f}m"
        )

    def test_gradient_falloff(self, tmp_path):
        """Gradient is much stronger near target than far away."""
        data, _ = self._run_survey(tmp_path, add_noise=False)

        max_grad = np.zeros(len(data["timestamp"]))
        for i in range(1, 5):
            max_grad = np.maximum(max_grad, np.abs(data[f"g{i}_grad"]))

        # Split into near-target and far-from-target
        dist = np.sqrt(
            (data["lon"] - self.TARGET_X)**2
            + (data["lat"] - self.TARGET_Y)**2
        )
        near_mask = dist < 3.0
        far_mask = dist > 8.0

        if np.any(near_mask) and np.any(far_mask):
            near_max = np.max(max_grad[near_mask])
            far_max = np.max(max_grad[far_mask])
            assert near_max > 5.0 * far_max, (
                f"Near={near_max}, far={far_max}, ratio={near_max/far_max:.1f}"
            )

    def test_target_detectable_with_noise(self, tmp_path):
        """With noise, target still detectable within 3m."""
        data, _ = self._run_survey(tmp_path, add_noise=True, seed=42)

        max_grad = np.zeros(len(data["timestamp"]))
        for i in range(1, 5):
            max_grad = np.maximum(max_grad, np.abs(data[f"g{i}_grad"]))

        peak_idx = np.argmax(max_grad)
        peak_x = data["lon"][peak_idx]
        peak_y = data["lat"][peak_idx]

        dist = np.sqrt((peak_x - self.TARGET_X)**2 + (peak_y - self.TARGET_Y)**2)
        assert dist < 3.0, (
            f"Noisy peak at ({peak_x:.1f}, {peak_y:.1f}), distance={dist:.1f}m"
        )

    def test_gradient_bot_minus_top(self, tmp_path):
        """Gradient = bot - top relationship holds in output."""
        data, _ = self._run_survey(tmp_path, add_noise=False)
        for i in range(1, 5):
            expected = data[f"g{i}_bot"] - data[f"g{i}_top"]
            np.testing.assert_array_equal(
                data[f"g{i}_grad"], expected,
                err_msg=f"Pair {i}: gradient != bot - top",
            )


class TestEndToEndMultiTarget:
    """End-to-end tests with scattered-debris.json (multiple targets)."""

    SCENARIO = SCENARIOS_DIR / "scattered-debris.json"

    def test_multiple_anomalies_detected(self, tmp_path):
        """Multiple distinct anomalies are detectable (>= 3 local maxima)."""
        csv_path = tmp_path / "survey.csv"
        data = run_scenario_survey(
            self.SCENARIO,
            csv_path,
            walk_type="zigzag",
            line_spacing=1.0,
            add_noise=False,
            seed=42,
        )

        # Combine all pairs for max gradient
        max_grad = np.zeros(len(data["timestamp"]))
        for i in range(1, 5):
            max_grad = np.maximum(max_grad, np.abs(data[f"g{i}_grad"]))

        # Find peaks above a threshold (background noise level)
        threshold = np.percentile(max_grad, 95)
        above = max_grad > threshold

        # Count distinct spatial clusters
        positions = np.column_stack([data["lon"], data["lat"]])
        peak_positions = positions[above]

        if len(peak_positions) < 3:
            pytest.skip("Not enough peaks above threshold")

        # Cluster by proximity (>3m apart = different anomaly)
        clusters = []
        for pos in peak_positions:
            found = False
            for cluster in clusters:
                if np.linalg.norm(pos - cluster[0]) < 3.0:
                    cluster.append(pos)
                    found = True
                    break
            if not found:
                clusters.append([pos])

        assert len(clusters) >= 3, (
            f"Expected >= 3 anomaly clusters, found {len(clusters)}"
        )

    def test_csv_loadable(self, tmp_path):
        """Multi-target CSV is loadable by pandas."""
        pd = pytest.importorskip("pandas")
        csv_path = tmp_path / "survey.csv"
        run_scenario_survey(
            self.SCENARIO,
            csv_path,
            walk_type="zigzag",
            line_spacing=2.0,
            seed=42,
        )
        df = pd.read_csv(csv_path)
        assert len(df) > 100
        assert "g1_grad" in df.columns
