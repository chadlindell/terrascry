"""HIRT validation tests — the 4 scenarios from the HIRT field guide.

Each scenario validates a different aspect of the physics engine:
1. Single ferrous target — basic magnetic + EM detection
2. Bomb crater — heterogeneous fill, multiple targets
3. Clandestine burial — ERT-dominant, disturbed soil contrast
4. Swamp crash site — high conductivity background, multi-modal

These tests use only analytical backends (no SimPEG/pyGIMLi required).
"""

from pathlib import Path

import numpy as np
import pytest

from geosim.em.fdem import fdem_forward, secondary_field_conductive_sphere
from geosim.em.skin_depth import skin_depth
from geosim.resistivity.ert import ert_forward
from geosim.resistivity.geometric import geometric_factor_wenner
from geosim.scenarios.loader import load_scenario
from geosim.sensors.pathfinder import run_scenario_survey

SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"


class TestScenario1SingleFerrous:
    """Scenario 1: Single ferrous target — extend for EM."""

    SCENARIO = SCENARIOS_DIR / "single-ferrous-target.json"

    def test_magnetic_detection(self, tmp_path):
        """Pathfinder detects the single target."""
        data = run_scenario_survey(
            self.SCENARIO, tmp_path / "s1.csv",
            walk_type="zigzag", add_noise=False, seed=42,
        )
        max_grad = np.zeros(len(data["timestamp"]))
        for i in range(1, 5):
            max_grad = np.maximum(max_grad, np.abs(data[f"g{i}_grad"]))
        assert np.max(max_grad) > 0, "Should detect magnetic anomaly"

    def test_em_response_at_target(self):
        """EM response is nonzero at the target location."""
        scenario = load_scenario(self.SCENARIO)
        em_sources = scenario.em_sources
        assert len(em_sources) >= 1

        src = em_sources[0]
        resp = secondary_field_conductive_sphere(
            radius=src['radius'],
            conductivity=src['conductivity'],
            frequency=10000.0,
            r_obs=1.5,  # 0.5m above target at -1.0
        )
        assert abs(resp) > 0

    def test_skin_depth_appropriate(self):
        """Skin depth in site soils allows penetration to target depth."""
        scenario = load_scenario(self.SCENARIO)
        # Use first soil layer conductivity
        sigma = scenario.terrain.layers[1].conductivity  # sandy loam, 0.02 S/m
        delta = skin_depth(10000.0, sigma)
        # Skin depth should be > target depth (1m) for detection
        assert delta > 1.0, f"Skin depth {delta:.1f}m too shallow"

    def test_resistivity_model_layers(self):
        """Resistivity model has correct number of layers."""
        scenario = load_scenario(self.SCENARIO)
        model = scenario.resistivity_model
        assert len(model['resistivities']) == 2
        assert len(model['thicknesses']) == 1


class TestScenario2BombCrater:
    """Scenario 2: Bomb crater with heterogeneous fill."""

    SCENARIO = SCENARIOS_DIR / "bomb-crater-heterogeneous.json"

    def test_multiple_magnetic_targets(self, tmp_path):
        """Multiple magnetic targets detected in survey."""
        data = run_scenario_survey(
            self.SCENARIO, tmp_path / "s2.csv",
            walk_type="zigzag", line_spacing=2.0, add_noise=False, seed=42,
        )
        max_grad = np.zeros(len(data["timestamp"]))
        for i in range(1, 5):
            max_grad = np.maximum(max_grad, np.abs(data[f"g{i}_grad"]))

        # Should see multiple peaks
        threshold = np.percentile(max_grad, 90)
        above = max_grad > threshold
        assert np.sum(above) > 10, "Should have significant anomalous zone"

    def test_em_sources_exist(self):
        """Scenario has EM-relevant sources."""
        scenario = load_scenario(self.SCENARIO)
        assert len(scenario.em_sources) >= 3

    def test_uxb_strongest_em_target(self):
        """UXB (largest object) has the strongest EM response."""
        scenario = load_scenario(self.SCENARIO)
        em_sources = scenario.em_sources

        responses = []
        for src in em_sources:
            r = secondary_field_conductive_sphere(
                radius=src['radius'],
                conductivity=src['conductivity'],
                frequency=5000.0,
                r_obs=max(src['radius'] * 2, 1.0),
            )
            responses.append((abs(r), src['name']))

        responses.sort(reverse=True)
        assert "UXB" in responses[0][1] or "bomb" in responses[0][1].lower()


class TestScenario3ClandestineBurial:
    """Scenario 3: Clandestine burial — ERT-dominant."""

    SCENARIO = SCENARIOS_DIR / "clandestine-burial.json"

    def test_clay_soil_higher_conductivity(self):
        """Clay soil has higher conductivity than typical (aids ERT)."""
        scenario = load_scenario(self.SCENARIO)
        clay_layer = [l for l in scenario.terrain.layers if "clay" in l.name.lower()]
        assert len(clay_layer) > 0
        assert clay_layer[0].conductivity > 0.1

    def test_small_ferrous_detected_magnetically(self, tmp_path):
        """Small ferrous objects (belt buckle, etc.) produce magnetic anomaly."""
        data = run_scenario_survey(
            self.SCENARIO, tmp_path / "s3.csv",
            walk_type="zigzag", add_noise=False, seed=42,
        )
        max_grad = np.zeros(len(data["timestamp"]))
        for i in range(1, 5):
            max_grad = np.maximum(max_grad, np.abs(data[f"g{i}_grad"]))
        # Shovel head is shallow and moderately sized — should be detectable
        assert np.max(max_grad) > 0

    def test_ert_forward_runs(self):
        """ERT forward model runs for this site's resistivity structure."""
        scenario = load_scenario(self.SCENARIO)
        model = scenario.resistivity_model

        # Simple Wenner array over the site
        electrodes = np.array([[x, 7.5] for x in np.arange(5, 11)])
        measurements = [(0, 3, 1, 2), (1, 4, 2, 3), (2, 5, 3, 4)]

        result = ert_forward(
            electrode_positions=electrodes,
            measurements=measurements,
            resistivities=model['resistivities'],
            thicknesses=model['thicknesses'],
            backend='analytical',
        )
        assert len(result['apparent_resistivity']) == 3
        assert all(r > 0 for r in result['apparent_resistivity'])


class TestScenario4SwampCrashSite:
    """Scenario 4: Swamp/marsh crash site — new HIRT scenario."""

    SCENARIO = SCENARIOS_DIR / "swamp-crash-site.json"

    def test_scenario_loads(self):
        """New scenario loads correctly with all extensions."""
        scenario = load_scenario(self.SCENARIO)
        assert scenario.name == "Swamp/Marsh Crash Site"
        assert len(scenario.objects) >= 4
        assert len(scenario.anomaly_zones) >= 1
        assert scenario.hirt_config is not None

    def test_hirt_config_present(self):
        """HIRT config has probes and frequencies."""
        scenario = load_scenario(self.SCENARIO)
        hirt = scenario.hirt_config
        assert len(hirt.probes) == 2
        assert len(hirt.frequencies) == 5
        assert hirt.injection_current > 0

    def test_anomaly_zones_present(self):
        """Anomaly zones describe the crash disturbance."""
        scenario = load_scenario(self.SCENARIO)
        assert len(scenario.anomaly_zones) >= 2
        impact = [az for az in scenario.anomaly_zones if "impact" in az.name.lower()]
        assert len(impact) >= 1

    def test_high_background_conductivity(self):
        """Saturated marsh has high background conductivity."""
        scenario = load_scenario(self.SCENARIO)
        # Saturated clay layer should be highly conductive
        clay = [l for l in scenario.terrain.layers if "clay" in l.name.lower()]
        assert len(clay) > 0
        assert clay[0].conductivity >= 0.2

    def test_shallow_skin_depth(self):
        """High conductivity means shallow skin depth at HIRT frequencies."""
        scenario = load_scenario(self.SCENARIO)
        sigma = 0.3  # saturated clay
        for freq in scenario.hirt_config.frequencies:
            delta = skin_depth(freq, sigma)
            # Skin depth should be reasonable (meters, not km)
            assert 0.1 < delta < 100.0

    def test_magnetic_survey_completes(self, tmp_path):
        """Pathfinder magnetic survey runs on this scenario."""
        data = run_scenario_survey(
            self.SCENARIO, tmp_path / "s4.csv",
            walk_type="zigzag", line_spacing=2.0, add_noise=False, seed=42,
        )
        assert len(data["timestamp"]) > 100

    def test_conductivity_model(self):
        """Build conductivity model includes background and anomalies."""
        scenario = load_scenario(self.SCENARIO)
        model = scenario.build_conductivity_model()
        assert len(model['background']) >= 2
        assert len(model['anomalies']) >= 2

    def test_fdem_forward_analytical(self):
        """FDEM analytical forward runs for site model."""
        scenario = load_scenario(self.SCENARIO)
        model = scenario.resistivity_model

        conductivities = [1.0 / r for r in model['resistivities']]
        result = fdem_forward(
            thicknesses=model['thicknesses'],
            conductivities=conductivities,
            frequencies=scenario.hirt_config.frequencies,
            coil_separation=0.3,
            backend='analytical',
        )
        assert len(result['real']) == len(scenario.hirt_config.frequencies)
        assert result['backend'] == 'analytical'


class TestServerNewCommands:
    """Test the 3 new ZMQ server commands via handle_request."""

    def _make_server(self):
        from geosim.server import PhysicsServer
        server = PhysicsServer()
        server.load_scenario(str(SCENARIOS_DIR / "single-ferrous-target.json"))
        return server

    def test_query_skin_depth(self):
        """query_skin_depth returns valid result without scenario."""
        from geosim.server import PhysicsServer
        server = PhysicsServer()
        result = server.handle_request({
            "command": "query_skin_depth",
            "params": {"frequency": 1000.0, "conductivity": 0.01},
        })
        assert result["status"] == "ok"
        delta = result["data"]["skin_depth"]
        assert 10.0 < delta < 1000.0
        assert result["data"]["unit"] == "meters"

    def test_query_em_response(self):
        """query_em_response returns in-phase and quadrature."""
        server = self._make_server()
        result = server.handle_request({
            "command": "query_em_response",
            "params": {
                "positions": [[10.0, 10.0, 0.3]],
                "frequency": 10000.0,
            },
        })
        assert result["status"] == "ok"
        assert len(result["data"]["response_real"]) == 1
        assert len(result["data"]["response_imag"]) == 1

    def test_query_em_response_no_scenario(self):
        """query_em_response errors without loaded scenario."""
        from geosim.server import PhysicsServer
        server = PhysicsServer()
        result = server.handle_request({
            "command": "query_em_response",
            "params": {"positions": [[0, 0, 0]], "frequency": 1000.0},
        })
        assert result["status"] == "error"

    def test_query_apparent_resistivity(self):
        """query_apparent_resistivity returns valid results."""
        server = self._make_server()
        result = server.handle_request({
            "command": "query_apparent_resistivity",
            "params": {
                "electrode_positions": [[0, 0], [1, 0], [2, 0], [3, 0]],
                "measurements": [[0, 3, 1, 2]],
            },
        })
        assert result["status"] == "ok"
        assert len(result["data"]["apparent_resistivity"]) == 1
        assert result["data"]["apparent_resistivity"][0] > 0
        assert len(result["data"]["geometric_factors"]) == 1

    def test_query_apparent_resistivity_no_scenario(self):
        """query_apparent_resistivity errors without scenario."""
        from geosim.server import PhysicsServer
        server = PhysicsServer()
        result = server.handle_request({
            "command": "query_apparent_resistivity",
            "params": {
                "electrode_positions": [[0, 0], [1, 0], [2, 0], [3, 0]],
                "measurements": [[0, 3, 1, 2]],
            },
        })
        assert result["status"] == "error"
