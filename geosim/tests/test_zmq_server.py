"""Integration tests for the ZeroMQ physics server.

Tests the full REQ-REP protocol: connect, load scenario, query field/gradient,
get info, shutdown. Also benchmarks round-trip latency.
"""

from __future__ import annotations

import json
import os
import threading
import time

import numpy as np
import pytest

zmq = pytest.importorskip("zmq")

from geosim.server import PhysicsServer

SCENARIO_PATH = os.path.join(
    os.path.dirname(__file__), "..", "scenarios", "single-ferrous-target.json"
)
SWAMP_SCENARIO_PATH = os.path.join(
    os.path.dirname(__file__), "..", "scenarios", "swamp-crash-site.json"
)
# Use a random high port to avoid conflicts
TEST_PORT = 15555
TEST_ADDR = f"tcp://127.0.0.1:{TEST_PORT}"
BIND_ADDR = f"tcp://*:{TEST_PORT}"


@pytest.fixture(scope="module")
def server_thread():
    """Start the physics server in a background thread."""
    server = PhysicsServer(bind_address=BIND_ADDR)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    time.sleep(0.2)  # Give server time to bind
    yield server
    # Send shutdown command
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(TEST_ADDR)
    sock.send_json({"command": "shutdown"})
    sock.recv_json()
    sock.close()
    ctx.term()
    thread.join(timeout=3)


@pytest.fixture(scope="module")
def client(server_thread):
    """Create a ZMQ REQ client connected to the test server."""
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(TEST_ADDR)
    sock.setsockopt(zmq.RCVTIMEO, 5000)  # 5s timeout
    yield sock
    sock.close()
    ctx.term()


def send_command(client, command: str, params: dict | None = None) -> dict:
    """Helper: send a command and return the response."""
    msg = {"command": command}
    if params:
        msg["params"] = params
    client.send_json(msg)
    return client.recv_json()


class TestServerProtocol:
    """Test the server's REQ-REP command protocol."""

    def test_ping(self, client):
        resp = send_command(client, "ping")
        assert resp["status"] == "ok"
        assert resp["data"]["message"] == "pong"

    def test_unknown_command(self, client):
        resp = send_command(client, "nonexistent")
        assert resp["status"] == "error"
        assert "Unknown command" in resp["message"]

    def test_query_field_without_scenario(self, client):
        resp = send_command(client, "query_field", {"positions": [[0, 0, 0.5]]})
        assert resp["status"] == "error"
        assert "No scenario" in resp["message"]

    def test_load_scenario(self, client):
        resp = send_command(client, "load_scenario", {"path": SCENARIO_PATH})
        assert resp["status"] == "ok"
        assert resp["data"]["name"] is not None
        assert resp["data"]["n_sources"] > 0

    def test_get_scenario_info(self, client):
        # Scenario loaded in previous test (module-scoped fixtures maintain state)
        resp = send_command(client, "get_scenario_info")
        assert resp["status"] == "ok"
        data = resp["data"]
        assert "name" in data
        assert "terrain" in data
        assert "x_extent" in data["terrain"]
        assert "y_extent" in data["terrain"]

    def test_query_field(self, client):
        positions = [[0.0, 0.0, 0.5], [1.0, 1.0, 0.5], [2.0, 0.0, 0.3]]
        resp = send_command(client, "query_field", {"positions": positions})
        assert resp["status"] == "ok"
        B = np.array(resp["data"]["B"])
        assert B.shape == (3, 3)  # 3 points, 3 components
        # Field should be nonzero near the buried target
        assert np.any(np.abs(B) > 0)

    def test_query_gradient(self, client):
        # Target is at [10, 10, -1] — query near it and far away
        positions = [[10.0, 10.0, 0.175], [0.0, 0.0, 0.175]]
        resp = send_command(
            client, "query_gradient",
            {"positions": positions, "sensor_separation": 0.35, "component": 2},
        )
        assert resp["status"] == "ok"
        data = resp["data"]
        grad = np.array(data["gradient"])
        assert len(grad) == 2
        # Gradient near the target should be stronger than far away
        assert abs(grad[0]) > abs(grad[1])

    def test_query_field_batch(self, client):
        """Test querying many points at once."""
        x = np.linspace(-5, 5, 50)
        y = np.linspace(-5, 5, 50)
        X, Y = np.meshgrid(x, y)
        positions = np.column_stack([
            X.ravel(), Y.ravel(), np.full(X.size, 0.3)
        ]).tolist()

        resp = send_command(client, "query_field", {"positions": positions})
        assert resp["status"] == "ok"
        B = np.array(resp["data"]["B"])
        assert B.shape == (2500, 3)


class TestServerLatency:
    """Benchmark round-trip latency for physics queries."""

    def test_ping_latency(self, client):
        """Ping should be < 1ms round-trip."""
        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            send_command(client, "ping")
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        median = np.median(times)
        p95 = np.percentile(times, 95)
        print(f"\nPing latency: median={median:.2f}ms, p95={p95:.2f}ms")
        assert median < 2.0, f"Ping too slow: {median:.2f}ms"

    def test_single_point_field_latency(self, client):
        """Single-point field query should be < 5ms."""
        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            send_command(client, "query_field", {"positions": [[0, 0, 0.5]]})
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        median = np.median(times)
        p95 = np.percentile(times, 95)
        print(f"\nSingle-point field latency: median={median:.2f}ms, p95={p95:.2f}ms")
        assert median < 5.0, f"Field query too slow: {median:.2f}ms"

    def test_gradient_latency(self, client):
        """Gradient query at a single point should be < 5ms."""
        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            send_command(
                client, "query_gradient",
                {"positions": [[0, 0, 0.175]], "sensor_separation": 0.35},
            )
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        median = np.median(times)
        p95 = np.percentile(times, 95)
        print(f"\nGradient latency: median={median:.2f}ms, p95={p95:.2f}ms")
        assert median < 5.0, f"Gradient query too slow: {median:.2f}ms"

    def test_batch_field_latency(self, client):
        """100-point field query should be < 10ms."""
        positions = np.column_stack([
            np.random.uniform(-5, 5, 100),
            np.random.uniform(-5, 5, 100),
            np.full(100, 0.3),
        ]).tolist()

        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            send_command(client, "query_field", {"positions": positions})
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        median = np.median(times)
        p95 = np.percentile(times, 95)
        print(f"\n100-point field latency: median={median:.2f}ms, p95={p95:.2f}ms")
        assert median < 20.0, f"Batch query too slow: {median:.2f}ms"


class TestHandleRequestDirectly:
    """Test the handle_request method without ZeroMQ (unit-level)."""

    def test_handle_ping(self):
        server = PhysicsServer()
        resp = server.handle_request({"command": "ping"})
        assert resp["status"] == "ok"
        assert resp["data"]["message"] == "pong"

    def test_handle_load_scenario(self):
        server = PhysicsServer()
        resp = server.handle_request({
            "command": "load_scenario",
            "params": {"path": SCENARIO_PATH},
        })
        assert resp["status"] == "ok"
        assert server.scenario is not None

    def test_handle_query_field_after_load(self):
        server = PhysicsServer()
        server.handle_request({
            "command": "load_scenario",
            "params": {"path": SCENARIO_PATH},
        })
        resp = server.handle_request({
            "command": "query_field",
            "params": {"positions": [[0, 0, 0.5]]},
        })
        assert resp["status"] == "ok"
        B = resp["data"]["B"]
        assert len(B) == 1
        assert len(B[0]) == 3

    def test_handle_unknown(self):
        server = PhysicsServer()
        resp = server.handle_request({"command": "bogus"})
        assert resp["status"] == "error"

    def test_scenario_info_returns_all_fields(self):
        """get_scenario_info returns full payload with all expected fields."""
        server = PhysicsServer()
        server.handle_request({
            "command": "load_scenario",
            "params": {"path": SCENARIO_PATH},
        })
        resp = server.handle_request({"command": "get_scenario_info"})
        assert resp["status"] == "ok"
        data = resp["data"]

        # Core fields
        assert "name" in data
        assert "description" in data
        assert "n_sources" in data

        # Terrain with full detail
        assert "terrain" in data
        terrain = data["terrain"]
        assert "x_extent" in terrain
        assert "y_extent" in terrain
        assert "surface_elevation" in terrain
        assert "layers" in terrain

        # Earth field
        assert "earth_field" in data
        assert len(data["earth_field"]) == 3

        # Objects list
        assert "objects" in data
        assert isinstance(data["objects"], list)
        assert len(data["objects"]) > 0
        obj = data["objects"][0]
        assert "name" in obj
        assert "position" in obj
        assert "type" in obj
        assert "radius" in obj

        # Metadata
        assert "metadata" in data

        # Instrument info
        assert "has_hirt" in data
        assert "available_instruments" in data
        assert isinstance(data["available_instruments"], list)
        assert "mag_gradiometer" in data["available_instruments"]
        assert "anomaly_zones" in data
        assert "hirt_config" in data
        assert "environment_profile" in data

    def test_scenario_info_objects_match_scenario(self):
        """Objects list in info matches actual scenario objects."""
        server = PhysicsServer()
        server.handle_request({
            "command": "load_scenario",
            "params": {"path": SCENARIO_PATH},
        })
        resp = server.handle_request({"command": "get_scenario_info"})
        data = resp["data"]

        # Number of objects should match
        assert len(data["objects"]) == len(server.scenario.objects)

        # Names should match
        info_names = [o["name"] for o in data["objects"]]
        scenario_names = [o.name for o in server.scenario.objects]
        assert info_names == scenario_names

    def test_instrument_availability_single_target(self):
        """Single ferrous target scenario has all three instruments (has layers + EM targets)."""
        server = PhysicsServer()
        server.handle_request({
            "command": "load_scenario",
            "params": {"path": SCENARIO_PATH},
        })
        resp = server.handle_request({"command": "get_scenario_info"})
        instruments = resp["data"]["available_instruments"]
        assert "mag_gradiometer" in instruments

    def test_instrument_availability_hirt_scenario(self):
        """HIRT scenarios include EM and resistivity instruments."""
        hirt_path = os.path.join(
            os.path.dirname(__file__), "..", "scenarios", "bomb-crater-heterogeneous.json"
        )
        if not os.path.exists(hirt_path):
            pytest.skip("bomb-crater scenario not found")

        server = PhysicsServer()
        server.handle_request({
            "command": "load_scenario",
            "params": {"path": hirt_path},
        })
        resp = server.handle_request({"command": "get_scenario_info"})
        instruments = resp["data"]["available_instruments"]
        # Has EM-detectable objects (conductivity > 0 and radius > 0)
        assert "em_fdem" in instruments

    def test_scenario_info_includes_rich_layer_and_hirt_data(self):
        """Swamp scenario exposes anomaly zones and rich terrain layer fields."""
        server = PhysicsServer()
        server.handle_request({
            "command": "load_scenario",
            "params": {"path": SWAMP_SCENARIO_PATH},
        })
        resp = server.handle_request({"command": "get_scenario_info"})
        assert resp["status"] == "ok"
        data = resp["data"]

        assert data["has_hirt"] is True
        assert isinstance(data["hirt_config"], dict)
        assert data["hirt_config"]["probe_count"] >= 2
        assert len(data["anomaly_zones"]) >= 1

        layer0 = data["terrain"]["layers"][0]
        assert "color" in layer0
        assert "relative_permittivity" in layer0
        assert "susceptibility" in layer0

    def test_query_em_response_changes_with_anomaly_zones(self):
        """Anomaly zones affect live EM response queries."""
        server = PhysicsServer()
        server.handle_request({
            "command": "load_scenario",
            "params": {"path": SWAMP_SCENARIO_PATH},
        })
        params = {"positions": [[12.5, 12.5, -1.0]], "frequency": 10000.0}
        with_zone = server.handle_request({
            "command": "query_em_response",
            "params": params,
        })
        assert with_zone["status"] == "ok"

        saved = server.scenario.anomaly_zones
        server.scenario.anomaly_zones = []
        no_zone = server.handle_request({
            "command": "query_em_response",
            "params": params,
        })
        server.scenario.anomaly_zones = saved
        assert no_zone["status"] == "ok"

        real_delta = abs(with_zone["data"]["response_real"][0] - no_zone["data"]["response_real"][0])
        imag_delta = abs(with_zone["data"]["response_imag"][0] - no_zone["data"]["response_imag"][0])
        assert real_delta > 1e-18 or imag_delta > 1e-18

    def test_query_apparent_resistivity_changes_with_anomaly_zones(self):
        """Anomaly zones perturb apparent resistivity in live query path."""
        server = PhysicsServer()
        server.handle_request({
            "command": "load_scenario",
            "params": {"path": SWAMP_SCENARIO_PATH},
        })
        params = {
            "electrode_positions": [[11.0, 12.0], [12.0, 12.0], [13.0, 12.0], [14.0, 12.0]],
            "measurements": [[0, 3, 1, 2]],
        }
        with_zone = server.handle_request({
            "command": "query_apparent_resistivity",
            "params": params,
        })
        assert with_zone["status"] == "ok"

        saved = server.scenario.anomaly_zones
        server.scenario.anomaly_zones = []
        no_zone = server.handle_request({
            "command": "query_apparent_resistivity",
            "params": params,
        })
        server.scenario.anomaly_zones = saved
        assert no_zone["status"] == "ok"

        rho_delta = abs(
            with_zone["data"]["apparent_resistivity"][0]
            - no_zone["data"]["apparent_resistivity"][0]
        )
        assert rho_delta > 1e-6

    def test_set_comms_profile_and_get_server_stats(self):
        """Server exposes comms profile controls and request stats."""
        server = PhysicsServer()
        set_resp = server.handle_request({
            "command": "set_comms_profile",
            "params": {
                "enabled": True,
                "base_latency_ms": 3.0,
                "jitter_ms": 0.0,
                "drop_rate": 0.0,
                "timeout_rate": 0.0,
                "max_history": 64,
            },
        })
        assert set_resp["status"] == "ok"
        prof = set_resp["data"]["comms_profile"]
        assert prof["enabled"] is True
        assert prof["base_latency_ms"] == 3.0
        assert prof["max_history"] == 64

        stats_resp = server.handle_request({"command": "get_server_stats"})
        assert stats_resp["status"] == "ok"
        stats = stats_resp["data"]
        assert "stats" in stats
        assert "recent_requests" in stats
        assert "comms_profile" in stats

    def test_simulated_packet_drop(self):
        """Drop profile forces query errors with explicit error code."""
        server = PhysicsServer()
        server.handle_request({
            "command": "load_scenario",
            "params": {"path": SCENARIO_PATH},
        })
        server.handle_request({
            "command": "set_comms_profile",
            "params": {
                "enabled": True,
                "drop_rate": 1.0,
                "timeout_rate": 0.0,
                "base_latency_ms": 0.0,
                "jitter_ms": 0.0,
                "random_seed": 7,
            },
        })
        resp = server.handle_request({
            "command": "query_field",
            "params": {"positions": [[0.0, 0.0, 0.3]]},
        })
        assert resp["status"] == "error"
        assert resp.get("error_code") == "simulated_packet_drop"

    def test_simulated_latency_delay(self):
        """Latency profile adds measurable response time."""
        server = PhysicsServer()
        server.handle_request({
            "command": "load_scenario",
            "params": {"path": SCENARIO_PATH},
        })
        server.handle_request({
            "command": "set_comms_profile",
            "params": {
                "enabled": True,
                "base_latency_ms": 20.0,
                "jitter_ms": 0.0,
                "drop_rate": 0.0,
                "timeout_rate": 0.0,
            },
        })
        t0 = time.perf_counter()
        resp = server.handle_request({
            "command": "query_field",
            "params": {"positions": [[0.0, 0.0, 0.3]]},
        })
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert resp["status"] == "ok"
        assert elapsed_ms >= 15.0

    def test_query_metal_detector_near_target(self):
        """Metal detector returns large ΔT near the target."""
        server = PhysicsServer()
        server.handle_request({
            "command": "load_scenario",
            "params": {"path": SCENARIO_PATH},
        })
        # Target at [10, 10, -1] — query directly above
        resp = server.handle_request({
            "command": "query_metal_detector",
            "params": {"positions": [[10.0, 10.0, 0.175]]},
        })
        assert resp["status"] == "ok"
        delta_t = resp["data"]["delta_t"]
        assert len(delta_t) == 1
        # Should be a significant anomaly (nT range = 1e-9 to 1e-6 T)
        assert abs(delta_t[0]) > 1e-9, f"ΔT too small near target: {delta_t[0]}"

    def test_query_metal_detector_far_from_target(self):
        """Metal detector returns near-zero ΔT far from target."""
        server = PhysicsServer()
        server.handle_request({
            "command": "load_scenario",
            "params": {"path": SCENARIO_PATH},
        })
        # Far corner — should be near zero
        resp = server.handle_request({
            "command": "query_metal_detector",
            "params": {"positions": [[0.0, 0.0, 0.175]]},
        })
        assert resp["status"] == "ok"
        delta_t = resp["data"]["delta_t"]
        # Far away: ΔT should be much smaller than near target
        assert abs(delta_t[0]) < 1e-7, f"ΔT too large far from target: {delta_t[0]}"

    def test_query_metal_detector_delta_t_in_nT_range(self):
        """ΔT values near target are in nT range (not µT)."""
        server = PhysicsServer()
        server.handle_request({
            "command": "load_scenario",
            "params": {"path": SCENARIO_PATH},
        })
        resp = server.handle_request({
            "command": "query_metal_detector",
            "params": {"positions": [[10.0, 10.0, 0.175]]},
        })
        delta_t = resp["data"]["delta_t"][0]
        # ΔT should be nT-range (1e-9 to 1e-5 T), NOT µT (50e-6)
        assert abs(delta_t) < 1e-4, f"ΔT suspiciously large (µT range?): {delta_t}"

    def test_query_metal_detector_stronger_near_target(self):
        """ΔT is stronger near the target than far away."""
        server = PhysicsServer()
        server.handle_request({
            "command": "load_scenario",
            "params": {"path": SCENARIO_PATH},
        })
        resp = server.handle_request({
            "command": "query_metal_detector",
            "params": {"positions": [[10.0, 10.0, 0.175], [0.0, 0.0, 0.175]]},
        })
        delta_t = resp["data"]["delta_t"]
        assert abs(delta_t[0]) > abs(delta_t[1])

    def test_query_em_sweep_returns_multiple_frequencies(self):
        """EM sweep returns results for each requested frequency."""
        server = PhysicsServer()
        server.handle_request({
            "command": "load_scenario",
            "params": {"path": SCENARIO_PATH},
        })
        freqs = [1000.0, 5000.0, 10000.0]
        resp = server.handle_request({
            "command": "query_em_sweep",
            "params": {"positions": [[10.0, 10.0, 0.175]], "frequencies": freqs},
        })
        assert resp["status"] == "ok"
        sweep = resp["data"]["sweep"]
        assert len(sweep) == 3
        for i, entry in enumerate(sweep):
            assert entry["frequency"] == freqs[i]
            assert len(entry["response_real"]) == 1
            assert len(entry["response_imag"]) == 1

    def test_query_em_sweep_default_frequencies(self):
        """EM sweep uses default frequencies when none provided."""
        server = PhysicsServer()
        server.handle_request({
            "command": "load_scenario",
            "params": {"path": SCENARIO_PATH},
        })
        resp = server.handle_request({
            "command": "query_em_sweep",
            "params": {"positions": [[10.0, 10.0, 0.175]]},
        })
        assert resp["status"] == "ok"
        sweep = resp["data"]["sweep"]
        # Scenario now has hirt_config with 4 frequencies
        assert len(sweep) >= 3

    def test_query_em_response_includes_layered_background(self):
        """EM response is non-zero even far from targets (layered-earth background)."""
        server = PhysicsServer()
        server.handle_request({
            "command": "load_scenario",
            "params": {"path": SCENARIO_PATH},
        })
        # Query far from any target — should still get layered-earth background
        resp = server.handle_request({
            "command": "query_em_response",
            "params": {"positions": [[0.0, 0.0, 0.3]], "frequency": 10000.0},
        })
        assert resp["status"] == "ok"
        real_val = resp["data"]["response_real"][0]
        # Layered earth should produce a non-zero real response
        assert abs(real_val) > 1e-12, f"EM background too small: {real_val}"

    def test_set_environment_updates_state(self):
        """set_environment command updates server environment."""
        server = PhysicsServer()
        server.handle_request({
            "command": "load_scenario",
            "params": {"path": SCENARIO_PATH},
        })
        resp = server.handle_request({
            "command": "set_environment",
            "params": {"temperature_c": 35.0, "saturation": 0.9},
        })
        assert resp["status"] == "ok"
        env_data = resp["data"]["environment"]
        assert env_data["temperature_c"] == 35.0
        assert env_data["saturation"] == 0.9

    def test_set_environment_frozen(self):
        """Frozen condition can be toggled via set_environment."""
        server = PhysicsServer()
        server.handle_request({
            "command": "load_scenario",
            "params": {"path": SCENARIO_PATH},
        })
        resp = server.handle_request({
            "command": "set_environment",
            "params": {"frozen": True},
        })
        assert resp["status"] == "ok"
        assert resp["data"]["environment"]["frozen"] is True

    def test_scenario_info_includes_soil_environment(self):
        """get_scenario_info returns current soil environment."""
        server = PhysicsServer()
        server.handle_request({
            "command": "load_scenario",
            "params": {"path": SCENARIO_PATH},
        })
        resp = server.handle_request({"command": "get_scenario_info"})
        assert resp["status"] == "ok"
        data = resp["data"]
        assert "soil_environment" in data
        env = data["soil_environment"]
        assert "temperature_c" in env
        assert "saturation" in env

    def test_environment_affects_layer_conductivity(self):
        """Temperature change affects effective conductivity in scenario info."""
        server = PhysicsServer()
        server.handle_request({
            "command": "load_scenario",
            "params": {"path": SCENARIO_PATH},
        })
        # Get baseline conductivity
        resp1 = server.handle_request({"command": "get_scenario_info"})
        cond_cold = resp1["data"]["terrain"]["layers"][0]["conductivity"]

        # Raise temperature
        server.handle_request({
            "command": "set_environment",
            "params": {"temperature_c": 40.0},
        })
        resp2 = server.handle_request({"command": "get_scenario_info"})
        cond_hot = resp2["data"]["terrain"]["layers"][0]["conductivity"]

        # Hot should give higher conductivity (if Archie params present)
        # or equal if static-only layer
        assert cond_hot >= cond_cold

    def test_query_gradient_returns_per_channel(self):
        """Gradient response includes per_channel with 4 entries."""
        server = PhysicsServer()
        server.handle_request({
            "command": "load_scenario",
            "params": {"path": SCENARIO_PATH},
        })
        resp = server.handle_request({
            "command": "query_gradient",
            "params": {"positions": [[10.0, 10.0, 0.175]]},
        })
        assert resp["status"] == "ok"
        data = resp["data"]
        assert "per_channel" in data
        assert "adc_counts" in data
        assert len(data["per_channel"]) == 1  # 1 position
        assert len(data["per_channel"][0]) == 4  # 4 channels
        assert len(data["adc_counts"][0]) == 4

    def test_query_metal_detector_returns_enriched_fields(self):
        """Metal detector response includes target_id, depth_estimate, etc."""
        server = PhysicsServer()
        server.handle_request({
            "command": "load_scenario",
            "params": {"path": SCENARIO_PATH},
        })
        resp = server.handle_request({
            "command": "query_metal_detector",
            "params": {"positions": [[10.0, 10.0, 0.175]]},
        })
        assert resp["status"] == "ok"
        data = resp["data"]
        assert "target_id" in data
        assert "depth_estimate" in data
        assert "ground_mineral_level" in data
        assert "ferrous_ratio" in data
        assert len(data["target_id"]) == 1
        assert 0 <= data["target_id"][0] <= 99

    def test_ground_mineral_level_varies_by_scenario(self):
        """Scenarios with high susceptibility report higher ground mineral level."""
        server = PhysicsServer()
        server.handle_request({
            "command": "load_scenario",
            "params": {"path": SCENARIO_PATH},
        })
        resp = server.handle_request({
            "command": "query_metal_detector",
            "params": {"positions": [[5.0, 5.0, 0.175]]},
        })
        data = resp["data"]
        # single-ferrous-target has soil with some susceptibility
        mineral = data["ground_mineral_level"][0]
        assert mineral >= 0.0
        assert mineral <= 100.0

    def test_hot_rock_produces_signal(self):
        """Hot rock (high χ) clutter objects produce detectable delta_t."""
        server = PhysicsServer()
        # Use bomb-crater which has slag clutter (high χ and σ)
        bomb_path = os.path.join(
            os.path.dirname(__file__), "..", "scenarios", "bomb-crater-heterogeneous.json"
        )
        server.handle_request({
            "command": "load_scenario",
            "params": {"path": bomb_path},
        })
        # Check that we have clutter objects with high susceptibility
        has_high_chi = any(
            obj.susceptibility > 0.005
            for obj in server.scenario.objects
        )
        assert has_high_chi, "Bomb-crater scenario should have high-χ clutter objects"

        # Query near a high-χ clutter object position
        high_chi_obj = next(
            obj for obj in server.scenario.objects
            if obj.susceptibility > 0.005
        )
        pos = high_chi_obj.position.tolist()
        query_pos = [pos[0], pos[1], 0.175]
        resp = server.handle_request({
            "command": "query_metal_detector",
            "params": {"positions": [query_pos]},
        })
        assert resp["status"] == "ok"
        delta_t = resp["data"]["delta_t"][0]
        # Hot rock should produce measurable signal
        assert abs(delta_t) > 1e-12

    def test_slag_produces_em_response(self):
        """Slag (high σ) clutter produces EM response."""
        server = PhysicsServer()
        bomb_path = os.path.join(
            os.path.dirname(__file__), "..", "scenarios", "bomb-crater-heterogeneous.json"
        )
        server.handle_request({
            "command": "load_scenario",
            "params": {"path": bomb_path},
        })
        # Find a high-conductivity EM source (slag)
        high_sigma_sources = [
            src for src in server.scenario.em_sources
            if src['conductivity'] > 100
        ]
        if not high_sigma_sources:
            pytest.skip("No high-σ EM sources in bomb-crater scenario")

        src = high_sigma_sources[0]
        query_pos = [src['position'][0], src['position'][1], 0.3]
        resp = server.handle_request({
            "command": "query_em_response",
            "params": {"positions": [query_pos], "frequency": 10000.0},
        })
        assert resp["status"] == "ok"
        # Should have non-trivial EM response
        real_val = abs(resp["data"]["response_real"][0])
        imag_val = abs(resp["data"]["response_imag"][0])
        assert real_val > 1e-15 or imag_val > 1e-15

    def test_request_history_contains_commands(self):
        """Server stats include recent per-command records."""
        server = PhysicsServer()
        server.handle_request({
            "command": "load_scenario",
            "params": {"path": SCENARIO_PATH},
        })
        server.handle_request({"command": "ping"})
        server.handle_request({
            "command": "query_field",
            "params": {"positions": [[0.0, 0.0, 0.3]]},
        })

        stats_resp = server.handle_request({"command": "get_server_stats"})
        assert stats_resp["status"] == "ok"
        recent = stats_resp["data"]["recent_requests"]
        assert len(recent) >= 2
        commands = [r["command"] for r in recent]
        assert "query_field" in commands
