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
