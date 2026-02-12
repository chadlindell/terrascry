"""ZeroMQ server wrapping the physics engine.

Provides a REQ-REP interface for external clients (e.g., Godot) to
query the physics engine. Also supports PUB-SUB for streaming sensor
data during walk simulations.

Protocol (REQ-REP):
    Client sends JSON: {"command": "...", "params": {...}}
    Server returns JSON: {"status": "ok", "data": {...}}

Commands:
    "load_scenario": Load a scenario file
        params: {"path": "/path/to/scenario.json"}

    "query_field": Compute magnetic field at points
        params: {"positions": [[x,y,z], ...]}
        returns: {"B": [[Bx,By,Bz], ...]}

    "query_gradient": Compute gradiometer readings at points
        params: {"positions": [[x,y,z], ...],
                 "sensor_separation": 0.35,
                 "component": 2}
        returns: {"B_bottom": [...], "B_top": [...], "gradient": [...]}

    "get_scenario_info": Return loaded scenario metadata
        returns: {"name": "...", "n_sources": N, "terrain": {...}}

    "ping": Health check
        returns: {"message": "pong"}

    "shutdown": Stop the server
"""

from __future__ import annotations

import json
import logging
import sys

import numpy as np

logger = logging.getLogger(__name__)


def _make_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    return obj


class PhysicsServer:
    """ZeroMQ REQ-REP server for the physics engine.

    Parameters
    ----------
    bind_address : str
        ZeroMQ bind address (default: "tcp://*:5555").
    """

    def __init__(self, bind_address: str = "tcp://*:5555"):
        self.bind_address = bind_address
        self.scenario = None
        self.sources = []
        self._running = False

    def load_scenario(self, path: str) -> dict:
        """Load a scenario file."""
        from geosim.scenarios.loader import load_scenario

        self.scenario = load_scenario(path)
        self.sources = self.scenario.magnetic_sources
        return {
            "name": self.scenario.name,
            "n_sources": len(self.sources),
            "n_objects": len(self.scenario.objects),
        }

    def query_field(self, positions: list) -> dict:
        """Compute magnetic field at given positions."""
        from geosim.magnetics.dipole import superposition_field

        r_obs = np.array(positions, dtype=np.float64)
        B = superposition_field(r_obs, self.sources)
        return {"B": _make_serializable(B)}

    def query_gradient(
        self,
        positions: list,
        sensor_separation: float = 0.35,
        component: int = 2,
    ) -> dict:
        """Compute gradiometer readings at given positions."""
        from geosim.magnetics.dipole import gradiometer_reading

        r_obs = np.array(positions, dtype=np.float64)
        B_bot, B_top, grad = gradiometer_reading(
            r_obs, self.sources, sensor_separation, component
        )
        return {
            "B_bottom": _make_serializable(B_bot),
            "B_top": _make_serializable(B_top),
            "gradient": _make_serializable(grad),
        }

    def handle_request(self, request: dict) -> dict:
        """Route a request to the appropriate handler."""
        command = request.get("command", "")
        params = request.get("params", {})

        try:
            if command == "ping":
                return {"status": "ok", "data": {"message": "pong"}}

            elif command == "load_scenario":
                info = self.load_scenario(params["path"])
                return {"status": "ok", "data": info}

            elif command == "query_field":
                if not self.sources:
                    return {"status": "error", "message": "No scenario loaded"}
                result = self.query_field(params["positions"])
                return {"status": "ok", "data": result}

            elif command == "query_gradient":
                if not self.sources:
                    return {"status": "error", "message": "No scenario loaded"}
                result = self.query_gradient(
                    params["positions"],
                    params.get("sensor_separation", 0.35),
                    params.get("component", 2),
                )
                return {"status": "ok", "data": result}

            elif command == "get_scenario_info":
                if self.scenario is None:
                    return {"status": "error", "message": "No scenario loaded"}
                return {
                    "status": "ok",
                    "data": {
                        "name": self.scenario.name,
                        "description": self.scenario.description,
                        "n_sources": len(self.sources),
                        "terrain": {
                            "x_extent": list(self.scenario.terrain.x_extent),
                            "y_extent": list(self.scenario.terrain.y_extent),
                        },
                    },
                }

            elif command == "shutdown":
                self._running = False
                return {"status": "ok", "data": {"message": "shutting down"}}

            else:
                return {"status": "error", "message": f"Unknown command: {command}"}

        except Exception as e:
            logger.exception("Error handling request")
            return {"status": "error", "message": str(e)}

    def run(self) -> None:
        """Start the REQ-REP server loop."""
        import zmq

        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(self.bind_address)

        logger.info(f"GeoSim physics server listening on {self.bind_address}")
        self._running = True

        try:
            while self._running:
                # Wait for request with timeout so we can check _running
                if socket.poll(timeout=1000):
                    message = socket.recv_json()
                    response = self.handle_request(message)
                    socket.send_json(response)
        except KeyboardInterrupt:
            logger.info("Server interrupted")
        finally:
            socket.close()
            context.term()
            logger.info("Server stopped")


def main():
    """Entry point for geosim-server command."""
    import argparse

    parser = argparse.ArgumentParser(description="GeoSim physics ZeroMQ server")
    parser.add_argument(
        "--bind", default="tcp://*:5555",
        help="ZeroMQ bind address (default: tcp://*:5555)",
    )
    parser.add_argument(
        "--scenario", default=None,
        help="Pre-load a scenario file on startup",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    server = PhysicsServer(bind_address=args.bind)

    if args.scenario:
        info = server.load_scenario(args.scenario)
        logger.info(f"Pre-loaded scenario: {info['name']} ({info['n_sources']} sources)")

    server.run()


if __name__ == "__main__":
    main()
