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

    "query_em_response": Compute FDEM secondary field at receiver positions
        params: {"positions": [[x,y,z], ...], "frequency": 1000.0}
        returns: {"response_real": [...], "response_imag": [...]}

    "query_apparent_resistivity": Compute ERT apparent resistivity
        params: {"electrode_positions": [[x,y], ...],
                 "measurements": [[c1,c2,p1,p2], ...]}
        returns: {"apparent_resistivity": [...], "geometric_factors": [...]}

    "query_skin_depth": Quick analytical skin depth calculation
        params: {"frequency": 1000.0, "conductivity": 0.01}
        returns: {"skin_depth": 503.29, "unit": "meters"}

    "ping": Health check
        returns: {"message": "pong"}

    "shutdown": Stop the server
"""

from __future__ import annotations

import logging

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

    def query_em_response(
        self,
        positions: list,
        frequency: float = 1000.0,
    ) -> dict:
        """Compute FDEM secondary field response at positions."""
        from geosim.em.fdem import secondary_field_conductive_sphere

        em_sources = self.scenario.em_sources
        results_real = []
        results_imag = []

        for pos in positions:
            r_obs = np.array(pos, dtype=np.float64)
            total = 0.0 + 0.0j

            for src in em_sources:
                r_src = np.array(src['position'], dtype=np.float64)
                dist = float(np.linalg.norm(r_obs - r_src))
                if dist < src['radius']:
                    dist = src['radius'] * 1.01  # avoid inside-sphere

                resp = secondary_field_conductive_sphere(
                    radius=src['radius'],
                    conductivity=src['conductivity'],
                    frequency=frequency,
                    r_obs=dist,
                )
                total += resp

            results_real.append(float(total.real))
            results_imag.append(float(total.imag))

        return {
            "response_real": results_real,
            "response_imag": results_imag,
            "frequency": frequency,
        }

    def query_apparent_resistivity(
        self,
        electrode_positions: list,
        measurements: list,
    ) -> dict:
        """Compute ERT apparent resistivity."""
        from geosim.resistivity.ert import ert_forward

        positions = np.array(electrode_positions, dtype=np.float64)
        meas_tuples = [tuple(m) for m in measurements]

        res_model = self.scenario.resistivity_model
        result = ert_forward(
            electrode_positions=positions,
            measurements=meas_tuples,
            resistivities=res_model['resistivities'],
            thicknesses=res_model['thicknesses'] or None,
            backend='analytical',
        )
        return {
            "apparent_resistivity": result['apparent_resistivity'],
            "geometric_factors": result['geometric_factors'],
        }

    @staticmethod
    def query_skin_depth(frequency: float, conductivity: float) -> dict:
        """Quick analytical skin depth calculation."""
        from geosim.em.skin_depth import skin_depth

        delta = float(skin_depth(frequency, conductivity))
        return {"skin_depth": delta, "unit": "meters"}

    def _build_scenario_info(self) -> dict:
        """Build the full scenario info payload."""
        s = self.scenario

        # Terrain info with layers
        terrain_info = {
            "x_extent": list(s.terrain.x_extent),
            "y_extent": list(s.terrain.y_extent),
            "surface_elevation": s.terrain.surface_elevation,
            "layers": [
                {
                    "name": layer.name,
                    "z_top": layer.z_top,
                    "z_bottom": layer.z_bottom,
                    "conductivity": layer.conductivity,
                }
                for layer in s.terrain.layers
            ],
        }

        # Objects list (safe subset — positions, types, radii)
        objects_info = [
            {
                "name": obj.name,
                "position": _make_serializable(obj.position),
                "type": obj.object_type,
                "radius": obj.radius,
            }
            for obj in s.objects
        ]

        return {
            "name": s.name,
            "description": s.description,
            "n_sources": len(self.sources),
            "terrain": terrain_info,
            "earth_field": _make_serializable(s.earth_field),
            "objects": objects_info,
            "metadata": s.metadata,
            "has_hirt": s.hirt_config is not None,
            "available_instruments": self._determine_instruments(),
        }

    def _determine_instruments(self) -> list[str]:
        """Determine available instruments from scenario data."""
        instruments = ["mag_gradiometer"]

        if self.scenario is None:
            return instruments

        # EM available if any object has conductivity and radius (EM-detectable)
        has_em_targets = any(
            obj.conductivity > 0 and obj.radius > 0
            for obj in self.scenario.objects
        )

        # HIRT config explicitly enables EM and ERT
        has_hirt = self.scenario.hirt_config is not None

        if has_em_targets or has_hirt:
            instruments.append("em_fdem")

        # ERT available if terrain has layers with varying conductivity, or HIRT
        has_layered = len(self.scenario.terrain.layers) >= 2
        if has_hirt or has_layered:
            instruments.append("resistivity")

        return instruments

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
                    "data": self._build_scenario_info(),
                }

            elif command == "query_em_response":
                if self.scenario is None:
                    return {"status": "error", "message": "No scenario loaded"}
                result = self.query_em_response(
                    params["positions"],
                    params.get("frequency", 1000.0),
                )
                return {"status": "ok", "data": result}

            elif command == "query_apparent_resistivity":
                if self.scenario is None:
                    return {"status": "error", "message": "No scenario loaded"}
                result = self.query_apparent_resistivity(
                    params["electrode_positions"],
                    params["measurements"],
                )
                return {"status": "ok", "data": result}

            elif command == "query_skin_depth":
                result = self.query_skin_depth(
                    params["frequency"],
                    params["conductivity"],
                )
                return {"status": "ok", "data": result}

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
