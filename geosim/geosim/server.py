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

    "query_metal_detector": Compute ground-balanced differential total-field anomaly
        params: {"positions": [[x,y,z], ...]}
        returns: {"delta_t": [...], "unit": "T"}

    "query_em_response": Compute FDEM secondary field at receiver positions
        params: {"positions": [[x,y,z], ...], "frequency": 1000.0}
        returns: {"response_real": [...], "response_imag": [...]}

    "query_em_sweep": Batch multi-frequency EM query
        params: {"positions": [[x,y,z], ...],
                 "frequencies": [1000, 5000, 10000]}
        returns: {"sweep": [{"frequency": f, "response_real": [...],
                             "response_imag": [...]}, ...]}

    "query_apparent_resistivity": Compute ERT apparent resistivity
        params: {"electrode_positions": [[x,y], ...],
                 "measurements": [[c1,c2,p1,p2], ...]}
        returns: {"apparent_resistivity": [...], "geometric_factors": [...]}

    "query_skin_depth": Quick analytical skin depth calculation
        params: {"frequency": 1000.0, "conductivity": 0.01}
        returns: {"skin_depth": 503.29, "unit": "meters"}

    "set_comms_profile": Configure simulated transport impairments
        params: {"enabled": true, "base_latency_ms": 8, "jitter_ms": 2,
                 "drop_rate": 0.02, "timeout_rate": 0.01}
        returns: {"comms_profile": {...}}

    "get_server_stats": Return rolling request and comms stats
        returns: {"stats": {...}, "recent_requests": [...], "comms_profile": {...}}

    "ping": Health check
        returns: {"message": "pong"}

    "shutdown": Stop the server
"""

from __future__ import annotations

from collections import deque
import logging
import time

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


def _effective_zone_radius(zone) -> float:
    """Convert a zone geometry into an effective radius in meters."""
    dims = zone.dimensions or {}
    if zone.shape == "sphere":
        return max(float(dims.get("radius", 0.5)), 0.05)
    if zone.shape == "cylinder":
        r = float(dims.get("radius", 0.5))
        h = float(dims.get("height", 1.0))
        return max(float((r * r * h) ** (1.0 / 3.0)), 0.05)
    if zone.shape == "box":
        l = float(dims.get("length", 1.0))
        w = float(dims.get("width", 1.0))
        d = float(dims.get("depth", 1.0))
        return max(float((l * w * d / (4.0 / 3.0 * np.pi)) ** (1.0 / 3.0)), 0.05)
    return 0.5


def _derive_environment_profile(scenario) -> dict:
    """Derive a compact environment profile for rendering/audio/UX tuning."""
    layers = scenario.terrain.layers
    metadata = scenario.metadata or {}
    name = str(scenario.name).lower()
    desc = str(scenario.description).lower()
    category = str(metadata.get("category", "")).lower()

    mean_conductivity = 0.05
    if layers:
        mean_conductivity = float(np.mean([float(layer.conductivity) for layer in layers]))

    wetness = float(np.clip(mean_conductivity / 0.25, 0.0, 1.0))
    water_table_depth = metadata.get("water_table_depth")
    if water_table_depth is not None:
        try:
            wt = float(water_table_depth)
            wetness = float(np.clip(wetness + np.clip((1.0 - wt) / 1.0, 0.0, 0.6), 0.0, 1.0))
        except (TypeError, ValueError):
            pass
    if "swamp" in name or "marsh" in name or "waterlogged" in desc:
        wetness = float(np.clip(wetness + 0.35, 0.0, 1.0))

    ruggedness = float(np.clip(len(scenario.anomaly_zones) * 0.08, 0.0, 0.35))
    if "crater" in name or "crater" in desc or metadata.get("crater"):
        ruggedness = float(np.clip(ruggedness + 0.4, 0.0, 1.0))
    if "uxo" in category:
        ruggedness = float(np.clip(ruggedness + 0.2, 0.0, 1.0))
    if "forensic" in category:
        ruggedness = float(np.clip(ruggedness - 0.15, 0.05, 1.0))

    return {
        "wetness": wetness,
        "ruggedness": ruggedness,
        "wind_intensity": float(np.clip(0.3 + 0.5 * wetness, 0.0, 1.0)),
        "vegetation_density": float(np.clip(0.55 + 0.35 * wetness - 0.2 * ruggedness, 0.1, 1.0)),
        "fog_density": float(np.clip(0.003 + 0.01 * wetness, 0.0, 0.02)),
    }


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
        self._rng = np.random.default_rng(2026)
        self._request_history = deque(maxlen=200)
        self._stats = {
            "request_count": 0,
            "error_count": 0,
            "dropped_count": 0,
            "timed_out_count": 0,
            "avg_latency_ms": 0.0,
        }
        self._comms_profile = self._sanitize_comms_profile({})
        self._environment = None  # SoilEnvironment instance

    def load_scenario(self, path: str) -> dict:
        """Load a scenario file."""
        from geosim.environment import SoilEnvironment
        from geosim.scenarios.loader import load_scenario

        self.scenario = load_scenario(path)
        self.sources = self.scenario.magnetic_sources
        self._apply_scenario_comms_profile()

        # Initialize environment from scenario defaults
        env_data = self.scenario.soil_environment
        if env_data and isinstance(env_data, dict):
            self._environment = SoilEnvironment.from_dict(env_data)
        else:
            self._environment = SoilEnvironment()

        return {
            "name": self.scenario.name,
            "n_sources": len(self.sources),
            "n_objects": len(self.scenario.objects),
        }

    def set_environment(self, params: dict) -> dict:
        """Update soil environment conditions at runtime."""
        from geosim.environment import SoilEnvironment

        if self._environment is None:
            self._environment = SoilEnvironment()

        if "temperature_c" in params:
            self._environment.temperature_c = float(params["temperature_c"])
        if "saturation" in params:
            self._environment.saturation = float(params["saturation"])
        if "pore_water_sigma" in params:
            self._environment.pore_water_sigma = float(params["pore_water_sigma"])
        if "frozen" in params:
            self._environment.frozen = bool(params["frozen"])

        return {"environment": self._environment.to_dict()}

    def _sanitize_comms_profile(self, profile: dict, merge: bool = False) -> dict:
        """Validate and clamp comms profile values."""
        base = {
            "enabled": False,
            "base_latency_ms": 0.0,
            "jitter_ms": 0.0,
            "drop_rate": 0.0,
            "timeout_rate": 0.0,
            "per_command_latency_ms": {},
            "max_history": 200,
            "random_seed": None,
        }
        if merge:
            base.update(self._comms_profile)
        if profile:
            base.update(profile)

        per_cmd = base.get("per_command_latency_ms", {})
        if not isinstance(per_cmd, dict):
            per_cmd = {}
        sanitized_per_cmd = {}
        for k, v in per_cmd.items():
            try:
                sanitized_per_cmd[str(k)] = max(0.0, float(v))
            except (TypeError, ValueError):
                continue

        random_seed = base.get("random_seed", None)
        if random_seed is not None:
            try:
                random_seed = int(random_seed)
            except (TypeError, ValueError):
                random_seed = None

        return {
            "enabled": bool(base.get("enabled", False)),
            "base_latency_ms": max(0.0, float(base.get("base_latency_ms", 0.0))),
            "jitter_ms": max(0.0, float(base.get("jitter_ms", 0.0))),
            "drop_rate": float(np.clip(float(base.get("drop_rate", 0.0)), 0.0, 1.0)),
            "timeout_rate": float(np.clip(float(base.get("timeout_rate", 0.0)), 0.0, 1.0)),
            "per_command_latency_ms": sanitized_per_cmd,
            "max_history": int(np.clip(int(base.get("max_history", 200)), 20, 5000)),
            "random_seed": random_seed,
        }

    def _public_comms_profile(self) -> dict:
        """Profile representation safe for API responses."""
        prof = dict(self._comms_profile)
        return prof

    def _apply_scenario_comms_profile(self) -> None:
        """Load scenario-provided comms profile, or reset to defaults."""
        if self.scenario is None:
            self._comms_profile = self._sanitize_comms_profile({})
            return
        metadata = self.scenario.metadata or {}
        profile = metadata.get("comms_profile", {})
        if isinstance(profile, dict):
            self._comms_profile = self._sanitize_comms_profile(profile)
            seed = self._comms_profile.get("random_seed")
            if seed is not None:
                self._rng = np.random.default_rng(seed)
        else:
            self._comms_profile = self._sanitize_comms_profile({})
        self._reset_history_capacity(self._comms_profile["max_history"])

    def _reset_history_capacity(self, max_history: int) -> None:
        history = list(self._request_history)
        self._request_history = deque(history[-max_history:], maxlen=max_history)

    def set_comms_profile(self, profile: dict) -> dict:
        """Update comms simulation profile at runtime."""
        self._comms_profile = self._sanitize_comms_profile(profile, merge=True)
        seed = self._comms_profile.get("random_seed")
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._reset_history_capacity(self._comms_profile["max_history"])
        return self._public_comms_profile()

    def _apply_comms_impairment(self, command: str) -> dict | None:
        """Apply configured transport impairments and return error response if dropped."""
        prof = self._comms_profile
        if not prof.get("enabled", False):
            return None
        if command in {"ping", "shutdown", "set_comms_profile", "get_server_stats"}:
            return None

        if self._rng.random() < prof["drop_rate"]:
            return {
                "status": "error",
                "message": f"Simulated packet drop for command '{command}'",
                "error_code": "simulated_packet_drop",
            }
        if self._rng.random() < prof["timeout_rate"]:
            timeout_ms = prof["base_latency_ms"] + 2.5 * prof["jitter_ms"] + 50.0
            if timeout_ms > 0:
                time.sleep(timeout_ms / 1000.0)
            return {
                "status": "error",
                "message": f"Simulated timeout for command '{command}'",
                "error_code": "simulated_timeout",
            }

        base = prof["base_latency_ms"] + prof["per_command_latency_ms"].get(command, 0.0)
        jitter = prof["jitter_ms"]
        latency_ms = max(0.0, float(base + self._rng.normal(0.0, jitter)))
        if latency_ms > 0:
            time.sleep(latency_ms / 1000.0)
        return None

    def _record_request(self, command: str, response: dict, latency_ms: float) -> None:
        """Record request stats and keep bounded history."""
        status = str(response.get("status", "error"))
        error_code = response.get("error_code")

        self._stats["request_count"] += 1
        n = self._stats["request_count"]
        prev_avg = float(self._stats["avg_latency_ms"])
        self._stats["avg_latency_ms"] = prev_avg + (latency_ms - prev_avg) / float(max(n, 1))
        if status != "ok":
            self._stats["error_count"] += 1
        if error_code == "simulated_packet_drop":
            self._stats["dropped_count"] += 1
        if error_code == "simulated_timeout":
            self._stats["timed_out_count"] += 1

        self._request_history.append({
            "timestamp_unix": time.time(),
            "command": command,
            "status": status,
            "latency_ms": float(latency_ms),
            "error_code": error_code,
        })

    def _build_server_stats(self) -> dict:
        """Build observability payload for comms and request behavior."""
        return {
            "stats": dict(self._stats),
            "comms_profile": self._public_comms_profile(),
            "recent_requests": list(self._request_history),
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
        """Compute gradiometer readings at given positions.

        Returns standard gradient plus per-channel readings for 4 sensor
        pairs (Pathfinder instrument model: 0.50m horizontal spacing).
        """
        from geosim.magnetics.dipole import gradiometer_reading

        r_obs = np.array(positions, dtype=np.float64)
        single = r_obs.ndim == 1
        if single:
            r_obs = r_obs[np.newaxis, :]

        B_bot, B_top, grad = gradiometer_reading(
            r_obs, self.sources, sensor_separation, component
        )

        # Compute per-channel readings for 4 sensor pairs
        # Pathfinder has 4 pairs at 0.50m horizontal spacing
        pair_offsets = [-0.75, -0.25, 0.25, 0.75]  # meters from center
        per_channel = []
        adc_counts = []
        adc_scale = 2.5e9  # counts per Tesla

        for pos in r_obs:
            channels = []
            counts = []
            for offset in pair_offsets:
                # Offset in X (East) direction
                offset_pos = pos.copy()
                offset_pos[0] += offset
                _, _, ch_grad = gradiometer_reading(
                    offset_pos[np.newaxis, :], self.sources,
                    sensor_separation, component,
                )
                g = float(ch_grad[0]) if hasattr(ch_grad, '__len__') else float(ch_grad)
                channels.append(g)
                counts.append(int(np.clip(g * adc_scale, -32000, 32000)))
            per_channel.append(channels)
            adc_counts.append(counts)

        return {
            "B_bottom": _make_serializable(B_bot),
            "B_top": _make_serializable(B_top),
            "gradient": _make_serializable(grad),
            "per_channel": per_channel,
            "adc_counts": adc_counts,
        }

    def query_metal_detector(
        self,
        positions: list,
        frequency: float = 10000.0,
    ) -> dict:
        """Compute metal detector response combining magnetic and EM components.

        Real VLF metal detectors use a transmit coil to generate an AC field.
        The response has two components:

        1. **Magnetic (ferrous)**: induced + remanent dipole moment produces
           a total-field anomaly ΔT = B̂_earth · B_anomaly (dominant for
           ferrous targets like steel, iron).

        2. **EM induction (conductive)**: eddy currents in conductive targets
           produce a secondary field detectable as in-phase/quadrature response.
           This makes non-ferrous metals (brass, copper, aluminium) detectable.

        The combined response is the magnetic ΔT plus the magnitude of the
        EM secondary field scaled to equivalent ΔT units, giving a single
        scalar that increases for both ferrous and conductive targets.

        Also computes target ID (0-99 ferrous→non-ferrous), depth estimate,
        ferrous ratio, and ground mineral level from soil susceptibility.

        Parameters
        ----------
        positions : list
            Observation positions [[x, y, z], ...].
        frequency : float
            Operating frequency in Hz (default 10 kHz, typical VLF).

        Returns
        -------
        dict
            ``{"delta_t": [...], "target_id": [...], "depth_estimate": [...],
               "ground_mineral_level": [...], "ferrous_ratio": [...], "unit": "T"}``
        """
        from geosim.em.fdem import secondary_field_conductive_sphere
        from geosim.magnetics.dipole import superposition_field

        r_obs = np.array(positions, dtype=np.float64)
        single = r_obs.ndim == 1
        if single:
            r_obs = r_obs[np.newaxis, :]

        # Component 1: Magnetic anomaly ΔT from ferrous dipole moments
        B_anomaly = superposition_field(r_obs, self.sources)
        if B_anomaly.ndim == 1:
            B_anomaly = B_anomaly[np.newaxis, :]

        B_earth = self.scenario.earth_field
        B_earth_mag = float(np.linalg.norm(B_earth))
        if B_earth_mag < 1e-20:
            n = len(r_obs)
            return {
                "delta_t": [0.0] * n, "target_id": [0] * n,
                "depth_estimate": [0.0] * n, "ground_mineral_level": [0.0] * n,
                "ferrous_ratio": [0.0] * n, "unit": "T",
            }

        B_hat = B_earth / B_earth_mag
        delta_t = np.dot(B_anomaly, B_hat)  # shape (N,)

        # Track magnetic and EM components separately for target ID
        mag_component = delta_t.copy()
        em_component = np.zeros(len(r_obs))

        # Component 2: EM induction from conductive targets
        em_sources = self.scenario.em_sources
        for i, pos in enumerate(r_obs):
            em_total = 0.0
            for src in em_sources:
                r_src = np.array(src['position'], dtype=np.float64)
                dist = float(np.linalg.norm(pos - r_src))
                if dist < src['radius']:
                    dist = src['radius'] * 1.01
                resp = secondary_field_conductive_sphere(
                    radius=src['radius'],
                    conductivity=src['conductivity'],
                    frequency=frequency,
                    r_obs=dist,
                )
                em_total += abs(resp) * B_earth_mag
            delta_t[i] += em_total
            em_component[i] = em_total

        # Ground susceptibility contribution (mineralized ground)
        ground_mineral = np.zeros(len(r_obs))
        layers = self.scenario.terrain.layers
        if layers:
            # Average susceptibility of near-surface layers
            avg_chi = float(np.mean([
                layer.susceptibility for layer in layers
                if layer.susceptibility > 0
            ])) if any(layer.susceptibility > 0 for layer in layers) else 0.0

            if avg_chi > 0:
                # Ground signal: B_earth × χ_soil × volume_factor
                # Scale to 0-100 where χ=0.01 → ~100
                ground_mineral[:] = float(np.clip(avg_chi / 0.01 * 100.0, 0.0, 100.0))
                # Add ground noise to delta_t
                ground_signal = B_earth_mag * avg_chi * 0.1  # volume coupling factor
                delta_t += ground_signal

        # Compute target ID (0=iron → 99=silver/copper) from phase angle
        target_id = np.zeros(len(r_obs), dtype=int)
        ferrous_ratio = np.zeros(len(r_obs))
        for i in range(len(r_obs)):
            abs_mag = abs(mag_component[i])
            abs_em = em_component[i]
            total = abs_mag + abs_em
            if total > 1e-20:
                # Ferrous ratio: 1.0 = pure ferrous, 0.0 = pure conductive
                fr = abs_mag / total
                ferrous_ratio[i] = fr
                # Target ID: 0-99, where low = ferrous, high = non-ferrous
                target_id[i] = int(np.clip((1.0 - fr) * 99, 0, 99))

        # Depth estimate from inverse-cube signal decay
        depth_estimate = np.zeros(len(r_obs))
        for i, pos in enumerate(r_obs):
            if abs(delta_t[i]) > 1e-15:
                # Find nearest object and estimate depth from signal strength
                min_dist = 1e10
                for obj in self.scenario.objects:
                    dist = float(np.linalg.norm(pos - obj.position))
                    if dist < min_dist:
                        min_dist = dist
                depth_estimate[i] = max(min_dist, 0.0)

        return {
            "delta_t": _make_serializable(delta_t),
            "target_id": [int(x) for x in target_id],
            "depth_estimate": _make_serializable(depth_estimate),
            "ground_mineral_level": _make_serializable(ground_mineral),
            "ferrous_ratio": _make_serializable(ferrous_ratio),
            "unit": "T",
        }

    def query_em_response(
        self,
        positions: list,
        frequency: float = 1000.0,
    ) -> dict:
        """Compute FDEM secondary field response at positions.

        Combines conductive sphere responses from buried objects and
        anomaly zones with a McNeill 1D layered-earth background.
        """
        from geosim.em.fdem import fdem_response_1d, secondary_field_conductive_sphere

        em_sources = self.scenario.em_sources
        anomaly_zones = self.scenario.anomaly_zones

        # Compute layered-earth background (McNeill 1D) once per query
        res_model = self.scenario.resistivity_model
        layered_bg = 0.0 + 0.0j
        thicknesses = res_model.get('thicknesses', [])
        resistivities = res_model.get('resistivities', [])
        if resistivities:
            conductivities_1d = [1.0 / max(r, 1e-6) for r in resistivities]
            coil_sep = 1.0  # GEM-2 style horizontal coplanar
            layered_bg = fdem_response_1d(
                thicknesses=thicknesses,
                conductivities=conductivities_1d,
                frequency=frequency,
                coil_separation=coil_sep,
                height=0.3,  # typical sensor height above ground
            )

        results_real = []
        results_imag = []

        for pos in positions:
            r_obs = np.array(pos, dtype=np.float64)
            total = layered_bg  # start with layered-earth background

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

            # Effective response from conductive anomaly zones.
            for zone in anomaly_zones:
                sigma_zone = float(zone.conductivity)
                if sigma_zone <= 0:
                    continue
                r_src = np.array(zone.center, dtype=np.float64)
                radius = _effective_zone_radius(zone)
                dist = float(np.linalg.norm(r_obs - r_src))
                if dist < radius:
                    dist = radius * 1.01
                resp = secondary_field_conductive_sphere(
                    radius=radius,
                    conductivity=sigma_zone,
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

    def query_em_sweep(
        self,
        positions: list,
        frequencies: list | None = None,
    ) -> dict:
        """Batch multi-frequency EM query.

        Parameters
        ----------
        positions : list
            Observation positions [[x, y, z], ...].
        frequencies : list[float] or None
            Frequencies to sweep. Defaults to scenario HIRT config
            frequencies or [1000, 5000, 10000].

        Returns
        -------
        dict
            ``{"sweep": [{"frequency": f, "response_real": [...],
                          "response_imag": [...]}, ...]}``
        """
        if frequencies is None:
            if self.scenario.hirt_config is not None:
                frequencies = list(self.scenario.hirt_config.frequencies)
            else:
                frequencies = [1000.0, 5000.0, 10000.0]

        sweep = []
        for freq in frequencies:
            result = self.query_em_response(positions, float(freq))
            sweep.append({
                "frequency": float(freq),
                "response_real": result["response_real"],
                "response_imag": result["response_imag"],
            })

        return {"sweep": sweep}

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
        rho_a = np.array(result['apparent_resistivity'], dtype=np.float64)

        # Approximate 3D anomaly-zone influence over analytical layered-earth response.
        zones = self.scenario.anomaly_zones
        if zones and len(rho_a) > 0:
            if positions.shape[1] == 2:
                surf_z = float(self.scenario.terrain.surface_elevation)
                pos_3d = np.column_stack([positions[:, 0], positions[:, 1], np.full(len(positions), surf_z)])
            else:
                pos_3d = positions

            resistivities = res_model.get('resistivities', [])
            background_rho = float(np.median(resistivities)) if resistivities else 100.0

            for i, (c1, c2, p1, p2) in enumerate(meas_tuples):
                midpoint = (pos_3d[c1] + pos_3d[c2] + pos_3d[p1] + pos_3d[p2]) / 4.0
                perturb = 0.0
                for zone in zones:
                    if zone.resistivity > 0:
                        zone_rho = float(zone.resistivity)
                    elif zone.conductivity > 0:
                        zone_rho = 1.0 / float(zone.conductivity)
                    else:
                        continue

                    center = np.array(zone.center, dtype=np.float64)
                    radius = _effective_zone_radius(zone)
                    dist = float(np.linalg.norm(midpoint - center))
                    contrast = (zone_rho - background_rho) / max(background_rho, 1e-6)
                    influence = np.exp(-((dist / radius) ** 2))
                    perturb += 0.35 * contrast * influence

                rho_a[i] *= max(0.05, 1.0 + perturb)

        return {
            "apparent_resistivity": _make_serializable(rho_a),
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
                    "conductivity": layer.effective_conductivity(self._environment),
                    "static_conductivity": layer.conductivity,
                    "relative_permittivity": layer.relative_permittivity,
                    "susceptibility": layer.susceptibility,
                    "color": layer.color,
                }
                for layer in s.terrain.layers
            ],
        }

        # Objects list (safe subset for client visualization + instrument previews)
        objects_info = [
            {
                "name": obj.name,
                "position": _make_serializable(obj.position),
                "type": obj.object_type,
                "radius": obj.radius,
                "conductivity": obj.conductivity,
                "susceptibility": obj.susceptibility,
                "metadata": obj.metadata,
            }
            for obj in s.objects
        ]

        anomalies_info = [
            {
                "name": az.name,
                "center": _make_serializable(np.array(az.center, dtype=np.float64)),
                "shape": az.shape,
                "dimensions": az.dimensions,
                "conductivity": az.conductivity,
                "resistivity": az.resistivity,
                "relative_permittivity": az.relative_permittivity,
                "susceptibility": az.susceptibility,
            }
            for az in s.anomaly_zones
        ]

        hirt_info = None
        if s.hirt_config is not None:
            hirt_info = {
                "frequencies": list(s.hirt_config.frequencies),
                "injection_current": s.hirt_config.injection_current,
                "mit_tx_current_mA": s.hirt_config.mit_tx_current_mA,
                "ert_current_mA": s.hirt_config.ert_current_mA,
                "mit_settle_ms": s.hirt_config.mit_settle_ms,
                "ert_settle_ms": s.hirt_config.ert_settle_ms,
                "adc_averaging": s.hirt_config.adc_averaging,
                "reciprocity_qc_pct": s.hirt_config.reciprocity_qc_pct,
                "section_id": s.hirt_config.section_id,
                "zone_id": s.hirt_config.zone_id,
                "include_intra_probe": s.hirt_config.include_intra_probe,
                "array_type": s.hirt_config.array_type,
                "probe_count": len(s.hirt_config.probes),
            }
        pathfinder_info = {}
        pf_meta = (s.metadata or {}).get("pathfinder", {})
        if isinstance(pf_meta, dict):
            pathfinder_info = dict(pf_meta)

        return {
            "name": s.name,
            "description": s.description,
            "n_sources": len(self.sources),
            "terrain": terrain_info,
            "earth_field": _make_serializable(s.earth_field),
            "objects": objects_info,
            "anomaly_zones": anomalies_info,
            "metadata": s.metadata,
            "has_hirt": s.hirt_config is not None,
            "hirt_config": hirt_info,
            "pathfinder_config": pathfinder_info,
            "environment_profile": _derive_environment_profile(s),
            "soil_environment": self._environment.to_dict() if self._environment else None,
            "comms_profile": self._public_comms_profile(),
            "available_instruments": self._determine_instruments(),
        }

    def _determine_instruments(self) -> list[str]:
        """Determine available instruments from scenario data."""
        instruments = ["mag_gradiometer", "metal_detector"]

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
        t0 = time.perf_counter()
        response: dict

        try:
            impairment = self._apply_comms_impairment(command)
            if impairment is not None:
                response = impairment
            elif command == "ping":
                response = {"status": "ok", "data": {"message": "pong"}}

            elif command == "load_scenario":
                info = self.load_scenario(params["path"])
                response = {"status": "ok", "data": info}

            elif command == "query_field":
                if self.scenario is None:
                    response = {"status": "error", "message": "No scenario loaded"}
                else:
                    result = self.query_field(params["positions"])
                    response = {"status": "ok", "data": result}

            elif command == "query_gradient":
                if self.scenario is None:
                    response = {"status": "error", "message": "No scenario loaded"}
                else:
                    result = self.query_gradient(
                        params["positions"],
                        params.get("sensor_separation", 0.35),
                        params.get("component", 2),
                    )
                    response = {"status": "ok", "data": result}

            elif command == "get_scenario_info":
                if self.scenario is None:
                    response = {"status": "error", "message": "No scenario loaded"}
                else:
                    response = {
                        "status": "ok",
                        "data": self._build_scenario_info(),
                    }

            elif command == "query_metal_detector":
                if self.scenario is None:
                    response = {"status": "error", "message": "No scenario loaded"}
                else:
                    result = self.query_metal_detector(params["positions"])
                    response = {"status": "ok", "data": result}

            elif command == "query_em_response":
                if self.scenario is None:
                    response = {"status": "error", "message": "No scenario loaded"}
                else:
                    result = self.query_em_response(
                        params["positions"],
                        params.get("frequency", 1000.0),
                    )
                    response = {"status": "ok", "data": result}

            elif command == "query_em_sweep":
                if self.scenario is None:
                    response = {"status": "error", "message": "No scenario loaded"}
                else:
                    result = self.query_em_sweep(
                        params["positions"],
                        params.get("frequencies"),
                    )
                    response = {"status": "ok", "data": result}

            elif command == "query_apparent_resistivity":
                if self.scenario is None:
                    response = {"status": "error", "message": "No scenario loaded"}
                else:
                    result = self.query_apparent_resistivity(
                        params["electrode_positions"],
                        params["measurements"],
                    )
                    response = {"status": "ok", "data": result}

            elif command == "query_skin_depth":
                result = self.query_skin_depth(
                    params["frequency"],
                    params["conductivity"],
                )
                response = {"status": "ok", "data": result}

            elif command == "set_environment":
                result = self.set_environment(params)
                response = {"status": "ok", "data": result}

            elif command == "set_comms_profile":
                profile = self.set_comms_profile(params)
                response = {"status": "ok", "data": {"comms_profile": profile}}

            elif command == "get_server_stats":
                response = {"status": "ok", "data": self._build_server_stats()}

            elif command == "shutdown":
                self._running = False
                response = {"status": "ok", "data": {"message": "shutting down"}}

            else:
                response = {"status": "error", "message": f"Unknown command: {command}"}

        except Exception as e:
            logger.exception("Error handling request")
            response = {"status": "error", "message": str(e), "error_code": "exception"}

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._record_request(command, response, elapsed_ms)
        return response

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
