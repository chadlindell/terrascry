"""HIRT borehole survey pipeline.

Simulates FDEM and ERT measurements for the HIRT dual-probe
borehole instrument. Two probes are inserted vertically into
the ground; the "survey" is a measurement sequence of coil
pairs (FDEM) at multiple frequencies plus electrode quadrupoles
(ERT). Output is two CSVs.

Key difference from Pathfinder: HIRT is borehole-based, not surface
walking. Probes remain stationary. The survey is a measurement
sequence, not a walk path.

Coordinate convention:
    X=East, Y=North, Z=Up (right-handed).
    Probe-local z: 0 at probe top (surface), positive downward.
    Global z: negative = underground.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from geosim.em.coil import ProbeCoilSet, hirt_default_coils
from geosim.em.fdem import (
    fdem_response_1d,
    secondary_field_conductive_sphere,
)
from geosim.em.skin_depth import skin_depth
from geosim.resistivity.electrodes import ElectrodeArray, hirt_default_electrodes
from geosim.resistivity.ert import ert_forward
from geosim.scenarios.loader import AnomalyZone, ProbeConfig

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FDEMMeasurement:
    """Single FDEM measurement specification.

    Parameters
    ----------
    tx_index : int
        Transmitter coil index within the ProbeCoilSet.
    rx_index : int
        Receiver coil index within the ProbeCoilSet.
    probe_pair : tuple[int, int]
        (tx_probe_index, rx_probe_index).
    frequency : float
        Operating frequency in Hz.
    tx_position : ndarray, shape (3,)
        Global [x, y, z] of transmitter coil.
    rx_position : ndarray, shape (3,)
        Global [x, y, z] of receiver coil.
    coil_separation : float
        Distance between TX and RX in meters.
    """

    tx_index: int
    rx_index: int
    probe_pair: tuple[int, int]
    frequency: float
    tx_position: np.ndarray
    rx_position: np.ndarray
    coil_separation: float


@dataclass
class ERTMeasurement:
    """Single ERT quadrupole measurement specification.

    Parameters
    ----------
    c1_index : int
        Current electrode 1 global index.
    c2_index : int
        Current electrode 2 global index.
    p1_index : int
        Potential electrode 1 global index.
    p2_index : int
        Potential electrode 2 global index.
    c1_position, c2_position, p1_position, p2_position : ndarray, shape (3,)
        Global electrode positions.
    c1_label, c2_label, p1_label, p2_label : str
        Electrode labels (e.g. 'A1', 'B3').
    """

    c1_index: int
    c2_index: int
    p1_index: int
    p2_index: int
    c1_position: np.ndarray
    c2_position: np.ndarray
    p1_position: np.ndarray
    p2_position: np.ndarray
    c1_label: str = ""
    c2_label: str = ""
    p1_label: str = ""
    p2_label: str = ""


@dataclass
class HIRTSurveyConfig:
    """HIRT instrument configuration for survey simulation.

    Parameters
    ----------
    frequencies : list[float]
        FDEM operating frequencies in Hz.
    coil_set : ProbeCoilSet
        Coil geometry definition.
    electrode_array : ElectrodeArray
        Electrode geometry definition (default geometry reference).
    fdem_noise_floor : float
        Base FDEM noise floor (dimensionless response units).
    ert_noise_floor : float
        Relative noise floor for ERT apparent resistivity.
    enable_fdem : bool
        Whether FDEM measurements are enabled.
    enable_ert : bool
        Whether ERT measurements are enabled.
    """

    frequencies: list[float] = field(
        default_factory=lambda: [1000.0, 5000.0, 25000.0]
    )
    coil_set: ProbeCoilSet | None = None
    electrode_array: ElectrodeArray | None = None
    fdem_noise_floor: float = 1e-4
    ert_noise_floor: float = 0.02
    enable_fdem: bool = True
    enable_ert: bool = True
    mit_tx_current_mA: float = 10.0
    ert_current_mA: float = 1.0
    mit_settle_ms: int = 10
    ert_settle_ms: int = 50
    adc_averaging: int = 16
    reciprocity_target_pct: float = 5.0

    def __post_init__(self):
        if self.coil_set is None:
            self.coil_set = hirt_default_coils()
        if self.electrode_array is None:
            self.electrode_array = hirt_default_electrodes()


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def probe_to_global(
    probe_local_z: float | np.ndarray,
    probe_config: ProbeConfig,
) -> np.ndarray:
    """Transform probe-local depth to global [x, y, z] coordinates.

    Parameters
    ----------
    probe_local_z : float or ndarray
        Depth from probe top (positive = downward into ground).
        0 = probe top (at surface), probe_length = probe bottom.
    probe_config : ProbeConfig
        Probe configuration with surface insertion position.

    Returns
    -------
    position : ndarray, shape (3,) or (N, 3)
        Global position(s). Z is negative underground.
    """
    pos = np.asarray(probe_config.position, dtype=np.float64)
    local_z = np.asarray(probe_local_z, dtype=np.float64)

    tilt_deg = float(getattr(probe_config, "tilt_deg", 0.0) or 0.0)
    azimuth_deg = float(getattr(probe_config, "azimuth_deg", 0.0) or 0.0)
    is_angled = str(getattr(probe_config, "orientation", "vertical")).lower() == "angled"
    if is_angled or abs(tilt_deg) > 1e-9:
        tilt = np.deg2rad(tilt_deg)
        az = np.deg2rad(azimuth_deg)
        # Depth direction vector, tilt measured from vertical, azimuth clockwise from North.
        depth_dir = np.array([
            np.sin(tilt) * np.sin(az),  # +X east
            np.sin(tilt) * np.cos(az),  # +Y north
            -np.cos(tilt),              # +depth is downward in global Z
        ], dtype=np.float64)
    else:
        depth_dir = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    if local_z.ndim == 0:
        return pos + depth_dir * float(local_z)

    n = len(local_z)
    result = np.empty((n, 3))
    result[:, 0] = pos[0] + depth_dir[0] * local_z
    result[:, 1] = pos[1] + depth_dir[1] * local_z
    result[:, 2] = pos[2] + depth_dir[2] * local_z
    return result


# ---------------------------------------------------------------------------
# Measurement generation
# ---------------------------------------------------------------------------

def generate_fdem_measurements(
    probe_configs: list[ProbeConfig],
    coil_set: ProbeCoilSet,
    frequencies: list[float],
    include_intra_probe: bool = True,
) -> list[FDEMMeasurement]:
    """Build ordered FDEM measurement sequence.

    Generates all TX-RX coil pair combinations for each probe
    (intra-probe) and across probes (cross-probe), at each frequency.

    For 2 probes with 3 TX + 3 RX coils and 3 frequencies:
    - Intra-probe: 2 * 9 * 3 = 54
    - Cross-probe: 2 * 9 * 3 = 54
    - Total: 108

    Parameters
    ----------
    probe_configs : list[ProbeConfig]
        Probe positions and geometry.
    coil_set : ProbeCoilSet
        Coil geometry (TX/RX definitions with positions from probe bottom).
    frequencies : list[float]
        Operating frequencies in Hz.
    include_intra_probe : bool
        Include TX/RX combinations within the same probe (legacy mode).

    Returns
    -------
    measurements : list[FDEMMeasurement]
        Ordered measurement sequence.
    """
    transmitters = coil_set.transmitters
    receivers = coil_set.receivers
    measurements: list[FDEMMeasurement] = []

    # Optional per-probe depth overrides from scenario hirt_config.probes[*].coil_depths.
    # Depth values are expected relative to probe top, matching ProbeConfig semantics.
    probe_depth_overrides: dict[int, dict[int, float]] = {}
    n_coils = len(coil_set.coils)
    for probe_idx, probe in enumerate(probe_configs):
        if not probe.coil_depths:
            continue
        if len(probe.coil_depths) != n_coils:
            raise ValueError(
                f"Probe {probe_idx} has {len(probe.coil_depths)} coil_depths, "
                f"but coil_set defines {n_coils} coils."
            )
        override: dict[int, float] = {}
        for coil_idx, depth_from_top in enumerate(probe.coil_depths):
            d = float(depth_from_top)
            if d < 0 or d > probe.length:
                raise ValueError(
                    f"Probe {probe_idx} coil depth {d}m is outside probe length {probe.length}m."
                )
            override[coil_idx] = d
        probe_depth_overrides[probe_idx] = override

    def _coil_global(coil, probe_idx: int, probe):
        """Convert coil position (from probe bottom) to global coords."""
        coil_idx = coil_set.coils.index(coil)
        if probe_idx in probe_depth_overrides:
            depth_from_top = probe_depth_overrides[probe_idx][coil_idx]
        else:
            depth_from_top = probe.length - coil.position
        return probe_to_global(depth_from_top, probe)

    # Intra-probe: TX and RX in the same probe (legacy mode).
    if include_intra_probe:
        for probe_idx, probe in enumerate(probe_configs):
            for tx in transmitters:
                tx_pos = _coil_global(tx, probe_idx, probe)
                for rx in receivers:
                    rx_pos = _coil_global(rx, probe_idx, probe)
                    separation = float(np.linalg.norm(tx_pos - rx_pos))
                    for freq in frequencies:
                        measurements.append(FDEMMeasurement(
                            tx_index=coil_set.coils.index(tx),
                            rx_index=coil_set.coils.index(rx),
                            probe_pair=(probe_idx, probe_idx),
                            frequency=freq,
                            tx_position=tx_pos.copy(),
                            rx_position=rx_pos.copy(),
                            coil_separation=separation,
                        ))

    # Cross-probe: TX in one probe, RX in another
    n_probes = len(probe_configs)
    for pi in range(n_probes):
        for pj in range(n_probes):
            if pi == pj:
                continue
            probe_tx = probe_configs[pi]
            probe_rx = probe_configs[pj]
            for tx in transmitters:
                tx_pos = _coil_global(tx, pi, probe_tx)
                for rx in receivers:
                    rx_pos = _coil_global(rx, pj, probe_rx)
                    separation = float(np.linalg.norm(tx_pos - rx_pos))
                    for freq in frequencies:
                        measurements.append(FDEMMeasurement(
                            tx_index=coil_set.coils.index(tx),
                            rx_index=coil_set.coils.index(rx),
                            probe_pair=(pi, pj),
                            frequency=freq,
                            tx_position=tx_pos.copy(),
                            rx_position=rx_pos.copy(),
                            coil_separation=separation,
                        ))

    return measurements


def generate_ert_measurements(
    probe_configs: list[ProbeConfig],
    electrode_array: ElectrodeArray | None = None,
    array_type: str = "crosshole",
) -> list[ERTMeasurement]:
    """Build ordered ERT quadrupole measurement sequence.

    For cross-hole arrays, generates bipole-bipole measurements with
    adjacent electrode pairs: current pair from one borehole, potential
    pair from the other, then swap directions.

    For 2 probes with 8 ring electrodes each:
    - 7 adjacent pairs per probe
    - Direction 1: 7 * 7 = 49
    - Direction 2: 7 * 7 = 49
    - Total: 98

    Parameters
    ----------
    probe_configs : list[ProbeConfig]
        Probe positions with ring_depths.
    electrode_array : ElectrodeArray, optional
        Electrode geometry reference (not used for positioning;
        positions derived from probe configs' ring_depths).
    array_type : str
        Array type ('crosshole' supported).

    Returns
    -------
    measurements : list[ERTMeasurement]
        Ordered measurement sequence.
    """
    # Build electrode list from probe ring depths
    electrodes_by_probe: dict[int, list[dict]] = {}
    global_index = 0

    for probe_idx, probe in enumerate(probe_configs):
        electrodes_by_probe[probe_idx] = []
        prefix = chr(ord('A') + probe_idx)
        for ring_idx, depth in enumerate(probe.ring_depths):
            global_pos = probe_to_global(depth, probe)
            electrodes_by_probe[probe_idx].append({
                'position': global_pos,
                'label': f"{prefix}{ring_idx + 1}",
                'global_index': global_index,
            })
            global_index += 1

    measurements: list[ERTMeasurement] = []

    if array_type == "crosshole":
        if len(probe_configs) < 2:
            return measurements

        # Ordered probe pairs implement the TX/RX style matrix: i -> j for i != j.
        for c_probe_idx, elecs in electrodes_by_probe.items():
            c_pairs = [(elecs[i], elecs[i + 1]) for i in range(len(elecs) - 1)]
            if not c_pairs:
                continue

            for p_probe_idx, other_elecs in electrodes_by_probe.items():
                if p_probe_idx == c_probe_idx:
                    continue
                p_pairs = [
                    (other_elecs[i], other_elecs[i + 1])
                    for i in range(len(other_elecs) - 1)
                ]
                if not p_pairs:
                    continue

                for c1, c2 in c_pairs:
                    for p1, p2 in p_pairs:
                        measurements.append(ERTMeasurement(
                            c1_index=c1['global_index'],
                            c2_index=c2['global_index'],
                            p1_index=p1['global_index'],
                            p2_index=p2['global_index'],
                            c1_position=c1['position'].copy(),
                            c2_position=c2['position'].copy(),
                            p1_position=p1['position'].copy(),
                            p2_position=p2['position'].copy(),
                            c1_label=c1['label'],
                            c2_label=c2['label'],
                            p1_label=p1['label'],
                            p2_label=p2['label'],
                        ))
    elif array_type == "wenner":
        # Local single-borehole Wenner quadruples:
        # C1 - P1 - P2 - C2 with equal spacing a.
        for elecs in electrodes_by_probe.values():
            n = len(elecs)
            for a in range(1, max(1, n // 3 + 1)):
                max_start = n - 3 * a
                for i in range(max_start):
                    c1 = elecs[i]
                    p1 = elecs[i + a]
                    p2 = elecs[i + 2 * a]
                    c2 = elecs[i + 3 * a]
                    measurements.append(ERTMeasurement(
                        c1_index=c1['global_index'],
                        c2_index=c2['global_index'],
                        p1_index=p1['global_index'],
                        p2_index=p2['global_index'],
                        c1_position=c1['position'].copy(),
                        c2_position=c2['position'].copy(),
                        p1_position=p1['position'].copy(),
                        p2_position=p2['position'].copy(),
                        c1_label=c1['label'],
                        c2_label=c2['label'],
                        p1_label=p1['label'],
                        p2_label=p2['label'],
                    ))
    elif array_type == "dipole-dipole":
        # Local single-borehole dipole-dipole:
        # C1-C2 is one dipole (length a), P1-P2 is a second dipole with separation n*a.
        for elecs in electrodes_by_probe.values():
            n = len(elecs)
            for a in range(1, max(1, n // 4 + 1)):
                for n_sep in range(1, 4):
                    max_start = n - (n_sep + 2) * a
                    for i in range(max_start):
                        c1 = elecs[i]
                        c2 = elecs[i + a]
                        p1 = elecs[i + (n_sep + 1) * a]
                        p2 = elecs[i + (n_sep + 2) * a]
                        measurements.append(ERTMeasurement(
                            c1_index=c1['global_index'],
                            c2_index=c2['global_index'],
                            p1_index=p1['global_index'],
                            p2_index=p2['global_index'],
                            c1_position=c1['position'].copy(),
                            c2_position=c2['position'].copy(),
                            p1_position=p1['position'].copy(),
                            p2_position=p2['position'].copy(),
                            c1_label=c1['label'],
                            c2_label=c2['label'],
                            p1_label=p1['label'],
                            p2_label=p2['label'],
                        ))
    else:
        raise ValueError(
            f"Unsupported ERT array_type '{array_type}'. "
            "Supported: crosshole, wenner, dipole-dipole."
        )

    return measurements


# ---------------------------------------------------------------------------
# FDEM simulation
# ---------------------------------------------------------------------------

def simulate_fdem(
    measurements: list[FDEMMeasurement],
    conductivity_model: dict,
    em_sources: list[dict],
    anomaly_zones: list[AnomalyZone] | None,
    config: HIRTSurveyConfig,
    rng: np.random.Generator | None = None,
    add_noise: bool = True,
) -> dict:
    """Simulate FDEM responses for all measurements.

    For each measurement:
    1. Compute 1D background via fdem_response_1d
    2. Superpose conductive sphere responses from em_sources
    3. Compute skin depth
    4. Apply frequency-dependent noise

    Parameters
    ----------
    measurements : list[FDEMMeasurement]
        Measurement sequence from generate_fdem_measurements.
    conductivity_model : dict
        Keys: 'thicknesses' (list[float]), 'conductivities' (list[float]).
    em_sources : list[dict]
        Conductive objects: each has 'position', 'radius', 'conductivity'.
    config : HIRTSurveyConfig
        Instrument configuration.
    rng : Generator, optional
        Random number generator.
    add_noise : bool
        Whether to add measurement noise.

    Returns
    -------
    data : dict
        FDEM survey data with keys matching CSV columns:
        measurement_id, probe_pair, tx_depth, rx_depth,
        coil_separation, frequency, response_real, response_imag,
        skin_depth.
    """
    if rng is None:
        rng = np.random.default_rng()

    thicknesses = conductivity_model.get('thicknesses', [])
    conductivities = conductivity_model.get('conductivities', [0.01])
    avg_sigma = float(np.mean(conductivities)) if conductivities else 0.01

    data: dict[str, list] = {
        'measurement_id': [],
        'probe_pair': [],
        'tx_depth': [],
        'rx_depth': [],
        'coil_separation': [],
        'frequency': [],
        'response_real': [],
        'response_imag': [],
        'skin_depth': [],
    }

    zones = anomaly_zones or []

    def _zone_effective_radius(zone: AnomalyZone) -> float:
        dims = zone.dimensions or {}
        if zone.shape == "sphere":
            return float(dims.get("radius", 0.5))
        if zone.shape == "cylinder":
            r = float(dims.get("radius", 0.5))
            h = float(dims.get("height", 1.0))
            return float((r * r * h) ** (1.0 / 3.0))
        if zone.shape == "box":
            l = float(dims.get("length", 1.0))
            w = float(dims.get("width", 1.0))
            d = float(dims.get("depth", 1.0))
            return float((l * w * d / (4.0 / 3.0 * np.pi)) ** (1.0 / 3.0))
        return 0.5

    for i, meas in enumerate(measurements):
        # 1D layered-earth background
        response = fdem_response_1d(
            thicknesses, conductivities,
            meas.frequency, meas.coil_separation,
        )

        # Superpose conductive sphere contributions
        midpoint = (meas.tx_position + meas.rx_position) / 2.0
        for src in em_sources:
            src_pos = np.asarray(src['position'], dtype=np.float64)
            r_obs = float(np.linalg.norm(src_pos - midpoint))
            radius = src['radius']
            if r_obs > radius:
                sphere_resp = secondary_field_conductive_sphere(
                    radius, src['conductivity'], meas.frequency, r_obs,
                )
                response += sphere_resp

        # Add conductive anomaly zones as effective bodies.
        for zone in zones:
            sigma_zone = float(zone.conductivity)
            if sigma_zone <= 0:
                continue
            center = np.asarray(zone.center, dtype=np.float64)
            radius = max(_zone_effective_radius(zone), 0.05)
            r_obs = float(np.linalg.norm(center - midpoint))
            if r_obs <= radius:
                r_obs = radius * 1.01
            zone_resp = secondary_field_conductive_sphere(
                radius, sigma_zone, meas.frequency, r_obs
            )
            response += zone_resp

        # Skin depth at this frequency
        sd = float(skin_depth(meas.frequency, avg_sigma))

        # Frequency-dependent noise: scale with sqrt(freq/1000)
        if add_noise:
            noise_scale = config.fdem_noise_floor * np.sqrt(
                meas.frequency / 1000.0
            )
            response += noise_scale * (
                rng.standard_normal() + 1j * rng.standard_normal()
            )

        data['measurement_id'].append(i)
        data['probe_pair'].append(
            f"{meas.probe_pair[0]}-{meas.probe_pair[1]}"
        )
        data['tx_depth'].append(float(-meas.tx_position[2]))
        data['rx_depth'].append(float(-meas.rx_position[2]))
        data['coil_separation'].append(meas.coil_separation)
        data['frequency'].append(meas.frequency)
        data['response_real'].append(float(np.real(response)))
        data['response_imag'].append(float(np.imag(response)))
        data['skin_depth'].append(sd)

    return data


# ---------------------------------------------------------------------------
# ERT simulation
# ---------------------------------------------------------------------------

def simulate_ert(
    measurements: list[ERTMeasurement],
    resistivity_model: dict,
    anomaly_zones: list[AnomalyZone] | None,
    config: HIRTSurveyConfig,
    rng: np.random.Generator | None = None,
    add_noise: bool = True,
) -> dict:
    """Simulate ERT responses for all measurements.

    1. Build electrode position array and measurement tuples
    2. Call ert_forward (analytical backend)
    3. Apply relative Gaussian noise to apparent resistivity

    Parameters
    ----------
    measurements : list[ERTMeasurement]
        Measurement sequence from generate_ert_measurements.
    resistivity_model : dict
        Keys: 'thicknesses' (list[float]), 'resistivities' (list[float]).
    config : HIRTSurveyConfig
        Instrument configuration.
    rng : Generator, optional
        Random number generator.
    add_noise : bool
        Whether to add measurement noise.

    Returns
    -------
    data : dict
        ERT survey data with keys matching CSV columns:
        measurement_id, c1_electrode, c2_electrode, p1_electrode,
        p2_electrode, geometric_factor, apparent_resistivity.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Build unique electrode position array and index mapping
    pos_map: dict[tuple, int] = {}
    all_positions: list[np.ndarray] = []
    meas_tuples: list[tuple[int, int, int, int]] = []

    for meas in measurements:
        indices = []
        for pos in [meas.c1_position, meas.c2_position,
                     meas.p1_position, meas.p2_position]:
            key = tuple(pos.tolist())
            if key not in pos_map:
                pos_map[key] = len(all_positions)
                all_positions.append(pos)
            indices.append(pos_map[key])
        meas_tuples.append((indices[0], indices[1], indices[2], indices[3]))

    positions = np.array(all_positions)
    resistivities = resistivity_model.get('resistivities', [100.0])
    thicknesses = resistivity_model.get('thicknesses', [])
    thick_arg = thicknesses if thicknesses else None

    result = ert_forward(
        positions, meas_tuples, resistivities, thick_arg,
        backend='analytical',
    )

    rho_a = np.array(result['apparent_resistivity'], dtype=np.float64)
    k_factors = np.array(result['geometric_factors'], dtype=np.float64)

    if add_noise:
        rho_a *= 1.0 + config.ert_noise_floor * rng.standard_normal(len(rho_a))

    # Approximate 3D anomaly influence by perturbing 1D rho_a with distance-
    # weighted contrast terms. This preserves analytical speed while making
    # anomaly_zones materially affect synthetic data.
    zones = anomaly_zones or []
    if zones:
        background_rho = float(np.median(resistivities)) if resistivities else 100.0
        for i, meas in enumerate(measurements):
            midpoint = (
                np.asarray(meas.c1_position)
                + np.asarray(meas.c2_position)
                + np.asarray(meas.p1_position)
                + np.asarray(meas.p2_position)
            ) / 4.0

            perturb = 0.0
            for zone in zones:
                center = np.asarray(zone.center, dtype=np.float64)
                dist = float(np.linalg.norm(midpoint - center))

                dims = zone.dimensions or {}
                if zone.shape == "sphere":
                    radius = float(dims.get("radius", 0.6))
                elif zone.shape == "cylinder":
                    r = float(dims.get("radius", 0.6))
                    h = float(dims.get("height", 1.2))
                    radius = float((r * r * h) ** (1.0 / 3.0))
                else:
                    l = float(dims.get("length", 1.2))
                    w = float(dims.get("width", 1.2))
                    d = float(dims.get("depth", 1.2))
                    radius = float((l * w * d / (4.0 / 3.0 * np.pi)) ** (1.0 / 3.0))
                radius = max(radius, 0.1)

                if zone.resistivity > 0:
                    zone_rho = float(zone.resistivity)
                elif zone.conductivity > 0:
                    zone_rho = 1.0 / float(zone.conductivity)
                else:
                    continue

                contrast = (zone_rho - background_rho) / max(background_rho, 1e-6)
                influence = np.exp(-((dist / radius) ** 2))
                perturb += 0.35 * contrast * influence

            rho_a[i] *= max(0.05, 1.0 + perturb)

    data = {
        'measurement_id': list(range(len(measurements))),
        'c1_electrode': [m.c1_label for m in measurements],
        'c2_electrode': [m.c2_label for m in measurements],
        'p1_electrode': [m.p1_label for m in measurements],
        'p2_electrode': [m.p2_label for m in measurements],
        'geometric_factor': k_factors.tolist(),
        'apparent_resistivity': rho_a.tolist(),
    }

    return data


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

FDEM_COLUMNS = [
    'measurement_id', 'probe_pair', 'tx_depth', 'rx_depth',
    'coil_separation', 'frequency', 'response_real', 'response_imag',
    'skin_depth',
]

ERT_COLUMNS = [
    'measurement_id', 'c1_electrode', 'c2_electrode',
    'p1_electrode', 'p2_electrode', 'geometric_factor',
    'apparent_resistivity',
]

HIRT_MIT_COLUMNS = [
    'timestamp', 'section_id', 'zone_id', 'tx_probe_id', 'rx_probe_id',
    'freq_hz', 'amp', 'phase_deg', 'tx_current_mA',
]

HIRT_ERT_COLUMNS = [
    'timestamp', 'section_id', 'zone_id', 'inject_pos_id', 'inject_neg_id',
    'sense_id', 'volt_mV', 'current_mA', 'polarity', 'notes',
]


def export_fdem_csv(data: dict, path: str | Path) -> None:
    """Write FDEM survey data to CSV.

    Parameters
    ----------
    data : dict
        FDEM data from simulate_fdem().
    path : str or Path
        Output CSV file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = len(data['measurement_id'])
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(FDEM_COLUMNS)
        for i in range(n_rows):
            row = []
            for col in FDEM_COLUMNS:
                val = data[col][i]
                if isinstance(val, float):
                    row.append(f'{val:.8e}')
                else:
                    row.append(val)
            writer.writerow(row)


def export_ert_csv(data: dict, path: str | Path) -> None:
    """Write ERT survey data to CSV.

    Parameters
    ----------
    data : dict
        ERT data from simulate_ert().
    path : str or Path
        Output CSV file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = len(data['measurement_id'])
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(ERT_COLUMNS)
        for i in range(n_rows):
            row = []
            for col in ERT_COLUMNS:
                val = data[col][i]
                if isinstance(val, float):
                    row.append(f'{val:.8e}')
                else:
                    row.append(val)
            writer.writerow(row)


def export_hirt_mit_csv(
    data: dict,
    path: str | Path,
    section_id: str = "S01",
    zone_id: str = "ZA",
    start_time: datetime | None = None,
    tx_current_mA: float = 10.0,
    measurement_interval_s: float = 1.0,
) -> None:
    """Write MIT records in HIRT data-format style."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if start_time is None:
        start_time = datetime.now(timezone.utc)

    n_rows = len(data['measurement_id'])
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(HIRT_MIT_COLUMNS)
        for i in range(n_rows):
            timestamp = (start_time + timedelta(seconds=i * measurement_interval_s)).isoformat()
            probe_pair = str(data['probe_pair'][i]).split("-")
            tx_probe = f"P{int(probe_pair[0]) + 1:02d}" if len(probe_pair) >= 1 else "P01"
            rx_probe = f"P{int(probe_pair[1]) + 1:02d}" if len(probe_pair) >= 2 else "P02"
            real = float(data['response_real'][i])
            imag = float(data['response_imag'][i])
            amp = float(np.hypot(real, imag))
            phase_deg = float(np.degrees(np.arctan2(imag, real)))
            writer.writerow([
                timestamp,
                section_id,
                zone_id,
                tx_probe,
                rx_probe,
                float(data['frequency'][i]),
                f"{amp:.8e}",
                f"{phase_deg:.6f}",
                f"{tx_current_mA:.3f}",
            ])


def export_hirt_ert_csv(
    data: dict,
    path: str | Path,
    section_id: str = "S01",
    zone_id: str = "ZA",
    start_time: datetime | None = None,
    current_mA: float = 1.0,
    measurement_interval_s: float = 1.0,
    include_polarity_reversal: bool = True,
) -> None:
    """Write ERT records in HIRT data-format style."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if start_time is None:
        start_time = datetime.now(timezone.utc)

    n_rows = len(data['measurement_id'])
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(HIRT_ERT_COLUMNS)
        row_idx = 0
        for i in range(n_rows):
            timestamp = (start_time + timedelta(seconds=row_idx * measurement_interval_s)).isoformat()
            rho_a = float(data['apparent_resistivity'][i])
            k = float(data['geometric_factor'][i])
            current_a = max(current_mA / 1000.0, 1e-9)
            volt_mV = (rho_a / max(k, 1e-9)) * current_a * 1000.0
            writer.writerow([
                timestamp,
                section_id,
                zone_id,
                data['c1_electrode'][i],
                data['c2_electrode'][i],
                data['p1_electrode'][i],
                f"{volt_mV:.6f}",
                f"{current_mA:.3f}",
                1,
                "",
            ])
            row_idx += 1

            if include_polarity_reversal:
                timestamp = (
                    start_time + timedelta(seconds=row_idx * measurement_interval_s)
                ).isoformat()
                writer.writerow([
                    timestamp,
                    section_id,
                    zone_id,
                    data['c1_electrode'][i],
                    data['c2_electrode'][i],
                    data['p1_electrode'][i],
                    f"{-volt_mV:.6f}",
                    f"{current_mA:.3f}",
                    -1,
                    "reversed polarity",
                ])
                row_idx += 1


# ---------------------------------------------------------------------------
# QC and sidecar exports
# ---------------------------------------------------------------------------

def _measurement_interval_s(config: HIRTSurveyConfig, mode: str) -> float:
    """Estimate acquisition interval from settle times and averaging settings."""
    avg = max(int(config.adc_averaging), 1)
    if mode == "mit":
        return max(0.02, (float(config.mit_settle_ms) + 4.0 * avg) / 1000.0)
    return max(0.03, (float(config.ert_settle_ms) + 6.0 * avg) / 1000.0)


def _fdem_reciprocity_errors_pct(data: dict) -> list[float]:
    """Compute reciprocity errors (%) from reciprocal probe-pair measurements."""
    by_key: dict[tuple, list[float]] = {}
    n_rows = len(data.get("measurement_id", []))
    for i in range(n_rows):
        pair_raw = str(data["probe_pair"][i]).split("-")
        if len(pair_raw) != 2:
            continue
        a = int(pair_raw[0])
        b = int(pair_raw[1])
        if a == b:
            continue
        freq = round(float(data["frequency"][i]), 3)
        tx_depth = round(float(data["tx_depth"][i]), 3)
        rx_depth = round(float(data["rx_depth"][i]), 3)
        depth_key = (min(tx_depth, rx_depth), max(tx_depth, rx_depth))
        pair_key = (min(a, b), max(a, b))
        key = (pair_key, depth_key, freq)
        amp = float(np.hypot(float(data["response_real"][i]), float(data["response_imag"][i])))
        by_key.setdefault(key, []).append(amp)

    errors: list[float] = []
    for vals in by_key.values():
        if len(vals) < 2:
            continue
        m1 = float(vals[0])
        m2 = float(vals[1])
        denom = max((abs(m1) + abs(m2)) * 0.5, 1e-12)
        errors.append(abs(m1 - m2) / denom * 100.0)
    return errors


def _ert_reciprocity_errors_pct(data: dict) -> list[float]:
    """Compute reciprocity errors (%) from reciprocal ERT quadrupoles."""
    by_key: dict[tuple, list[float]] = {}
    n_rows = len(data.get("measurement_id", []))
    for i in range(n_rows):
        c_pair = tuple(sorted((str(data["c1_electrode"][i]), str(data["c2_electrode"][i]))))
        p_pair = tuple(sorted((str(data["p1_electrode"][i]), str(data["p2_electrode"][i]))))
        key = tuple(sorted((c_pair, p_pair)))
        by_key.setdefault(key, []).append(float(data["apparent_resistivity"][i]))

    errors: list[float] = []
    for vals in by_key.values():
        if len(vals) < 2:
            continue
        r1 = float(vals[0])
        r2 = float(vals[1])
        denom = max((abs(r1) + abs(r2)) * 0.5, 1e-12)
        errors.append(abs(r1 - r2) / denom * 100.0)
    return errors


def build_hirt_qc_summary(
    fdem_data: dict | None,
    ert_data: dict | None,
    config: HIRTSurveyConfig,
) -> dict:
    """Build survey-level QC summary for inversion and ML pipelines."""
    summary: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "targets": {
            "reciprocity_pct": float(config.reciprocity_target_pct),
        },
    }

    if fdem_data is not None:
        errs = _fdem_reciprocity_errors_pct(fdem_data)
        amp = np.hypot(
            np.asarray(fdem_data.get("response_real", []), dtype=np.float64),
            np.asarray(fdem_data.get("response_imag", []), dtype=np.float64),
        )
        summary["mit"] = {
            "measurement_count": int(len(fdem_data.get("measurement_id", []))),
            "mean_amplitude": float(np.mean(amp)) if amp.size else 0.0,
            "p95_amplitude": float(np.percentile(amp, 95)) if amp.size else 0.0,
            "reciprocity": {
                "paired_count": int(len(errs)),
                "mean_error_pct": float(np.mean(errs)) if errs else 0.0,
                "max_error_pct": float(np.max(errs)) if errs else 0.0,
                "pass_rate": float(np.mean(np.array(errs) <= config.reciprocity_target_pct)) if errs else 1.0,
            },
        }

    if ert_data is not None:
        errs = _ert_reciprocity_errors_pct(ert_data)
        rho = np.asarray(ert_data.get("apparent_resistivity", []), dtype=np.float64)
        summary["ert"] = {
            "measurement_count": int(len(ert_data.get("measurement_id", []))),
            "mean_rho_a_ohm_m": float(np.mean(rho)) if rho.size else 0.0,
            "p95_rho_a_ohm_m": float(np.percentile(rho, 95)) if rho.size else 0.0,
            "reciprocity": {
                "paired_count": int(len(errs)),
                "mean_error_pct": float(np.mean(errs)) if errs else 0.0,
                "max_error_pct": float(np.max(errs)) if errs else 0.0,
                "pass_rate": float(np.mean(np.array(errs) <= config.reciprocity_target_pct)) if errs else 1.0,
            },
        }

    return summary


def export_probe_registry_csv(
    probe_configs: list[ProbeConfig],
    path: str | Path,
    firmware_rev: str = "v1.0-sim",
    calibration_date: str | None = None,
) -> None:
    """Export HIRT-style probe registry CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if calibration_date is None:
        calibration_date = datetime.now(timezone.utc).date().isoformat()

    columns = [
        "probe_id", "coil_L_mH", "coil_Q", "rx_gain_dB", "ring_depths_m",
        "firmware_rev", "calibration_date", "notes",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for i, probe in enumerate(probe_configs):
            probe_id = f"P{i + 1:02d}"
            coil_l = 1.45 + 0.03 * (i % 3)
            coil_q = 28 + (i % 4)
            rx_gain = 40.0
            ring_depths = ";".join(f"{float(d):.2f}" for d in probe.ring_depths)
            notes = "tilted" if abs(float(getattr(probe, "tilt_deg", 0.0) or 0.0)) > 0.1 else ""
            writer.writerow([
                probe_id,
                f"{coil_l:.3f}",
                f"{coil_q:.1f}",
                f"{rx_gain:.1f}",
                ring_depths,
                firmware_rev,
                calibration_date,
                notes,
            ])


def export_survey_geometry_csv(
    probe_configs: list[ProbeConfig],
    path: str | Path,
) -> None:
    """Export HIRT survey geometry CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "probe_id", "x_m", "y_m", "z_surface_m", "z_tip_m", "status", "notes",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for i, probe in enumerate(probe_configs):
            pos = np.asarray(probe.position, dtype=np.float64)
            tip = probe_to_global(float(probe.length), probe)
            status = "ACTIVE"
            notes = ""
            tilt = float(getattr(probe, "tilt_deg", 0.0) or 0.0)
            if abs(tilt) > 0.1:
                notes = f"tilt={tilt:.1f}deg"
            writer.writerow([
                f"P{i + 1:02d}",
                f"{float(pos[0]):.3f}",
                f"{float(pos[1]):.3f}",
                f"{float(pos[2]):.3f}",
                f"{float(tip[2]):.3f}",
                status,
                notes,
            ])


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run_hirt_survey(
    scenario_path: str | Path,
    output_dir: str | Path,
    fdem_enabled: bool = True,
    ert_enabled: bool = True,
    add_noise: bool = True,
    seed: int | None = None,
) -> dict:
    """Run a complete HIRT survey simulation from a scenario file.

    Chains: scenario -> measurement generation -> physics -> noise -> CSV.

    Parameters
    ----------
    scenario_path : str or Path
        Path to scenario JSON file (must contain hirt_config).
    output_dir : str or Path
        Directory for output CSV files.
    fdem_enabled : bool
        Run FDEM simulation.
    ert_enabled : bool
        Run ERT simulation.
    add_noise : bool
        Add realistic noise to measurements.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    result : dict
        Keys: 'fdem' (data dict), 'ert' (data dict),
        'fdem_csv' (path str), 'ert_csv' (path str).

    Raises
    ------
    ValueError
        If the scenario has no hirt_config.
    """
    from geosim.scenarios.loader import load_scenario

    scenario = load_scenario(scenario_path)
    if scenario.hirt_config is None:
        raise ValueError(
            f"Scenario '{scenario.name}' has no hirt_config. "
            "HIRT survey requires probe positions and frequencies."
        )

    hirt = scenario.hirt_config
    rng = np.random.default_rng(seed)
    config = HIRTSurveyConfig(
        frequencies=hirt.frequencies,
        mit_tx_current_mA=float(hirt.mit_tx_current_mA),
        ert_current_mA=float(hirt.ert_current_mA),
        mit_settle_ms=int(hirt.mit_settle_ms),
        ert_settle_ms=int(hirt.ert_settle_ms),
        adc_averaging=int(hirt.adc_averaging),
        reciprocity_target_pct=float(hirt.reciprocity_qc_pct),
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {}

    if fdem_enabled:
        fdem_meas = generate_fdem_measurements(
            hirt.probes, config.coil_set, config.frequencies,
            include_intra_probe=bool(hirt.include_intra_probe),
        )

        # Build 1D conductivity model from terrain layers
        layers = scenario.terrain.layers
        if layers:
            thicknesses = [
                abs(layer.z_top - layer.z_bottom)
                for layer in layers[:-1]
            ]
            conductivities = [layer.conductivity for layer in layers]
        else:
            thicknesses = []
            conductivities = [0.01]

        cond_model = {
            'thicknesses': thicknesses,
            'conductivities': conductivities,
        }

        fdem_data = simulate_fdem(
            fdem_meas, cond_model, scenario.em_sources,
            scenario.anomaly_zones, config, rng, add_noise,
        )

        fdem_path = output_dir / 'hirt_fdem.csv'
        export_fdem_csv(fdem_data, fdem_path)
        export_hirt_mit_csv(
            fdem_data,
            output_dir / 'hirt_mit_records.csv',
            section_id=hirt.section_id,
            zone_id=hirt.zone_id,
            tx_current_mA=config.mit_tx_current_mA,
            measurement_interval_s=_measurement_interval_s(config, "mit"),
        )
        result['fdem'] = fdem_data
        result['fdem_csv'] = str(fdem_path)
        result['mit_records_csv'] = str(output_dir / 'hirt_mit_records.csv')

    if ert_enabled:
        ert_meas = generate_ert_measurements(
            hirt.probes, config.electrode_array, hirt.array_type,
        )

        ert_data = simulate_ert(
            ert_meas, scenario.resistivity_model,
            scenario.anomaly_zones, config, rng, add_noise,
        )

        ert_path = output_dir / 'hirt_ert.csv'
        export_ert_csv(ert_data, ert_path)
        export_hirt_ert_csv(
            ert_data,
            output_dir / 'hirt_ert_records.csv',
            section_id=hirt.section_id,
            zone_id=hirt.zone_id,
            current_mA=config.ert_current_mA,
            measurement_interval_s=_measurement_interval_s(config, "ert"),
        )
        result['ert'] = ert_data
        result['ert_csv'] = str(ert_path)
        result['ert_records_csv'] = str(output_dir / 'hirt_ert_records.csv')

    # Sidecar exports for downstream reconstruction/analytics pipelines.
    export_survey_geometry_csv(hirt.probes, output_dir / "survey_geometry.csv")
    export_probe_registry_csv(
        hirt.probes,
        output_dir / "probe_registry.csv",
        firmware_rev=str(scenario.metadata.get("hirt_firmware_rev", "v1.0-sim")),
        calibration_date=str(
            scenario.metadata.get("calibration_date", datetime.now(timezone.utc).date().isoformat())
        ),
    )
    qc = build_hirt_qc_summary(result.get("fdem"), result.get("ert"), config)
    with open(output_dir / "hirt_qc_summary.json", "w") as f:
        json.dump(qc, f, indent=2)
    result["survey_geometry_csv"] = str(output_dir / "survey_geometry.csv")
    result["probe_registry_csv"] = str(output_dir / "probe_registry.csv")
    result["qc_summary_json"] = str(output_dir / "hirt_qc_summary.json")

    return result
