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
from dataclasses import dataclass, field
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
from geosim.scenarios.loader import ProbeConfig

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

    if local_z.ndim == 0:
        return np.array([pos[0], pos[1], pos[2] - float(local_z)])

    n = len(local_z)
    result = np.empty((n, 3))
    result[:, 0] = pos[0]
    result[:, 1] = pos[1]
    result[:, 2] = pos[2] - local_z
    return result


# ---------------------------------------------------------------------------
# Measurement generation
# ---------------------------------------------------------------------------

def generate_fdem_measurements(
    probe_configs: list[ProbeConfig],
    coil_set: ProbeCoilSet,
    frequencies: list[float],
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

    Returns
    -------
    measurements : list[FDEMMeasurement]
        Ordered measurement sequence.
    """
    transmitters = coil_set.transmitters
    receivers = coil_set.receivers
    measurements: list[FDEMMeasurement] = []

    def _coil_global(coil, probe):
        """Convert coil position (from probe bottom) to global coords."""
        depth_from_top = probe.length - coil.position
        return probe_to_global(depth_from_top, probe)

    # Intra-probe: TX and RX in the same probe
    for probe_idx, probe in enumerate(probe_configs):
        for tx in transmitters:
            tx_pos = _coil_global(tx, probe)
            for rx in receivers:
                rx_pos = _coil_global(rx, probe)
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
                tx_pos = _coil_global(tx, probe_tx)
                for rx in receivers:
                    rx_pos = _coil_global(rx, probe_rx)
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

    if array_type == "crosshole" and len(probe_configs) >= 2:
        for probe_idx in range(2):
            elecs = electrodes_by_probe[probe_idx]
            other_idx = 1 - probe_idx
            other_elecs = electrodes_by_probe[other_idx]

            # Adjacent pairs in each borehole
            c_pairs = [
                (elecs[i], elecs[i + 1])
                for i in range(len(elecs) - 1)
            ]
            p_pairs = [
                (other_elecs[i], other_elecs[i + 1])
                for i in range(len(other_elecs) - 1)
            ]

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

    return measurements


# ---------------------------------------------------------------------------
# FDEM simulation
# ---------------------------------------------------------------------------

def simulate_fdem(
    measurements: list[FDEMMeasurement],
    conductivity_model: dict,
    em_sources: list[dict],
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
    config = HIRTSurveyConfig(frequencies=hirt.frequencies)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {}

    if fdem_enabled:
        fdem_meas = generate_fdem_measurements(
            hirt.probes, config.coil_set, config.frequencies,
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
            config, rng, add_noise,
        )

        fdem_path = output_dir / 'hirt_fdem.csv'
        export_fdem_csv(fdem_data, fdem_path)
        result['fdem'] = fdem_data
        result['fdem_csv'] = str(fdem_path)

    if ert_enabled:
        ert_meas = generate_ert_measurements(
            hirt.probes, config.electrode_array, hirt.array_type,
        )

        ert_data = simulate_ert(
            ert_meas, scenario.resistivity_model,
            config, rng, add_noise,
        )

        ert_path = output_dir / 'hirt_ert.csv'
        export_ert_csv(ert_data, ert_path)
        result['ert'] = ert_data
        result['ert_csv'] = str(ert_path)

    return result
