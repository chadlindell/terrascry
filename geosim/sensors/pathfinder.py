"""Pathfinder gradiometer instrument model.

Simulates the 4-pair fluxgate gradiometer in "trapeze" configuration.
Produces CSV output matching the Pathfinder firmware format exactly.

Physical configuration (from Pathfinder design-concept.md):
    - 4 sensor pairs on horizontal carbon fiber bar
    - 50 cm horizontal spacing between pairs
    - Bottom sensors: ~17.5 cm above ground (midpoint of 15-20 cm range)
    - Top sensors: ~52.5 cm above ground (35 cm baseline)
    - Sensor separation (baseline): 35 cm
    - Total swath: 1.5 m

Coordinate convention:
    Sensor array centered on operator position.
    X=East, Y=North, Z=Up.
    Pair 1 is leftmost (west), Pair 4 is rightmost (east)
    when walking North.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from geosim.magnetics.dipole import gradiometer_reading, superposition_field
from geosim.noise.models import NoiseModel, pathfinder_noise_model


# Default sensor geometry matching Pathfinder config.h and design-concept.md
DEFAULT_NUM_PAIRS = 4
DEFAULT_PAIR_SPACING = 0.50  # 50 cm horizontal spacing
DEFAULT_SENSOR_SEPARATION = 0.35  # 35 cm vertical baseline
DEFAULT_BOTTOM_HEIGHT = 0.175  # 17.5 cm above ground
DEFAULT_SAMPLE_RATE = 10.0  # Hz (config.h: SAMPLE_RATE_HZ)

# ADC conversion: ADS1115 at GAIN_ONE (±4.096V), 16-bit signed
# Fluxgate sensitivity assumed ~50 µV/nT, ADC LSB = 0.125 mV
# → ~2.5 ADC counts per nT, or ~400 pT per count
ADC_COUNTS_PER_TESLA = 2.5e9  # 2.5 counts/nT = 2.5e9 counts/T
ADC_SATURATION = 32000  # from config.h


@dataclass
class PathfinderConfig:
    """Pathfinder instrument configuration."""

    num_pairs: int = DEFAULT_NUM_PAIRS
    pair_spacing: float = DEFAULT_PAIR_SPACING
    sensor_separation: float = DEFAULT_SENSOR_SEPARATION
    bottom_height: float = DEFAULT_BOTTOM_HEIGHT
    sample_rate: float = DEFAULT_SAMPLE_RATE
    adc_counts_per_tesla: float = ADC_COUNTS_PER_TESLA
    adc_saturation: int = ADC_SATURATION
    noise_model: NoiseModel | None = None

    def __post_init__(self):
        if self.noise_model is None:
            self.noise_model = pathfinder_noise_model()

    @property
    def swath_width(self) -> float:
        """Total swath width in meters."""
        return (self.num_pairs - 1) * self.pair_spacing

    def sensor_offsets(self) -> np.ndarray:
        """Compute lateral offsets for each sensor pair relative to center.

        Returns
        -------
        offsets : ndarray, shape (num_pairs,)
            X-offsets in meters. Pair 1 is leftmost (most negative).
        """
        total = self.swath_width
        return np.linspace(-total / 2, total / 2, self.num_pairs)


def generate_walk_path(
    start: tuple[float, float],
    end: tuple[float, float],
    speed: float = 1.0,
    sample_rate: float = 10.0,
    surface_elevation: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a straight-line walk path.

    Parameters
    ----------
    start : tuple
        (x, y) start position in meters.
    end : tuple
        (x, y) end position in meters.
    speed : float
        Walking speed in m/s.
    sample_rate : float
        Sample rate in Hz.
    surface_elevation : float
        Ground surface z-coordinate in meters.

    Returns
    -------
    positions : ndarray, shape (N, 2)
        (x, y) positions along the path.
    timestamps : ndarray, shape (N,)
        Timestamps in seconds.
    headings : ndarray, shape (N,)
        Heading angles in radians (0=North, π/2=East).
    """
    start = np.array(start, dtype=np.float64)
    end = np.array(end, dtype=np.float64)

    distance = np.linalg.norm(end - start)
    duration = distance / speed
    n_samples = max(2, int(duration * sample_rate))

    t = np.linspace(0, duration, n_samples)
    frac = t / duration
    positions = start[np.newaxis, :] + frac[:, np.newaxis] * (end - start)[np.newaxis, :]

    # Heading: angle from North (Y-axis), clockwise positive
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    heading = np.arctan2(dx, dy)  # atan2(east, north)
    headings = np.full(n_samples, heading)

    return positions, t, headings


def generate_zigzag_path(
    origin: tuple[float, float],
    width: float,
    length: float,
    line_spacing: float = 1.0,
    speed: float = 1.0,
    sample_rate: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a zigzag survey path (boustrophedon).

    Parameters
    ----------
    origin : tuple
        (x, y) SW corner of survey area.
    width : float
        East-west extent in meters.
    length : float
        North-south extent in meters.
    line_spacing : float
        Distance between parallel lines in meters.
    speed : float
        Walking speed in m/s.
    sample_rate : float
        Sample rate in Hz.

    Returns
    -------
    positions, timestamps, headings
        Same as generate_walk_path.
    """
    x0, y0 = origin
    n_lines = max(1, int(width / line_spacing) + 1)

    all_positions = []
    all_timestamps = []
    all_headings = []
    t_offset = 0.0

    for i in range(n_lines):
        x = x0 + i * line_spacing
        if i % 2 == 0:
            start = (x, y0)
            end = (x, y0 + length)
        else:
            start = (x, y0 + length)
            end = (x, y0)

        pos, t, hdg = generate_walk_path(start, end, speed, sample_rate)
        all_positions.append(pos)
        all_timestamps.append(t + t_offset)
        all_headings.append(hdg)
        t_offset = all_timestamps[-1][-1] + 1.0 / sample_rate

        # Add turn between lines (if not last line)
        if i < n_lines - 1:
            turn_start = end
            turn_end = (x + line_spacing, end[1])
            pos_t, t_t, hdg_t = generate_walk_path(turn_start, turn_end, speed, sample_rate)
            all_positions.append(pos_t)
            all_timestamps.append(t_t + t_offset)
            all_headings.append(hdg_t)
            t_offset = all_timestamps[-1][-1] + 1.0 / sample_rate

    return (
        np.vstack(all_positions),
        np.concatenate(all_timestamps),
        np.concatenate(all_headings),
    )


def simulate_survey(
    positions: np.ndarray,
    timestamps: np.ndarray,
    headings: np.ndarray,
    sources: list[dict],
    config: PathfinderConfig | None = None,
    rng: np.random.Generator | None = None,
    add_noise: bool = True,
) -> dict:
    """Simulate a Pathfinder survey along a walk path.

    Parameters
    ----------
    positions : ndarray, shape (N, 2)
        (x, y) operator positions along the path.
    timestamps : ndarray, shape (N,)
        Timestamps in seconds.
    headings : ndarray, shape (N,)
        Heading angles in radians.
    sources : list of dict
        Magnetic dipole sources (from Scenario.magnetic_sources).
    config : PathfinderConfig, optional
        Instrument configuration.
    rng : Generator, optional
        Random number generator.
    add_noise : bool
        Whether to add realistic noise.

    Returns
    -------
    data : dict
        Keys match Pathfinder CSV columns:
        - 'timestamp': milliseconds (uint32)
        - 'lat', 'lon': GPS coordinates (simulated as local meters)
        - 'g{N}_top', 'g{N}_bot', 'g{N}_grad': ADC counts for each pair
    """
    if config is None:
        config = PathfinderConfig()
    if rng is None:
        rng = np.random.default_rng()

    n_samples = len(positions)
    offsets = config.sensor_offsets()

    # Initialize result arrays
    data = {
        'timestamp': (timestamps * 1000).astype(np.uint32),
        'lat': positions[:, 1],  # Y = northing
        'lon': positions[:, 0],  # X = easting
    }

    for pair_idx in range(config.num_pairs):
        pair_num = pair_idx + 1

        # Compute sensor positions for this pair
        # Lateral offset rotated by heading
        dx = offsets[pair_idx] * np.cos(headings)
        dy = -offsets[pair_idx] * np.sin(headings)

        sensor_x = positions[:, 0] + dx
        sensor_y = positions[:, 1] + dy

        # Bottom sensor positions (3D)
        r_bottom = np.column_stack([
            sensor_x,
            sensor_y,
            np.full(n_samples, config.bottom_height),
        ])

        # Compute field at both sensors
        B_bot, B_top, grad = gradiometer_reading(
            r_bottom, sources, config.sensor_separation, component=2
        )

        # Add noise to each channel independently
        if add_noise and config.noise_model is not None:
            B_bot = config.noise_model.apply(
                B_bot, timestamps, headings, config.sample_rate, rng
            )
            B_top = config.noise_model.apply(
                B_top, timestamps, headings, config.sample_rate, rng
            )
            grad = B_bot - B_top

        # Convert to ADC counts
        top_adc = np.clip(
            np.round(B_top * config.adc_counts_per_tesla).astype(np.int32),
            -config.adc_saturation, config.adc_saturation,
        )
        bot_adc = np.clip(
            np.round(B_bot * config.adc_counts_per_tesla).astype(np.int32),
            -config.adc_saturation, config.adc_saturation,
        )
        grad_adc = bot_adc - top_adc  # Match firmware: bottom - top

        data[f'g{pair_num}_top'] = top_adc
        data[f'g{pair_num}_bot'] = bot_adc
        data[f'g{pair_num}_grad'] = grad_adc

    return data


def export_csv(
    data: dict,
    path: str | Path,
    num_pairs: int = DEFAULT_NUM_PAIRS,
    gps_quality: bool = False,
) -> None:
    """Export survey data as Pathfinder-compatible CSV.

    Output matches the firmware CSV format exactly, so it can be
    loaded by Pathfinder's visualize_data.py tool.

    Parameters
    ----------
    data : dict
        Survey data from simulate_survey().
    path : str or Path
        Output CSV file path.
    num_pairs : int
        Number of sensor pairs (determines columns).
    gps_quality : bool
        Include fix_quality, hdop, altitude columns.
    """
    # Build header matching firmware's writeCSVHeader()
    columns = ['timestamp', 'lat', 'lon']
    if gps_quality:
        columns.extend(['fix_quality', 'hdop', 'altitude'])
    for i in range(1, num_pairs + 1):
        columns.extend([f'g{i}_top', f'g{i}_bot', f'g{i}_grad'])

    n_rows = len(data['timestamp'])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for i in range(n_rows):
            row = []
            for col in columns:
                val = data.get(col)
                if val is None:
                    row.append(0)
                elif isinstance(val, np.ndarray):
                    v = val[i]
                    if col in ('lat', 'lon'):
                        row.append(f'{v:.6f}')
                    else:
                        row.append(int(v))
                else:
                    row.append(val)
            writer.writerow(row)


def run_scenario_survey(
    scenario_path: str | Path,
    output_csv: str | Path,
    walk_type: str = 'zigzag',
    speed: float = 1.0,
    line_spacing: float = 1.0,
    add_noise: bool = True,
    seed: int | None = None,
) -> dict:
    """Run a complete Pathfinder survey simulation from a scenario file.

    This is the high-level convenience function that chains:
    scenario → walk path → physics → noise → CSV export.

    Parameters
    ----------
    scenario_path : str or Path
        Path to scenario JSON file.
    output_csv : str or Path
        Output CSV file path.
    walk_type : str
        'zigzag' for boustrophedon, 'straight' for single line.
    speed : float
        Walking speed in m/s.
    line_spacing : float
        Line spacing for zigzag in meters.
    add_noise : bool
        Add realistic noise.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    data : dict
        Survey data (also written to CSV).
    """
    from geosim.scenarios.loader import load_scenario

    scenario = load_scenario(scenario_path)
    sources = scenario.magnetic_sources

    rng = np.random.default_rng(seed)
    config = PathfinderConfig()

    x_min, x_max = scenario.terrain.x_extent
    y_min, y_max = scenario.terrain.y_extent

    if walk_type == 'zigzag':
        positions, timestamps, headings = generate_zigzag_path(
            origin=(x_min, y_min),
            width=x_max - x_min,
            length=y_max - y_min,
            line_spacing=line_spacing,
            speed=speed,
            sample_rate=config.sample_rate,
        )
    else:
        positions, timestamps, headings = generate_walk_path(
            start=((x_min + x_max) / 2, y_min),
            end=((x_min + x_max) / 2, y_max),
            speed=speed,
            sample_rate=config.sample_rate,
        )

    data = simulate_survey(
        positions, timestamps, headings, sources, config, rng, add_noise
    )

    export_csv(data, output_csv, config.num_pairs)

    return data
