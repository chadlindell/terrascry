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
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from geosim.magnetics.dipole import gradiometer_reading
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
DEFAULT_PLATFORM = "handheld"
PLATFORM_DEFAULTS = {
    "handheld": {"num_pairs": 4, "sample_rate": 10.0, "gps_quality": False},
    "backpack": {"num_pairs": 4, "sample_rate": 10.0, "gps_quality": False},
    "drone": {"num_pairs": 2, "sample_rate": 20.0, "gps_quality": True},
}


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
    gps_origin: tuple[float, float] | None = None  # (lat0, lon0)
    gps_altitude: float = 0.0  # meters
    gps_noise_std_m: float = 0.0  # horizontal position noise
    gps_quality: bool = False  # include fix_quality/hdop/altitude columns
    gps_dropout_rate: float = 0.0  # probability of fix loss [0..1]
    gps_fix_behavior: str = "zero"  # 'zero' matches firmware no-fix output
    gps_quality_mode: str = "binary"  # 'binary' or 'rtk'
    gps_rtk_fix_rate: float = 0.65
    hdop_base: float = 1.2
    hdop_jitter: float = 0.25
    altitude_noise_std_m: float = 0.2
    platform: str = DEFAULT_PLATFORM
    channel_offset_top_adc: list[float] = field(default_factory=list)
    channel_offset_bot_adc: list[float] = field(default_factory=list)
    channel_gain_top: list[float] = field(default_factory=list)
    channel_gain_bot: list[float] = field(default_factory=list)
    ambient_temp_c: float = 22.0
    temp_jitter_c: float = 0.6
    thermal_drift_adc_per_c: float = 0.5
    battery_start_v: float = 8.2
    battery_end_v: float = 7.3
    noise_model: NoiseModel | None = None

    def __post_init__(self):
        self.platform = str(self.platform or DEFAULT_PLATFORM).lower()
        if self.platform not in PLATFORM_DEFAULTS:
            self.platform = DEFAULT_PLATFORM

        defaults = PLATFORM_DEFAULTS[self.platform]
        if self.platform == "drone":
            if self.num_pairs == DEFAULT_NUM_PAIRS:
                self.num_pairs = defaults["num_pairs"]
            if self.sample_rate == DEFAULT_SAMPLE_RATE:
                self.sample_rate = defaults["sample_rate"]
            if not self.gps_quality:
                self.gps_quality = bool(defaults["gps_quality"])

        self.num_pairs = int(np.clip(int(self.num_pairs), 1, 4))
        self.sample_rate = float(max(self.sample_rate, 1.0))

        self.gps_fix_behavior = str(self.gps_fix_behavior or "zero").lower()
        if self.gps_fix_behavior not in {"zero", "hold_last"}:
            self.gps_fix_behavior = "zero"
        self.gps_quality_mode = str(self.gps_quality_mode or "binary").lower()
        if self.gps_quality_mode not in {"binary", "rtk"}:
            self.gps_quality_mode = "binary"
        self.gps_rtk_fix_rate = float(np.clip(self.gps_rtk_fix_rate, 0.0, 1.0))

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
    if speed <= 0:
        raise ValueError("speed must be > 0")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")

    start = np.array(start, dtype=np.float64)
    end = np.array(end, dtype=np.float64)

    distance = np.linalg.norm(end - start)
    if distance < 1e-12:
        positions = np.vstack([start, start])
        timestamps = np.array([0.0, 1.0 / sample_rate], dtype=np.float64)
        headings = np.zeros(2, dtype=np.float64)
        return positions, timestamps, headings

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
    if width <= 0:
        raise ValueError("width must be > 0")
    if length <= 0:
        raise ValueError("length must be > 0")
    if line_spacing <= 0:
        raise ValueError("line_spacing must be > 0")

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

    # Optional conversion from local EN (meters) to geodetic lat/lon.
    # If no origin is set, keep historical behavior (local meters in lat/lon fields).
    if config.gps_origin is not None:
        lat0, lon0 = config.gps_origin
        x_e = positions[:, 0].copy()
        y_n = positions[:, 1].copy()
        if config.gps_noise_std_m > 0:
            x_e += rng.normal(0.0, config.gps_noise_std_m, n_samples)
            y_n += rng.normal(0.0, config.gps_noise_std_m, n_samples)
        lat = lat0 + y_n / 111_320.0
        lon = lon0 + x_e / (111_320.0 * np.cos(np.deg2rad(lat0)))
    else:
        lat = positions[:, 1]
        lon = positions[:, 0]

    # Compute speeds from position differences for motion noise
    speeds = np.zeros(n_samples)
    cumulative_distance = np.zeros(n_samples)
    if n_samples > 1:
        dt = np.diff(timestamps)
        dt = np.where(dt > 0, dt, 1.0 / config.sample_rate)
        dpos = np.diff(positions, axis=0)
        step_dist = np.sqrt(dpos[:, 0] ** 2 + dpos[:, 1] ** 2)
        speeds[1:] = step_dist / dt
        speeds[0] = speeds[1] if n_samples > 1 else 0.0
        cumulative_distance[1:] = np.cumsum(step_dist)
    heading_unwrapped = np.unwrap(headings)
    heading_deg = np.rad2deg(heading_unwrapped)

    # Derive line IDs from heading and lateral line transitions:
    # -1 for turns/transits, otherwise non-negative line index.
    line_id = np.full(n_samples, -1, dtype=np.int32)
    current_line = -1
    last_line_x = float(positions[0, 0]) if n_samples > 0 else 0.0
    line_transition_threshold = 0.35
    for i in range(n_samples):
        # Heading is measured from North, so |cos|>|sin| means mostly N/S line traversal.
        in_line = abs(np.cos(heading_unwrapped[i])) >= abs(np.sin(heading_unwrapped[i]))
        if not in_line:
            line_id[i] = -1
            continue
        if current_line < 0 or abs(float(positions[i, 0]) - last_line_x) > line_transition_threshold:
            current_line += 1
            last_line_x = float(positions[i, 0])
        line_id[i] = current_line

    # Synthetic IMU attitude from dynamics.
    yaw_rate = np.zeros(n_samples)
    accel = np.zeros(n_samples)
    if n_samples > 1:
        dt = np.diff(timestamps)
        dt = np.where(dt > 0, dt, 1.0 / config.sample_rate)
        yaw_rate[1:] = np.diff(heading_unwrapped) / dt
        yaw_rate[0] = yaw_rate[1]
        accel[1:] = np.diff(speeds) / dt
        accel[0] = accel[1]

    g = 9.81
    roll_deg = np.rad2deg(np.arctan2(speeds * yaw_rate, g))
    pitch_deg = np.rad2deg(np.arctan2(-accel, g))
    if add_noise:
        roll_deg += rng.normal(0.0, 0.35, n_samples)
        pitch_deg += rng.normal(0.0, 0.25, n_samples)
    yaw_rate_dps = np.rad2deg(yaw_rate)

    # Device telemetry profiles (for downstream calibration/diagnostics workflows).
    t_norm = np.linspace(0.0, 1.0, n_samples) if n_samples > 1 else np.zeros(n_samples)
    sensor_temp = np.full(n_samples, config.ambient_temp_c, dtype=np.float64)
    sensor_temp += 0.8 * np.sin(2.0 * np.pi * t_norm)
    if add_noise:
        sensor_temp += rng.normal(0.0, config.temp_jitter_c, n_samples)
    battery_v = config.battery_start_v + (config.battery_end_v - config.battery_start_v) * t_norm
    if add_noise:
        battery_v += rng.normal(0.0, 0.015, n_samples)

    # Dynamic GNSS quality model with optional dropouts.
    fix_quality = None
    hdop = None
    altitude = None
    if config.gps_quality:
        fix_ok = rng.random(n_samples) >= np.clip(config.gps_dropout_rate, 0.0, 1.0)
        if n_samples > 0:
            fix_ok[0] = True

        if config.gps_quality_mode == "rtk":
            # 0=no fix, 1=autonomous, 4=RTK float, 5=RTK fixed.
            fix_quality = np.zeros(n_samples, dtype=np.int32)
            for i in range(n_samples):
                if not fix_ok[i]:
                    fix_quality[i] = 0
                    continue
                if rng.random() < config.gps_rtk_fix_rate:
                    fix_quality[i] = 5
                elif rng.random() < 0.55:
                    fix_quality[i] = 4
                else:
                    fix_quality[i] = 1
        else:
            fix_quality = fix_ok.astype(np.int32)

        if config.gps_fix_behavior == "hold_last":
            for i in range(1, n_samples):
                if not fix_ok[i]:
                    lat[i] = lat[i - 1]
                    lon[i] = lon[i - 1]
        else:
            lat = np.where(fix_ok, lat, 0.0)
            lon = np.where(fix_ok, lon, 0.0)

        hdop = config.hdop_base + np.abs(rng.normal(0.0, config.hdop_jitter, n_samples))
        hdop += 0.35 * (line_id < 0)
        if config.gps_quality_mode == "rtk":
            hdop += np.where(fix_quality == 5, -0.45, 0.0)
            hdop += np.where(fix_quality == 4, -0.2, 0.0)
            hdop += np.where(fix_quality == 1, 0.8, 0.0)
            hdop += np.where(fix_quality == 0, 3.0, 0.0)
            hdop = np.where(fix_quality == 0, 99.9, hdop)
        else:
            hdop += 1.1 * (~fix_ok)
        hdop = np.clip(hdop, 0.6, 99.9)

        altitude = np.full(n_samples, config.gps_altitude, dtype=np.float64)
        if config.altitude_noise_std_m > 0:
            altitude += rng.normal(0.0, config.altitude_noise_std_m, n_samples)
        altitude = np.where(fix_ok, altitude, 0.0)

    # Initialize result arrays
    data = {
        'timestamp': (timestamps * 1000).astype(np.uint32),
        'lat': lat,
        'lon': lon,
        'x_east': positions[:, 0],
        'y_north': positions[:, 1],
        'heading_deg': heading_deg,
        'speed_mps': speeds,
        'distance_m': cumulative_distance,
        'line_id': line_id,
        'imu_roll_deg': roll_deg,
        'imu_pitch_deg': pitch_deg,
        'imu_yaw_rate_dps': yaw_rate_dps,
        'imu_accel_mps2': accel,
        'sensor_temp_c': sensor_temp,
        'battery_v': battery_v,
        'adc_saturation_count': np.zeros(n_samples, dtype=np.int32),
    }
    if config.gps_quality:
        data['fix_quality'] = fix_quality
        data['hdop'] = hdop
        data['altitude'] = altitude

    def _resolve_pair_values(raw_values: list[float], fallback: np.ndarray) -> np.ndarray:
        if len(raw_values) == 0:
            return fallback
        vals = np.asarray(raw_values, dtype=np.float64)
        if vals.size == config.num_pairs:
            return vals
        if vals.size == 1:
            return np.full(config.num_pairs, float(vals[0]), dtype=np.float64)
        out = np.full(config.num_pairs, float(vals[-1]), dtype=np.float64)
        n = min(config.num_pairs, vals.size)
        out[:n] = vals[:n]
        return out

    if add_noise:
        default_top_offsets = rng.normal(0.0, 12.0, config.num_pairs)
        default_bot_offsets = rng.normal(0.0, 12.0, config.num_pairs)
        default_top_gains = 1.0 + rng.normal(0.0, 0.02, config.num_pairs)
        default_bot_gains = 1.0 + rng.normal(0.0, 0.02, config.num_pairs)
    else:
        default_top_offsets = np.zeros(config.num_pairs, dtype=np.float64)
        default_bot_offsets = np.zeros(config.num_pairs, dtype=np.float64)
        default_top_gains = np.ones(config.num_pairs, dtype=np.float64)
        default_bot_gains = np.ones(config.num_pairs, dtype=np.float64)
    top_offsets = _resolve_pair_values(config.channel_offset_top_adc, default_top_offsets)
    bot_offsets = _resolve_pair_values(config.channel_offset_bot_adc, default_bot_offsets)
    top_gains = _resolve_pair_values(config.channel_gain_top, default_top_gains)
    bot_gains = _resolve_pair_values(config.channel_gain_bot, default_bot_gains)
    thermal_adc = (sensor_temp - config.ambient_temp_c) * config.thermal_drift_adc_per_c

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
            # Compute vertical field gradient for motion noise
            field_grad = grad / config.sensor_separation

            # Random phase offset per pair (pairs don't swing identically)
            phase_bot = rng.uniform(0, 2 * np.pi)
            phase_top = rng.uniform(0, 2 * np.pi)

            B_bot = config.noise_model.apply(
                B_bot, timestamps, headings, config.sample_rate, rng,
                speeds=speeds, field_gradient=field_grad,
                phase_offset=phase_bot,
            )
            B_top = config.noise_model.apply(
                B_top, timestamps, headings, config.sample_rate, rng,
                speeds=speeds, field_gradient=field_grad,
                phase_offset=phase_top,
            )
            grad = B_bot - B_top

        # Convert to ADC counts
        top_adc = np.round(
            B_top * config.adc_counts_per_tesla * top_gains[pair_idx]
            + top_offsets[pair_idx]
            + thermal_adc
        ).astype(np.int32)
        bot_adc = np.round(
            B_bot * config.adc_counts_per_tesla * bot_gains[pair_idx]
            + bot_offsets[pair_idx]
            + thermal_adc
        ).astype(np.int32)
        if add_noise:
            top_adc += rng.integers(-1, 2, n_samples)
            bot_adc += rng.integers(-1, 2, n_samples)
        top_sat = np.abs(top_adc) >= config.adc_saturation
        bot_sat = np.abs(bot_adc) >= config.adc_saturation
        sat = np.logical_or(top_sat, bot_sat).astype(np.int32)
        data['adc_saturation_count'] += sat
        data[f'g{pair_num}_sat'] = sat
        top_adc = np.clip(top_adc, -config.adc_saturation, config.adc_saturation)
        bot_adc = np.clip(bot_adc, -config.adc_saturation, config.adc_saturation)
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
    if gps_quality or (
        'fix_quality' in data and 'hdop' in data and 'altitude' in data
    ):
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
                        row.append(f'{v:.7f}')
                    elif col in ('hdop', 'altitude'):
                        row.append(f'{float(v):.1f}')
                    else:
                        row.append(int(v))
                else:
                    row.append(val)
            writer.writerow(row)


def export_telemetry_csv(
    data: dict,
    path: str | Path,
    num_pairs: int = DEFAULT_NUM_PAIRS,
) -> None:
    """Export enriched telemetry CSV for analytics/model-development workflows.

    Includes Pathfinder channels plus derived kinematics/IMU/GNSS quality fields.
    """
    columns = [
        'timestamp', 'lat', 'lon',
        'x_east', 'y_north',
        'heading_deg', 'speed_mps', 'distance_m', 'line_id',
        'imu_roll_deg', 'imu_pitch_deg',
        'imu_yaw_rate_dps', 'imu_accel_mps2',
        'sensor_temp_c', 'battery_v', 'adc_saturation_count',
    ]
    if 'fix_quality' in data and 'hdop' in data and 'altitude' in data:
        columns.extend(['fix_quality', 'hdop', 'altitude'])
    for i in range(1, num_pairs + 1):
        columns.extend([f'g{i}_top', f'g{i}_bot', f'g{i}_grad'])
        sat_col = f'g{i}_sat'
        if sat_col in data:
            columns.append(sat_col)

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
                    continue
                if isinstance(val, np.ndarray):
                    v = val[i]
                    if col in ('lat', 'lon'):
                        row.append(f'{float(v):.7f}')
                    elif col in ('x_east', 'y_north', 'distance_m'):
                        row.append(f'{float(v):.3f}')
                    elif col in (
                        'heading_deg', 'speed_mps', 'imu_roll_deg', 'imu_pitch_deg',
                        'imu_yaw_rate_dps', 'imu_accel_mps2', 'sensor_temp_c', 'battery_v',
                    ):
                        row.append(f'{float(v):.3f}')
                    elif col in ('hdop', 'altitude'):
                        row.append(f'{float(v):.2f}')
                    else:
                        row.append(int(v))
                else:
                    row.append(val)
            writer.writerow(row)


def _extract_anomaly_candidates(
    data: dict,
    num_pairs: int = DEFAULT_NUM_PAIRS,
    top_k: int = 10,
    min_separation_m: float = 1.5,
) -> list[dict]:
    """Extract top anomaly candidates from gradient peaks."""
    if 'timestamp' not in data:
        return []

    n_samples = len(data['timestamp'])
    if n_samples == 0:
        return []

    max_grad = np.zeros(n_samples, dtype=np.float64)
    for i in range(1, num_pairs + 1):
        col = f'g{i}_grad'
        if col in data:
            max_grad = np.maximum(max_grad, np.abs(np.asarray(data[col], dtype=np.float64)))

    idx_sorted = np.argsort(max_grad)[::-1]
    used_positions: list[np.ndarray] = []
    candidates: list[dict] = []
    x = np.asarray(data.get('x_east', data.get('lon')), dtype=np.float64)
    y = np.asarray(data.get('y_north', data.get('lat')), dtype=np.float64)
    line_ids = np.asarray(data.get('line_id', np.full(n_samples, -1)))

    for idx in idx_sorted:
        if len(candidates) >= top_k:
            break
        pos = np.array([x[idx], y[idx]], dtype=np.float64)
        if any(np.linalg.norm(pos - p) < min_separation_m for p in used_positions):
            continue
        used_positions.append(pos)
        candidates.append({
            'sample_index': int(idx),
            'timestamp_ms': int(data['timestamp'][idx]),
            'x_east': float(x[idx]),
            'y_north': float(y[idx]),
            'line_id': int(line_ids[idx]) if line_ids.size > idx else -1,
            'score_adc': float(max_grad[idx]),
        })

    return candidates


def build_survey_summary(
    data: dict,
    scenario_name: str = "",
    num_pairs: int = DEFAULT_NUM_PAIRS,
) -> dict:
    """Build summary metadata and candidate targets for a synthetic survey."""
    n_samples = len(data.get('timestamp', []))
    duration_s = 0.0
    if n_samples > 1:
        duration_s = float((int(data['timestamp'][-1]) - int(data['timestamp'][0])) / 1000.0)

    pair_stats: dict[str, dict] = {}
    for i in range(1, num_pairs + 1):
        col = f'g{i}_grad'
        if col not in data:
            continue
        g = np.asarray(data[col], dtype=np.float64)
        pair_stats[f'g{i}'] = {
            'peak_abs_adc': float(np.max(np.abs(g))) if g.size else 0.0,
            'rms_adc': float(np.sqrt(np.mean(g * g))) if g.size else 0.0,
        }

    distance_m = 0.0
    if 'distance_m' in data and len(data['distance_m']) > 0:
        distance_m = float(data['distance_m'][-1])

    sample_rate_hz = 0.0
    if duration_s > 0:
        sample_rate_hz = float(n_samples / duration_s)
    gps_fix_rate = None
    if 'fix_quality' in data:
        fq = np.asarray(data['fix_quality'], dtype=np.int32)
        gps_fix_rate = float(np.mean(fq > 0)) if fq.size else 0.0
    sat_total = int(np.sum(np.asarray(data.get('adc_saturation_count', []), dtype=np.int32)))
    temp_series = np.asarray(data.get('sensor_temp_c', []), dtype=np.float64)
    battery_series = np.asarray(data.get('battery_v', []), dtype=np.float64)

    return {
        'scenario': scenario_name,
        'sample_count': n_samples,
        'duration_s': duration_s,
        'sample_rate_hz': sample_rate_hz,
        'distance_m': distance_m,
        'gps_fix_rate': gps_fix_rate,
        'adc_saturation_total': sat_total,
        'temperature_c': {
            'mean': float(np.mean(temp_series)) if temp_series.size else None,
            'min': float(np.min(temp_series)) if temp_series.size else None,
            'max': float(np.max(temp_series)) if temp_series.size else None,
        },
        'battery_v': {
            'start': float(battery_series[0]) if battery_series.size else None,
            'end': float(battery_series[-1]) if battery_series.size else None,
        },
        'pair_stats': pair_stats,
        'anomaly_candidates': _extract_anomaly_candidates(data, num_pairs=num_pairs),
    }


def build_sample_labels(
    data: dict,
    scenario_objects: list,
    num_pairs: int = DEFAULT_NUM_PAIRS,
) -> list[dict]:
    """Build per-sample labels for ML-oriented anomaly classification."""
    n_samples = len(data.get('timestamp', []))
    if n_samples == 0:
        return []

    x = np.asarray(data.get('x_east', data.get('lon')), dtype=np.float64)
    y = np.asarray(data.get('y_north', data.get('lat')), dtype=np.float64)
    z = np.zeros(n_samples, dtype=np.float64)
    max_grad = np.zeros(n_samples, dtype=np.float64)
    for i in range(1, num_pairs + 1):
        col = f'g{i}_grad'
        if col in data:
            max_grad = np.maximum(max_grad, np.abs(np.asarray(data[col], dtype=np.float64)))

    labels: list[dict] = []
    for i in range(n_samples):
        nearest_name = ""
        nearest_type = ""
        nearest_depth = 0.0
        nearest_dist = np.inf
        anomaly_radius = 0.8
        for obj in scenario_objects:
            pos = np.asarray(obj.position, dtype=np.float64)
            dist = float(np.hypot(x[i] - pos[0], y[i] - pos[1]))
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_name = str(obj.name)
                nearest_type = str(getattr(obj, "object_type", "unknown"))
                nearest_depth = float(abs(pos[2]))
                anomaly_radius = max(0.8, float(getattr(obj, "radius", 0.1)) * 8.0)

        score = float(max_grad[i])
        is_anomaly = bool((nearest_dist <= anomaly_radius) and (score > 20.0))
        labels.append({
            "sample_index": i,
            "timestamp_ms": int(data["timestamp"][i]),
            "x_east": float(x[i]),
            "y_north": float(y[i]),
            "z_up": float(z[i]),
            "nearest_target": nearest_name,
            "target_type": nearest_type,
            "target_depth_m": nearest_depth,
            "target_distance_m": float(nearest_dist) if np.isfinite(nearest_dist) else None,
            "max_grad_adc": score,
            "is_anomaly": 1 if is_anomaly else 0,
        })
    return labels


def export_labels_csv(
    labels: list[dict],
    path: str | Path,
) -> None:
    """Export sample labels for ML pipelines."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "sample_index",
        "timestamp_ms",
        "x_east",
        "y_north",
        "z_up",
        "nearest_target",
        "target_type",
        "target_depth_m",
        "target_distance_m",
        "max_grad_adc",
        "is_anomaly",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in labels:
            writer.writerow([row.get(c, "") for c in columns])


def run_scenario_survey(
    scenario_path: str | Path,
    output_csv: str | Path,
    walk_type: str = 'zigzag',
    speed: float = 1.0,
    line_spacing: float = 1.0,
    add_noise: bool = True,
    seed: int | None = None,
    export_extended: bool = True,
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
    export_extended : bool
        Write telemetry + summary artifacts for analytics workflows.

    Returns
    -------
    data : dict
        Survey data (also written to CSV).
    """
    from geosim.scenarios.loader import load_scenario

    scenario = load_scenario(scenario_path)
    sources = scenario.magnetic_sources

    rng = np.random.default_rng(seed)
    meta = scenario.metadata or {}
    pf_meta = meta.get("pathfinder", {})
    if not isinstance(pf_meta, dict):
        pf_meta = {}
    platform = str(pf_meta.get("platform", DEFAULT_PLATFORM)).lower()
    config = PathfinderConfig(platform=platform)

    if "num_pairs" in pf_meta:
        config.num_pairs = int(pf_meta.get("num_pairs", config.num_pairs))
    if "sample_rate_hz" in pf_meta:
        config.sample_rate = float(pf_meta.get("sample_rate_hz", config.sample_rate))
    if "pair_spacing_m" in pf_meta:
        config.pair_spacing = float(pf_meta.get("pair_spacing_m", config.pair_spacing))
    if "sensor_baseline_m" in pf_meta:
        config.sensor_separation = float(pf_meta.get("sensor_baseline_m", config.sensor_separation))
    if "bottom_height_m" in pf_meta:
        config.bottom_height = float(pf_meta.get("bottom_height_m", config.bottom_height))

    gps_meta = pf_meta.get("gps_origin", meta.get('gps_origin', {}))
    if isinstance(gps_meta, dict) and 'lat' in gps_meta and 'lon' in gps_meta:
        config.gps_origin = (float(gps_meta['lat']), float(gps_meta['lon']))
        config.gps_altitude = float(gps_meta.get('altitude', 0.0))
        config.gps_noise_std_m = float(pf_meta.get("gps_noise_m", meta.get('gps_noise_m', 0.0)))
        config.gps_quality = bool(pf_meta.get("gps_quality", meta.get('gps_quality', True)))
        config.gps_dropout_rate = float(pf_meta.get("gps_dropout_rate", meta.get('gps_dropout_rate', 0.0)))
        config.gps_fix_behavior = str(pf_meta.get("gps_fix_behavior", "zero"))
        config.gps_quality_mode = str(pf_meta.get("gps_quality_mode", "binary"))
        config.gps_rtk_fix_rate = float(pf_meta.get("gps_rtk_fix_rate", 0.65))
        config.hdop_base = float(pf_meta.get("gps_hdop_base", meta.get('gps_hdop_base', 1.2)))
        config.hdop_jitter = float(pf_meta.get("gps_hdop_jitter", meta.get('gps_hdop_jitter', 0.25)))
        config.altitude_noise_std_m = float(
            pf_meta.get("gps_altitude_noise_m", meta.get('gps_altitude_noise_m', 0.2))
        )

    calib_meta = pf_meta.get("calibration", {})
    if isinstance(calib_meta, dict):
        config.channel_offset_top_adc = list(calib_meta.get("offset_top_adc", []))
        config.channel_offset_bot_adc = list(calib_meta.get("offset_bot_adc", []))
        config.channel_gain_top = list(calib_meta.get("gain_top", []))
        config.channel_gain_bot = list(calib_meta.get("gain_bot", []))
        config.ambient_temp_c = float(calib_meta.get("ambient_temp_c", config.ambient_temp_c))
        config.thermal_drift_adc_per_c = float(
            calib_meta.get("thermal_drift_adc_per_c", config.thermal_drift_adc_per_c)
        )
    power_meta = pf_meta.get("power", {})
    if isinstance(power_meta, dict):
        config.battery_start_v = float(power_meta.get("battery_start_v", config.battery_start_v))
        config.battery_end_v = float(power_meta.get("battery_end_v", config.battery_end_v))
    config.num_pairs = int(np.clip(int(config.num_pairs), 1, 4))
    config.sample_rate = float(max(config.sample_rate, 1.0))
    config.gps_fix_behavior = str(config.gps_fix_behavior or "zero").lower()
    if config.gps_fix_behavior not in {"zero", "hold_last"}:
        config.gps_fix_behavior = "zero"
    config.gps_quality_mode = str(config.gps_quality_mode or "binary").lower()
    if config.gps_quality_mode not in {"binary", "rtk"}:
        config.gps_quality_mode = "binary"
    config.gps_rtk_fix_rate = float(np.clip(config.gps_rtk_fix_rate, 0.0, 1.0))
    if config.noise_model is None:
        config.noise_model = pathfinder_noise_model()

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

    if export_extended:
        output_csv = Path(output_csv)
        telemetry_path = output_csv.with_name(output_csv.stem + "_telemetry.csv")
        summary_path = output_csv.with_name(output_csv.stem + "_summary.json")
        labels_path = output_csv.with_name(output_csv.stem + "_labels.csv")
        session_path = output_csv.with_name(output_csv.stem + "_session.json")
        export_telemetry_csv(data, telemetry_path, config.num_pairs)
        summary = build_survey_summary(data, scenario_name=scenario.name, num_pairs=config.num_pairs)
        labels = build_sample_labels(data, scenario.objects, num_pairs=config.num_pairs)
        export_labels_csv(labels, labels_path)
        session = {
            "scenario": scenario.name,
            "platform": config.platform,
            "sample_rate_hz": config.sample_rate,
            "num_pairs": config.num_pairs,
            "pair_spacing_m": config.pair_spacing,
            "sensor_baseline_m": config.sensor_separation,
            "bottom_height_m": config.bottom_height,
            "gps_quality": config.gps_quality,
            "gps_quality_mode": config.gps_quality_mode,
            "gps_fix_behavior": config.gps_fix_behavior,
            "seed": seed,
            "artifacts": {
                "csv": str(output_csv),
                "telemetry_csv": str(telemetry_path),
                "summary_json": str(summary_path),
                "labels_csv": str(labels_path),
            },
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        with open(session_path, "w") as f:
            json.dump(session, f, indent=2)
        data['telemetry_csv'] = str(telemetry_path)
        data['summary_json'] = str(summary_path)
        data['labels_csv'] = str(labels_path)
        data['session_json'] = str(session_path)
        data['sample_labels'] = labels
        data['anomaly_candidates'] = summary['anomaly_candidates']

    return data
