"""
MQTT message schemas for instrument data streaming.

All messages are JSON-serialized. Timestamps are ISO 8601 UTC.
"""

import json
from dataclasses import asdict, dataclass


@dataclass
class PathfinderRawMessage:
    """Raw Pathfinder gradiometer reading streamed over MQTT."""

    timestamp_utc: str
    lat: float
    lon: float
    gradients: list[float]  # One per sensor pair (nT)
    top_raw: list[int]      # Raw ADC values
    bot_raw: list[int]
    imu_pitch_deg: float | None = None   # BNO055 pitch
    imu_roll_deg: float | None = None    # BNO055 roll
    imu_heading_deg: float | None = None
    gps_fix_quality: int | None = None
    hdop: float | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> "PathfinderRawMessage":
        return cls(**json.loads(data))


@dataclass
class PathfinderCorrectedMessage:
    """Tilt-corrected Pathfinder reading."""

    timestamp_utc: str
    lat: float
    lon: float
    corrected_gradients: list[float]  # After tilt correction (nT)
    raw_gradients: list[float]
    tilt_correction: list[float]      # Correction applied (nT)
    pitch_deg: float
    roll_deg: float

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> "PathfinderCorrectedMessage":
        return cls(**json.loads(data))


@dataclass
class AnomalyDetectedMessage:
    """Real-time anomaly detection alert."""

    timestamp_utc: str
    lat: float
    lon: float
    anomaly_strength_nt: float
    anomaly_type: str  # "ferrous_dipole", "broad_anomaly", "linear_feature", "unknown"
    confidence: float  # 0.0 to 1.0
    sensor_pair: int

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class HIRTMITRawMessage:
    """Raw HIRT MIT measurement."""

    timestamp_utc: str
    tx_probe: int
    rx_probe: int
    frequency_hz: int
    amplitude: float
    phase_deg: float

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class HIRTERTRawMessage:
    """Raw HIRT ERT measurement."""

    timestamp_utc: str
    a: int  # Current injection electrode A
    b: int  # Current injection electrode B
    m: int  # Potential measurement electrode M
    n: int  # Potential measurement electrode N
    voltage_mv: float
    current_ma: float
    apparent_resistivity_ohm_m: float

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class ProbeOrientationMessage:
    """HIRT probe inclinometer reading."""

    timestamp_utc: str
    probe_index: int
    tilt_x_deg: float  # Tilt from vertical, X axis
    tilt_y_deg: float  # Tilt from vertical, Y axis
    tilt_magnitude_deg: float

    def to_json(self) -> str:
        return json.dumps(asdict(self))
