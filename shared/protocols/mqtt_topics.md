# TERRASCRY MQTT Topic Hierarchy

All instrument data flows through MQTT using the topic prefix `terrascry/`.

## Topic Structure

```
terrascry/{instrument}/{category}/{subcategory}
```

## Pathfinder Topics

| Topic | Description | Rate | QoS |
|-------|-------------|------|-----|
| `terrascry/pathfinder/data/raw` | Raw gradiometer + EMI + IMU + GPS + IR | 10 Hz | 0 |
| `terrascry/pathfinder/data/corrected` | After tilt correction | 10 Hz | 1 |
| `terrascry/pathfinder/data/emi` | EMI conductivity (I, Q, apparent conductivity) | 10 Hz | 0 |
| `terrascry/pathfinder/data/thermal` | IR temperature (object + ambient) | 10 Hz | 0 |
| `terrascry/pathfinder/anomaly/detected` | Multi-physics anomaly alerts | Event-driven | 2 |

## HIRT Topics

| Topic | Description | Rate | QoS |
|-------|-------------|------|-----|
| `terrascry/hirt/data/mit/raw` | MIT quadrature measurements | Per sequence | 0 |
| `terrascry/hirt/data/ert/raw` | ERT four-electrode readings | Per sequence | 0 |
| `terrascry/hirt/model/update` | Latest inversion result | ~1/min | 1 |
| `terrascry/hirt/probe/orientation` | Inclinometer readings | 1 Hz | 0 |
| `terrascry/hirt/probe/position` | GPS position of probe insertion points | Event-driven | 1 |

## System Topics

| Topic | Description | Rate | QoS |
|-------|-------------|------|-----|
| `terrascry/status/pathfinder` | Pathfinder system health | 1 Hz | 1 |
| `terrascry/status/hirt` | HIRT system health | 1 Hz | 1 |
| `terrascry/status/+` | Wildcard for all instrument health | 1 Hz | 1 |

## QoS Policy

| Level | Usage | Rationale |
|-------|-------|-----------|
| QoS 0 (at most once) | Raw high-rate data | Loss tolerable at 10 Hz; SD card backup exists |
| QoS 1 (at least once) | Corrected data, status, probe positions | Must be logged; duplicates handled by sequence number |
| QoS 2 (exactly once) | Anomaly alerts | Critical notifications; no false duplicates |

## Message Format

All messages are JSON-serialized. Timestamps are ISO 8601 UTC. Numeric values use SI-compatible units: nanotesla (nT) for gradients, degrees for angles, meters for positions.

### Pathfinder Raw Message

```json
{
  "timestamp_utc": "2026-02-18T14:30:12.345Z",
  "seq": 12345,
  "boot_id": "a1b2c3",
  "lat": 51.2345678,
  "lon": -1.4567890,
  "gradients": [12.3, -5.1, 8.7, -2.4],
  "top_raw": [16384, 16512, 16256, 16400],
  "bot_raw": [16520, 16480, 16380, 16440],
  "imu_pitch_deg": 1.2,
  "imu_roll_deg": -0.8,
  "imu_heading_deg": 45.3,
  "emi_i": 0.0023,
  "emi_q": 0.0015,
  "emi_sigma_a": 0.045,
  "ir_object_c": 12.3,
  "ir_ambient_c": 15.1,
  "gps_fix_quality": 5,
  "hdop": 0.8,
  "satellites": 14
}
```

### HIRT MIT Raw Message

```json
{
  "timestamp_utc": "2026-02-18T15:10:45.123Z",
  "tx_probe": 1,
  "rx_probe": 3,
  "frequency_hz": 15000,
  "amplitude": 0.00234,
  "phase_deg": 12.45
}
```

### HIRT ERT Raw Message

```json
{
  "timestamp_utc": "2026-02-18T15:11:02.456Z",
  "a": 1,
  "b": 4,
  "m": 2,
  "n": 3,
  "voltage_mv": 23.45,
  "current_ma": 1.2,
  "apparent_resistivity_ohm_m": 145.6
}
```

### Anomaly Detected Message

```json
{
  "timestamp_utc": "2026-02-18T14:31:05.678Z",
  "lat": 51.2345680,
  "lon": -1.4567892,
  "anomaly_strength_nt": 45.2,
  "anomaly_type": "ferrous_dipole",
  "confidence": 0.87,
  "sensor_pair": 2
}
```

## Broker Configuration

See `../edge/mosquitto.conf` for the Eclipse Mosquitto broker configuration.

Default broker: `terrascry-jetson.local:1883` (mDNS on Jetson).

## Python Implementation

Message dataclasses and topic constants are defined in `geosim/geosim/streaming/`:
- `mqtt_topics.py` — Topic string constants
- `messages.py` — Dataclass schemas with `to_json()` / `from_json()`

## ESP32 Implementation

On ESP32 firmware, use the `esp-mqtt` library (not PubSubClient) for deterministic timing compatibility with TDM. Messages are serialized with `snprintf` (not ArduinoJson) for deterministic memory usage at 10 Hz.
