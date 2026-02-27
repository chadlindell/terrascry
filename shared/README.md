# TERRASCRY Shared Components

Cross-instrument libraries, protocols, hardware designs, and deployment configurations shared between Pathfinder and HIRT.

## Directory Structure

```
shared/
├── firmware/
│   ├── sensor_pod/     # I2C drivers for ZED-F9P, BNO055, BMP390, DS3231
│   ├── mqtt/           # MQTT client helpers, message schemas
│   └── sd_logger/      # Common SD card CSV logging
├── protocols/
│   ├── mqtt_topics.md  # Topic hierarchy: terrascry/{instrument}/{channel}
│   ├── csv_schemas.md  # Shared CSV column definitions
│   └── protobuf/       # Binary message definitions (future)
├── hardware/
│   ├── sensor_pod/     # Pod schematics, BOM, PCB files
│   └── connectors/     # M8 8-pin connector specs, pinouts
├── edge/
│   ├── docker-compose.yml  # Jetson edge deployment
│   ├── mosquitto.conf      # MQTT broker config
│   └── containers/         # Per-instrument processing containers
└── workflow/
    └── field_handoff.md    # Pathfinder → HIRT handoff procedure
```

## Sensor Pod

The sensor pod is the key shared hardware between Pathfinder and HIRT. It contains positioning and environmental sensors in an IP67 enclosure:

| Sensor | I2C Address | Function |
|--------|-------------|----------|
| ZED-F9P | 0x42 | RTK GPS (cm-accuracy positioning) |
| BNO055 | 0x29 | 9-axis IMU (orientation, tilt correction) |
| BMP390 | 0x77 | Barometric pressure (altitude) |
| DS3231 | 0x68 | Real-time clock (timestamp sync) |

**Connection:** M8 8-pin connector → PCA9615 differential I2C → Cat5 STP cable (1-2m) → instrument I2C Bus 1.

Both instruments use the same GPS receiver, eliminating coordinate transformation errors and enabling seamless data fusion in joint inversion.

## PlatformIO Integration

Both Pathfinder and HIRT firmware projects reference shared libraries:

```ini
# In pathfinder/firmware/platformio.ini or hirt/firmware/platformio.ini
lib_extra_dirs = ../../shared/firmware
```

## MQTT Topic Hierarchy

All instruments publish under `terrascry/`:

```
terrascry/pathfinder/data/raw          — 10 Hz raw sensor data
terrascry/pathfinder/data/corrected    — Tilt-corrected readings
terrascry/pathfinder/anomaly/detected  — Anomaly alerts
terrascry/hirt/data/mit/raw            — MIT measurements
terrascry/hirt/data/ert/raw            — ERT measurements
terrascry/hirt/model/update            — Inversion results
terrascry/status/{instrument}          — Health heartbeat (1 Hz)
```

See `protocols/mqtt_topics.md` for full specification.
