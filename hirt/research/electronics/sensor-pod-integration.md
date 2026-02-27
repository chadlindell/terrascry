# Sensor Pod Integration with HIRT

## Overview

The shared sensor pod (ZED-F9P GPS, BNO055 IMU, BMP390 barometer, DS3231 RTC) provides precise positioning and orientation data for HIRT probe deployment. This document describes how the pod integrates with HIRT's base hub electronics and the operational workflow for probe GPS verification.

## M8 Connector Interface

The HIRT base hub adds an M8 4-pin female connector identical to Pathfinder's, enabling the same sensor pod to plug into either instrument without modification.

### Connector Pinout (M8 4-pin)

| Pin | Signal | Direction | Notes |
|-----|--------|-----------|-------|
| 1 | VCC (3.3V) | Hub → Pod | From HIRT's 3.3V LDO |
| 2 | GND | Common | Shared ground |
| 3 | SDA_D+ | Bidirectional | Differential I2C data (PCA9615) |
| 4 | SCL_D- | Bidirectional | Differential I2C clock (PCA9615) |

### PCA9615 I2C Bus Buffer

HIRT's base hub requires the same PCA9615 differential-to-single-ended I2C buffer used by Pathfinder:

```
HIRT Base Hub                       Sensor Pod
┌──────────────┐   Cat5 STP 1-2m   ┌──────────────┐
│ ESP32        │                    │ PCA9615      │
│ I2C Bus 1    │──► PCA9615 ──►────│ (pod-side)   │──► ZED-F9P
│ (GPIO 16/17) │    (hub-side)     │              │──► BNO055
│              │                    │              │──► BMP390
│              │                    │              │──► DS3231
└──────────────┘                    └──────────────┘
```

The PCA9615 hub-side breakout mounts inside the HIRT base hub enclosure, adjacent to the M8 connector.

## Probe Position Verification Workflow

The primary use case for GPS in HIRT is recording the exact position of each probe insertion point. This enables:

1. Accurate geometric factor calculation for ERT inversion
2. Spatial registration of HIRT data with Pathfinder survey maps
3. Post-survey verification that probes were placed at planned grid positions

### Workflow: GPS Each Hole Before Insertion

```
For each probe position in the survey grid:

1. Operator places pod (or entire Pathfinder with pod) at planned hole location
2. Wait for RTK fix (typically <10 seconds if NTRIP corrections active)
3. Press "Record Position" button on HIRT controller
4. HIRT firmware reads ZED-F9P position (lat, lon, altitude)
5. Firmware stores position in survey metadata CSV:
   probe_id, timestamp, lat, lon, alt_m, fix_quality, hdop
6. Operator inserts pilot rod, then probe
7. Probe inclinometer (ADXL345 in probe) records insertion angle
8. Repeat for next probe position
```

### GPS Position Data Format

```csv
# HIRT Probe Position Log
# Generated: 2026-02-18T14:30:00Z
# RTK Base: UNAVCO CORS station
probe_id,timestamp,latitude,longitude,altitude_m,fix_type,hdop,satellites
P1,2026-02-18T14:30:12Z,51.2345678,-1.4567890,102.34,RTK_FIXED,0.8,14
P2,2026-02-18T14:31:45Z,51.2345712,-1.4567823,102.31,RTK_FIXED,0.7,15
P3,2026-02-18T14:33:22Z,51.2345690,-1.4567756,102.35,RTK_FIXED,0.9,13
```

## Inclinometer vs Pod IMU: Role Separation

HIRT uses two different inertial sensors for different purposes:

| Sensor | Location | Role | Why Separate |
|--------|----------|------|-------------|
| ADXL345 (inclinometer) | Inside each probe, at probe head | Measures probe tilt/inclination after insertion | Must be in the borehole, waterproof, small |
| BNO055 (pod IMU) | In sensor pod, on surface | Measures crossbar/surface orientation, tilt correction during GPS recording | Too large for probe; not needed underground |

### ADXL345 Probe Inclinometer

- Mounted in each HIRT probe head
- Reports X, Y, Z acceleration → computes tilt angle from vertical
- Used to correct geometric factor: `K_corrected = K_geometric × cos(tilt)`
- Communicates via probe's wired connection to base hub
- I2C address: 0x1D or 0x53 (per probe, on probe's local bus)

### BNO055 Pod IMU

- Located in the shared sensor pod
- Not used during HIRT measurements (probes are underground, pod is on surface)
- Used during GPS position recording to verify the pod is level
- Used when pod is on Pathfinder for real-time tilt correction of gradient readings

## Hardware Integration Checklist

For HIRT base hub modification:

- [ ] Add M8 4-pin female panel-mount connector to base hub enclosure
- [ ] Mount PCA9615 breakout board inside base hub
- [ ] Wire PCA9615 to ESP32 I2C Bus 1 (GPIO 16/17)
- [ ] Add 3.3V power tap from HIRT's LDO to M8 pin 1
- [ ] Test I2C communication with all pod sensors over 1.5m Cat5 cable
- [ ] Update HIRT firmware: I2C bus scan at startup, pod detection
- [ ] Update HIRT firmware: GPS position recording function
- [ ] Update survey metadata CSV format to include probe positions

## Firmware Changes Required

### Startup I2C Bus Scan

```c
void detect_sensor_pod() {
    Wire1.beginTransmission(0x42);  // ZED-F9P
    if (Wire1.endTransmission() == 0) {
        pod_present = true;
        gps_init(&Wire1);
        imu_init(&Wire1);
        baro_init(&Wire1);
        rtc_init(&Wire1);
        Serial.println("Sensor pod detected on I2C Bus 1");
    } else {
        pod_present = false;
        Serial.println("No sensor pod - using manual position entry");
    }
}
```

### Graceful Degradation

If the pod is not connected:
- HIRT continues to operate normally (all measurement functions work)
- Probe positions must be entered manually (grid coordinates from survey plan)
- No RTK GPS, no automatic position recording
- Warning displayed: "No pod — manual positions required"

## References

- ADXL345 datasheet: Analog Devices
- PCA9615 datasheet: NXP Semiconductors
- HIRT electronics documentation: `HIRT/docs/build-guide/electronics.qmd`
- Shared sensor pod design: `Pathfinder/research/multi-sensor-architecture/sensor-pod-design.md`
