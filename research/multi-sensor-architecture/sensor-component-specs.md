# Sensor Component Specifications

## Overview

This document specifies all sensor components for the multi-sensor Pathfinder, including I2C addresses, data rates, power requirements, and integration notes. Components are organized by subsystem.

---

## GPS: u-blox ZED-F9P

| Parameter | Value |
|-----------|-------|
| I2C Address | 0x42 (default) |
| Update rate | Up to 10 Hz with RTK |
| Accuracy | 0.01 m + 1 ppm (RTK fixed) |
| Cost | $200-275 (module + antenna) |
| GNSS bands | L1 + L2 (GPS, GLONASS, Galileo, BeiDou) |
| Power | 68 mA @ 3.3V typical |
| Interface | I2C, UART, SPI |
| RTK corrections | NTRIP via ESP32 WiFi |

### Integration Notes

- Located in shared sensor pod (see `sensor-pod-design.md`)
- Requires external L1+L2 antenna (active helical recommended)
- SMA connector on sensor pod for antenna cable
- NTRIP client runs on ESP32 Core 0, forwards corrections to ZED-F9P via I2C
- Falls back to standalone mode (1-2m accuracy) if NTRIP unavailable

### NTRIP Correction Sources

| Provider | Coverage | Cost | Latency |
|----------|----------|------|---------|
| RTK2go (community) | Variable | Free | 1-5 s |
| UNAVCO/CORS | US | Free | 1-10 s |
| Emlid Caster | Global | $0 (own base) | <1 s |
| Commercial (e.g., PointOne) | Global | $50-100/mo | <1 s |

---

## IMU: Bosch BNO055

| Parameter | Value |
|-----------|-------|
| I2C Address | 0x29 (AD0 = HIGH) |
| Fusion rate | 100 Hz (IMUPLUS mode) |
| Accelerometer | ±2/4/8/16 g, 14-bit |
| Gyroscope | ±125-2000 °/s, 16-bit |
| Magnetometer | ±1300 μT (DISABLED in our config) |
| Power | 12.3 mA @ 3.3V (IMUPLUS) |
| Cost | $25-35 (breakout board) |

### Configuration for Pathfinder

```
Mode: IMUPLUS (0x08)
```

**IMUPLUS mode** uses accelerometer + gyroscope fusion only. The internal magnetometer is **disabled** to prevent interference with the fluxgate sensors.

### Placement Constraints

- Must be >20 cm from nearest fluxgate sensor
- Located in sensor pod, which mounts at the crossbar center
- The pod is 25 cm from the nearest fluxgate (Pair 1)
- Orientation: X-axis along crossbar, Z-axis vertical

### Tilt Correction Application

The BNO055 provides Euler angles (heading, roll, pitch) at 100 Hz. These are used to:

1. Correct gradient readings for crossbar tilt (pitch/roll affect sensor height above ground)
2. Detect and flag operator turns (heading changes > threshold → discard readings)
3. Compensate for sensor orientation relative to Earth's field

---

## Barometric Pressure/Altitude: Bosch BMP390

| Parameter | Value |
|-----------|-------|
| I2C Address | 0x77 (default) |
| Pressure range | 300-1250 hPa |
| Pressure resolution | 0.02 hPa (~0.17 m altitude) |
| Temperature range | -40 to +85°C |
| Power | 3.4 μA @ 1 Hz |
| Cost | $5-10 (breakout board) |

### Integration Notes

- Located in sensor pod alongside GPS and IMU
- Provides barometric altitude for GPS altitude cross-check
- Temperature reading useful for thermal drift compensation of fluxgate/LM2917 chain
- Ultra-low power consumption makes it "free" in the power budget

---

## IR Temperature: Melexis MLX90614xAC

| Parameter | Value |
|-----------|-------|
| I2C Address | 0x5A (default) |
| FOV | 35° (xAC variant) |
| Accuracy | ±0.5°C (0-50°C range) |
| Object temp range | -70 to +380°C |
| Ambient temp range | -40 to +125°C |
| Power | 1.5 mA @ 3.3V |
| Cost | $12-18 |

### Integration Notes

- Mounted on crossbar pointing downward at ground surface
- 35° FOV at 20 cm height → ~12 cm diameter footprint on ground
- Detects thermal anomalies: disturbed soil (different thermal inertia), underground voids, buried infrastructure
- Complements magnetic and conductivity data for multi-physics interpretation
- Located on I2C Bus 0 (local sensors) to avoid pod cable latency

---

## LiDAR: Slamtec RPLiDAR C1

| Parameter | Value |
|-----------|-------|
| Interface | UART (USB adapter to Jetson) |
| Range | 0.1-12 m |
| Sample rate | 5000 samples/s |
| Scan rate | 10 Hz (360° scans) |
| Angular resolution | 0.72° |
| Power | 3.5W @ 5V |
| Cost | $65-80 |

### Integration Notes

- **NOT connected through ESP32** — direct USB to Jetson Nano/Orin
- Brushless DC motor for rotation → magnetic interference source (see `interference-matrix.md`)
- Mounted at far end of crossbar (-100 cm position) for maximum separation from fluxgates
- ROS2 rplidar driver on Jetson handles point cloud acquisition
- Data used for: micro-topography mapping, obstacle avoidance, DEM generation

---

## Real-Time Clock: Maxim DS3231

| Parameter | Value |
|-----------|-------|
| I2C Address | 0x68 (fixed) |
| Accuracy | ±2 ppm (0-40°C) |
| Battery backup | CR2032 coin cell |
| Temperature sensor | ±3°C internal |
| Power | 200 μA @ 3.3V active |
| Cost | $5-10 (breakout board) |

### Integration Notes

- Located in sensor pod for precise timestamping
- Provides accurate time even without GPS fix
- Used to correlate measurements across Pathfinder and HIRT (shared pod)
- I2C address 0x68 has no conflicts in the address map

---

## Camera Options

### Option A: ESP32-CAM Standalone Module

| Parameter | Value |
|-----------|-------|
| Sensor | OV2640, 2 MP |
| Interface | WiFi (standalone ESP32) or SD card |
| Resolution | Up to 1600×1200 |
| Cost | $8-15 |
| Power | ~300 mA @ 5V during capture |

### Option B: OV2640 on Main ESP32

| Parameter | Value |
|-----------|-------|
| Sensor | Same OV2640 |
| Interface | Camera parallel interface on ESP32 |
| Cost | $5 (module only) |
| Complexity | Requires 8+ GPIO pins, DMA, firmware complexity |

**Recommendation**: ESP32-CAM standalone. Simpler integration, independent operation, built-in SD card for image buffering. Triggered by main ESP32 via GPIO pulse at each measurement cycle.

---

## Summary Table

| Sensor | I2C Addr | Bus | Power | Data Rate | Location |
|--------|----------|-----|-------|-----------|----------|
| ADS1115 #1 | 0x48 | Bus 0 | 0.2 mA | 128 SPS | Main PCB |
| ADS1115 #2 | 0x49 | Bus 0 | 0.2 mA | 128 SPS | Main PCB |
| MLX90614 | 0x5A | Bus 0 | 1.5 mA | 10 Hz | Main PCB |
| ZED-F9P | 0x42 | Bus 1 | 68 mA | 10 Hz | Sensor Pod |
| BNO055 | 0x29 | Bus 1 | 12.3 mA | 100 Hz | Sensor Pod |
| BMP390 | 0x77 | Bus 1 | <0.01 mA | 1 Hz | Sensor Pod |
| DS3231 | 0x68 | Bus 1 | 0.2 mA | 1 Hz | Sensor Pod |
| RPLiDAR C1 | N/A (UART/USB) | USB to Jetson | 700 mA | 10 Hz scans | Crossbar end |
| ESP32-CAM | N/A (GPIO trigger) | Standalone | 300 mA | 1 Hz | Crossbar |
