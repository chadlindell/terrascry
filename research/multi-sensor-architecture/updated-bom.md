# Updated Bill of Materials — Multi-Sensor Pathfinder

## Overview

This BOM reflects the expanded multi-sensor architecture: magnetic gradiometry + EMI conductivity + RTK GPS + IMU + IR temperature + LiDAR + camera. Costs include the shared sensor pod (amortized across Pathfinder and HIRT).

## BOM by Subsystem

### Existing Components (from original design)

| Component | Qty | Est. Cost | Notes |
|-----------|-----|-----------|-------|
| **Sensors** |
| FG-3+ fluxgate sensors | 8 | $480-640 | 4 gradiometer pairs |
| **Frame** |
| Carbon fiber tube 2m × 25mm | 1 | $40 | Main crossbar |
| PVC conduit 20mm × 50cm | 4 | $10 | Sensor drop tubes |
| 3D printed mounts | 8+ | $20 | Sensor clips + brackets |
| **Harness** |
| Backpack harness (salvage) | 1 | $0-30 | Old backpack straps |
| Bungee cord | 2m | $5 | Vibration dampening |
| Carabiners | 4 | $10 | Quick-release |
| **Original Electronics** |
| ADS1115 16-bit ADC | 2 | $10 | 4 channels each |
| LM2917 F-to-V converter | 8 | $16 | One per fluxgate |
| SD card module | 1 | $5 | Data logging backup |
| LiPo 7.4V 2000mAh | 1 | $20 | Belt-mounted |
| Speaker/buzzer | 1 | $2 | Pace beeper |
| Enclosure (IP65) | 1 | $15 | Electronics housing |
| Cables, connectors (original) | - | $25 | |
| **Subtotal (existing)** | | **$658-848** | |

### ESP32 Upgrade (replacing Arduino Nano)

| Component | Qty | Est. Cost | Notes |
|-----------|-----|-----------|-------|
| ESP32-DevKitC V4 | 1 | $5 | Dual-core, WiFi, BT, 2× I2C |
| **Subtotal** | | **$5** | |

### Power Supply Upgrade

| Component | Qty | Est. Cost | Notes |
|-----------|-----|-----------|-------|
| LM2596 buck converter module | 1 | $2 | 7.4V → 5.5V |
| TPS7A4901 LDO (or breakout) | 1 | $3 | Ultra-low-noise 5.0V analog rail |
| LM78L05 voltage regulators | 8 | $4 | One per FG-3+ sensor |
| Ferrite beads (BLM18PG601) | 4 | $1 | EMI filtering |
| Shielded inductor 10 μH | 1 | $2 | LC filter |
| Low-ESR capacitors (assorted) | - | $3 | Decoupling throughout |
| **Subtotal** | | **~$15** | |

### EMI Conductivity Channel

| Component | Qty | Est. Cost | Notes |
|-----------|-----|-----------|-------|
| AD9833 DDS module | 1 | $5 | 15 kHz sine wave generation |
| OPA549 power op-amp | 1 | $8 | TX coil driver |
| AD8421 instrumentation amp | 1 | $8 | RX preamp (3 nV/√Hz) |
| AD630 balanced modulator | 1 | $15 | Phase-sensitive detector (I/Q) |
| TX coil (30 turns, 12cm, hand-wound) | 1 | $5 | Enameled copper wire |
| RX coil (30 turns, 8cm, hand-wound) | 1 | $3 | Enameled copper wire |
| Bucking coil (calculated) | 1 | $2 | Small nulling coil |
| Passive components (R, C, ferrite) | - | $5 | Signal chain filtering |
| Coil mounting hardware | - | $5 | 3D printed brackets |
| Connectors (M8 for TX, M8 for RX) | 2 | $15 | IP67 cable connections |
| **Subtotal** | | **~$71** | |

### IR Temperature Sensor

| Component | Qty | Est. Cost | Notes |
|-----------|-----|-----------|-------|
| MLX90614xAC (35° FOV) | 1 | $15 | I2C, non-contact IR |
| **Subtotal** | | **$15** | |

### Camera Module

| Component | Qty | Est. Cost | Notes |
|-----------|-----|-----------|-------|
| ESP32-CAM module | 1 | $10 | Standalone, OV2640 |
| 3D printed mount | 1 | $2 | Under-crossbar, pointing down |
| Cable + GPIO trigger | 1 | $3 | Trigger from main ESP32 |
| **Subtotal** | | **~$15** | |

### LiDAR

| Component | Qty | Est. Cost | Notes |
|-----------|-----|-----------|-------|
| RPLiDAR C1 | 1 | $65-80 | 360° scanning, UART/USB |
| USB cable to Jetson | 1 | $5 | Direct connection, bypasses ESP32 |
| 3D printed mount | 1 | $3 | End-of-crossbar bracket |
| **Subtotal** | | **~$73-93** | |

### Shared Sensor Pod

| Component | Qty | Est. Cost | Notes |
|-----------|-----|-----------|-------|
| ZED-F9P breakout (SparkFun) | 1 | $220 | RTK GPS receiver |
| L1+L2 active helical antenna | 1 | $40 | Multi-band GNSS |
| BNO055 breakout (Adafruit) | 1 | $30 | 9-axis IMU |
| BMP390 breakout (Adafruit) | 1 | $10 | Barometer + temp |
| DS3231 RTC breakout | 1 | $5 | Precision clock |
| PCA9615 breakout ×2 | 2 | $10 | Differential I2C |
| IP67 enclosure | 1 | $8 | Weatherproof housing |
| M8 connector pair | 1 | $8 | Quick-disconnect |
| SMA bulkhead | 1 | $3 | GPS antenna connector |
| Cat5 STP cable 1.5m | 1 | $10 | Pod cable |
| ESD protection IC | 1 | $2 | TPD4E05U06 |
| GPS antenna mast (telescoping) | 1 | $10 | 50cm above crossbar |
| Misc (battery, wire, board) | - | $8 | Assembly materials |
| **Subtotal** | | **~$364** | |
| **Pathfinder share (50%)** | | **~$182** | Shared with HIRT |

### EMI Mitigation / Shielding

| Component | Qty | Est. Cost | Notes |
|-----------|-----|-----------|-------|
| M8 IP67 connectors (additional) | 3 | $12 | Pod, fluxgate harnesses |
| Shielded cable (braided) | 2m | $5 | Fluxgate signal cables |
| 100 pF bypass caps (RF) | 10 | $1 | Anti-rectification on amp inputs |
| **Subtotal** | | **~$18** | |

## Cost Summary

| Category | Low | High | Notes |
|----------|-----|------|-------|
| Existing components | $658 | $848 | Fluxgate cost is main variable |
| ESP32 upgrade | $5 | $5 | Replaces Arduino Nano ($5-25 savings) |
| Power supply | $15 | $15 | New multi-stage design |
| EMI conductivity channel | $71 | $71 | Complete signal chain |
| IR temperature | $15 | $15 | MLX90614 |
| Camera | $15 | $15 | ESP32-CAM standalone |
| LiDAR | $73 | $93 | RPLiDAR C1 price varies |
| Sensor pod (50% share) | $182 | $182 | Shared with HIRT |
| EMI mitigation | $18 | $18 | Connectors + shielding |
| **TOTAL** | **$1,052** | **$1,262** | |

### With Full Sensor Pod Cost (if not sharing with HIRT)

| **TOTAL (pod not shared)** | **$1,234** | **$1,444** | |

## Comparison with Original Design

| Metric | Original | Multi-Sensor | Change |
|--------|----------|-------------|--------|
| Build cost | $660-870 | $1,052-1,262 | +$392-394 |
| Sensor modalities | 1 (magnetic) | 7 (mag, EMI, GPS, IMU, IR, LiDAR, camera) | +6 modalities |
| MCU | Arduino Nano | ESP32 | Dual-core, WiFi |
| GPS accuracy | 2-5m (NEO-6M) | 0.01m (RTK) | 200-500× improvement |
| Data connectivity | SD card only | MQTT + SD card | Real-time streaming |
| Weight (est.) | ~1.25 kg | ~1.45 kg | +200g |

The multi-sensor upgrade adds 7× the sensing capability for approximately 60% more cost. The GPS upgrade alone (from 2-5m to 0.01m accuracy) transforms data quality.

## Procurement Notes

- **ZED-F9P**: Available from SparkFun, Ardusimple, u-blox distributors. Lead time 1-4 weeks.
- **AD630**: Analog Devices, may require direct order. Available from Mouser/DigiKey.
- **OPA549**: Texas Instruments. In-stock at major distributors.
- **RPLiDAR C1**: Available from Slamtec, Amazon, RobotShop.
- **FG-3+ sensors**: Direct from FG Sensors. Lead time varies.
