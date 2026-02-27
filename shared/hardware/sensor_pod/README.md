# Shared Sensor Pod — Hardware Specification

The sensor pod is the primary shared hardware between Pathfinder and HIRT. It provides positioning, orientation, and environmental sensing in a ruggedized enclosure.

## Sensors

| Component | I2C Address | Function | Current Draw |
|-----------|-------------|----------|--------------|
| u-blox ZED-F9P | 0x42 | RTK GPS (cm-accuracy) | ~70 mA (tracking) |
| Bosch BNO055 | 0x29 | 9-axis IMU (IMUPLUS mode) | ~12 mA |
| Bosch BMP390 | 0x77 | Barometric pressure + temperature | ~3 mA |
| Maxim DS3231 | 0x68 | Temperature-compensated RTC | ~0.2 mA |
| **Total** | | | **~85 mA typical** |

Peak draw (GPS acquisition): ~150-200 mA.

## Enclosure

- **Type:** IP67 ABS enclosure, approximately 60 x 60 x 25 mm
- **Connector:** M8 8-pin panel-mount (NOT 4-pin — PCA9615 requires 6 conductors minimum)
- **Antenna:** SMA panel-mount for external GPS antenna (patch or helix)
- **Vent:** IP67 membrane vent for BMP390 barometric accuracy

## M8 8-Pin Connector Pinout

| Pin | Signal | Color (Cat5) | Description |
|-----|--------|-------------|-------------|
| 1 | VCC | Orange/White | 3.3V supply |
| 2 | GND | Orange | Ground |
| 3 | SDA_D+ | Green/White | PCA9615 differential SDA+ |
| 4 | SDA_D- | Green | PCA9615 differential SDA- |
| 5 | SCL_D+ | Blue/White | PCA9615 differential SCL+ |
| 6 | SCL_D- | Blue | PCA9615 differential SCL- |
| 7 | VBUS | Brown/White | 5V input (before pod's 3.3V LDO) |
| 8 | SHIELD | Brown | Cable shield / drain wire |

## PCA9615 Differential I2C

The PCA9615 converts standard I2C to differential signaling for reliable communication over 1-2m Cat5 STP cable. This eliminates I2C bus capacitance and noise issues at cable lengths beyond ~30 cm.

**Critical:** Both the pod side and instrument side need a PCA9615. The instrument-side PCA9615 connects to ESP32 I2C Bus 1 (GPIO 16 SDA, GPIO 17 SCL).

## ESD Protection

- TVS diodes (PESD5V0S2BT) on all I2C lines at the M8 connector
- ESD protection is critical for hot-plug scenarios (pod moved between instruments during field use)

## Power Architecture

- Input: 5V via VBUS (pin 7) from instrument
- LDO: AP2112K-3.3 (NOT LM78L05 — no 5V→3.3V dropout issue at this voltage)
- All sensors operate at 3.3V
- Cable voltage drop: ~0.05V at 200 mA over 2m Cat5 (negligible at 5V input)

## PCB Design Notes

- 4-layer PCB recommended (signal, ground, power, signal)
- ZED-F9P placed closest to SMA connector, ground plane unbroken beneath RF traces
- BNO055 placed away from GPS and PCA9615 (magnetic interference)
- Keep PCA9615 within 5 cm of M8 connector
- IP67 vent membrane directly above BMP390

See `pathfinder/research/multi-sensor-architecture/sensor-pod-pcb-requirements.md` for full consensus-validated PCB design requirements.

## BOM

| Component | Part Number | Qty | Est. Cost |
|-----------|-------------|-----|-----------|
| ZED-F9P RTK GPS module | u-blox ZED-F9P | 1 | $180 |
| BNO055 9-axis IMU | Bosch BNO055 | 1 | $10 |
| BMP390 barometer | Bosch BMP390 | 1 | $5 |
| DS3231 RTC | Maxim DS3231SN | 1 | $8 |
| PCA9615 diff I2C (pod side) | NXP PCA9615 | 1 | $3 |
| PCA9615 diff I2C (instrument side) | NXP PCA9615 | 1 | $3 |
| AP2112K-3.3 LDO | Diodes Inc | 1 | $0.50 |
| PESD5V0S2BT TVS diodes | Nexperia | 2 | $1 |
| TPD4E05U06 ESD protection | TI | 1 | $0.50 |
| M8 8-pin connector (panel) | Various | 1 | $8 |
| M8 8-pin connector (cable) | Various | 1 | $6 |
| Cat5 STP cable (2m) | Various | 1 | $3 |
| IP67 ABS enclosure | Hammond 1554B | 1 | $12 |
| IP67 vent membrane | Amphenol | 1 | $3 |
| SMA panel-mount connector | Various | 1 | $3 |
| GPS antenna (patch or helix) | Various | 1 | $30 |
| 4-layer PCB | JLCPCB | 1 | $15 |
| Passive components | Various | — | $5 |
| **Total** | | | **~$296** |

## Design References

- Sensor pod design: `pathfinder/research/multi-sensor-architecture/sensor-pod-design.md`
- PCB requirements: `pathfinder/research/multi-sensor-architecture/sensor-pod-pcb-requirements.md`
- Firmware consensus: `pathfinder/research/multi-sensor-architecture/sensor-pod-firmware-consensus.md`
- HIRT integration: `hirt/research/electronics/sensor-pod-integration.md`
- HIRT consensus: `hirt/research/electronics/sensor-pod-consensus.md`
