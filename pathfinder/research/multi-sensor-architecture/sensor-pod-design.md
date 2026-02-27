# Shared Sensor Pod Design

## Overview

The sensor pod is a self-contained, weatherproof module containing GPS, IMU, barometer, and RTC sensors. It is designed to be **shared between Pathfinder and HIRT** — physically moving the same pod between instruments. This ensures consistent positioning, timing, and orientation data across both instruments during a survey.

## Sensor Complement

| Sensor | Function | I2C Address |
|--------|----------|-------------|
| ZED-F9P | RTK GPS (cm-accuracy positioning) | 0x42 |
| BNO055 | 9-axis IMU (tilt/orientation, magnetometer disabled) | 0x29 |
| BMP390 | Barometric pressure/altitude + temperature | 0x77 |
| DS3231 | Precision RTC (±2 ppm, battery-backed) | 0x68 |

## Physical Design

### Enclosure

- **Type**: IP67 ABS enclosure, approximately 60×60×25mm internal
- **Rating**: IP67 (dust-tight, submersible to 1m for 30 min)
- **Material**: ABS plastic (non-magnetic, non-conductive)
- **Color**: White or light gray (minimize solar heating)
- **Mounting**: 3D-printed cradle with quick-release latch

### External Connections

| Connector | Type | Pins | Function |
|-----------|------|------|----------|
| M8 4-pin (male) | IP67 circular | VCC, GND, SDA_D+, SCL_D- | Differential I2C + power to host |
| SMA (female) | Panel-mount bulkhead | 1 | GPS L1+L2 antenna (external active helical) |

### Internal Layout

```
┌────────────────────────────────┐
│  ┌──────────┐  ┌──────────┐   │
│  │ ZED-F9P  │  │  BNO055  │   │
│  │ (largest)│  │  (IMU)   │   │
│  └──────────┘  └──────────┘   │
│                                │
│  ┌──────────┐  ┌──────────┐   │
│  │  BMP390  │  │  DS3231  │   │
│  │ (baro)   │  │  (RTC)   │   │
│  └──────────┘  └──────────┘   │
│                                │
│  ┌──────────────────────────┐  │
│  │  PCA9615 diff I2C buffer │  │
│  └──────────────────────────┘  │
│                                │
│  [M8 connector]   [SMA conn]  │
└────────────────────────────────┘
```

## I2C Communication: PCA9615 Differential Bus

### Problem

Standard I2C (SDA/SCL) is designed for short distances (<30 cm on a PCB). The sensor pod connects to the host electronics via a 1-2m cable. At this distance, standard I2C is unreliable due to:
- Capacitive loading (cable adds 100-200 pF)
- Ground noise between separated boards
- EMI pickup on long runs

### Solution: PCA9615 Differential I2C Buffer

The NXP PCA9615 converts standard I2C to differential signaling (like RS-485) for cable transmission:

```
Host ESP32                                          Sensor Pod
┌──────────┐     Cat5 STP Cable (1-2m)      ┌──────────────────┐
│          │                                 │                  │
│  SDA ────┤──► PCA9615 ──► SDA_D+/SDA_D- ──► PCA9615 ──► SDA │──► Sensors
│  SCL ────┤──► (host)  ──► SCL_D+/SCL_D- ──► (pod)   ──► SCL │
│          │              ──► VCC, GND      ──►               │
│          │                                 │                  │
└──────────┘                                 └──────────────────┘
```

### Cable Specification

| Parameter | Value |
|-----------|-------|
| Cable type | Cat5 STP (Shielded Twisted Pair) |
| Length | 1-2m |
| Pair 1 | SDA_D+ / SDA_D- (differential SDA) |
| Pair 2 | SCL_D+ / SCL_D- (differential SCL) |
| Pair 3 | VCC / GND (power) |
| Pair 4 | Reserved / Shield drain |
| Shield | Connected to GND at host end only (prevents ground loop) |

### I2C Speed

- Standard mode (100 kHz) for reliability over cable
- PCA9615 supports up to 400 kHz but 100 kHz provides margin
- At 100 kHz with 5 sensors, bus utilization is ~15% (plenty of headroom)

## Shared Usage: Pathfinder ↔ HIRT

### Pathfinder Usage

When connected to Pathfinder:
- GPS provides RTK position at each measurement cycle (10 Hz)
- IMU provides tilt/orientation for gradient correction (100 Hz, downsampled)
- Barometer provides altitude cross-check and temperature
- RTC provides precise timestamps for data correlation

### HIRT Usage

When connected to HIRT:
- GPS records probe insertion point positions (one-shot per probe)
- IMU provides probe inclination (if pod is attached to probe head)
- RTC provides timestamps for correlation with Pathfinder data
- Barometer provides environmental temperature for thermal correction

### Switchover Procedure

1. Power off current instrument
2. Disconnect M8 cable from current host
3. Move pod + GPS antenna to new instrument
4. Connect M8 cable to new host
5. Power on — host auto-detects pod via I2C bus scan

No configuration changes needed. Both instruments detect the same I2C addresses and configure sensors identically.

## Power Design

### From Host

The pod receives power through the M8 cable from the host instrument:

| Parameter | Value |
|-----------|-------|
| Supply voltage | 3.3V (from host's 3.3V LDO) |
| Total current | ~82 mA (ZED-F9P: 68 mA, BNO055: 12 mA, BMP390: <1 mA, DS3231: <1 mA, PCA9615: 1 mA) |
| Cable voltage drop | ~50 mV at 82 mA over 1m Cat5 (26 AWG) |
| Effective voltage at pod | ~3.25V — within spec for all sensors |

### Battery Backup (DS3231 only)

- CR2032 coin cell inside pod for DS3231 RTC backup
- Maintains timekeeping when pod is disconnected
- Battery life: >5 years (DS3231 draws <1 μA from backup battery)

## ESD Protection

All I2C lines at the M8 connector have ESD protection:

| Component | Part | Rating |
|-----------|------|--------|
| ESD clamp diodes | TPD4E05U06 (TI) | ±8 kV contact, ±15 kV air |
| Location | Inside pod, immediately after M8 connector |
| Note | Adds ~5 pF per line — within I2C capacitance budget |

## BOM (Sensor Pod)

| Component | Qty | Est. Cost |
|-----------|-----|-----------|
| u-blox ZED-F9P breakout (SparkFun GPS-RTK2) | 1 | $220 |
| L1+L2 active helical antenna | 1 | $40 |
| BNO055 breakout (Adafruit 4646) | 1 | $30 |
| BMP390 breakout (Adafruit 4816) | 1 | $10 |
| DS3231 breakout (generic) | 1 | $5 |
| PCA9615 breakout ×2 (one for pod, one for host) | 2 | $10 |
| IP67 enclosure (~80×80×35mm) | 1 | $8 |
| M8 4-pin connector pair (male + female) | 1 | $8 |
| SMA bulkhead connector | 1 | $3 |
| Cat5 STP cable 1.5m (with M8 termination) | 1 | $10 |
| TPD4E05U06 ESD protection | 1 | $2 |
| CR2032 battery | 1 | $1 |
| Prototype board, wire, headers | - | $5 |
| **Total** | | **~$352** |

**Note**: The ZED-F9P + antenna dominate the cost (~$260 of $352). The pod is shared between instruments, so this cost is amortized across both Pathfinder and HIRT.

## References

- PCA9615 datasheet: NXP Semiconductors
- ZED-F9P integration manual: u-blox F9 HPS
- BNO055 datasheet: Bosch Sensortec BST-BNO055-DS000
- IP67 enclosure specifications: Hammond Manufacturing 1554 series
