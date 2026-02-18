# I2C Address Map

## Overview

The Pathfinder ESP32 manages two separate I2C buses to isolate local high-speed analog sensors from the remote sensor pod. This document provides the complete address map and confirms there are no conflicts.

## Bus Architecture

```
ESP32
├── I2C Bus 0 (GPIO 21/22) ── Local sensors on main PCB
│   ├── ADS1115 #1 ── 0x48 (ADDR → GND)
│   ├── ADS1115 #2 ── 0x49 (ADDR → VDD)
│   └── MLX90614  ── 0x5A
│
└── I2C Bus 1 (GPIO 16/17) ── Sensor Pod via PCA9615
    ├── ZED-F9P   ── 0x42
    ├── BNO055    ── 0x29 (AD0 = HIGH)
    ├── BMP390    ── 0x77
    └── DS3231    ── 0x68
```

## Complete Address Table

### Bus 0: Local Sensors

| Address | Device | Config | Notes |
|---------|--------|--------|-------|
| 0x48 | ADS1115 #1 | ADDR → GND | Fluxgate channels 1-4 (pairs 1-2) |
| 0x49 | ADS1115 #2 | ADDR → VDD | Fluxgate channels 5-8 (pairs 3-4) |
| 0x5A | MLX90614 | Default | IR temperature sensor |

**Unused addresses on Bus 0**: 0x4A, 0x4B available for additional ADS1115 (EMI I/Q channels)

### Bus 1: Sensor Pod (via PCA9615 differential I2C)

| Address | Device | Config | Notes |
|---------|--------|--------|-------|
| 0x29 | BNO055 | AD0 = HIGH | IMU (IMUPLUS mode, magnetometer OFF) |
| 0x42 | ZED-F9P | Default | RTK GPS receiver |
| 0x68 | DS3231 | Fixed | Precision RTC |
| 0x77 | BMP390 | SDO → VDD | Barometric pressure + temperature |

## Conflict Analysis

### Bus 0 Conflicts: **NONE**

| Address | Device 1 | Device 2 | Conflict? |
|---------|----------|----------|-----------|
| 0x48 | ADS1115 #1 | — | No |
| 0x49 | ADS1115 #2 | — | No |
| 0x5A | MLX90614 | — | No |

### Bus 1 Conflicts: **NONE**

| Address | Device 1 | Device 2 | Conflict? |
|---------|----------|----------|-----------|
| 0x29 | BNO055 | — | No |
| 0x42 | ZED-F9P | — | No |
| 0x68 | DS3231 | — | No |
| 0x77 | BMP390 | — | No |

### Cross-Bus Conflicts: **N/A**

Buses are physically separate (different GPIO pins). No cross-bus conflicts possible.

## Address Configuration Details

### ADS1115 Address Selection

The ADS1115 address is set by connecting the ADDR pin:

| ADDR Pin | I2C Address |
|----------|-------------|
| GND | 0x48 |
| VDD | 0x49 |
| SDA | 0x4A |
| SCL | 0x4B |

For Pathfinder: #1 → GND (0x48), #2 → VDD (0x49). If EMI I/Q channels need dedicated ADC, use #3 → SDA (0x4A) on Bus 0.

### BNO055 Address Selection

| AD0 Pin | I2C Address |
|---------|-------------|
| LOW (GND) | 0x28 |
| HIGH (VDD) | 0x29 |

Using 0x29 (AD0 HIGH) to avoid potential conflict with future devices at 0x28.

### BMP390 Address Selection

| SDO Pin | I2C Address |
|---------|-------------|
| GND | 0x76 |
| VDD | 0x77 |

Using 0x77 (SDO HIGH). Note: 0x76 conflicts with some MS5611 barometer breakouts if those were ever added.

## Firmware Bus Initialization

```c
// ESP32 I2C bus initialization
#include <Wire.h>

// Bus 0: Local sensors (main PCB)
#define I2C_BUS0_SDA 21
#define I2C_BUS0_SCL 22
#define I2C_BUS0_FREQ 400000  // 400 kHz (short traces on PCB)

// Bus 1: Sensor pod (via PCA9615, over Cat5 cable)
#define I2C_BUS1_SDA 16
#define I2C_BUS1_SCL 17
#define I2C_BUS1_FREQ 100000  // 100 kHz (conservative for 1-2m cable)

TwoWire I2C_Local = TwoWire(0);   // Bus 0
TwoWire I2C_Pod   = TwoWire(1);   // Bus 1

void setup() {
    I2C_Local.begin(I2C_BUS0_SDA, I2C_BUS0_SCL, I2C_BUS0_FREQ);
    I2C_Pod.begin(I2C_BUS1_SDA, I2C_BUS1_SCL, I2C_BUS1_FREQ);

    // Scan both buses
    scan_bus(&I2C_Local, "Bus 0 (Local)");
    scan_bus(&I2C_Pod, "Bus 1 (Pod)");
}
```

## Future Expansion Capacity

### Bus 0 Available Addresses

| Address | Potential Device |
|---------|-----------------|
| 0x4A | ADS1115 #3 (EMI I channel) |
| 0x4B | ADS1115 #4 (EMI Q channel) |
| 0x50-0x57 | EEPROM (calibration data storage) |

### Bus 1 Available Addresses

| Address | Potential Device |
|---------|-----------------|
| 0x1D | ADXL345 accelerometer (if separate from BNO055) |
| 0x53 | ADXL345 alt address |
| 0x76 | Second barometer (if needed) |

## HIRT Compatibility

When the sensor pod is connected to HIRT, the same Bus 1 address map applies. HIRT's local I2C bus (Bus 0) has its own sensors that do not conflict with the pod addresses:

| HIRT Bus 0 Address | Device | Conflicts with Pod? |
|---------------------|--------|-------------------|
| 0x48 | AD7124-8 ADC | No (different bus) |
| 0x1D | ADXL345 inclinometer | No (different bus) |

**No conflicts between Pathfinder pod sensors and HIRT local sensors.**
