# Pathfinder Wiring Guide

Complete wiring instructions for assembling the Pathfinder gradiometer electronics.

## Overview

The system consists of:
- 1x Arduino Nano (microcontroller)
- 2x ADS1115 ADC modules (16-bit, I2C)
- 1x NEO-6M GPS module (UART)
- 1x SD card module (SPI)
- 1x Piezo buzzer (pace beeper)
- 1x LED (status indicator)
- 8x Fluxgate sensors (gradiometer pairs)
- 1x 7.4V LiPo battery with 5V regulator

## Power Distribution

### Power Supply Requirements

- **System voltage**: 5V regulated
- **Total current**: ~150 mA typical, 200 mA peak (Modeled, electronics only — excludes fluxgate sensors)
- **Battery**: 7.4V 2S LiPo (2000+ mAh recommended)
- **Regulator**: 5V buck converter or 7805 linear regulator

### Power Wiring

```
[7.4V LiPo Battery]
       |
       +--[Power Switch]--[5V Regulator]--+
                                          |
          +-------------------------------+
          |               |               |               |
    [Arduino Nano]    [ADS1115 #1]   [ADS1115 #2]    [GPS Module]
          |               |               |               |
          +---------------+---------------+---------------+
                              |
                         [Common GND]
```

**Important**: All grounds must be connected together (common ground).

## I2C Bus (ADS1115 ADC Modules)

### ADS1115 Address Configuration

The two ADS1115 modules must have different I2C addresses:

**Module 1 (Address 0x48)**: ADDR pin → GND
**Module 2 (Address 0x49)**: ADDR pin → VDD

| Pin | Module 1 | Module 2 | Arduino Nano |
|-----|----------|----------|--------------|
| VDD | 5V | 5V | 5V |
| GND | GND | GND | GND |
| SCL | → | → | A5 (SCL) |
| SDA | → | → | A4 (SDA) |
| ADDR | GND | VDD | - |
| ALRT | (not used) | (not used) | - |

### I2C Bus Notes

- Add 4.7k pull-up resistors on SDA and SCL if not already on modules
- Keep I2C wires short (<30 cm) to reduce noise
- Use twisted pair for SDA/SCL if possible
- Maximum bus length: ~1 meter for reliable operation

## Sensor Connections (Fluxgate Channels)

### Gradiometer Pair Assignments

| Pair | Top Sensor | Bottom Sensor | ADS Module | Channels |
|------|------------|---------------|------------|----------|
| 1 | Sensor 1T | Sensor 1B | ADS1 (0x48) | A0, A1 |
| 2 | Sensor 2T | Sensor 2B | ADS1 (0x48) | A2, A3 |
| 3 | Sensor 3T | Sensor 3B | ADS2 (0x49) | A0, A1 |
| 4 | Sensor 4T | Sensor 4B | ADS2 (0x49) | A2, A3 |

### ADS1115 Module 1 (0x48) - Pairs 1 and 2

```
ADS1115 Module 1
   A0 ← Pair 1 Top sensor output
   A1 ← Pair 1 Bottom sensor output
   A2 ← Pair 2 Top sensor output
   A3 ← Pair 2 Bottom sensor output
  GND ← Common ground to all sensors
```

### ADS1115 Module 2 (0x49) - Pairs 3 and 4

```
ADS1115 Module 2
   A0 ← Pair 3 Top sensor output
   A1 ← Pair 3 Bottom sensor output
   A2 ← Pair 4 Top sensor output
   A3 ← Pair 4 Bottom sensor output
  GND ← Common ground to all sensors
```

### Fluxgate Sensor Wiring

Each fluxgate sensor typically has 3 wires:

| Wire | Color (typical) | Connection |
|------|-----------------|------------|
| Signal Out | White/Yellow | ADS1115 analog input (A0-A3) |
| Power (+) | Red | 5V |
| Ground (-) | Black | Common GND |

**Check your sensor datasheet** - color coding varies by manufacturer.

### Sensor Cable Shielding

For best noise performance:
1. Use shielded cable for sensor connections
2. Connect shield to ground at electronics box end ONLY
3. Do not ground shield at sensor end (avoid ground loops)
4. Keep sensor cables away from power wires

## GPS Module (NEO-6M)

### GPS Connections

| NEO-6M Pin | Arduino Nano Pin | Notes |
|------------|------------------|-------|
| VCC | 5V | Power (check voltage - some need 3.3V!) |
| GND | GND | Common ground |
| TX | D4 (RX) | GPS transmit → Arduino receive |
| RX | D3 (TX) | GPS receive (not used for NEO-6M) |

**Voltage Warning**: Most NEO-6M modules accept 5V, but some require 3.3V. Check your module specs!

### GPS Antenna

- Use active GPS antenna with clear sky view
- Keep antenna away from metal objects
- Mount on non-conductive surface (plastic, fiberglass)
- Minimum 10 cm from electronics for best performance

## SD Card Module

### SD Card SPI Connections

| SD Module Pin | Arduino Nano Pin | SPI Signal |
|---------------|------------------|------------|
| VCC | 5V | Power |
| GND | GND | Ground |
| CS | D10 | Chip Select |
| MOSI | D11 | Master Out Slave In |
| MISO | D12 | Master In Slave Out |
| SCK | D13 | Serial Clock |

**Voltage Warning**: Some SD modules need 3.3V! Use a level-shifter module or 3.3V-compatible SD module.

### SD Card Format

- Use SD card ≤32 GB
- Format as FAT32 (not exFAT)
- Class 10 recommended for fast writes

## User Interface

### Status LED

| Component | Arduino Pin | Connection |
|-----------|-------------|------------|
| LED Anode (+) | D2 | Via 220Ω resistor |
| LED Cathode (-) | GND | Direct |

```
Arduino D2 ----[220Ω]---->(LED)---->GND
```

### Pace Beeper

| Component | Arduino Pin | Connection |
|-----------|-------------|------------|
| Piezo (+) | D9 | Direct or via transistor |
| Piezo (-) | GND | Direct |

**For passive buzzer**:
```
Arduino D9 ---->(Piezo)---->GND
```

**For active buzzer or speaker (louder)**:
```
Arduino D9 ----[1kΩ]---->| NPN transistor (2N2222)
                         |
                    (Buzzer+)
                         |
                        GND
```

## Complete Wiring Diagram

```
                           +5V BUS
                              |
        +---------------------+---------------------+
        |          |          |          |          |
    [Arduino]  [ADS1]    [ADS2]     [GPS]       [SD]
        |          |          |          |          |
        +---------------------+---------------------+
                          GND BUS

ARDUINO NANO:
  A4 (SDA) ----+-------- ADS1 SDA -------- ADS2 SDA
               |
  A5 (SCL) ----+-------- ADS1 SCL -------- ADS2 SCL
               |
  D4 ---------|----------|----------|------ GPS TX
  D3 (unused)
  D9 ---------|----------|----------|------ Buzzer (+)
  D2 ---------|----------|----------|------ LED (+) via 220Ω
  D10 --------|----------|----------|------ SD CS
  D11 --------|----------|----------|------ SD MOSI
  D12 --------|----------|----------|------ SD MISO
  D13 --------|----------|----------|------ SD SCK

ADS1115 #1 (ADDR=GND → 0x48):
  A0 ← Sensor 1 Top
  A1 ← Sensor 1 Bottom
  A2 ← Sensor 2 Top
  A3 ← Sensor 2 Bottom

ADS1115 #2 (ADDR=VDD → 0x49):
  A0 ← Sensor 3 Top
  A1 ← Sensor 3 Bottom
  A2 ← Sensor 4 Top
  A3 ← Sensor 4 Bottom
```

## Assembly Steps

### 1. Prepare Components

- [ ] Test all modules individually before assembly
- [ ] Check voltage requirements (5V vs 3.3V)
- [ ] Verify I2C addresses with scanner sketch
- [ ] Label all wires with masking tape

### 2. Build Power Distribution

- [ ] Connect 5V regulator to battery
- [ ] Add power switch in series
- [ ] Create common 5V and GND buses (use breadboard or terminal block)
- [ ] Test voltage output with multimeter (should be 5.0V ± 0.1V)

### 3. Wire I2C Bus

- [ ] Connect ADS1 to Arduino (SDA, SCL)
- [ ] Connect ADS2 to Arduino (SDA, SCL)
- [ ] Configure ADDR pins (ADS1→GND, ADS2→VDD)
- [ ] Add pull-up resistors if needed (4.7kΩ)

### 4. Connect Peripherals

- [ ] Wire GPS module to D4
- [ ] Wire SD card module (SPI pins)
- [ ] Connect status LED to D2 with resistor
- [ ] Connect buzzer to D9

### 5. Connect Sensors

- [ ] Label each sensor cable (1T, 1B, 2T, 2B, etc.)
- [ ] Connect sensors to correct ADS channels
- [ ] Ensure common ground to all sensors
- [ ] Route sensor cables away from power wires

### 6. Test and Debug

- [ ] Upload firmware
- [ ] Open serial monitor (115200 baud)
- [ ] Check for initialization messages
- [ ] Verify all modules detected
- [ ] Test GPS lock outdoors
- [ ] Verify SD card writes

## Troubleshooting

### Module Not Detected

**Symptom**: "ADS1115 not found" or "SD card failed"

**Solutions**:
1. Check power connections (5V and GND)
2. Verify wiring to correct pins
3. Test with multimeter for continuity
4. Try different I2C address (swap ADDR wiring)
5. Upload I2C scanner sketch to find devices

### Erratic Sensor Readings

**Symptom**: Noisy or unstable ADC values

**Solutions**:
1. Check sensor power supply (should be stable 5V)
2. Ensure good ground connections
3. Shield sensor cables
4. Separate sensor cables from power/data cables
5. Add 0.1µF capacitors across sensor power pins
6. Lower ADC data rate in config.h

### GPS Won't Lock

**Symptom**: LED blinks medium speed, no coordinates

**Solutions**:
1. Move to open area with sky view
2. Wait 1-2 minutes for cold start
3. Check GPS TX → Arduino D4 connection
4. Verify GPS power LED is lit
5. Try different GPS baud rate (some modules use 115200)

## Safety Notes

⚠️ **Battery Safety**
- Use proper LiPo charger (not car battery charger!)
- Never charge unattended
- Store in fireproof bag
- Check for damage before use
- Disconnect when not in use

⚠️ **Electrical Safety**
- Double-check polarity before powering on
- Use fuse or current limiter on battery
- Avoid short circuits (cover exposed connections)
- Work on non-conductive surface

## Enclosure Recommendations

### Electronics Box
- IP65 rated for weather resistance
- Minimum size: 15 x 10 x 5 cm
- Ventilation holes for heat
- Cable glands for sensor/GPS wires
- Mounting brackets for belt/harness

### Component Layout
- Arduino and modules on DIN rail or standoffs
- SD card accessible without opening box
- Status LED visible through window
- Buzzer holes for sound output

## Bill of Materials (Wiring)

| Item | Quantity | Notes |
|------|----------|-------|
| 22 AWG stranded wire (red) | 2m | Power (+) |
| 22 AWG stranded wire (black) | 2m | Ground (-) |
| 24 AWG stranded wire (colors) | 10m | Signals |
| Shielded cable 2-conductor | 5m | Sensor connections |
| Heat shrink tubing | 1m | Assorted sizes |
| Dupont connectors | 50 | Male/female |
| Terminal blocks 5mm | 4 | Power distribution |
| 220Ω resistor | 1 | LED current limit |
| 4.7kΩ resistor | 2 | I2C pull-ups (if needed) |
| Electrical tape | 1 roll | Wire bundling |
| Cable ties | 20 | Wire management |

## Next Steps

After wiring is complete:
1. See `README.md` for firmware upload instructions
2. See `QUICK_START.md` for operation guide
3. Perform calibration procedure (in development)
4. Conduct field test survey

## Support

For wiring issues, check:
- [Arduino Nano pinout](https://docs.arduino.cc/hardware/nano)
- [ADS1115 datasheet](https://www.ti.com/lit/ds/symlink/ads1115.pdf)
- Project repository issues page
