# Pathfinder Gradiometer Firmware

Arduino firmware for the Pathfinder 4-pair fluxgate gradiometer system.

## Overview

This firmware runs on an Arduino Nano and manages:
- 8 fluxgate sensors (4 pairs) via 2x ADS1115 ADCs
- GPS positioning via NEO-6M module
- SD card data logging
- Pace beeper for walking cadence
- Status LED for system health

## Hardware Requirements

### Electronics
- **Microcontroller**: Arduino Nano (ATmega328P)
- **ADCs**: 2x ADS1115 16-bit I2C modules
- **GPS**: NEO-6M or compatible UART GPS module
- **Storage**: SD card module (SPI interface)
- **Audio**: Piezo buzzer or small speaker
- **Power**: 7.4V LiPo battery with 5V regulator for Arduino

### Sensors
- 8x Fluxgate sensors (e.g., FG Sensors DIY kit)
- Configured as 4 gradiometer pairs (top/bottom)

## Pin Connections

### Arduino Nano Pin Assignments

| Pin | Function | Connection |
|-----|----------|------------|
| **I2C Bus** |
| A4 | SDA | ADS1115 modules (both) |
| A5 | SCL | ADS1115 modules (both) |
| **SPI Bus (SD Card)** |
| 10 | CS | SD card module CS |
| 11 | MOSI | SD card module DI |
| 12 | MISO | SD card module DO |
| 13 | SCK | SD card module CLK |
| **GPS (Software Serial)** |
| 4 | RX | GPS TX pin |
| 3 | TX | GPS RX pin (not used) |
| **User Interface** |
| 9 | Beeper | Piezo buzzer (+) |
| 2 | Status LED | LED anode (+ resistor) |
| GND | Ground | Common ground for all modules |
| 5V | Power | Power for GPS, SD card, ADCs |

### ADS1115 ADC Connections

**Module 1 (Address 0x48)**: Gradiometer Pairs 1 and 2

| Channel | Sensor Connection |
|---------|------------------|
| A0 | Pair 1 Top sensor output |
| A1 | Pair 1 Bottom sensor output |
| A2 | Pair 2 Top sensor output |
| A3 | Pair 2 Bottom sensor output |

**Module 2 (Address 0x49)**: Gradiometer Pairs 3 and 4

| Channel | Sensor Connection |
|---------|------------------|
| A0 | Pair 3 Top sensor output |
| A1 | Pair 3 Bottom sensor output |
| A2 | Pair 4 Top sensor output |
| A3 | Pair 4 Bottom sensor output |

**I2C Address Configuration:**
- Module 1: Connect ADDR pin to GND (address 0x48)
- Module 2: Connect ADDR pin to VDD (address 0x49)

## Building and Uploading

### Prerequisites

1. **Install PlatformIO**
   - [Visual Studio Code](https://code.visualstudio.com/) with PlatformIO extension, OR
   - [PlatformIO Core CLI](https://docs.platformio.org/en/latest/core/installation.html)

2. **Required Libraries** (auto-installed by PlatformIO):
   - `Adafruit ADS1X15` - ADS1115 ADC driver
   - `TinyGPSPlus` - GPS NMEA parsing
   - `SdFat` - Fast SD card library

### Build and Upload

#### Using VS Code + PlatformIO Extension

1. Open the `firmware/` folder in VS Code
2. PlatformIO will detect the project automatically
3. Click the "Upload" button (→) in the bottom toolbar
4. Monitor serial output with the "Serial Monitor" button

#### Using PlatformIO CLI

```bash
# Navigate to firmware directory
cd /development/projects/active/Pathfinder/firmware/

# Build firmware
pio run

# Upload to Arduino Nano
pio run --target upload

# Open serial monitor
pio device monitor
```

### First-Time Upload

If upload fails with "programmer not responding":
1. Check USB connection
2. Verify correct port in `platformio.ini` (update `upload_port`)
3. Try holding RESET button while uploading
4. Some clones need `upload_speed = 57600` instead of 115200

## Configuration

All user-configurable parameters are in `include/config.h`:

### Key Settings

```cpp
// Sample rate (Hz)
#define SAMPLE_RATE_HZ  10      // Change to 20 for faster acquisition

// Pace beeper interval (ms)
#define BEEP_INTERVAL_MS 1000   // 1 beep per second = 1 m/s pace

// GPS baud rate
#define GPS_BAUD 9600           // NEO-6M default

// ADC gain setting
#define ADC_GAIN GAIN_ONE       // +/- 4.096V range

// Enable debug output
#define SERIAL_DEBUG 1          // Set to 0 to disable for battery savings
```

Edit these values and re-upload to customize behavior.

## Operation

### Startup Sequence

1. Power on system
2. **Status LED blinks rapidly** during initialization
3. Two quick beeps indicate startup complete
4. **Status LED pattern** indicates system state:
   - **Fast blink (100ms)**: Error (SD card failure)
   - **Medium blink (500ms)**: No GPS fix
   - **Slow blink (2s)**: Logging normally

### In the Field

1. Wait for **slow blink** (GPS locked)
2. Walk at steady pace (1 beep per second ≈ 1 m/s)
3. Maintain level orientation
4. Data logs automatically to SD card as `PATHXXXX.CSV`

### Data Retrieval

1. Power off system
2. Remove SD card
3. Copy CSV files to computer
4. Files increment automatically: `PATH0001.CSV`, `PATH0002.CSV`, etc.

## Data Format

### CSV Structure

```csv
timestamp,lat,lon,g1_top,g1_bot,g1_grad,g2_top,g2_bot,g2_grad,g3_top,g3_bot,g3_grad,g4_top,g4_bot,g4_grad
1234567,51.2345678,18.3456789,12450,12678,228,12401,12623,222,12389,12598,209,12434,12656,222
```

### Column Definitions

- **timestamp**: Milliseconds since Arduino boot
- **lat/lon**: GPS coordinates (decimal degrees, WGS84)
- **gN_top**: Top sensor raw ADC value (16-bit signed)
- **gN_bot**: Bottom sensor raw ADC value
- **gN_grad**: Gradient (bottom - top)

### Converting to Physical Units

Raw ADC values → Voltage:
```
voltage = (ADC_value / 32768) × 4.096V
```

Voltage → Magnetic field depends on your fluxgate sensor specifications.
Consult sensor datasheet for mV/µT conversion factor.

## Troubleshooting

### No SD Card Detected

- Check wiring to SD module (SPI pins)
- Verify SD card is formatted as FAT32
- Ensure CS pin is correct (`SD_CS_PIN = 10`)
- Try different SD card (some old cards not compatible)

### GPS Not Locking

- Ensure clear view of sky (no metal/trees blocking)
- Wait 30-60 seconds for cold start
- Check GPS RX pin connection (Arduino D4 → GPS TX)
- Verify GPS baud rate (default 9600)

### ADS1115 Not Found

- Check I2C wiring (SDA=A4, SCL=A5)
- Verify I2C addresses:
  - Module 1: ADDR pin → GND (0x48)
  - Module 2: ADDR pin → VDD (0x49)
- Run I2C scanner to detect devices
- Ensure 5V power to ADS1115

### Erratic Readings

- Check sensor power supply stability
- Ensure proper grounding (all grounds connected)
- Move sensors away from electronics noise
- Shield cables with grounded braid
- Lower ADC data rate for less noise (`ADC_DATA_RATE = 64`)

### Serial Monitor Shows Garbage

- Set baud rate to 115200 in monitor
- Check `SERIAL_BAUD` in `config.h` matches monitor

## Advanced Modifications

### Adding Calibration

Insert calibration coefficients in `readGradiometers()`:

```cpp
// After reading raw values
reading.g1_top = (reading.g1_top - offset1_top) * scale1_top;
reading.g1_bot = (reading.g1_bot - offset1_bot) * scale1_bot;
```

### Changing Sample Rate Dynamically

Add button to cycle through rates:

```cpp
const uint8_t rates[] = {10, 20, 50};
uint8_t currentRate = 0;

// In button interrupt:
currentRate = (currentRate + 1) % 3;
SAMPLE_RATE_HZ = rates[currentRate];
```

### Adding Real-Time Clock

Replace `millis()` timestamps with DS3231 RTC for absolute time:

```cpp
#include <RTClib.h>
RTC_DS3231 rtc;

// In setup():
rtc.begin();

// In logReading():
DateTime now = rtc.now();
logFile.print(now.timestamp());
```

## Performance Notes

### Memory Usage

- **Flash**: ~18 KB / 32 KB (56% used)
- **RAM**: ~1.2 KB / 2 KB (60% used)

Remaining RAM is tight. Avoid large arrays or String objects.

### Sample Rate Limits

- **10 Hz**: Rock solid, recommended
- **20 Hz**: Stable for most sensors
- **50 Hz**: Maximum practical (depends on ADC data rate)

Higher rates reduce per-sample noise averaging. Balance speed vs. quality.

### Battery Life Estimate

At 10 Hz sample rate (Modeled, MCU + peripherals only — excludes fluxgate sensors):
- Arduino Nano: ~60 mA
- ADS1115 (2x): ~2 mA
- GPS module: ~40 mA
- SD writes: ~20 mA average
- **MCU + peripherals**: ~120 mA
- **Full system with 8 fluxgates**: ~305 mA typical (see `hardware/schematics/main-board.md`)

With 2000 mAh LiPo @ 7.4V (80% regulator efficiency):
- MCU + peripherals only: ~13 hours
- Full system: ~5.2 hours

## License

MIT License. See [LICENSE](../LICENSE) in the project root.

## Contributing

Improvements welcome! Please document changes and test thoroughly before submitting.

## References

- [ADS1115 Datasheet](https://www.ti.com/lit/ds/symlink/ads1115.pdf)
- [NEO-6M GPS Manual](https://www.u-blox.com/sites/default/files/products/documents/NEO-6_DataSheet_(GPS.G6-HW-09005).pdf)
- [TinyGPS++ Documentation](http://arduiniana.org/libraries/tinygpsplus/)
- [SdFat Library](https://github.com/greiman/SdFat)

## Support

For hardware questions, see `/hardware/` directory.
For design rationale, see `/docs/design-concept.md`.
