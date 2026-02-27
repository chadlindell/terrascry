# Pathfinder Firmware Documentation Index

Complete guide to building, configuring, and operating the Pathfinder gradiometer firmware.

## Quick Navigation

### For First-Time Builders
1. **[WIRING.md](WIRING.md)** - Complete wiring instructions and pin connections
2. **[README.md](README.md)** - Build and upload firmware
3. **[CALIBRATION.md](CALIBRATION.md)** - Calibrate and validate system
4. **[QUICK_START.md](QUICK_START.md)** - Field operation guide

### For Experienced Users
- **[platformio.ini](platformio.ini)** - PlatformIO project configuration
- **[include/config.h](include/config.h)** - User-configurable parameters
- **[src/main.cpp](src/main.cpp)** - Main firmware source code

### For Data Analysis
- **[tools/README.md](tools/README.md)** - Data processing utilities
- **[tools/visualize_data.py](tools/visualize_data.py)** - Quick field visualization
- **[tools/example_data.csv](tools/example_data.csv)** - Sample data for testing

## Document Overview

### WIRING.md
**Complete Hardware Assembly Guide**

- Power distribution and regulation
- I2C bus configuration (ADS1115 ADCs)
- Sensor channel assignments
- GPS and SD card connections
- LED and beeper wiring
- Troubleshooting hardware issues

**Start here** if assembling electronics for the first time.

### README.md
**Firmware Build and Upload Instructions**

- Installing PlatformIO
- Required libraries (auto-installed)
- Building and uploading firmware
- Configuration options
- Serial debugging
- CSV data format specification
- Troubleshooting compilation/upload issues

**Start here** once hardware is wired and ready for programming.

### CALIBRATION.md
**System Calibration and Validation**

- Level 1: Offset calibration
- Level 2: Noise floor measurement
- Level 3: Dynamic response tests
- Level 4: Inter-pair consistency
- Level 5: GPS accuracy validation
- Applying calibration in post-processing
- Recalibration schedule
- Temperature compensation (advanced)

**Start here** after firmware upload to establish baseline performance.

### QUICK_START.md
**Field Operations Reference Card**

- Pre-flight checklist
- Power-on sequence
- Status LED meanings
- Survey walking technique
- Data download procedure
- Quick troubleshooting
- Field tips (battery, weather, data quality)

**Print this** and keep in field equipment case for quick reference.

### include/config.h
**Configuration Parameters**

All user-customizable settings:
- Sample rate (default: 10 Hz)
- Beep interval (default: 1000 ms)
- GPS baud rate (default: 9600)
- ADC gain and data rate
- Pin assignments
- Debug output settings
- CSV filename pattern

Edit this file to customize behavior without modifying main firmware.

### src/main.cpp
**Main Firmware Source**

Complete Arduino code:
- Hardware initialization (ADC, GPS, SD card)
- Data acquisition loop
- Gradiometer reading and calculation
- GPS parsing and position logging
- CSV file writing
- User interface (LED patterns, beeper)

Well-commented for DIY builders to understand and extend.

### tools/visualize_data.py
**Field Data Visualization**

Python script for quick data checks:
```bash
# View all pairs as time series
python visualize_data.py PATH0001.CSV

# Generate spatial map
python visualize_data.py PATH0001.CSV --map

# Print statistics only
python visualize_data.py PATH0001.CSV --stats-only
```

Requires: `pandas`, `matplotlib` (install via pip)

## Typical Workflow

### Phase 1: Assembly
1. Read **WIRING.md** thoroughly
2. Gather all components (see BOM in WIRING.md)
3. Wire power distribution
4. Connect I2C bus (ADS1115 modules)
5. Wire peripherals (GPS, SD, LED, beeper)
6. Connect sensors to ADC channels
7. Double-check all connections

### Phase 2: Programming
1. Install PlatformIO (see **README.md**)
2. Open firmware folder in VS Code
3. Review **config.h** settings
4. Upload firmware to Arduino Nano
5. Open serial monitor (115200 baud)
6. Verify initialization messages

### Phase 3: Calibration
1. Follow **CALIBRATION.md** procedures
2. Perform static baseline measurement
3. Check noise floor
4. Test ferrous object response
5. Verify GPS lock
6. Record calibration values

### Phase 4: Field Operation
1. Use **QUICK_START.md** checklist
2. Wait for GPS lock (slow LED blink)
3. Conduct survey at beeper pace
4. Download data from SD card
5. Visualize with **visualize_data.py**

### Phase 5: Data Processing
1. Transfer CSV files to computer
2. Run visualization scripts
3. Apply calibration offsets (see CALIBRATION.md)
4. Import to GIS software (QGIS, ArcGIS)
5. Generate gradient maps
6. Identify anomalies for investigation

## Hardware Requirements Summary

### Electronics
- Arduino Nano (ATmega328P)
- 2× ADS1115 16-bit ADC modules
- NEO-6M GPS module
- SD card module (SPI)
- Piezo buzzer
- LED + 220Ω resistor
- 7.4V LiPo battery (2000+ mAh)
- 5V regulator (buck converter recommended)

### Sensors
- 8× Fluxgate sensors (e.g., FG Sensors)
- Configured as 4 vertical gradiometer pairs
- 50 cm top-bottom separation
- 50 cm horizontal spacing

### Tools Needed
- Soldering iron
- Multimeter
- Wire strippers
- Heat shrink gun
- Small screwdrivers
- Hot glue gun (optional, for strain relief)

## Software Requirements

### Build Environment
- PlatformIO IDE or PlatformIO Core CLI
- Visual Studio Code (recommended) or other editor
- USB driver for Arduino Nano (CH340 or FTDI)

### Data Processing
- Python 3.7+
- pandas library (`pip install pandas`)
- matplotlib library (`pip install matplotlib`)

### Optional Tools
- Arduino IDE (for troubleshooting)
- QGIS (for spatial visualization)
- OpenSCAD (for custom mechanical parts)

## Support and Resources

### Documentation
- All docs in `/firmware/` directory
- Design concept: `/docs/design-concept.md`
- Hardware specs: `/hardware/` directory

### Code Repository
- Firmware: `/firmware/src/main.cpp`
- Configuration: `/firmware/include/config.h`
- Tools: `/firmware/tools/`

### External References
- [ADS1115 Datasheet](https://www.ti.com/lit/ds/symlink/ads1115.pdf)
- [NEO-6M GPS Manual](https://www.u-blox.com/sites/default/files/products/documents/NEO-6_DataSheet_(GPS.G6-HW-09005).pdf)
- [TinyGPS++ Library](http://arduiniana.org/libraries/tinygpsplus/)
- [SdFat Library Docs](https://github.com/greiman/SdFat)
- [Arduino Nano Pinout](https://docs.arduino.cc/hardware/nano)

### Community Support
- GitHub Issues: [Project repository issues page]
- Discussions: [Project repository discussions]
- Email: [Maintainer contact info]

## Contributing

Improvements and extensions welcome! Areas for contribution:
- Additional sensor types (magnetometers, accelerometers)
- Real-time display (LCD/OLED)
- Bluetooth data streaming
- Advanced filtering algorithms
- Machine learning anomaly detection
- Alternative microcontroller ports (ESP32, STM32)

See project repository for contribution guidelines.

## Changelog

### Version 1.0 (Initial Release)
- Basic 8-channel gradiometer acquisition
- GPS position logging
- SD card CSV output
- Pace beeper for walking cadence
- Status LED patterns
- Configurable sample rate
- Debug serial output

### Planned Features
- Real-time clock (RTC) for absolute timestamps
- Battery voltage monitoring
- EEPROM storage for calibration values
- Advanced filtering (moving average, median)
- Anomaly detection alerts
- Multi-file session management

## License

[Specify your open-source license - e.g., MIT, GPL-3.0, Apache-2.0]

## Acknowledgments

Based on proven gradiometer designs:
- Bartington Grad601 trapeze configuration
- Commercial multi-channel arrays (Foerster, SENSYS)
- Open-source geophysics tools

Built with excellent open-source libraries:
- Adafruit ADS1X15 (Adafruit Industries)
- TinyGPS++ (Mikal Hart)
- SdFat (Bill Greiman)

---

**Last Updated**: 2026-02-02
**Firmware Version**: 1.0
**Documentation Status**: Complete
