# Cross-Sensor Electromagnetic Interference Matrix

## Overview

The multi-sensor Pathfinder carries 15+ electromagnetic sources and sensitive receivers in close proximity. This document catalogs all interference pathways, rates their severity, and specifies mitigations. The interference analysis drove the TDM firmware architecture and physical layout decisions.

## Methodology

Each sensor/subsystem is classified as either a **source** (emitter of electromagnetic energy) or a **victim** (sensitive receiver that can be corrupted). Some devices are both. All 15 source-victim pairs are analyzed for coupling mechanism, severity, and mitigation.

## Severity Rating Scale

| Rating | Definition | Action Required |
|--------|-----------|-----------------|
| **CRITICAL** | Signal corruption exceeds target noise floor by >10× | Mandatory TDM + physical separation |
| **HIGH** | Signal corruption comparable to noise floor | TDM recommended + filtering |
| **MODERATE** | Measurable but manageable with filtering | Passive filtering sufficient |
| **LOW** | Negligible in normal operation | Monitor during bench testing |

## Complete Interference Matrix

### CRITICAL Interference Paths

#### 1. EMI TX Coil → Fluxgate Sensors
- **Coupling**: Direct magnetic field from TX coil at 15 kHz
- **Level at 1m separation**: ~2,100 nT **(Modeled)** — calculated from TX coil dipole field at 1m
- **Severity**: **CRITICAL** — 4000× above Pathfinder's 0.5 nT noise floor
- **Mitigation**:
  - **Primary**: TDM — EMI TX is OFF during fluxgate measurement phase (mandatory)
  - **Secondary**: Physical separation — TX coil at -50 cm, fluxgates at +25 to +100 cm on crossbar (1.25-1.5m separation)
  - **Tertiary**: LM2917 bandwidth filter provides 63 dB rejection at 15 kHz for any residual coupling during settling

#### 2. LiDAR Motor → Fluxgate Sensors
- **Coupling**: DC magnetic field from brushless motor permanent magnets + switching transients
- **Level at typical distance**: 100-300 nT DC field **(Modeled)**, plus broadband switching noise
- **Severity**: **CRITICAL** — DC component directly corrupts magnetic measurement
- **Mitigation**:
  - **Primary**: Physical separation — LiDAR at -100 cm (far end of crossbar), fluxgates at +25 to +100 cm (1.25-2.0m separation)
  - **Secondary**: LiDAR motor field falls off as 1/r³ (dipole); at 1.5m, 100 nT → ~3 nT
  - **Tertiary**: The DC motor field is static relative to measurement timescale — gradiometer subtraction removes most of it as common-mode
  - **Quaternary**: Consider TDM for LiDAR (motor off during fluxgate phase) if bench testing shows >1 nT residual

### HIGH Interference Paths

#### 3. LM2596 Buck Converter → ADS1115 ADC
- **Coupling**: Conducted via power rail, radiated from switching inductor
- **Switching frequency**: 150 kHz
- **Level**: 10-50 mV ripple on power rail without filtering **(Modeled)**
- **Severity**: **HIGH** — ripple exceeds ADS1115 LSB (125 μV) by 100×
- **Mitigation**:
  - Ferrite bead + LC filter between buck converter and analog supply
  - TPS7A49 LDO as final regulation stage (>60 dB PSRR at 150 kHz)
  - Separate analog and digital power domains

#### 4. AD9833 63rd Harmonic → GPS L1 Band
- **Coupling**: 15 kHz × 63 = 945 kHz is not at GPS L1 (1575.42 MHz), but higher harmonics of the DDS clock (25 MHz MCLK) can reach GPS bands
- **Level**: Potentially significant if DDS output is unfiltered
- **Severity**: **HIGH** — GPS receiver sensitivity is -130 dBm; even weak harmonics can desensitize
- **Mitigation**:
  - Low-pass filter on AD9833 output (fc = 50 kHz, 5th-order Butterworth)
  - Physical separation: GPS antenna on mast, AD9833/TX coil at -50 cm on crossbar
  - TDM: EMI TX off during GPS position fix (if degradation observed)

#### 5. ESP32 WiFi TX → Analog Signal Chain
- **Coupling**: 2.4 GHz RF rectification in analog amplifier inputs (AD8421, LM2917)
- **Level**: Depends on shielding; potentially 0.1-10 mV at amplifier input **(Modeled)**
- **Severity**: **HIGH** during active transmission
- **Mitigation**:
  - **Primary**: TDM — WiFi TX disabled during fluxgate measurement phase
  - **Secondary**: 100 pF bypass capacitors on all analog amplifier inputs (fc = 160 MHz with 10 kΩ source impedance)
  - **Tertiary**: Metal enclosure for analog electronics provides ~20 dB shielding at 2.4 GHz

### MODERATE Interference Paths

#### 6. WiFi → Fluxgate (Residual)
- **Coupling**: RF rectification after TDM + shielding
- **Level**: Estimated <0.1 nT after TDM mitigation **(Modeled)**
- **Severity**: **MODERATE** — acceptable if TDM is properly implemented
- **Mitigation**: TDM + RC filter + enclosure shielding

#### 7. I2C Clock → Analog Signals
- **Coupling**: Capacitive crosstalk from I2C SCL (100-400 kHz) to analog traces
- **Level**: ~1 mV on adjacent traces without guard **(Modeled)**
- **Severity**: **MODERATE**
- **Mitigation**: Route I2C on separate cable/PCB layer; ground guard traces; keep I2C away from analog inputs

#### 8. SD Card SPI → Analog Signals
- **Coupling**: SPI clock at 4-25 MHz near analog traces
- **Level**: Similar to I2C but higher frequency (more capacitive coupling)
- **Severity**: **MODERATE**
- **Mitigation**: TDM — SD card writes only during settling phase; physical separation on PCB

#### 9. GPS NMEA Serial → Analog Signals
- **Coupling**: UART at 9600-115200 baud near analog traces
- **Level**: Low — UART is low-frequency and differential-like
- **Severity**: **MODERATE-LOW**
- **Mitigation**: Route UART on opposite side of PCB from analog traces

#### 10. EMI TX → GPS
- **Coupling**: 15 kHz harmonics into GPS RF front-end
- **Level**: Low if TX output is filtered
- **Severity**: **MODERATE**
- **Mitigation**: LP filter on TX output; physical separation (TX at -50 cm, GPS on mast)

### LOW Interference Paths

#### 11. BNO055 IMU → Fluxgate
- **Coupling**: BNO055 has internal magnetometer with low-level excitation
- **Level**: Negligible at >20 cm separation
- **Severity**: **LOW**
- **Mitigation**: BNO055 configured in IMUPLUS mode (magnetometer disabled); maintain >20 cm from fluxgates

#### 12. MLX90614 → Analog
- **Coupling**: I2C communication and internal chopper amplifier
- **Level**: Negligible
- **Severity**: **LOW**

#### 13. DS3231 RTC → Analog
- **Coupling**: 32.768 kHz crystal oscillator
- **Level**: Negligible at any reasonable distance
- **Severity**: **LOW**

#### 14. BMP390 → Analog
- **Coupling**: Internal ADC switching
- **Level**: Negligible
- **Severity**: **LOW**

#### 15. Camera → Fluxgate
- **Coupling**: Image sensor clock and digital processing
- **Level**: Low if camera has its own shielded enclosure
- **Severity**: **LOW**
- **Mitigation**: Use standalone ESP32-CAM module with metal shield

## Interference Budget Summary

| Source | Victim | Severity | Primary Mitigation | Residual After Mitigation |
|--------|--------|----------|-------------------|--------------------------|
| EMI TX | Fluxgate | CRITICAL | TDM + separation | <0.01 nT **(Target)** |
| LiDAR motor | Fluxgate | CRITICAL | Separation + gradiometer | <1 nT **(Target)** |
| Buck converter | ADC | HIGH | Ferrite + LC + LDO | <0.2 LSB **(Target)** |
| AD9833 harmonics | GPS | HIGH | LP filter + separation | No degradation **(Target)** |
| WiFi TX | Analog | HIGH | TDM | <0.1 nT **(Target)** |
| I2C clock | Analog | MODERATE | PCB layout | <0.1 LSB **(Target)** |
| SD card SPI | Analog | MODERATE | TDM | Zero during measurement |
| All LOW paths | Various | LOW | Distance | Negligible |

## TDM Phase Assignments

Based on this interference analysis, the TDM cycle must enforce:

| Phase | Duration | Active | Disabled |
|-------|----------|--------|----------|
| Fluxgate measurement | 50 ms | Fluxgates, ADC, GPS RX | EMI TX, WiFi TX, SD card, LiDAR (optional) |
| EMI TX/RX | 30 ms | EMI coils, ADC (I/Q) | WiFi TX |
| Settling / comms | 20 ms | WiFi TX, SD card write, MQTT | EMI TX |

See `tdm-firmware-design.md` for implementation details.
