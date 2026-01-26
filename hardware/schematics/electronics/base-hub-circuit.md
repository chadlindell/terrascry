# Base Hub Circuit Schematic - Detailed Design

## Overview

Complete circuit design for the base hub/control unit, acting as the central "Active Brain" for the passive probe array. This unit houses the MIT signal generator, ERT current source, all analog signal processing (TX drivers, RX amplifiers), and the main Data Acquisition (DAQ) system.

**Note:** ERT current source details are in [ERT Circuit](ert-circuit.md). This document focuses on base hub-specific circuits.

---

## Power Management System

### Battery Input and Protection

**Battery:** 12V sealed lead-acid (12 Ah) or LiFePO4

**Input Circuit:**
```
Battery+ ──→ Fuse (5A) ──→ Power Switch ──→ Distribution
            (0287005.PXCN)   (DPST switch)
```

**Fuse Protection:**
- Fuse: 5A fast-blow (0034.5002)
- Protects against shorts

**Power Switch:**
- DPDT switch for main power
- Allows battery disconnect

### Voltage Regulation

**12V Rail (High Power):**
- Powers: TX Drivers (High Voltage Op-Amps)
- Powers: ERT Current Source (High Compliance)
- Direct from battery (Fused)

**5V Rail (Analog Low Noise):**
```
12V Input ──→ LM2596 Module ──→ LDO (Low Noise) ──→ 5V Analog
              (Buck Pre-reg)    (TPS7A49 or similar)
```
- Powers: Instrumentation Amps, ADC Reference
- Critical: Needs low ripple/noise

**3.3V Rail (Digital):**
```
5V Input ──→ AMS1117-3.3 ──→ 3.3V Output
             (LDO)            (for MCU, digital)
```
- Powers: MCU (ESP32), Logic, Digital I/O

---

## Analog Measurement System (The "Passive Probe" Interface)

Since the probes are passive (coils/electrodes only), the Base Hub handles all signal conditioning.

### 1. MIT Receive Chain (Analog Front End)

**Inputs:** Differential pairs coming from the Probe RX MUX.

**Circuit:**
```
Multiplexed Input ──→ RX Preamp (LNA) ──→ Bandpass Filter ──→ ADC
                      (AD620/INA128)      (Active Filter)     (ADS1256)
```

**RX Preamp (LNA):**
- Component: AD620 or INA128
- Gain: Selectable (x10, x100) via relay or digital pot
- Purpose: Boost uV signals from passive coils to mV levels

**ADC Interface:**
- Component: ADS1256IDBR (24-bit ADC)
- Sample Rate: 30 kS/s (sufficient for digital lock-in at <5kHz)
- Input: Differential

### 2. ERT Voltage Measurement

**Inputs:** Differential pairs from the ERT MUX (connected to Probe Rings).

**Circuit:**
```
Multiplexed Rings ──→ Protection TVS ──→ Instrumentation Amp ──→ ADC
                                         (INA128)                (ADS1256)
```

**Protection:**
- Critical: TVS diodes to protect sensitive inputs from static or TX drive spikes.

---

## Data Logger / User Interface

**Purpose:** Connect the headless Base Hub to the user's Tablet/Laptop for control and visualization.

**Options:**

1.  **WiFi (ESP32 Default):**
    - Creates "HIRT_Base_Station" Access Point.
    - Tablet connects via Web Browser or App.

2.  **USB Serial (Backup):**
    - Standard USB-to-UART (CP2102/FT232).
    - Wired connection for reliability.

---

## Control Logic (MCU)

**Component:** ESP32-WROOM-32 or STM32F4

**Responsibilities:**
1.  **Waveform Generation:** Control AD9833 (DDS) for MIT TX.
2.  **Multiplexing:** Control the massive relay/analog-mux array to switch between probes.
3.  **Data Acquisition:** Read ADS1256 via SPI.
4.  **DSP:** Perform digital lock-in detection (Demodulation) on raw ADC data.
5.  **Comms:** Stream processed data to Tablet.

---

## Component Summary

### Power Components
| Component | Part Number | Quantity | Notes |
|-----------|-------------|----------|-------|
| Fuse Holder | 0287005.PXCN | 1 | Panel mount |
| 5V Regulator | LM2596 + LDO | 1 | Low noise design essential |
| 3.3V Regulator | AMS1117-3.3 | 1 | Digital logic |

### Analog Front End
| Component | Part Number | Quantity | Notes |
|-----------|-------------|----------|-------|
| Preamp | AD620 / INA128 | 1-2 | Low noise instrumentation amp |
| ADC | ADS1256IDBR | 1 | 24-bit, High precision |
| Mux Array | CD4051 / ADG732 | Multiple | Handles 20+ probe inputs |
| Protection | TVS Diodes | Many | ESD protection for all inputs |

### Control & Interface
| Component | Part Number | Quantity | Notes |
|-----------|-------------|----------|-------|
| MCU | ESP32-WROOM | 1 | Main controller |
| DDS | AD9833 | 1 | Signal Generator |
| USB-Serial | CP2102 | 1 | Data logger backup |

---

## Enclosure and Mechanical

**Requirements:**
- **Connector Panel:** Needs high-density connector strategy (e.g., DB25 or Circular Military) to handle 20+ probes. *See Wiring Guide.*
- **Shielding:** Internal metal shielding for Analog Front End to prevent noise from MCU/Power supply.

---

## Design Notes

### Power Budget (Revised for Passive System)
- **Base Hub Idle:** ~200 mA (MCU + amps)
- **MIT TX Active:** +100-300 mA (Driving coils)
- **ERT Active:** +2-10 mA (Negligible)
- **Total:** < 1A @ 12V.
- **Battery:** 12Ah battery provides >12 hours continuous operation.

### Signal Integrity (Critical Risk)
- Routing analog signals from 20+ probes requires careful shielding.
- **Recommendation:** Use a "Breakout Box" architecture rather than 20 individual cables running to the main hub. (See Future Development).


