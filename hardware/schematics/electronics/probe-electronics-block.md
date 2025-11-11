# Probe Electronics Block Diagram

## Overview

Complete system architecture for the HIRT probe electronics, showing MIT (Magneto-Inductive Tomography) and ERT (Electrical Resistivity Tomography) subsystems integrated into a single dual-role probe.

## Complete System Block Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PROBE ELECTRONICS (Each Probe)                  │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                    MIT-3D SUBSYSTEM                            │  │
│  │                                                                 │  │
│  │  MCU (ESP32)                                                    │  │
│  │    │                                                            │  │
│  │    ├─ SPI ──→ DDS Generator (AD9833)                          │  │
│  │    │            │                                               │  │
│  │    │            └─→ TX Driver (Op-Amp) ──→ TX Coil (Ferrite)  │  │
│  │    │                                                             │  │
│  │    │            RX Coil (Ferrite) ──→ RX Preamp (AD620)       │  │
│  │    │                                 │                          │  │
│  │    │                                 └─→ Inst. Amp (INA128)    │  │
│  │    │                                            │               │  │
│  │    │                                            └─→ ADC         │  │
│  │    │                                            (ADS1256)       │  │
│  │    │                                                │            │  │
│  │    └────────────────────────────────────────────────┘            │  │
│  │                    (Digital Lock-In via MCU DSP)                 │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                    ERT-LITE SUBSYSTEM                           │  │
│  │                                                                 │  │
│  │  ERT Ring Electrodes (2-3 rings)                                │  │
│  │    │                                                             │  │
│  │    └─→ Multiplexer (CD4051) ──→ Diff Amp ──→ ADC (ADS1256)     │  │
│  │                                      │            │              │  │
│  │                                      └────────────┘              │  │
│  │                                            │                      │  │
│  │                                            └─→ MCU (ESP32)       │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                    POWER & COMMUNICATIONS                        │  │
│  │                                                                 │  │
│  │  Power Input (5V/12V) ──→ Regulators ──→ 3.3V, 5V Rails       │  │
│  │                                                                 │  │
│  │  Communications:                                                │  │
│  │    - Wired: RJ45/CAT5 (Ethernet/RS485)                         │  │
│  │    - Wireless: LoRa/BLE (optional)                             │  │
│  │                                                                 │  │
│  │  Sync/Clock:                                                    │  │
│  │    - Wired sync line from base hub                              │  │
│  │    - Or: Distributed clock reference                            │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         BASE HUB / CONTROL UNIT                         │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                    ERT CURRENT SOURCE                            │  │
│  │                                                                 │  │
│  │  Voltage Ref (REF5025) ──→ Op-Amp Current Source ──→          │  │
│  │                                    │                            │  │
│  │                                    ├─→ Polarity Switch ──→      │  │
│  │                                    │   (Relay/SS)              │  │
│  │                                    │                            │  │
│  │                                    └─→ Current Monitor ──→     │  │
│  │                                        ADC                      │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                    DIFFERENTIAL VOLTMETER                      │  │
│  │                                                                 │  │
│  │  Probe Inputs ──→ Mux ──→ Inst. Amp ──→ ADC (ADS1256)         │  │
│  │    (20-24)         (CD4051)   (INA128)                         │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                    SYNC/CLOCK DISTRIBUTION                      │  │
│  │                                                                 │  │
│  │  Clock Gen ──→ Buffer ──→ Distribution ──→ All Probes          │  │
│  │  (Crystal/DDS)   (74HC244)    (Cables)                          │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                    POWER SYSTEM                                  │  │
│  │                                                                 │  │
│  │  Battery (12V/10-20Ah) ──→ Regulators ──→ Distribution        │  │
│  │                                    │                            │  │
│  │                                    ├─→ 12V (Probes)            │  │
│  │                                    ├─→ 5V (Logic)              │  │
│  │                                    └─→ 3.3V (MCU)              │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                    DATA LOGGER / INTERFACE                      │  │
│  │                                                                 │  │
│  │  MCU/Computer ──→ Data Storage ──→ Display/Tablet              │  │
│  │    (ESP32/PC)      (SD/USB)        (Field Tablet)              │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────┐
                    │   Field      │
                    │   Tablet/    │
                    │   Computer   │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Base Hub     │
                    │  (Control)    │
                    └───┬───────┬───┘
                        │       │
        ┌───────────────┼───────┼───────────────┐
        │               │       │               │
    ┌───▼───┐      ┌───▼───┐ ┌─▼───┐      ┌───▼───┐
    │ Probe │      │ Probe │ │ ... │      │ Probe │
    │  P01  │      │  P02  │ │     │      │  P20  │
    └───────┘      └───────┘ └─────┘      └───────┘
```

## Signal Flow Description

### MIT-3D Measurement Path

1. **Transmit Path:**
   - MCU configures DDS (AD9833) via SPI for desired frequency (2-50 kHz)
   - DDS generates sine wave output
   - TX driver amplifies signal to drive TX coil
   - TX coil generates magnetic field in surrounding medium

2. **Receive Path:**
   - RX coil picks up magnetic field (attenuated/phase-shifted by conductive objects)
   - RX preamp (AD620) provides first-stage amplification with low noise
   - Instrumentation amp (INA128) provides differential amplification and common-mode rejection
   - ADC (ADS1256) digitizes signal at high resolution (24-bit)
   - MCU performs digital lock-in detection (multiply by reference, filter)

### ERT Measurement Path

1. **Current Injection (Base Hub):**
   - Precision current source generates 0.5-2 mA
   - Polarity switch reverses current direction periodically
   - Current injected between two probe electrodes (via base hub control)

2. **Voltage Measurement (Probe):**
   - ERT ring electrodes sense voltage
   - Multiplexer (CD4051) selects which ring(s) to measure
   - Differential amplifier measures voltage between rings or relative to reference
   - ADC digitizes voltage
   - MCU logs measurement with current and geometry information

## Power Distribution

### Probe Power
- **Input:** 5V or 12V from base hub (via cable)
- **Regulation:** On-board regulators provide 3.3V (MCU) and 5V (analog)
- **Consumption:** ~100-200 mA per probe (depends on TX power)

### Base Hub Power
- **Source:** 12V battery (10-20 Ah)
- **Distribution:** 
  - 12V to probes (via cables)
  - 5V for logic circuits
  - 3.3V for MCU
- **Backup:** USB power option for charging/operation

## Communication Architecture

### Wired Option (Recommended)
- **Protocol:** RS485 or Ethernet over CAT5
- **Topology:** Bus or star (hub-and-spoke)
- **Advantages:** Reliable, low latency, no power for radios
- **Cables:** Shielded CAT5 with power conductors

### Wireless Option (Optional)
- **Protocol:** LoRa (long range) or BLE (short range)
- **Topology:** Star (all probes to base)
- **Advantages:** No cables, flexible deployment
- **Disadvantages:** Power consumption, potential interference

## Synchronization

### Clock Distribution
- Base hub generates reference clock (from crystal or DDS)
- Clock distributed to all probes via dedicated sync line
- Ensures phase coherence for MIT measurements
- Enables synchronized data collection

### Timebase
- All probes use same clock reference
- MCU timestamps all measurements
- Enables correlation of measurements across probes

## Component Interconnections

### Probe Internal Connections
- **SPI:** MCU ↔ DDS, MCU ↔ ADC
- **I2C:** MCU ↔ ADC (alternative interface)
- **GPIO:** MCU controls mux, switches, status LEDs
- **Analog:** Coils ↔ amplifiers ↔ ADC

### Probe-to-Base Connections
- **Power:** 2-wire (V+, GND)
- **Data:** 2-wire (RS485) or 4-wire (Ethernet)
- **Sync:** 1-wire clock reference
- **ERT:** Current injection via base hub control

## PCB Layout Considerations

- **Separation:** Keep TX and RX coils physically separated
- **Shielding:** Shield RX signal paths, use ground planes
- **Power:** Decouple power supplies, filter noise
- **Routing:** Minimize loop areas, use twisted pairs for signals
- **Size:** Fit in probe head capsule (Ø25-32 mm)

## Connector Pinouts

### Probe Connector (8-pin example)
- Pin 1: Power + (12V)
- Pin 2: Power GND
- Pin 3: Data + (RS485 A)
- Pin 4: Data - (RS485 B)
- Pin 5: Sync Clock
- Pin 6: Sync GND
- Pin 7: ERT Current + (from base)
- Pin 8: ERT Current - (to base)

*Note: Actual pinout depends on chosen connector and communication method*

