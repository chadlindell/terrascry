# Base Hub Circuit Schematic - Detailed Design

## Overview

Complete circuit design for the base hub/control unit, including ERT current source, differential voltmeter, sync/clock distribution, power management, and communication interfaces.

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
- Keep spares available

**Power Switch:**
- DPDT switch for main power
- Allows battery disconnect
- Optional: Electronic switch (MOSFET)

### Voltage Regulation

**12V Distribution:**
- Direct from battery (for probes)
- Fused distribution block
- Current capacity: 5A total

**5V Regulator (LM2596 Module):**
```
12V Input ──→ LM2596 Module ──→ 5V Output
              (Buck converter)    (for base hub circuits)
```

**Module Specifications:**
- Input: 7-40V
- Output: 5V, 3A max
- Efficiency: >80%
- Built-in protection

**5V Distribution:**
- For base hub analog circuits
- ERT current source
- Sync circuits
- Current: ~500 mA

**3.3V Regulator (AMS1117-3.3):**
```
5V Input ──→ AMS1117-3.3 ──→ 3.3V Output
             (LDO)            (for MCU, digital)
```

**Component Values:**
- Input cap: 10µF tantalum
- Output cap: 10µF tantalum
- Additional: 100nF ceramic near regulator

**3.3V Distribution:**
- For MCU (ESP32 or similar)
- Digital circuits
- Current: ~200 mA

### Power Distribution Block

**Terminal Block:**
- Multi-position terminal block
- 12V distribution to probes (20 connections)
- 5V and 3.3V distribution
- Fused outputs if needed

**Current Monitoring:**
- Optional: Current sense resistors
- Monitor total current draw
- Detect faults

---

## Sync/Clock Distribution

### Clock Generation

**Option 1: Crystal Oscillator**

**Component:** ECS-100-10-30B-TR (10 MHz crystal oscillator)

**Circuit:**
```
+5V ──→ Oscillator VCC ──→ Clock Output ──→ Buffer
       (ECS-100)         (10 MHz square)
       |
       ├── 100nF ── GND
       └── 10µF ── GND
```

**Specifications:**
- Frequency: 10 MHz
- Stability: ±30 ppm
- Output: TTL/CMOS compatible
- Power: 5V, <20 mA

**Option 2: DDS Generator** (Alternative)

- Use AD9833 (same as probe DDS)
- More flexible (can change frequency)
- More complex

### Clock Buffer/Distribution

**Component:** SN74HC244N (Octal buffer, non-inverting)

**Purpose:** Drive multiple clock outputs (one per probe)

**Circuit:**
```
Clock Input ──→ Buffer Input (74HC244)
                  │
                  ├── Output 1 ──→ Probe 1 Sync
                  ├── Output 2 ──→ Probe 2 Sync
                  ├── ...
                  └── Output 20 ──→ Probe 20 Sync
```

**Component Values:**
- Input: Clock from oscillator
- Outputs: 20 outputs (may need multiple buffers)
- Power: 5V
- Current: ~10 mA per output (depends on load)

**Buffering Strategy:**
- One 74HC244 drives 8 outputs
- Use 3 buffers for 20 probes (24 outputs total)
- Or use clock distribution IC (e.g., CDCLVC1108)

### Sync Cable Distribution

**Cable:**
- CAT5 cable or similar
- One pair per probe
- Shielded if possible

**Connectors:**
- RJ45 connectors or custom
- One connector per probe cable
- Distribution hub/panel

**Signal Levels:**
- TTL/CMOS levels (0-5V)
- 10 MHz square wave
- 50Ω termination (if long cables)

---

## Differential Voltmeter (Base Hub)

**Purpose:** Measure ERT voltages (if not done at probes)

**Note:** Primary ERT voltage measurement is done at probes. Base hub voltmeter is optional for verification or alternative measurement.

### Instrumentation Amplifier

**Component:** INA128PAG4 (Same as probe-side)

**Circuit:**
```
Probe Input ──→ +IN (INA128)
Reference    ──→ -IN (INA128)
                |
                ├── RG (gain resistor)
                │
                └── OUT ──→ ADC
```

**Gain:** Typically 1-10x (voltage measurement)

### ADC Interface

**Component:** ADS1256IDBR (24-bit ADC, same as probe-side)

**Configuration:**
- Input Range: ±2.5V
- Sample Rate: 1-30 kS/s
- Resolution: 24 bits

**Multiplexer (if multiple channels):**
- CD4051BE (8-channel mux)
- Select which probe to measure
- Or measure multiple probes sequentially

---

## Communication Interface

### RS485 Interface (Wired Option)

**Component:** MAX485ESA+ (RS485 transceiver)

**Circuit:**
```
MCU UART ──→ MAX485 ──→ RS485 Bus ──→ All Probes
            (RO, DI)    (A, B)
```

**Termination:**
- 120Ω termination resistors at bus ends
- Bias resistors: 10kΩ pull-up on A, pull-down on B

**Bus Topology:**
- Daisy-chain or star (hub-and-spoke)
- Maximum length: 1200 m
- Maximum nodes: 32 (with proper termination)

### Ethernet Interface (Alternative)

**Component:** W5500 or similar Ethernet controller

**Circuit:**
```
MCU SPI ──→ W5500 ──→ Ethernet PHY ──→ RJ45 ──→ Network
```

**Use Case:**
- If using Ethernet for data
- More complex but higher bandwidth
- Requires Ethernet switch/hub

### Wireless Interface (Optional)

**LoRa Module (SX1278):**
```
MCU SPI ──→ SX1278 Module ──→ Antenna
```

**Specifications:**
- Range: 1-5 km (line of sight)
- Data Rate: 0.3-37.5 kbps
- Frequency: 433/868/915 MHz

**Use Case:**
- When cables are impractical
- Adds power consumption
- Lower data rate than wired

### Data Logger Interface

**Raspberry Pi or Tablet Connection:**

**USB Interface:**
- Direct USB connection
- Or USB-to-serial adapter
- Standard USB cable

**Serial Interface:**
- UART from MCU
- USB-to-serial converter (FT232, CP2102)
- Connect to tablet/computer

---

## Control and Data Acquisition

### MCU (ESP32 or Similar)

**Purpose:** Control base hub, data acquisition, communication

**Functions:**
- Control ERT current source
- Control sync distribution
- Data acquisition (if base hub voltmeter used)
- Communication with probes
- Data logging
- User interface

**Interfaces:**
- SPI: To ADC, communication modules
- UART: To RS485, data logger
- GPIO: Control relays, switches, LEDs
- I2C: Optional (for additional peripherals)

### Data Logger/Computer Interface

**Connection Options:**

1. **USB Serial:**
   - MCU UART → USB-to-serial → Computer
   - Simple, reliable
   - Standard USB cable

2. **Ethernet:**
   - MCU → Ethernet controller → Network
   - Higher bandwidth
   - Network connectivity

3. **WiFi (if ESP32):**
   - Built-in WiFi
   - Wireless connectivity
   - Web interface possible

**Data Protocol:**
- Custom protocol or Modbus
- ASCII or binary format
- Error checking (CRC)
- Acknowledgment

---

## Component Summary

### Power Management Components
| Component | Part Number | Quantity | Notes |
|-----------|-------------|----------|-------|
| Fuse Holder | 0287005.PXCN | 1 | Panel mount |
| Fuse | 0034.5002 | 5 | 5A fast-blow, spares |
| 5V Regulator | LM2596 Module | 1 | Buck converter |
| 3.3V Regulator | AMS1117-3.3 | 1 | LDO |
| Distribution Block | Terminal block | 1 | Multi-position |
| Capacitors | 10µF, 100nF | Multiple | Tantalum/ceramic |

### Sync/Clock Components
| Component | Part Number | Quantity | Notes |
|-----------|-------------|----------|-------|
| Oscillator | ECS-100-10-30B-TR | 1 | 10 MHz |
| Clock Buffer | SN74HC244N | 3 | Octal buffer (8 outputs each) |
| Connectors | RJ45 or custom | 20 | For sync cables |
| Cables | CAT5 | 20m | Sync distribution |

### Communication Components
| Component | Part Number | Quantity | Notes |
|-----------|-------------|----------|-------|
| RS485 Transceiver | MAX485ESA+ | 1 | RS485 interface |
| Ethernet Controller | W5500 (optional) | 1 | If Ethernet used |
| LoRa Module | SX1278 (optional) | 1 | If wireless used |
| USB-Serial | FT232 or CP2102 | 1 | Data logger interface |

### Control Components
| Component | Part Number | Quantity | Notes |
|-----------|-------------|----------|-------|
| MCU | ESP32 or STM32 | 1 | Control and data acquisition |
| ADC | ADS1256IDBR | 1 | If base hub voltmeter used |
| Inst. Amp | INA128PAG4 | 1 | If base hub voltmeter used |

---

## Enclosure and Mechanical

### Enclosure Specifications

**Requirements:**
- Weatherproof (IP65 or better)
- Size: ~200×150×100 mm (or larger)
- Material: Plastic or metal
- Mounting: Panel or surface mount

### Component Mounting

**PCB Mounting:**
- Standoffs or mounting posts
- Secure mounting
- Ground connection

**Battery Mounting:**
- Battery holder or bracket
- Secure mounting
- Ventilation if needed

**Connector Panel:**
- Mount connectors on panel
- Label all connectors
- Strain reliefs for cables

### Cable Management

**Cable Entry:**
- Cable glands (PG or M series)
- Waterproof entry
- Strain reliefs

**Internal Routing:**
- Organize cables
- Use cable ties
- Avoid sharp bends

---

## Design Notes

### Power Budget

**Base Hub Power Consumption:**
- MCU: ~100-200 mA @ 3.3V
- Analog circuits: ~100-200 mA @ 5V
- Sync distribution: ~50-100 mA @ 5V
- Communication: ~50-100 mA @ 5V
- **Total:** ~300-600 mA @ 12V (3.6-7.2W)

**Battery Life:**
- 12V 12Ah battery
- Capacity: 144 Wh
- At 5W average: ~28 hours
- With 20 probes: Additional ~2-4A @ 12V

### Thermal Considerations
- Ensure adequate ventilation
- Heat sinks if needed
- Monitor temperature
- Shutdown if overheating

### Reliability
- Use quality components
- Proper derating
- Fuse protection
- Error detection and recovery

### Calibration
- Calibrate current source
- Calibrate voltmeter (if used)
- Verify sync timing
- Document calibration values

---

*For ERT current source details, see [ERT Circuit](ert-circuit.md)*
*For probe circuits, see [MIT Circuit](mit-circuit.md) and [ERT Circuit](ert-circuit.md)*

