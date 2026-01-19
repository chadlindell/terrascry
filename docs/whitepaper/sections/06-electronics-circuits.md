# 6. Electronics and Circuits

## Overview

This section consolidates all circuit designs for the HIRT system, including the MIT (Magneto-Inductive Tomography) circuits, ERT (Electrical Resistivity Tomography) circuits, and base hub electronics. The design uses centralized electronics with passive probes.

---

## Design Philosophy

**Centralized Electronics:** All electronics are at surface in central hub. Probes are passive (coils and electrodes only). One hub serves all probes (20-24 typical).

---

## System Block Diagram

### Central Electronics Architecture

```
Surface - Central Electronics Hub:
+---------------------------------------------+
|     Central Electronics Hub                 |
|                                             |
|  DDS --> TX Amp --+                         |
|                   |                         |
|  TX MUX ----------+---> Select Probe TX     |
|                   |                         |
|  RX MUX <---------+---> Select Probe RX     |
|                   |                         |
|  RX Chain <-------+---> ADC --> MCU         |
|  (LNA, IA)        |                         |
|                   |                         |
|  ERT Current -----+---> MUX --> Diff Amp    |
|  Source           |      |                  |
|                   |    ADC --> MCU          |
|                   |                         |
|  Sync/Clock ------+------------------------+|
|  Power -----------+------------------------+|
|  Comms -----------+------------------------+|
+-----------------------+---------------------+
                        |
            Multi-Probe Cable Harness
                        |
            +-----------+-----------+
            |                       |
        +---v---+              +---v---+
        | Probe |              | Probe |
        |   1   |              |   2   |
        +---+---+              +---+---+
            |                      |
Downhole (Passive):
        +---+---+              +---+---+
        |  TX   |              |  TX   |
        | Coil  |              | Coil  |
        |       |              |       |
        |  RX   |              |  RX   |
        | Coil  |              | Coil  |
        |       |              |       |
        | Ring  |              | Ring  |
        | Ring  |              | Ring  |
        +-------+              +-------+
```

---

## MIT Circuit Design

### Transmit (TX) Chain

**Block Diagram:**
```
MCU (ESP32)
    |
    | SPI
    v
DDS Generator (AD9833)
    |
    | Sine Wave (0.6V RMS)
    v
TX Driver (Op-Amp OPA454)
    |
    | Amplified Signal (1-5V RMS)
    v
TX Coil (1-2 mH Ferrite)
    |
    v
Magnetic Field into Medium
```

### DDS Sine Generator

**Component:** AD9833 or similar

| Parameter | Specification |
|-----------|---------------|
| Frequency Range | 0.1 Hz to 12.5 MHz |
| Output | 0.6V RMS sine wave |
| Interface | SPI (10 MHz max) |
| Resolution | 28-bit frequency, 12-bit phase |
| Power | 2.3-5.5V, 3mA typical |
| Location | Central hub (one for all probes) |

**Operating Range:** 2-50 kHz (lower frequencies preferred for micro-probes)

### TX Driver

**Component:** OPA454 or OPA2277

| Parameter | Specification |
|-----------|---------------|
| Gain | 2-10x (set by resistors) |
| Output | +/-2.5A peak, 0-5 Vrms into coil |
| Bandwidth | 2.5 MHz |
| Slew Rate | 19 V/us |
| Power | +/-5V to +/-40V |

**TX Driver Circuit:**
```
DDS Out ---[10k]---+---[- OPA454 +]---+--- To TX Coil
                   |         |        |
              [10k-100k]     |    [Series R 10-100 ohm]
                   |         |        |
                  GND    [Feedback]  Coil Return
                              |
                         [Gain Set R]
```

### TX Multiplexer

**Component:** CD4051 or similar
- Selects which probe TX to drive
- Output: 0-5 Vrms into selected probe TX coil
- Purpose: Drive one probe TX at a time

### Receive (RX) Chain

**Block Diagram:**
```
RX Coil (1-2 mH Ferrite)
    |
    | Induced Voltage (uV-mV)
    v
RX Preamp (AD620)
    |
    | Gained Signal (x10-100)
    v
Instrumentation Amp (INA128)
    |
    | Gained Signal (x10-100)
    v
ADC (ADS1256)
    |
    | Digital Data (24-bit)
    v
MCU (Digital Lock-In Processing)
```

### RX Preamp (AD620)

| Parameter | Specification |
|-----------|---------------|
| Gain | 1 to 1000 (set by R_G) |
| Noise | 9 nV/sqrt(Hz) |
| CMRR | 100 dB minimum |
| Bandwidth | 1 MHz (G=1) |
| Input Impedance | 10 G-ohm |

**Gain Setting:**
```
G = (49.4k / R_G) + 1

For G = 10: R_G = 5.49k
For G = 100: R_G = 499 ohm
For G = 1000: R_G = 49.9 ohm
```

### Instrumentation Amp (INA128)

| Parameter | Specification |
|-----------|---------------|
| Gain | 1 to 10000 |
| CMRR | 120 dB (G=100) |
| Noise | 8 nV/sqrt(Hz) |
| Bandwidth | 1.3 MHz (G=1) |

### ADC (ADS1256)

| Parameter | Specification |
|-----------|---------------|
| Resolution | 24 bits |
| Sample Rate | 30 kSPS max |
| Noise | 0.6 uV RMS (100 SPS) |
| Interface | SPI |
| PGA | 1, 2, 4, 8, 16, 32, 64 |

### Lock-In Detection

**Option A: Digital Lock-in (Recommended)**
- Sample RX at >=2-5 kS/s
- Demodulate at reference frequency in MCU
- Advantages: Flexible, software-configurable, good for micro-probes
- Requires: High-speed ADC, DSP capability

**Option B: Analog Lock-in**
- Analog multiplier (AD630)
- Modest ADC for digitization
- Advantages: Lower computational load, proven design
- Requires: Analog reference signal

**Digital Lock-In Algorithm:**
```c
// Simplified digital lock-in
float I_sum = 0, Q_sum = 0;
for (int i = 0; i < N_samples; i++) {
    float sample = read_adc();
    float ref_I = sin(2 * PI * f * i / Fs);
    float ref_Q = cos(2 * PI * f * i / Fs);
    I_sum += sample * ref_I;
    Q_sum += sample * ref_Q;
}
float amplitude = sqrt(I_sum*I_sum + Q_sum*Q_sum) / N_samples;
float phase = atan2(Q_sum, I_sum);
```

---

## ERT Circuit Design

### Design Requirements

| Parameter | Specification |
|-----------|---------------|
| Output Current | 0.5 - 2 mA (adjustable) |
| Current Accuracy | +/- 5% |
| Compliance Voltage | +/- 10V minimum |
| Output Impedance | >1 M-ohm |
| Polarity Reversal | Programmable |
| Load Range | 100 ohm - 10 k-ohm |

### ERT Current Source Block Diagram

```
+12V ----+
         |
     [Voltage Reference]    [MCU Control]
     REF5025 (2.5V)             |
         |                      |
         v                      v
     [Op-Amp Current Source]  [Polarity Switch]
     OPA277 or OPA177           |
         |                      |
         +----------+-----------+
                    |
                    v
              [Current Monitor]
              (Sense Resistor + ADC)
                    |
                    v
              [Output to Probes]
              (Ring A <-> Ring B)
```

### Voltage Reference

**Component:** REF5025AIDGKR (Texas Instruments)

**Purpose:** Provides stable 2.5V reference for current source

**Circuit:**
```
+5V ---[100nF]---+---[REF5025]---+--- 2.5V Reference
                 |       |       |
                GND    [10uF]   [100nF]
                        |        |
                       GND      GND
```

**Key Specifications:**
- Output Voltage: 2.500V +/- 0.05%
- Temperature Drift: 3 ppm/C
- Noise: 3 uV p-p (0.1-10 Hz)
- Supply: 4-18V

### Op-Amp Current Source (Howland Current Pump)

**Component:** OPA277PAG4 or OPA177GP

**Circuit:**
```
        R1 (10k)
Vref ---/\/\/\---+
                 |
        R2 (10k) |
GND ----/\/\/\---+---[- OPA277 +]---+--- I_out
                         |          |
                         |   R3 (10k)
                         +---/\/\/\--+
                         |           |
                    R_sense (100)    |
                         |           |
                        GND    R4 (10k)
                               /\/\/\
                                 |
                            (Feedback)
```

**Current Calculation:**
```
I_out = Vref / R_sense

For I_out = 1 mA and Vref = 2.5V:
R_sense = 2.5V / 0.001A = 2500 ohms

Use precision resistor: 2.49k or 2.50k (0.1%)
```

**Current Range:**
| R_sense | Current @ 2.5V |
|---------|----------------|
| 5000 ohm | 0.5 mA |
| 2500 ohm | 1.0 mA |
| 1667 ohm | 1.5 mA |
| 1250 ohm | 2.0 mA |

### Current Setting Network

**MCU-Controlled Current Adjustment:**
```
      [DAC Output]
      (MCP4725 or PWM+Filter)
           |
           v
      [Attenuator]---+--- V_control (0-2.5V)
           |         |
          GND    [To OPA277 +input]
```

**Alternatively: Resistor Selection:**
```
MCU GPIO ---[CD4051 Mux]---+--- R_sense_1 (5k) --- 0.5 mA
                           +--- R_sense_2 (2.5k) --- 1.0 mA
                           +--- R_sense_3 (1.67k) --- 1.5 mA
                           +--- R_sense_4 (1.25k) --- 2.0 mA
```

### Polarity Reversal Circuit

**Component:** G5V-2-H1 (Omron) or ADG1219 (Analog Devices)

**Purpose:** Reverses current direction to eliminate electrode polarization

**Relay-Based Circuit:**
```
Current Source Output
        |
        +---[NO1]---+---[COM1]---+--- To Ring A
        |           |            |
        +---[NC2]---+  [Relay]   +--- To Ring B
                    |            |
        +---[NC1]---+---[COM2]---+
        |
       GND (Return)
```

**Relay Control:**
- DPDT relay (Double-Pole Double-Throw)
- Driven by MCU GPIO via transistor
- Reversal frequency: 0.5 Hz (every 2 seconds)

### Current Monitor

**Purpose:** Measure actual injected current for verification

**Circuit:**
```
I_out ---[Sense Resistor (10 ohm)]---+--- To Load
                |                     |
                +---[Diff Amp]--------+
                    (INA128)
                        |
                        v
                    [ADC Input]
                    (ADS1256)
```

**Calculation:**
```
V_sense = I_out x R_sense
V_sense = 1 mA x 10 ohm = 10 mV

With INA128 gain = 100:
V_adc = 10 mV x 100 = 1.0 V
```

### ERT Voltage Measurement

**Multiplexer:** Select which probe/ring pairs to measure
**Differential amplifier:** INA128 for high precision
**ADC:** 24-bit ADS1256 for voltage readings
**Purpose:** Measure voltage between ring pairs

### ERT Operation Modes

**DC Mode (Standard):**
1. Set current level (e.g., 1 mA)
2. Inject positive polarity for 2 seconds
3. Measure voltage during injection
4. Reverse polarity
5. Inject negative for 2 seconds
6. Measure voltage during injection
7. Average positive and negative readings

**AC Mode (Optional):**
1. Set current level
2. Modulate polarity at fixed frequency (e.g., 1 Hz)
3. Use lock-in detection for voltage measurement
4. Reduces electrode polarization effects

---

## Power Distribution

### Central Hub Power

| Rail | Source | Voltage | Purpose |
|------|--------|---------|---------|
| Main | Battery | 12V | System power |
| Logic | LM2596 | 5V | Digital circuits |
| MCU | AMS1117 | 3.3V | ESP32, ADC |

**Battery Input:**
```
Battery+ --> Fuse (5A) --> Power Switch --> Distribution
            (fast-blow)   (DPST switch)
```

**Voltage Regulation:**
```
12V Input --> LM2596 Module --> 5V Output (for base hub circuits)
              (Buck converter)

5V Input --> AMS1117-3.3 --> 3.3V Output (for MCU, digital)
             (LDO)
```

### Probe Power

- **No power needed in probe** (passive design)
- Power only needed for:
  - Cable drivers (if long cables)
  - Optional buffer amps (if needed)
- Much lower power per probe than old design

### Power Supply Requirements

| Rail | Voltage | Current | Notes |
|------|---------|---------|-------|
| +12V | 12V | 50 mA | Op-amp positive |
| -12V | -12V | 50 mA | Op-amp negative |
| +5V | 5V | 20 mA | Reference, logic |

---

## Communications and Sync

### Sync/Clock Distribution

**Clock Generation:**
```
+5V --> Oscillator VCC --> Clock Output --> Buffer
       (ECS-100)         (10 MHz square)
       |
       +-- 100nF -- GND
       +-- 10uF -- GND
```

**Clock Buffer (SN74HC244):**
```
Clock Input --> Buffer Input (74HC244)
                  |
                  +-- Output 1 --> Probe 1 Sync
                  +-- Output 2 --> Probe 2 Sync
                  +-- ...
                  +-- Output 20 --> Probe 20 Sync
```

### Communication Interface

**RS485 (MAX485):**
```
MCU UART --> MAX485 --> RS485 Bus --> All Probes
            (RO, DI)    (A, B)
```

**Termination:**
- 120 ohm termination resistors at bus ends
- Bias resistors: 10k pull-up on A, pull-down on B

### Communication Options

| Method | Advantages | Notes |
|--------|------------|-------|
| RS485 over CAT5 | Reliable, low latency | Primary method |
| Ethernet | Direct connection | For RPi4 |
| LoRa/BLE | Wireless, flexible | Optional |

### Data Rate Requirements

- **MIT:** Moderate (sweep takes seconds per frequency)
- **ERT:** Low (voltage readings every 1-2 seconds)
- **Total:** <100 kbps per probe typically sufficient
- **Central hub:** Handles all probes sequentially

---

## Complete Schematics

### Component Specifications

#### Key ICs

| Component | Part Number | Function | Package |
|-----------|-------------|----------|---------|
| DDS | AD9833BRMZ | Signal generator | MSOP-10 |
| Op-Amp | OPA454AIDDAR | TX driver | SOIC-8 |
| Inst. Amp | AD620ARZ | RX preamp | SOIC-8 |
| Inst. Amp | INA128PAG4 | Diff amp | DIP-8 |
| ADC | ADS1256IDBR | 24-bit ADC | SSOP-28 |
| Mux | CD4051BE | 8-channel mux | DIP-16 |
| MCU | ESP32-WROOM-32 | Controller | Module |
| RS485 | MAX485ESA+ | Transceiver | SOIC-8 |
| V-Ref | REF5025AIDGKR | 2.5V reference | SOIC-8 |
| Clock | ECS-100-10-30B-TR | 10 MHz osc | DIP-14 |
| Buffer | SN74HC244N | Octal buffer | DIP-20 |
| LDO | AMS1117-3.3 | 3.3V regulator | SOT-223 |

#### ERT Current Source Components

| Ref | Component | Value | Notes |
|-----|-----------|-------|-------|
| U1 | REF5025AIDGKR | 2.5V | Voltage reference |
| U2 | OPA277PAG4 | - | Current source op-amp |
| U3 | INA128PAG4 | - | Current monitor amp |
| U4 | ADG1219 or Relay | - | Polarity switch |
| R1-R4 | Resistor | 10k 0.1% | Current source network |
| R_sense | Resistor | 2.5k 0.1% | Current setting |
| R_mon | Resistor | 10 ohm 0.1% | Current monitor |
| C1-C3 | Capacitor | 100nF | Bypass caps |
| C4 | Capacitor | 10uF | Reference filter |

#### Passive Components

| Component | Value | Tolerance | Notes |
|-----------|-------|-----------|-------|
| Gain resistors | 49.9-5.49k | 0.1% | Metal film |
| Sense resistors | 10-100 ohm | 0.1% | Precision |
| Bypass caps | 100nF | 10% | MLCC, X7R |
| Filter caps | 10uF | 20% | Tantalum |
| ESD protection | TVS diodes | - | At inputs |

### Connector Pinouts

**Probe Connector (12-pin Phoenix Contact 1757248):**

| Pin | Signal | Notes |
|-----|--------|-------|
| 1 | TX+ | To probe TX coil |
| 2 | TX- | Return path |
| 3 | RX+ | Differential RX |
| 4 | RX- | Differential RX return |
| 5 | Guard | Analog ground |
| 6 | Ring A | Upper ERT electrode |
| 7 | Ring B | Mid ERT electrode |
| 8 | Ring C | Deep electrode |
| 9 | ID Sense | Auto-ID (future) |
| 10 | Spare+ | Reserved |
| 11 | Spare- | Reserved |
| 12 | Shield | Cable shield clamp |

---

## PCB Layout Guidelines

### General Rules

1. **Ground Planes:**
   - Use solid ground plane on bottom layer
   - Separate analog and digital grounds
   - Connect at single star point near ADC

2. **Power Distribution:**
   - Wide traces for power (>20 mil)
   - Decoupling caps at each IC
   - Use power planes where possible

3. **Signal Routing:**
   - Keep analog signals short
   - Use differential pairs for RX signals
   - Shield sensitive traces

4. **Component Placement:**
   - Group by function (TX, RX, power, digital)
   - Place bypass caps close to IC pins
   - Keep TX and RX coils physically separated

### EMI Considerations

- Enclose RX preamp in shielded area
- Use ferrite beads on power lines
- Ground shields at one end only
- Minimize loop areas in signal paths

---

## Shielding and Noise Reduction

### Best Practices

- **Twist and shield** all signal cables
- **Single-point ground** to avoid ground loops
- Use **shielded cables** throughout multi-probe harness
- Minimize **loop areas** in wiring
- **Separate TX and RX cables** in harness
- **Proper termination** of shields

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Direct coupling | TX->RX coupling | Orthogonal coils |
| Ground loops | Multiple ground paths | Single-point ground |
| EMI | External interference | Shielding, filtering |
| Cable crosstalk | Multiple probes in harness | Proper cable routing |

---

## Probe Components (Passive)

### MIT Coils

| Parameter | Specification |
|-----------|---------------|
| TX Coil | Ferrite core (6-8 mm) with 200-400 turns |
| RX Coil | Ferrite core (6-8 mm) with 200-400 turns |
| Wire | 34-38 AWG fine wire |
| Connection | Thin cable to central hub |
| Electronics | None (passive coils only) |

### ERT Rings

| Parameter | Specification |
|-----------|---------------|
| Rings | Narrow bands (3-5 mm) of stainless steel or copper |
| Connection | Thin twisted pair to central hub |
| Electronics | None (passive electrodes only) |

---

## Measurement Sequence

1. Select probe TX (via TX multiplexer)
2. Drive TX at desired frequency
3. Select probe RX (via RX multiplexer)
4. Measure RX signal (amplify, digitize, lock-in)
5. Repeat for all TX/RX pairs
6. Switch to ERT measurements
7. Select probe/ring pairs (via ERT multiplexer)
8. Inject current, measure voltage
9. Repeat for all ring pairs

---

## Safety Considerations

- **Maximum Output Current:** Limited to 5 mA by design
- **Compliance Voltage:** +/- 12V (safe for soil contact)
- **Fusing:** Include 10 mA fuse on output
- **Isolation:** Use opto-isolated relay control
- **Grounding:** Ensure proper earth ground for safety

---

## Troubleshooting

| Symptom | Cause | Solution |
|---------|-------|----------|
| No current output | Open circuit, bad op-amp | Check connections, replace op-amp |
| Low current | Reference drift, resistor tolerance | Calibrate, replace precision resistors |
| Unstable current | Oscillation, noise | Add compensation cap, check layout |
| No polarity reversal | Relay stuck, control signal | Check relay, verify MCU output |
| Compliance exceeded | Load too high | Reduce current, check electrode contact |

---

## Advantages of Centralized Design

1. **Lower Cost:** One set of electronics vs many
2. **Easier Maintenance:** All electronics accessible
3. **Better Reliability:** Passive probes more robust
4. **Easier Updates:** Firmware updates in one place
5. **Better Power Management:** Centralized power control
6. **Simpler Probes:** No electronics to fail in probe

## Trade-offs

1. **More Cables:** Multi-probe harness needed
2. **Central Failure Point:** Hub failure affects all probes (mitigate with redundancy)
3. **Cable Management:** More cables to manage in field
4. **Sequential Measurement:** Measure one probe at a time (acceptable for most applications)

---

*For mechanical assembly, see Section 5: Mechanical Design. For BOM, see Section 4: Bill of Materials.*
