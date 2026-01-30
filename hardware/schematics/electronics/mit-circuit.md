# MIT Circuit Schematic - Detailed Design

## Overview

Complete circuit design for the Magneto-Inductive Tomography (MIT) subsystem, including TX (transmit) chain, RX (receive) chain, and digital lock-in detection.

---

## TX Chain Design

### DDS Generator Circuit (AD9833)

**Component:** AD9833BRMZ (MSOP-10 package)

**Power Supply:**
- VDD: +5V (from 5V regulator)
- AGND, DGND: Common ground
- Decoupling: 100nF ceramic + 10µF tantalum near VDD pin

**SPI Interface:**
```
MCU (ESP32)          AD9833
MOSI  ───────────→  SDATA
SCLK  ───────────→  SCLK
CS    ───────────→  FSYNC
```

**Clock Input:**
- MCLK: 25 MHz crystal oscillator or external clock
- Connect via 33Ω series resistor
- Decouple with 100nF ceramic

**Output Circuit:**
```
AD9833 VOUT ──→ 10nF DC blocking cap ──→ 10kΩ load ──→ TX Driver Input
```

**Output Specifications:**
- Frequency Range: 0-12.5 MHz (use 2-50 kHz for MIT)
- Output: 0.6V RMS (typical)
- Output Impedance: High (requires buffer)

**Configuration:**
- Control via SPI (3-wire interface)
- Frequency resolution: 0.1 Hz at 25 MHz clock
- Phase resolution: 12 bits

### TX Driver Amplifier Circuit

**Component:** OPA2277 (Dual op-amp, use one channel)

**Circuit Topology:** Non-inverting amplifier

**Schematic:**
```
                    +5V
                     |
                     ├── 100nF ── GND
                     |
AD9833 Output ──→ 10nF ──→ +IN (OPA2277)
                              |
                              ├── R1 (10kΩ) ── GND
                              |
                              ├── R2 (20kΩ) ── OUT
                              |
                              └── OUT ──→ 100nF ──→ TX Coil
```

**Gain Calculation:**
- Gain = 1 + (R2/R1) = 1 + (20k/10k) = 3x
- Adjust R2 for desired gain (2-5x typical)

**Component Values:**
- R1: 10kΩ, 1% tolerance
- R2: 20kΩ (for 3x gain), 1% tolerance
- C1 (input): 10nF ceramic (DC blocking)
- C2 (output): 100nF ceramic (coupling to coil)
- Power: +5V and GND, decouple with 100nF + 10µF

**Output Specifications:**
- Output Voltage: 0-5V RMS (into coil)
- Output Current: 10-50 mA RMS (depends on coil impedance)
- Bandwidth: >100 kHz (for 50 kHz operation)

**Current Monitoring:**
```
TX Coil ──→ 1Ω sense resistor ──→ GND
                |
                └──→ Diff amp ──→ ADC (optional, for monitoring)
```

### TX Coil Interface

**Optimized Coil Specifications (2-50 kHz broadband):**

| Parameter | Target | Notes |
|-----------|--------|-------|
| Inductance | 1.5 mH ±10% | Measured at 1 kHz |
| DC Resistance | <8 Ω | 34 AWG, ~300 turns |
| Q Factor @ 10 kHz | ≥30 | Primary design frequency |
| Q Factor @ 2-50 kHz | ≥20 | Full operating range |
| Self-Resonant Freq | >200 kHz | Above operating range |
| Core Material | NiZn ferrite | Fair-Rite 61 or equivalent |
| Core Dimensions | Ø8 mm × 100 mm | |

**Coil Driver Circuit:**
```
TX Driver Output ──→ 100nF ──→ TX Coil ──→ GND
                              (1.5 mH)
```

**Impedance vs Frequency:**

| Frequency | XL (Ω) | Z (Ω) | Current @ 3V |
|-----------|--------|-------|--------------|
| 2 kHz | 19 | 21 | 143 mA |
| 10 kHz | 94 | 94 | 32 mA |
| 20 kHz | 188 | 188 | 16 mA |
| 50 kHz | 471 | 471 | 6 mA |

Note: At low frequencies, current limiting resistor may be needed.

**Protection:**
- Add series resistor (10-50Ω) to limit current if needed
- Add back-to-back diodes for overvoltage protection

---

## RX Chain Design

### RX Coil Interface

**Optimized Coil Specifications (matched to TX):**

| Parameter | Target | Notes |
|-----------|--------|-------|
| Inductance | 1.5 mH ±10% | Match TX within 5% |
| DC Resistance | <8 Ω | 34 AWG, ~300 turns |
| Q Factor @ 10 kHz | ≥30 | Higher Q = better sensitivity |
| Q Factor @ 2-50 kHz | ≥20 | Full operating range |
| Self-Resonant Freq | >200 kHz | Above operating range |
| TX-RX Coupling | <-40 dB | Via orthogonal mounting |
| Orientation | 90° ±5° to TX | Critical for coupling rejection |

**Coil Connection:**
```
RX Coil ──→ Twisted pair (shielded) ──→ RX Preamp Input
(1-2 mH)      (keep short, <10 cm)
```

**Shielding:**
- Use shielded twisted pair cable
- Connect shield to ground at preamp end only
- Keep cable short to minimize pickup

### RX Preamplifier Circuit (AD620)

**Component:** AD620ANZ (Instrumentation amplifier)

**Circuit Configuration:**
```
RX Coil+ ──→ +IN (AD620)
RX Coil- ──→ -IN (AD620)
              |
              ├── RG (gain resistor)
              │
              └── OUT ──→ Next stage
```

**Gain Setting:**
- Gain = 1 + (49.4kΩ / RG)
- RG = 49.4kΩ / (Gain - 1)

**Typical Values:**
- Gain = 10: RG = 5.49kΩ (use 5.6kΩ)
- Gain = 100: RG = 499Ω (use 510Ω)
- Gain = 1000: RG = 49.4Ω (use 51Ω)

**Component Values:**
- RG: Selected for desired gain (see above)
- Power: ±5V (or single +5V with virtual ground)
- Decoupling: 100nF ceramic + 10µF tantalum on each supply
- Input protection: 1kΩ series resistors + diodes to supplies

**Output Specifications:**
- Input noise: <1.3 nV/√Hz
- Bandwidth: >100 kHz
- Common-mode rejection: >100 dB (at 60 Hz)

### Instrumentation Amplifier Stage (INA128)

**Component:** INA128PAG4 (Precision instrumentation amplifier)

**Purpose:** Second-stage amplification and common-mode rejection

**Circuit Configuration:**
```
RX Preamp Output ──→ +IN (INA128)
Reference (GND)  ──→ -IN (INA128)
                      |
                      ├── RG (gain resistor)
                      │
                      └── OUT ──→ ADC Input
```

**Gain Setting:**
- Gain = 1 + (50kΩ / RG)
- RG = 50kΩ / (Gain - 1)

**Typical Values:**
- Gain = 10: RG = 5.56kΩ (use 5.6kΩ)
- Gain = 100: RG = 505Ω (use 510Ω)

**Component Values:**
- RG: Selected for desired gain
- Power: ±5V (or single +5V with virtual ground)
- Decoupling: 100nF + 10µF on each supply
- Reference: Connect REF pin to ground or mid-supply

**Total RX Chain Gain:**
- Preamp gain: 10-100x
- Inst. amp gain: 10-100x
- Total gain: 100-10,000x (adjust based on signal levels)

### ADC Interface (ADS1256)

**Component:** ADS1256IDBR (24-bit delta-sigma ADC)

**Input Circuit:**
```
INA128 Output ──→ 1kΩ ──→ +AIN (ADS1256)
                      │
                      └── 100nF ── GND (anti-aliasing)

GND ──→ -AIN (ADS1256)
```

**SPI Interface:**
```
MCU (ESP32)          ADS1256
MOSI  ───────────→  DIN
MISO  ←───────────  DOUT
SCLK  ───────────→  SCLK
CS    ───────────→  CS
DRDY  ←───────────  DRDY (data ready)
```

**Reference Voltage:**
- VREF: 2.5V precision reference (REF5025)
- Connect VREF+ and VREF- to ADC
- Decouple with 10µF tantalum

**Power Supply:**
- AVDD: +5V
- DVDD: +3.3V (digital supply)
- AGND, DGND: Separate grounds, connect at one point

**Sampling Configuration:**
- Sample Rate: 30 kS/s (for digital lock-in)
- Input Range: ±2.5V (with 2.5V reference)
- Resolution: 24 bits = 0.3 µV per LSB

**Input Protection:**
- Series resistors: 1kΩ (current limiting)
- Clamp diodes: BAT54S (to supplies)
- ESD protection: TVS diodes (optional)

---

## Digital Lock-in Detection

### Implementation in MCU

**Principle:**
- Multiply RX signal by reference (sine/cosine)
- Low-pass filter result
- Extract amplitude and phase

**Algorithm:**
```
1. Sample RX signal at high rate (≥2× highest frequency)
2. Generate reference sine/cosine at TX frequency
3. Multiply: I = RX × sin(ωt), Q = RX × cos(ωt)
4. Low-pass filter I and Q
5. Calculate: Amplitude = √(I² + Q²)
6. Calculate: Phase = atan2(Q, I)
```

**MCU Implementation (ESP32):**
- Use ADC to sample RX signal (via ADS1256)
- Generate reference in software (lookup table or DDS)
- Perform multiplication and filtering
- Calculate amplitude and phase

**Filter Design:**
- Low-pass filter: 2nd order IIR (Butterworth)
- Cutoff frequency: 1-10% of measurement frequency
- Example: For 10 kHz measurement, use 100-1000 Hz cutoff

**Performance:**
- SNR improvement: ~√(sample_rate / bandwidth)
- Example: 30 kS/s sample rate, 100 Hz bandwidth → ~17 dB improvement

---

## Power Supply Design

### Probe Power Distribution

**Input:** 12V from base hub (via cable)

**Regulators:**

**3.3V Regulator (AMS1117-3.3):**
```
12V Input ──→ 10µF ──→ AMS1117-3.3 ──→ 10µF ──→ 3.3V Output
                      (VIN)          (VOUT)        (MCU, ADC digital)
```

**5V Regulator (AMS1117-5.0):**
```
12V Input ──→ 10µF ──→ AMS1117-5.0 ──→ 10µF ──→ 5V Output
                      (VIN)          (VOUT)        (Analog circuits)
```

**Component Values:**
- Input capacitors: 10µF tantalum or ceramic
- Output capacitors: 10µF tantalum or ceramic
- Additional: 100nF ceramic near each regulator

**Current Requirements:**
- 3.3V: ~50-100 mA (MCU, ADC digital)
- 5V: ~50-100 mA (analog circuits, DDS, op-amps)
- Total: ~100-200 mA per probe

---

## Component Summary

### TX Chain Components
| Component | Part Number | Quantity | Notes |
|-----------|-------------|----------|-------|
| DDS | AD9833BRMZ-REEL7 | 1 | Sine generator |
| TX Driver | OPA2277PA | 1 | Amplifier |
| Gain Resistors | 10kΩ, 20kΩ | 2 | 1% tolerance |
| Coupling Caps | 10nF, 100nF | 2 | Ceramic |
| TX Coil | Custom | 1 | 1-2 mH |

### RX Chain Components
| Component | Part Number | Quantity | Notes |
|-----------|-------------|----------|-------|
| RX Preamp | AD620ANZ | 1 | Instrumentation amp |
| Inst. Amp | INA128PAG4 | 1 | Second stage |
| Gain Resistors | Various | 2 | Per gain setting |
| ADC | ADS1256IDBR | 1 | 24-bit ADC |
| Reference | REF5025AIDGKT | 1 | 2.5V reference |
| RX Coil | Custom | 1 | 1-2 mH |

### Power Components
| Component | Part Number | Quantity | Notes |
|-----------|-------------|----------|-------|
| 3.3V Reg | AMS1117-3.3 | 1 | LDO regulator |
| 5V Reg | AMS1117-5.0 | 1 | LDO regulator |
| Capacitors | 10µF, 100nF | Multiple | Tantalum/ceramic |

---

## Design Notes

### Noise Considerations
- Keep RX signal paths short
- Use shielded cables
- Separate analog and digital grounds
- Decouple all power supplies
- Use low-noise op-amps

### Frequency Response
- Ensure all stages have bandwidth >50 kHz
- Check for resonances in coil circuits
- Verify flat response across frequency range

### Calibration
- Measure actual gains at each stage
- Calibrate ADC reference
- Document coil constants
- Verify frequency accuracy

### PCB Layout Considerations
- Keep TX and RX physically separated
- Use ground planes
- Minimize loop areas
- Route sensitive signals carefully
- Keep power traces wide

---

*For base hub circuits, see [Base Hub Circuit](base-hub-circuit.md)*
*For ERT circuits, see [ERT Circuit](ert-circuit.md)*

