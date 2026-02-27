# ERT Circuit Schematic - Detailed Design

## Overview

Complete circuit design for the Electrical Resistivity Tomography (ERT) subsystem, including current source (base hub), probe-side voltage measurement circuits, and multiplexer control.

---

## Base Hub: ERT Current Source

### Precision Current Source Circuit

**Component:** OPA177GP (Precision op-amp)

**Circuit Topology:** Howland current source (voltage-controlled current source)

**Schematic:**
```
                    +5V
                     |
                     ├── 100nF ── GND
                     |
Vref (2.5V) ──→ R1 ──→ +IN (OPA177)
REF5025            |
                   ├── R2 ──→ OUT ──→ R3 ──→ Current Output (+)
                   │                    │
                   └── R4 ──→ -IN       │
                        │               │
                        └── R5 ──→ ─────┘
                                  │
                                  └── Load (Probe Electrodes)
                                  │
                                  └── R_sense ──→ GND
```

**Component Values:**
- R1: 10kΩ, 0.1% tolerance
- R2: 10kΩ, 0.1% tolerance (match R1)
- R3: 1kΩ, 0.1% tolerance
- R4: 10kΩ, 0.1% tolerance (match R1)
- R5: 1kΩ, 0.1% tolerance (match R3)
- R_sense: 1.25kΩ, 0.1% tolerance (for 2 mA full scale)

**Current Calculation:**
- I_out = Vref × (R2/R1) / R_sense
- For Vref = 2.5V, R2/R1 = 1, R_sense = 1.25kΩ:
- I_out = 2.5V / 1.25kΩ = 2.0 mA

**Adjustability:**
- Use digital potentiometer or switchable resistors
- Or vary Vref using DAC
- Range: 0.5-2 mA

**Voltage Reference (REF5025):**
```
+5V ──→ REF5025 V+ ──→ 2.5V Output ──→ Current Source
       (VIN)          (VOUT)
       |
       ├── 10µF ── GND
       └── 100nF ── GND
```

**Reference Specifications:**
- Output: 2.5V ± 0.05%
- Temperature coefficient: 5 ppm/°C
- Load regulation: Excellent

### Polarity Switching Circuit

**Component:** G5V-2-5DC (DPDT relay, 5V coil)

**Circuit:**
```
Current Source Output ──→ Relay Common (2 poles)
                              │
                              ├── NO1 ──→ Output+ (to probe)
                              ├── NC1 ──→ Output- (to probe)
                              ├── NO2 ──→ Output- (to probe)
                              └── NC2 ──→ Output+ (to probe)

Relay Control:
MCU GPIO ──→ 2N7002 (MOSFET) ──→ Relay Coil ──→ GND
            (Gate)              (Drain-Source)
```

**Relay Driver (2N7002):**
- Gate resistor: 1kΩ (to MCU GPIO)
- Pull-down resistor: 10kΩ (to GND)
- Flyback diode: 1N4148 across relay coil
- Current: ~40 mA (relay coil current)

**Control Sequence:**
- Normal: Relay OFF → Output+ and Output- as configured
- Reversed: Relay ON → Output+ and Output- swapped
- Switch every 1-2 seconds

### Current Monitoring

**Purpose:** Monitor actual current output for feedback and verification

**Circuit:**
```
R_sense ──→ Diff Amp (INA128) ──→ ADC
(1.25kΩ)      (voltage across      (ADS1256)
               sense resistor)
```

**Calculation:**
- Voltage across R_sense: V = I × R = 2mA × 1.25kΩ = 2.5V
- Measure with ADC, calculate: I = V / R_sense

---

## Probe-Side: ERT Voltage Measurement

### ERT Ring Electrode Interface

**Ring Specifications:**
- Material: Stainless steel (304SS)
- Width: 12 mm
- Thickness: 0.5 mm
- Mounting: On rod at 0.5m, 1.5m, 2.5m from tip

**Connection:**
```
ERT Ring ──→ Shielded Twisted Pair ──→ Multiplexer Input
(2-3 rings)    (3m length)              (CD4051)
```

**Isolation Requirements:**
- Ring-to-ring: >1 MΩ (dry)
- Ring-to-rod: >1 MΩ
- Use shielded cable, connect shield to ground at mux end only

### Multiplexer Circuit (CD4051)

**Component:** CD4051BE (8-channel analog multiplexer)

**Purpose:** Select which ERT ring(s) to measure

**Circuit:**
```
Ring 1 ──→ IN0 (CD4051)
Ring 2 ──→ IN1 (CD4051)
Ring 3 ──→ IN2 (CD4051)
GND    ──→ IN3 (CD4051)  (reference)
           ...
           |
           └── COM (Common) ──→ Differential Amp Input

Control:
MCU GPIO ──→ A, B, C (address lines)
MCU GPIO ──→ INH (inhibit, active low)
```

**Channel Selection:**
- A, B, C = 000 → Select Ring 1 (IN0)
- A, B, C = 001 → Select Ring 2 (IN1)
- A, B, C = 010 → Select Ring 3 (IN2)
- A, B, C = 011 → Select GND reference (IN3)

**Power Supply:**
- VDD: +5V
- VSS: GND
- VEE: Can be -5V for bipolar signals (or GND if unipolar)

**On-Resistance:**
- Typical: 125Ω
- Maximum: 1000Ω
- Consider in gain calculations

### Differential Amplifier (INA128)

**Component:** INA128PAG4 (Can share with MIT subsystem)

**Purpose:** Measure voltage between selected ring and reference

**Circuit:**
```
Mux Output ──→ +IN (INA128)
Reference   ──→ -IN (INA128)  (GND or another ring)
                |
                ├── RG (gain resistor)
                │
                └── OUT ──→ ADC Input
```

**Gain Setting:**
- Gain = 1 + (50kΩ / RG)
- For voltage measurement: Gain = 1-10x typically
- RG = 50kΩ / (Gain - 1)
- Example: Gain = 2 → RG = 50kΩ

**Component Values:**
- RG: Selected for desired gain (see above)
- Power: ±5V (or single +5V with virtual ground)
- Decoupling: 100nF + 10µF on each supply
- Reference: Connect REF pin to mid-supply or GND

**Input Protection:**
- Series resistors: 1kΩ (current limiting)
- Clamp diodes: BAT54S (to supplies)
- ESD protection: TVS diodes (optional)

### ADC Interface (ADS1256)

**Component:** ADS1256IDBR (24-bit ADC, shared with MIT)

**Input Circuit:**
```
INA128 Output ──→ 1kΩ ──→ +AIN (ADS1256)
                      │
                      └── 100nF ── GND (anti-aliasing)

Reference (GND) ──→ -AIN (ADS1256)
```

**Configuration:**
- Input Range: ±2.5V (with 2.5V reference)
- Sample Rate: 30 kS/s (or lower for ERT, e.g., 1 kS/s)
- Resolution: 24 bits = 0.3 µV per LSB

**Measurement Sequence:**
1. Select ERT ring via mux
2. Configure ADC channel
3. Wait for settling (ring contact stabilization)
4. Take multiple samples (average for noise reduction)
5. Record voltage and current

---

## Measurement Procedure

### Current Injection Sequence

**From Base Hub:**

1. **Configure Current Source:**
   - Set current level (typically 1 mA)
   - Enable current source

2. **Select Injection Electrodes:**
   - Set probe A, ring X as positive (+)
   - Set probe B, ring Y as negative (-)
   - Via base hub control and probe muxes

3. **Inject Current:**
   - Enable current source
   - Wait for stabilization (~100 ms)
   - Monitor current (verify 1.0 mA)

4. **Measure Voltages:**
   - All other probes measure voltage at their rings
   - Select ring via mux
   - Measure with ADC
   - Record voltage and current

5. **Reverse Polarity:**
   - Switch relay (reverse + and -)
   - Wait for stabilization
   - Repeat measurements
   - Average results (reduces polarization effects)

### Voltage Measurement Sequence

**At Each Probe:**

1. **Select Ring:**
   - Configure mux to select ring (0.5m, 1.5m, or 2.5m)
   - Wait for mux settling (~10 µs)

2. **Configure ADC:**
   - Select ADC channel
   - Configure gain and sample rate
   - Wait for ADC ready

3. **Measure:**
   - Take multiple samples (e.g., 100 samples)
   - Average for noise reduction
   - Record voltage

4. **Calculate Apparent Resistivity:**
   - ρ_a = (V / I) × K
   - K = geometric factor (depends on electrode positions)
   - Record in data file

---

## Component Summary

### Base Hub ERT Components
| Component | Part Number | Quantity | Notes |
|-----------|-------------|----------|-------|
| Current Source Op-Amp | OPA177GP | 1 | Precision op-amp |
| Voltage Reference | REF5025AIDGKT | 1 | 2.5V reference |
| Current Sense Resistor | RN73R2BTTD1251B10 | 1 | 1.25kΩ, 0.1% |
| Polarity Relay | G5V-2-5DC | 1 | DPDT, 5V |
| Relay Driver | 2N7002 | 1 | MOSFET |
| Current Set Resistors | Various | 5 | 0.1% tolerance |

### Probe-Side ERT Components
| Component | Part Number | Quantity | Notes |
|-----------|-------------|----------|-------|
| Multiplexer | CD4051BE | 1 | 8-channel mux |
| Differential Amp | INA128PAG4 | 1 | Can share with MIT |
| ADC | ADS1256IDBR | 1 | Shared with MIT |
| ERT Rings | Custom | 2-3 | Stainless steel |
| Shielded Cable | Custom | 3m | Twisted pair |

---

## Safety Considerations

### Current Limiting
- Maximum current: 2 mA (safe for human contact)
- Current source has inherent limiting
- Add fuse in series if needed (100 mA fast-blow)

### Overvoltage Protection
- Input protection on ADC (series resistors, clamp diodes)
- TVS diodes on electrode inputs (optional)
- Limit maximum voltage to ±5V

### Isolation
- Ensure rings are isolated from each other
- Verify isolation resistance (>1 MΩ)
- Check for ground faults

### Ground Fault Protection
- Monitor for unexpected current paths
- Detect ground faults
- Shut down if fault detected

---

## Design Notes

### Contact Resistance
- ERT rings must have good contact with soil
- Pre-moisten holes if soil is dry
- Use conductive gel if needed
- Monitor contact resistance

### Measurement Accuracy
- Use precision resistors (0.1% tolerance)
- Calibrate current source
- Verify voltage measurement accuracy
- Account for mux on-resistance

### Noise Reduction
- Average multiple samples
- Use low-pass filtering
- Reverse polarity and average
- Shield all signal paths

### Calibration
- Calibrate current source (measure actual current)
- Calibrate voltage measurement (apply known voltage)
- Verify mux operation (test each channel)
- Document calibration values

---

*For MIT circuits, see [MIT Circuit](mit-circuit.md)*
*For base hub circuits, see [Base Hub Circuit](base-hub-circuit.md)*

