# 9. Calibration

## Overview

This section provides detailed calibration procedures for HIRT probes before field deployment. Calibration ensures accurate and consistent measurements across all probes in the array.

---

## Table of Contents

1. [Calibration Principles](#calibration-principles)
2. [Required Equipment](#required-equipment)
3. [Coil Calibration](#coil-calibration)
4. [TX Calibration](#tx-calibration)
5. [RX Calibration](#rx-calibration)
6. [ERT Calibration](#ert-calibration)
7. [System-Level Calibration](#system-level-calibration)
8. [Field Quick-Check Procedure](#field-quick-check-procedure)
9. [Recalibration Schedule](#recalibration-schedule)

---

## Calibration Principles

### Why Calibrate?

1. **Probe-to-probe consistency** - Ensure all probes give comparable readings
2. **Accuracy** - Relate measurements to physical units
3. **Drift compensation** - Account for component aging
4. **Quality assurance** - Verify proper assembly and function

### Calibration Hierarchy

1. **Component-level** - Individual parts (coils, resistors)
2. **Subsystem-level** - TX chain, RX chain, ERT chain
3. **System-level** - Complete probe in known conditions
4. **Array-level** - Cross-calibration between probes

### Key Parameters

| Parameter | Target | Tolerance |
|-----------|--------|-----------|
| Coil Inductance | 1-2 mH | +/-10% |
| Coil Q Factor | >20 | Minimum |
| DDS Frequency | Commanded | +/-1% |
| TX Amplitude | Design spec | +/-10% |
| RX Gain | Design spec | +/-10% |
| ERT Current | Commanded | +/-5% |
| Reciprocity | A->B = B->A | +/-5% |

---

## Required Equipment

### Calibration Equipment

| Equipment | Purpose | Notes |
|-----------|---------|-------|
| LCR Meter | Coil measurements | Set to 10 kHz |
| Oscilloscope | Signal verification | 50 MHz minimum |
| Multimeter | Voltage/current/resistance | 6-digit preferred |
| Function Generator | Signal injection | 1 Hz - 100 kHz |
| Known Test Targets | Coupling verification | Aluminum plate, steel bar |
| Test Medium | ERT testing | Sand box with known resistivity |
| Reference Probe | Cross-calibration | If available |

### Environment Requirements

- Temperature: 20-25C (stable)
- EMI: Minimize interference
- Warm-up time: Allow 15-30 minutes for equipment stabilization

### Calibration Sheet

Prepare a calibration sheet for each probe (template at end of section).

---

## Coil Calibration

### Step 1: TX Coil Inductance

**Purpose:** Verify TX coil inductance is within specification.

**Procedure:**
1. Connect LCR meter to TX coil leads
2. Set measurement frequency to 10 kHz
3. Measure inductance (L)
4. Record value in mH
5. Compare to spec (target: 1-2 mH, tolerance +/-10%)

**Record:**
- TX Coil Inductance: _______ mH
- Test Frequency: 10 kHz
- Pass/Fail: _______

### Step 2: TX Coil Q Factor

**Purpose:** Verify TX coil quality factor.

**Procedure:**
1. Using same LCR meter setup
2. Measure Q factor directly (if available)
3. Or calculate: Q = (2 x pi x f x L) / R
4. Record value
5. Compare to spec (target: >20)

**Record:**
- TX Coil Q Factor: _______
- Pass/Fail: _______

### Step 3: TX Coil DC Resistance

**Procedure:**
1. Use multimeter (ohmmeter mode)
2. Measure DC resistance
3. Record value in ohms
4. Compare to spec (target: <10 ohm)

**Record:**
- TX Coil DC Resistance: _______ ohm

### Step 4: RX Coil Parameters

Repeat Steps 1-3 for RX coil.

**Record:**
- RX Coil Inductance: _______ mH
- RX Coil Q Factor: _______
- RX Coil DC Resistance: _______ ohm

### Step 5: Coil Constants

**Calculate:**
- Coil constant: K = L / (turns squared)
- Coil efficiency: eta = Q / (2 x pi x f x L)

**Record:**
- TX Coil Constant K: _______
- RX Coil Constant K: _______

---

## TX Calibration

### Step 6: DDS Frequency Accuracy

**Purpose:** Verify DDS generates correct frequencies.

**Procedure:**
1. Configure DDS for test frequency (10 kHz)
2. Measure output frequency with oscilloscope
3. Calculate frequency error: Error = (f_actual - f_commanded) / f_commanded
4. Test at: 2, 5, 10, 20, 50 kHz
5. Record all values

**Expected:** Frequency accuracy +/-1% or better

**Record:**
| Commanded (kHz) | Actual (kHz) | Error (%) |
|-----------------|--------------|-----------|
| 2 | _______ | _______ |
| 5 | _______ | _______ |
| 10 | _______ | _______ |
| 20 | _______ | _______ |
| 50 | _______ | _______ |

### Step 7: TX Output Amplitude

**Procedure:**
1. Configure DDS for 10 kHz
2. Measure DDS output amplitude (before driver)
3. Measure driver output amplitude (after driver)
4. Calculate gain: G = V_out / V_in
5. Test across frequency range

**Expected:**
- DDS output: ~0.6 V RMS typical
- Driver output: 1-5 V RMS
- Gain: 2-5x (design dependent)

**Record:**
| Freq (kHz) | DDS Out (V) | Driver Out (V) | Gain |
|------------|-------------|----------------|------|
| 10 | _______ | _______ | _______ |
| 20 | _______ | _______ | _______ |
| 50 | _______ | _______ | _______ |

### Step 8: TX Output Distortion

**Procedure:**
1. Configure DDS for 10 kHz
2. Observe waveform on oscilloscope
3. Check for clean sine wave
4. Measure THD if equipment allows
5. Record observations

**Expected:** THD <1%, clean sine wave

**Record:**
- Waveform Quality: Good / Fair / Poor
- THD: _______ % (if measured)

---

## RX Calibration

### Step 9: RX Chain Gain

**Purpose:** Verify RX amplification chain gain.

**Procedure:**
1. Inject known test signal into RX input (e.g., 1 mV at 10 kHz)
2. Measure at preamp output
3. Measure at inst. amp output
4. Measure at ADC input
5. Calculate gain at each stage

**Expected:** Match design specifications

**Record:**
| Stage | Input | Output | Gain |
|-------|-------|--------|------|
| Preamp | 1 mV | _______ | _______ |
| Inst Amp | _______ | _______ | _______ |
| Total | 1 mV | _______ | _______ |

### Step 10: RX Noise Floor

**Procedure:**
1. Place probe in quiet environment (away from metal objects)
2. No input signal (or shorted input)
3. Measure output noise amplitude
4. Record multiple measurements
5. Calculate standard deviation

**Expected:** Noise floor <1% of full scale

**Record:**
- Baseline amplitude logged
- Baseline phase logged
- Noise Floor: _______ (units)
- Noise Floor (% of full scale): _______ %

### Step 11: Lock-In Calibration

**Procedure:**
1. Apply known test signal to RX input
2. Configure lock-in for signal frequency
3. Measure detected amplitude
4. Compare to input amplitude
5. Calculate calibration factor: Cal = V_actual / V_detected

**Record:**
- Lock-In Calibration Factor: _______
- Amplitude Accuracy: _______ %

---

## ERT Calibration

### Step 12: Ring Isolation Verification

**Procedure:**
1. Measure resistance between rings
2. Measure resistance ring to ground
3. Measure resistance ring to rod
4. Record all values

**Expected:** All >1 M-ohm

**Record:**
- Ring A to Ring B: _______ M-ohm
- Ring B to Ring C: _______ M-ohm
- Ring A to Ground: _______ M-ohm
- Ring A to Rod: _______ M-ohm

### Step 13: Contact Resistance Measurement

**Procedure:**
1. Insert probe into test medium (sand box)
2. Inject test current (1 mA)
3. Measure voltage between rings
4. Calculate resistance: R = V / I
5. Test with different current levels

**Record:**
| Current (mA) | Voltage (mV) | Resistance (ohm) |
|--------------|--------------|------------------|
| 0.5 | _______ | _______ |
| 1.0 | _______ | _______ |
| 2.0 | _______ | _______ |

### Step 14: ERT Measurement Chain Calibration

**Procedure:**
1. Apply known voltage to ERT ring input
2. Select ring via multiplexer
3. Read ADC value
4. Calculate calibration factor: Cal = V_known / ADC_value
5. Repeat for all rings

**Record:**
| Ring | V_known (mV) | ADC Value | Cal Factor |
|------|--------------|-----------|------------|
| A | 100 | _______ | _______ |
| B | 100 | _______ | _______ |
| C | 100 | _______ | _______ |

### Step 15: Current Source Calibration

**Procedure:**
1. Connect current source to test load (1 k-ohm)
2. Set current level (1 mA)
3. Measure actual current with multimeter
4. Calculate calibration factor: Cal = I_actual / I_commanded
5. Test across current range

**Record:**
| Commanded (mA) | Actual (mA) | Cal Factor |
|----------------|-------------|------------|
| 0.5 | _______ | _______ |
| 1.0 | _______ | _______ |
| 1.5 | _______ | _______ |
| 2.0 | _______ | _______ |

---

## System-Level Calibration

### Step 16: Coupling Test Calibration

**Purpose:** Verify MIT system detects conductive targets.

**Procedure:**
1. Set up two probes 1-3 m apart
2. Place known target (aluminum plate) between probes
3. Configure one probe TX, other RX
4. Measure amplitude and phase at multiple frequencies
5. Remove target, measure baseline
6. Calculate coupling response

**Expected Results:**
- Amplitude reduction: 10-50% depending on target size
- Phase lag: 5-30 degrees depending on frequency
- Lower frequencies show deeper sensitivity

**Record:**
| Freq (kHz) | Baseline Amp | Target Amp | Drop (%) | Phase Shift |
|------------|--------------|------------|----------|-------------|
| 2 | _______ | _______ | _______ | _______ |
| 10 | _______ | _______ | _______ | _______ |
| 50 | _______ | _______ | _______ | _______ |

### Step 17: Reciprocity Calibration

**Procedure:**
1. Set up two probes (A and B)
2. Measure A to B (A as TX, B as RX)
3. Measure B to A (B as TX, A as RX)
4. Calculate reciprocity error: Error = |A->B - B->A| / |A->B|
5. Repeat for multiple frequencies

**Expected:** Reciprocity error <5%

**Record:**
| Freq (kHz) | A->B | B->A | Error (%) |
|------------|------|------|-----------|
| 10 | _______ | _______ | _______ |
| 20 | _______ | _______ | _______ |
| 50 | _______ | _______ | _______ |

### Step 18: System Baseline

**Procedure:**
1. Set up probes in test area (away from metal)
2. Measure baseline amplitude and phase
3. Record multiple measurements
4. Calculate average and standard deviation

**Record:**
- Baseline Amplitude: _______ (avg) +/- _______ (std)
- Baseline Phase: _______ (avg) +/- _______ (std)

---

## Field Quick-Check Procedure

### Pre-Deployment Quick Check

Before each field deployment, perform these abbreviated checks:

#### 1. Visual Inspection
- [ ] No visible damage
- [ ] All connections secure
- [ ] Cables intact

#### 2. Power Check
- [ ] Power on, no errors
- [ ] LEDs functioning
- [ ] Communication established

#### 3. Coil Quick Test
- [ ] TX coil connected (measure ~1-2 mH)
- [ ] RX coil connected (measure ~1-2 mH)
- [ ] Q factor reasonable (>20)

#### 4. ERT Quick Test
- [ ] Rings isolated (>1 M-ohm)
- [ ] No shorts to ground

#### 5. Coupling Verification
- [ ] Wave hand near coil, see response
- [ ] Or use small metal target

### Field Calibration Check

If readings seem anomalous in the field:

1. **Check baseline** - Move probe away from targets, verify baseline
2. **Check reciprocity** - Swap TX/RX, verify similar readings
3. **Check ERT** - Verify ring isolation, check for moisture
4. **Check connections** - Wiggle cables, look for intermittent issues

---

## Recalibration Schedule

### When to Re-Calibrate

| Trigger | Action |
|---------|--------|
| Before each field deployment | Quick check |
| After any repairs or modifications | Full calibration |
| If measurements become inconsistent | Full calibration |
| Periodically (every 6 months minimum) | Full calibration |
| After environmental exposure (extreme temps, water ingress) | Full calibration |

### Calibration Validity

Calibration is valid if:
- All measurements within specifications
- No repairs or modifications since calibration
- System performance consistent
- Within calibration validity period (6 months typical)

### Documentation Requirements

Maintain calibration records:
- [ ] All probes calibrated
- [ ] Calibration sheets stored
- [ ] Calibration dates recorded
- [ ] Re-calibration schedule maintained
- [ ] Issues documented

---

## Calibration Sheet Template

```
HIRT Probe Calibration Sheet

Probe ID: _____________
Date: _____________
Calibrated By: _____________

COIL PARAMETERS
TX Coil Inductance: _______ mH
TX Coil Q Factor: _______
TX Coil DC Resistance: _______ ohm
RX Coil Inductance: _______ mH
RX Coil Q Factor: _______
RX Coil DC Resistance: _______ ohm

TX OUTPUT
DDS Frequency Accuracy: _______ %
TX Output Amplitude: _______ V RMS
TX Output Distortion: _______ % THD
TX Current Draw: _______ mA

RX CHAIN
RX Chain Gain: _______
RX Noise Floor: _______ % of full scale
Lock-In Calibration Factor: _______

ERT
Ring Isolation: _______ M-ohm (min)
Contact Resistance: _______ ohm
ERT Calibration Factor: _______
Current Source Cal Factor: _______

SYSTEM
Coupling Response: _______
Reciprocity Error: _______ %
Baseline: _______

NOTES:
_________________________________
_________________________________

Calibration Valid Until: _____________

Signature: _____________
```

---

## Pre-Field Calibration Checklist

Before deploying to field:
- [ ] All probes calibrated
- [ ] Calibration sheets completed
- [ ] Coupling tests passed
- [ ] ERT rings verified
- [ ] Spare probes tested
- [ ] Base hub tested
- [ ] Communications verified
- [ ] Power systems tested
- [ ] Field tools packed

---

## Calibration Summary Table

Keep a calibration summary table for per-probe data:

| Probe ID | TX Coil L (mH) | TX Coil Q | RX Coil L (mH) | RX Coil Q | RX Gain (dB) | Ring Depths (m) | Firmware Rev | Date |
|----------|----------------|-----------|----------------|-----------|--------------|-----------------|--------------|------|
| P01      |                |           |                |           |              |                 |              |      |
| P02      |                |           |                |           |              |                 |              |      |
| P03      |                |           |                |           |              |                 |              |      |
| ...      |                |           |                |           |              |                 |              |      |

---

*For testing procedures, see Section 8: Testing and Verification. For field operation, see Section 10: Field Operation Manual.*
