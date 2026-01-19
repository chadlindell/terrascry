# 8. Testing and Verification

## Overview

This section provides comprehensive testing procedures and quality control checklists for HIRT probes before field deployment. All probes must pass these tests to ensure reliable field operation.

---

## Table of Contents

1. [Testing Workflow](#testing-workflow)
2. [Pre-Testing Setup](#pre-testing-setup)
3. [Mechanical Tests](#mechanical-tests)
4. [Electrical Tests](#electrical-tests)
5. [MIT Subsystem Tests](#mit-subsystem-tests)
6. [ERT Subsystem Tests](#ert-subsystem-tests)
7. [System Integration Tests](#system-integration-tests)
8. [Environmental Tests](#environmental-tests)
9. [QC Checklist](#qc-checklist)

---

## Testing Workflow

### Sequence

1. **Mechanical Testing** - Verify physical integrity
2. **Electrical Testing** - Verify power and continuity
3. **MIT Subsystem Testing** - Verify coils and signal chain
4. **ERT Subsystem Testing** - Verify electrodes and measurement
5. **System Integration Testing** - Verify complete system
6. **Environmental Testing** - Verify robustness
7. **QC Sign-off** - Final approval

### Pass/Fail Criteria

- All tests must PASS for field deployment
- CONDITIONAL status requires documented workarounds
- FAIL status requires repair and re-test

---

## Pre-Testing Setup

### Test Equipment Required

| Equipment | Purpose | Minimum Spec |
|-----------|---------|--------------|
| Power Supply | Probe power | 5V or 12V, current-limited |
| Digital Multimeter | Resistance, voltage, current | 0.1% accuracy |
| LCR Meter | Coil inductance and Q factor | 10 kHz test frequency |
| Oscilloscope | Signal verification | 50 MHz, 2-channel |
| Function Generator | Signal injection | 1 Hz - 100 kHz |
| Base Hub | System-level testing | Complete unit |

### Test Environment

- Clean, well-lit workbench
- Temperature: 20-25C (room temperature)
- Minimize EMI sources (move phones, radios away)
- Follow electrical safety procedures

### Documentation Preparation

Before testing, prepare:
- [ ] Test log template (see end of section)
- [ ] Probe registry entry
- [ ] Calibration sheet (see Section 9)
- [ ] Pass/fail criteria checklist

---

## Mechanical Tests

### Test 1: Rod Integrity

**Purpose:** Verify rod is straight, undamaged, and properly assembled.

**Procedure:**
1. Inspect rod visually for cracks, bends, or damage
2. Measure rod length (should match specification +/-5 mm)
3. Roll rod on flat surface to check straightness
4. Verify thread engagement on all joints
5. Perform gentle pull test on joints

**Pass Criteria:**
- [ ] No visible damage
- [ ] Length within specification
- [ ] Rod straight (no visible bends)
- [ ] Threads engaged properly
- [ ] Joints secure (no movement)

### Test 2: Probe Head Integrity

**Purpose:** Verify probe head (junction box) is properly sealed.

**Procedure:**
1. Inspect for cracks or damage
2. Check cap seal
3. Verify cable gland seal
4. Test attachment to rod
5. Check for loose components

**Pass Criteria:**
- [ ] No visible damage
- [ ] Cap sealed properly
- [ ] Cable gland sealed
- [ ] Securely attached to rod
- [ ] No loose components

### Test 3: ERT Ring Mounting

**Purpose:** Verify ERT rings are properly mounted and positioned.

**Procedure:**
1. Measure ring positions (0.5 m, 1.5 m, 2.5 m from tip)
2. Check ring attachment (no movement)
3. Verify electrical isolation from rod
4. Inspect ring connections
5. Check for damage or corrosion

**Pass Criteria:**
- [ ] Ring positions within +/-2 cm
- [ ] Rings securely mounted
- [ ] Rings isolated from rod (>1 M-ohm)
- [ ] Connections intact
- [ ] No visible damage

### Test 4: Cable Integrity

**Purpose:** Verify cable is properly routed and undamaged.

**Procedure:**
1. Visual inspection for damage
2. Test continuity of all conductors
3. Check cable routing (no sharp bends)
4. Verify strain relief
5. Gentle pull test on cable

**Pass Criteria:**
- [ ] No visible damage
- [ ] All conductors continuous
- [ ] Properly routed
- [ ] Strain relief effective
- [ ] Pull strength adequate

---

## Electrical Tests

### Test 5: Power Supply

**Purpose:** Verify power and regulation circuits.

**Procedure:**
1. Connect probe to power supply
2. Measure input voltage
3. Measure 3.3V rail (should be 3.3V +/-0.1V)
4. Measure 5V rail (should be 5.0V +/-0.1V)
5. Measure current draw
6. Check for voltage stability

**Pass Criteria:**
- [ ] 3.3V rail: 3.3V +/-0.1V
- [ ] 5V rail: 5.0V +/-0.1V
- [ ] Current draw within spec
- [ ] No voltage fluctuations
- [ ] No excessive heating

### Test 6: Continuity and Shorts

**Purpose:** Verify no shorts or open circuits.

**Procedure:**
1. Power off probe
2. Measure resistance between power and ground (should be >100 ohm)
3. Check for shorts between signal lines
4. Verify all connections continuous
5. Check shield connections

**Pass Criteria:**
- [ ] No shorts detected
- [ ] All connections continuous
- [ ] Shield connected properly
- [ ] No unexpected low resistance

### Test 7: Communication Interface

**Purpose:** Verify communication is functional.

**Procedure:**
1. Connect probe to base hub
2. Send test command
3. Verify response received
4. Test data transfer
5. Check reliability (multiple attempts)

**Pass Criteria:**
- [ ] Probe responds to commands
- [ ] Data transfer successful
- [ ] Reliability >95%
- [ ] No errors or timeouts

---

## MIT Subsystem Tests

### Test 8: Coil Parameters

**Purpose:** Verify TX and RX coils meet specifications.

**Procedure:**
1. Measure TX coil inductance (target: 1-2 mH)
2. Measure TX coil Q factor (target: >20)
3. Measure RX coil inductance (target: 1-2 mH)
4. Measure RX coil Q factor (target: >20)
5. Measure DC resistance of each coil

**Pass Criteria:**
- [ ] TX inductance: 1-2 mH
- [ ] TX Q factor: >20
- [ ] RX inductance: 1-2 mH
- [ ] RX Q factor: >20
- [ ] DC resistance: <10 ohm per coil

### Test 9: DDS Output

**Purpose:** Verify DDS generates correct frequencies.

**Procedure:**
1. Configure DDS for test frequency (10 kHz)
2. Measure output frequency with oscilloscope
3. Measure output amplitude
4. Check for distortion
5. Test multiple frequencies (2, 5, 10, 20, 50 kHz)

**Pass Criteria:**
- [ ] Frequency accuracy: +/-1%
- [ ] Output amplitude stable
- [ ] Distortion: <1% THD
- [ ] All test frequencies functional

### Test 10: TX Driver Output

**Purpose:** Verify TX driver amplifies correctly.

**Procedure:**
1. Configure DDS for test frequency
2. Measure DDS output (before driver)
3. Measure driver output (after amplification)
4. Calculate gain
5. Test across frequency range

**Pass Criteria:**
- [ ] Gain matches design (+/-10%)
- [ ] Output stable across frequencies
- [ ] No clipping or distortion
- [ ] Current draw within limits

### Test 11: RX Chain Response

**Purpose:** Verify RX amplification chain functions.

**Procedure:**
1. Inject test signal into RX input
2. Measure signal at each stage
3. Calculate gain at each stage
4. Verify total gain
5. Measure noise floor

**Pass Criteria:**
- [ ] Gain matches design (+/-10%)
- [ ] Noise floor <1% of full scale
- [ ] No saturation or clipping
- [ ] Signal chain functional

### Test 12: MIT Coupling Test

**Purpose:** Verify MIT system detects conductive targets.

**Procedure:**
1. Set up two probes 1-3 m apart
2. Place aluminum plate between probes
3. Configure one probe TX, other RX
4. Measure amplitude and phase at multiple frequencies
5. Remove target, measure baseline
6. Compare results

**Pass Criteria:**
- [ ] Amplitude drop: 10-50% with target
- [ ] Phase lag: 5-30 degrees with target
- [ ] Response varies with frequency
- [ ] Response varies with distance

---

## ERT Subsystem Tests

### Test 13: ERT Ring Isolation

**Purpose:** Verify ERT rings are electrically isolated.

**Procedure:**
1. Measure resistance between rings
2. Measure resistance ring to ground
3. Measure resistance ring to rod
4. Test in dry conditions
5. Test after water exposure

**Pass Criteria:**
- [ ] Ring-to-ring: >1 M-ohm
- [ ] Ring-to-ground: >1 M-ohm
- [ ] Ring-to-rod: >1 M-ohm
- [ ] Isolation maintained after water

### Test 14: ERT Contact Resistance

**Purpose:** Verify ERT rings make good contact.

**Procedure:**
1. Insert probe into test medium (sand box)
2. Inject test current (0.5-1 mA)
3. Measure voltage between rings
4. Calculate contact resistance
5. Verify stable readings

**Pass Criteria:**
- [ ] Contact resistance: <1000 ohm
- [ ] Readings stable
- [ ] Response linear with current
- [ ] Reasonable for test medium

### Test 15: ERT Measurement Chain

**Purpose:** Verify ERT measurement chain functions.

**Procedure:**
1. Apply known voltage to ERT input
2. Select ring via multiplexer
3. Measure at amplifier output
4. Read ADC value
5. Test all rings

**Pass Criteria:**
- [ ] Mux selects correct ring
- [ ] Amplification correct
- [ ] ADC reading matches input (+/-5%)
- [ ] All rings functional

### Test 16: ERT Current Source

**Purpose:** Verify ERT current source (base hub).

**Procedure:**
1. Connect to test load (known resistance)
2. Configure current level (0.5, 1.0, 1.5, 2.0 mA)
3. Measure actual current
4. Verify accuracy
5. Test stability

**Pass Criteria:**
- [ ] Current accuracy: +/-5%
- [ ] Current stable
- [ ] Works across range
- [ ] Works with different loads

---

## System Integration Tests

### Test 17: Full System Test

**Purpose:** Verify complete probe system functions.

**Procedure:**
1. Power on probe
2. Verify communication with base hub
3. Test MIT measurement
4. Test ERT measurement
5. Verify data collection
6. Test synchronization

**Pass Criteria:**
- [ ] All subsystems functional
- [ ] Communication reliable
- [ ] MIT measurements successful
- [ ] ERT measurements successful
- [ ] Data collection working
- [ ] Sync working

### Test 18: Reciprocity Test

**Purpose:** Verify measurement reciprocity.

**Procedure:**
1. Set up two probes (A and B)
2. Measure A to B
3. Measure B to A
4. Compare (should match within 5%)
5. Repeat for multiple pairs

**Pass Criteria:**
- [ ] Reciprocity within 5%
- [ ] Consistent across pairs
- [ ] Consistent across frequencies

### Test 19: Repeatability Test

**Purpose:** Verify measurements are repeatable.

**Procedure:**
1. Set up fixed test configuration
2. Take measurement
3. Wait 1 minute
4. Take same measurement
5. Repeat 5-10 times
6. Calculate standard deviation

**Pass Criteria:**
- [ ] Repeatability: std dev <5% of mean
- [ ] No drift over time
- [ ] Consistent across measurements

---

## Environmental Tests

### Test 20: Temperature Stability

**Purpose:** Verify performance over temperature range.

**Procedure:**
1. Test at room temperature (baseline)
2. Test at low temperature (5C if possible)
3. Test at high temperature (40C if possible)
4. Compare measurements

**Pass Criteria:**
- [ ] Performance stable across range
- [ ] No significant drift
- [ ] Calibration remains valid

### Test 21: Waterproofing Test

**Purpose:** Verify probe is waterproof.

**Procedure:**
1. Submerge probe head in water (10-20 cm depth)
2. Leave for 30 minutes
3. Remove and inspect for water ingress
4. Test electronics functionality
5. Measure insulation resistance

**Pass Criteria:**
- [ ] No water ingress
- [ ] Electronics functional
- [ ] Insulation resistance maintained

---

## QC Checklist

### Probe Information

**Probe ID:** _____________
**Date:** _____________
**Inspector:** _____________
**Status:** [ ] Pass  [ ] Fail  [ ] Conditional

---

### Mechanical Quality Control

#### Rod Assembly
- [ ] Rod straight and undamaged
- [ ] Rod length within specification (+/-5 mm)
- [ ] Threads engaged properly (if multi-section)
- [ ] Joints secure (no movement)
- [ ] No visible cracks or damage

#### Probe Head (Junction Box)
- [ ] Capsule intact (no cracks or damage)
- [ ] Cap sealed properly
- [ ] Cable gland sealed and secure
- [ ] Attached securely to rod
- [ ] No loose components

#### ERT Rings
- [ ] Rings at correct positions (0.5 m, 1.5 m, 2.5 m +/-2 cm)
- [ ] Rings securely mounted (no movement)
- [ ] Rings electrically isolated from rod (>1 M-ohm)
- [ ] Ring connections intact
- [ ] No visible damage or corrosion

#### Cable
- [ ] Cable undamaged (no kinks, cuts, abrasion)
- [ ] All conductors continuous
- [ ] Cable properly routed (no sharp bends)
- [ ] Strain relief effective
- [ ] Cable gland sealed

#### Waterproofing
- [ ] Probe head waterproof (submersion test passed)
- [ ] All seals intact
- [ ] No moisture ingress
- [ ] Electronics functional after water test

---

### Electrical Quality Control

#### Power Supply
- [ ] 3.3V rail: 3.3V +/-0.1V
- [ ] 5V rail: 5.0V +/-0.1V
- [ ] Current draw within specifications
- [ ] No voltage fluctuations
- [ ] No excessive heating

#### Continuity and Shorts
- [ ] No shorts detected
- [ ] All connections continuous
- [ ] Shield connected properly
- [ ] No unexpected low resistance

#### Communication
- [ ] Probe responds to commands
- [ ] Data transfer successful
- [ ] Communication reliable (>95% success rate)
- [ ] No errors or timeouts

---

### MIT Subsystem Quality Control

#### Coils
- [ ] TX coil inductance: 1-2 mH
- [ ] TX coil Q factor: >20
- [ ] RX coil inductance: 1-2 mH
- [ ] RX coil Q factor: >20
- [ ] Coils orthogonal (90 degree separation)
- [ ] Coils secured (no movement)
- [ ] No shorts between coils

#### DDS Output
- [ ] Frequency accuracy: +/-1% or better
- [ ] Output amplitude stable
- [ ] Distortion: <1% THD (if measurable)
- [ ] All test frequencies functional

#### TX Driver
- [ ] Gain matches design (+/-10%)
- [ ] Output stable across frequencies
- [ ] No clipping or distortion
- [ ] Current draw within limits

#### RX Chain
- [ ] Gain matches design (+/-10%)
- [ ] Noise floor acceptable (<1% of full scale)
- [ ] No saturation or clipping
- [ ] Signal chain functional

#### Lock-In Detection
- [ ] Amplitude detection accurate (+/-5%)
- [ ] Phase detection functional (if applicable)
- [ ] Works across frequency range
- [ ] Noise rejection adequate

#### MIT Coupling Test
- [ ] Amplitude drop measurable (10-50%)
- [ ] Phase lag measurable (5-30 degrees)
- [ ] Response varies with frequency
- [ ] Response varies with distance

---

### ERT Subsystem Quality Control

#### ERT Ring Isolation
- [ ] Ring-to-ring resistance: >1 M-ohm
- [ ] Ring-to-ground resistance: >1 M-ohm
- [ ] Ring-to-rod resistance: >1 M-ohm
- [ ] Isolation maintained after water exposure

#### ERT Contact Resistance
- [ ] Contact resistance: <1000 ohm (typical)
- [ ] Readings stable (no drift)
- [ ] Response linear with current
- [ ] Readings reasonable for test medium

#### ERT Measurement Chain
- [ ] Multiplexer selects correct ring
- [ ] Amplification correct
- [ ] ADC reading matches input (+/-5%)
- [ ] All rings functional
- [ ] Measurement chain stable

#### ERT Current Source (Base Hub)
- [ ] Current accuracy: +/-5%
- [ ] Current stable (no drift)
- [ ] Works across current range
- [ ] Works with different loads

---

### System Integration Quality Control

#### Full System Test
- [ ] All subsystems functional
- [ ] Communication reliable
- [ ] MIT measurements successful
- [ ] ERT measurements successful
- [ ] Data collection working
- [ ] Synchronization working (if applicable)

#### Reciprocity Test
- [ ] Reciprocity: A->B approximately equals B->A (within 5%)
- [ ] Consistent across probe pairs
- [ ] Consistent across frequencies

#### Repeatability Test
- [ ] Measurements repeatable (std dev <5% of mean)
- [ ] No drift over time
- [ ] Consistent across measurements

---

### Calibration Quality Control

#### Coil Calibration
- [ ] TX coil inductance measured and recorded
- [ ] TX coil Q factor measured and recorded
- [ ] RX coil inductance measured and recorded
- [ ] RX coil Q factor measured and recorded
- [ ] All values within specifications

#### TX Output Calibration
- [ ] DDS frequency accuracy verified
- [ ] TX output amplitude measured
- [ ] TX output distortion checked
- [ ] TX current draw measured
- [ ] All values within specifications

#### RX Chain Calibration
- [ ] RX chain gain measured
- [ ] RX noise floor measured
- [ ] Lock-in calibration factor determined
- [ ] All values within specifications

#### ERT Calibration
- [ ] ERT ring isolation verified
- [ ] Contact resistance measured
- [ ] ERT calibration factor determined
- [ ] All values within specifications

#### System Calibration
- [ ] Coupling test completed
- [ ] Reciprocity verified
- [ ] Baseline established
- [ ] All values within specifications

---

### Documentation Quality Control

#### Calibration Sheet
- [ ] Calibration sheet completed
- [ ] All measurements recorded
- [ ] Dates and signatures present
- [ ] Any issues documented

#### Test Log
- [ ] Test log completed
- [ ] All tests documented
- [ ] Pass/fail status recorded
- [ ] Any issues documented

#### Probe Registry
- [ ] Probe ID recorded
- [ ] Build date recorded
- [ ] Calibration date recorded
- [ ] Status updated

---

### Environmental Quality Control

#### Temperature Stability
- [ ] Performance stable across temperature range
- [ ] No significant drift
- [ ] Calibration remains valid

#### Waterproofing
- [ ] No water ingress
- [ ] Electronics functional after test
- [ ] Insulation resistance maintained

#### Vibration (Optional)
- [ ] No damage
- [ ] Performance unchanged
- [ ] All connections intact

---

### Pre-Field Quality Control

#### Final Inspection
- [ ] All mechanical checks passed
- [ ] All electrical checks passed
- [ ] All MIT checks passed
- [ ] All ERT checks passed
- [ ] All system integration checks passed
- [ ] All calibration checks passed
- [ ] All documentation complete

#### Field Readiness
- [ ] Probe labeled clearly
- [ ] Probe registered
- [ ] Calibration valid
- [ ] Test results documented
- [ ] Spare parts available (if needed)
- [ ] Field tools ready

---

## Quality Control Summary

### Overall Status

- [ ] **PASS** - All checks passed, ready for field deployment
- [ ] **CONDITIONAL** - Minor issues, acceptable for field use with notes
- [ ] **FAIL** - Issues found, requires repair/re-calibration

### Issues Found

List any issues found during QC:

1. _____________________________________________
2. _____________________________________________
3. _____________________________________________

### Actions Required

List any actions required before field deployment:

1. _____________________________________________
2. _____________________________________________
3. _____________________________________________

### Notes

_________________________________________________
_________________________________________________
_________________________________________________

---

## Sign-Off

**Inspector:** _____________
**Date:** _____________
**Signature:** _____________

**Approved for Field Deployment:** [ ] Yes  [ ] No  [ ] Conditional

**Conditions (if conditional):**
_________________________________________________
_________________________________________________

---

## Test Log Template

```
HIRT Probe Test Log

Probe ID: _____________
Date: _____________
Tester: _____________

MECHANICAL TESTS
| Test | Pass | Fail | Notes |
|------|------|------|-------|
| Rod Integrity | [ ] | [ ] | |
| Probe Head | [ ] | [ ] | |
| ERT Rings | [ ] | [ ] | |
| Cable | [ ] | [ ] | |

ELECTRICAL TESTS
| Test | Pass | Fail | Notes |
|------|------|------|-------|
| Power Supply | [ ] | [ ] | |
| Continuity | [ ] | [ ] | |
| Communication | [ ] | [ ] | |

MIT TESTS
| Test | Pass | Fail | Notes |
|------|------|------|-------|
| Coil Parameters | [ ] | [ ] | |
| DDS Output | [ ] | [ ] | |
| TX Driver | [ ] | [ ] | |
| RX Chain | [ ] | [ ] | |
| Coupling Test | [ ] | [ ] | |

ERT TESTS
| Test | Pass | Fail | Notes |
|------|------|------|-------|
| Ring Isolation | [ ] | [ ] | |
| Contact Resistance | [ ] | [ ] | |
| Measurement Chain | [ ] | [ ] | |
| Current Source | [ ] | [ ] | |

INTEGRATION TESTS
| Test | Pass | Fail | Notes |
|------|------|------|-------|
| Full System | [ ] | [ ] | |
| Reciprocity | [ ] | [ ] | |
| Repeatability | [ ] | [ ] | |

ENVIRONMENTAL TESTS
| Test | Pass | Fail | Notes |
|------|------|------|-------|
| Temperature | [ ] | [ ] | |
| Waterproofing | [ ] | [ ] | |

OVERALL RESULT: [ ] PASS [ ] FAIL

Issues Found:
_________________________________
_________________________________

Actions Required:
_________________________________
_________________________________

Signature: _____________ Date: _____________
```

---

## Next Steps

**If PASS:**
- [ ] Probe ready for field deployment
- [ ] Review Field Operation Manual
- [ ] Pack probe with documentation

**If CONDITIONAL:**
- [ ] Address minor issues
- [ ] Re-test affected systems
- [ ] Update documentation
- [ ] Re-evaluate for field deployment

**If FAIL:**
- [ ] Document all issues
- [ ] Repair or replace components
- [ ] Re-test all systems
- [ ] Re-calibrate if needed
- [ ] Re-run QC checklist

---

*For calibration procedures, see Section 9: Calibration. For troubleshooting, see Section 13: Troubleshooting.*
