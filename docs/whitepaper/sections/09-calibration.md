# 9. Bench Calibration & QC (before field)

## Pre-Field Calibration Checklist

### 1. Coil Checks

**Measure inductance & Q factor:**
- Use LCR meter if available
- Target inductance: **1–2 mH** for TX/RX coils
- Q factor: >20 preferred (indicates low losses)
- Document per-probe coil constants

**Verification:**
- [ ] TX coil inductance within spec
- [ ] RX coil inductance within spec
- [ ] Q factor acceptable
- [ ] Coil resistance measured

### 2. TX Output Verification

**Confirm sine amplitude across frequencies:**
- Sweep through frequency range (2–50 kHz)
- Measure output voltage into coil
- Verify amplitude stability
- Check for distortion (THD < 1% preferred)

**Verification:**
- [ ] Output amplitude stable across frequencies
- [ ] No unexpected resonances
- [ ] Distortion acceptable
- [ ] Current draw within limits

### 3. RX Chain Noise Floor

**Measure in air, probe far from metal:**
- Place probe in open area away from metal objects
- Record baseline amplitude/phase
- Measure noise floor
- Verify lock‑in operation

**Verification:**
- [ ] Baseline amplitude logged
- [ ] Baseline phase logged
- [ ] Noise floor acceptable (< 1% of full scale)
- [ ] Lock‑in functioning correctly

### 4. Coupling Test

**Place aluminum plate or steel bar between two probes:**
- Distance: 1–3 m between probes
- Sweep frequencies (2, 5, 10, 20, 50 kHz)
- Verify measurable amplitude drop
- Verify phase lag

**Expected Results:**
- Amplitude reduction: 10–50% depending on target size
- Phase lag: 5–30 degrees depending on frequency
- Lower frequencies show deeper sensitivity

**Verification:**
- [ ] Amplitude drop measurable
- [ ] Phase lag measurable
- [ ] Response varies with frequency as expected
- [ ] Response varies with distance as expected

### 5. ERT Ring Continuity

**Verify insulation between rings:**
- Measure resistance between rings (should be >1 MΩ)
- Check for shorts to ground
- Test in sand box: inject **0.5–1 mA**
- Confirm stable voltages

**Verification:**
- [ ] Rings electrically isolated (>1 MΩ)
- [ ] No shorts to ground
- [ ] Stable voltage readings in test medium
- [ ] Contact resistance acceptable

## Calibration Sheet Template

Keep a **calibration sheet** with per‑probe data:

| Probe ID | TX Coil L (mH) | TX Coil Q | RX Coil L (mH) | RX Coil Q | RX Gain (dB) | Ring Depths (m) | Firmware Rev | Date |
|----------|----------------|-----------|----------------|-----------|--------------|-----------------|--------------|------|
| P01      |                |           |                |           |              |                 |              |      |
| P02      |                |           |                |           |              |                 |              |      |

## Quality Control Procedures

### Reciprocity Check
- Measure TX→RX path (A→B)
- Measure RX→TX path (B→A)
- Values should match within 5%
- If not, investigate coupling issues

### Repeatability Check
- Repeat 5–10% of measurements
- Verify consistency within noise floor
- Document any outliers

### Temperature Stability
- Test over expected temperature range
- Verify calibration stability
- Note any temperature-dependent behavior

## Pre-Field Checklist

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

