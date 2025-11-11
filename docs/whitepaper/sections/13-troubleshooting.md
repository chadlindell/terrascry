# 13. Troubleshooting (common issues)

## Common Problems and Solutions

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| **RX saturation on nearby probes** | TX too strong; direct coupling | Lower TX level; increase TX–RX separation; rotate coils |
| **Noisy MIT data** | Poor shielding; ground loop | Use twisted/shielded pairs; single-point ground; increase integration time |
| **Unstable ERT voltages** | Poor ring contact / dry sand | Pre‑moisten hole; saline gel; ensure solid ring‑soil contact |
| **No depth sensitivity** | Spacing too small; too high freq | Add longer offsets; include 2–5 kHz MIT; use longer ERT baselines |
| **Inconsistent repeats** | Loose connectors; timebase drift | Re-seat connectors; verify sync; redo calibration/QC pass |

## Detailed Troubleshooting Guide

### RX Saturation on Nearby Probes

**Symptoms:**
- Amplitude readings maxed out on nearby probes
- Phase readings erratic or stuck
- Only affects probes close to TX

**Causes:**
- TX output too high for close-range measurements
- Direct magnetic coupling between TX and RX coils
- Insufficient separation or poor coil orientation

**Solutions:**
1. **Reduce TX level:** Lower DDS output or TX driver gain
2. **Increase separation:** Use wider probe spacing for close pairs
3. **Rotate coils:** Ensure TX and RX coils are orthogonal
4. **Add attenuation:** Use lower gain on RX for nearby probes
5. **Skip close pairs:** Don't measure TX→RX pairs < 0.5 m apart

### Noisy MIT Data

**Symptoms:**
- High noise floor
- Erratic amplitude/phase readings
- Inconsistent measurements

**Causes:**
- Poor cable shielding
- Ground loops (multiple ground paths)
- EMI from nearby sources
- Insufficient integration time

**Solutions:**
1. **Check shielding:** Verify all cables are properly shielded
2. **Single-point ground:** Ensure only one ground connection
3. **Twist pairs:** Use twisted-pair cables for signal lines
4. **Increase integration:** Longer measurement time reduces noise
5. **Check EMI sources:** Move away from power lines, vehicles, radios
6. **Verify connections:** Check all connectors are tight and clean

### Unstable ERT Voltages

**Symptoms:**
- Voltage readings drift or jump
- Inconsistent measurements
- Poor contact indicated

**Causes:**
- Dry soil preventing good electrical contact
- Poor ring-to-soil contact
- Loose connections
- Polarization effects

**Solutions:**
1. **Pre-moisten hole:** Add water to improve contact
2. **Use saline gel:** Apply conductive gel around rings
3. **Check ring contact:** Ensure rings are flush with soil
4. **Reverse polarity:** Use AC or periodic polarity reversal
5. **Check connections:** Verify all wiring is secure
6. **Increase current:** Slightly higher current may improve SNR

### No Depth Sensitivity

**Symptoms:**
- Measurements don't change with depth
- All readings similar regardless of target depth
- No response to known deep targets

**Causes:**
- Probe spacing too small (all measurements near-surface)
- Frequency too high (limited penetration)
- ERT baselines too short
- Targets too deep for configuration

**Solutions:**
1. **Add longer offsets:** Include TX→RX pairs with >3 m spacing
2. **Lower frequencies:** Use 2–5 kHz for MIT (deeper penetration)
3. **Longer ERT baselines:** Corner-to-corner, edge-to-edge injections
4. **Deeper probes:** Insert probes deeper if possible
5. **Check targets:** Verify expected target depths are realistic

### Inconsistent Repeats

**Symptoms:**
- Same measurement gives different results
- Poor reciprocity (A→B ≠ B→A)
- Measurements drift over time

**Causes:**
- Loose connectors
- Timebase/sync drift
- Temperature effects
- Calibration issues

**Solutions:**
1. **Re-seat connectors:** Unplug and reconnect all cables
2. **Verify sync:** Check timebase distribution is working
3. **Check calibration:** Re-run calibration procedures
4. **Temperature:** Note temperature variations, may need compensation
5. **Stabilization time:** Allow system to warm up before measurements
6. **QC pass:** Repeat calibration/QC procedures

## Diagnostic Procedures

### System Check Procedure

1. **Power check:** Verify all probes have power (LED indicators)
2. **Communication check:** Ping each probe, verify response
3. **Sync check:** Verify timebase is distributed correctly
4. **Calibration check:** Run quick calibration on one probe
5. **Coupling test:** Test TX→RX on two probes in air

### Field Diagnostic Kit

Essential items for troubleshooting:
- Multimeter (check continuity, resistance)
- Oscilloscope (optional, check signals)
- Spare cables and connectors
- Contact cleaner
- Electrical tape, heat-shrink
- Spare probes (2–4 recommended)
- Calibration sheet

## Prevention

### Pre-Field Checks
- [ ] All connectors tight and clean
- [ ] Cables tested for continuity
- [ ] Calibration up to date
- [ ] Spare parts available
- [ ] Field diagnostic kit packed

### During Survey
- Monitor data quality in real-time
- Check reciprocity periodically
- Note any anomalies immediately
- Keep spare probes ready

### Post-Survey
- Inspect all equipment
- Note any issues for repair
- Update calibration records
- Clean and store properly

