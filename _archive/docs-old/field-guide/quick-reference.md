# HIRT Field Quick Reference

**One-Page Field Reference** - Print and laminate for field use

## Pre-Deployment

- [ ] Permits/UXO clearance confirmed
- [ ] Calibration sheets packed
- [ ] Batteries charged + spares
- [ ] Pilot rods, driver, extraction tools
- [ ] Flags, GPS, notebooks

## Grid Layout

| Scenario | Rod Length | Spacing | Section Size | Probe Count |
|----------|-----------|---------|--------------|-------------|
| Woods | 1.6 m | 1–1.5 m | 8×8 m | 12–16 |
| Crater | 3.0 m | 1.5–2 m | 10×10 m | 20–24 |
| Swamp | 1.5–2 m | 2–3 m | Variable | Variable |

## Measurement Parameters

### MIT Frequencies
- **Deep:** 2–5 kHz
- **Standard:** 5–20 kHz  
- **Shallow:** 20–50 kHz

### ERT Current
- **Level:** 0.5–2 mA
- **Polarity:** Reverse every 1–2 s
- **Baselines:** Corner-to-corner, edge-to-edge

## Deployment Workflow

1. **Install all probes** → Mark with flags → Record coordinates/depths
2. **Background scan** → Control area outside suspected zone
3. **MIT sweep** → Each probe TX, others RX at 3–5 frequencies
4. **ERT patterns** → 2–4 long baselines, all probes log voltages
5. **QC checks** → Repeat 5–10% pairs, verify reciprocity
6. **Extract** → Move to next section (keep overlap)

## Data Records

### MIT Record
`timestamp, section_id, tx_probe_id, rx_probe_id, freq_hz, amp, phase_deg, tx_current_mA, notes`

### ERT Record  
`timestamp, section_id, inject_pos_probe_id, inject_neg_probe_id, sense_probe_id, volt_mV, current_mA, polarity, notes`

## Troubleshooting Quick Fixes

| Problem | Quick Fix |
|---------|-----------|
| RX saturation | Lower TX level, increase separation |
| Noisy data | Check shielding, single-point ground |
| Unstable ERT | Pre-moisten hole, check ring contact |
| No depth sensitivity | Add longer offsets, lower frequency |
| Inconsistent repeats | Re-seat connectors, verify sync |

## Depth Expectations

| Probe Depth | Spacing | Depth Range |
|-------------|---------|-------------|
| 1.5 m | 1.0–1.5 m | 1.5–2.5 m |
| 3.0 m | 1.5–2.0 m | 3–5 m |
| 3.0 m | 2.5–3.0 m | 4–6 m |

## Emergency Contacts

- Site Supervisor: ________________
- EOD Contact: ________________
- Emergency Services: ________________

## Field Log Template

**Date:** _______ **Site:** _______ **Section:** _______

**Team:** _______ **Weather:** _______ **Soil:** _______

**Probes:** _______ **Issues:** _______

**Notes:**
_________________________________________________________________
_________________________________________________________________

