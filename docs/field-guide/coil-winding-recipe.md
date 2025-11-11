# Coil Winding Recipe

## Overview

This document provides detailed specifications for winding the MIT TX and RX coils used in the HIRT probe system.

## Target Specifications

- **Inductance:** 1–2 mH (target ~1.5 mH)
- **Q Factor:** >20 (preferred)
- **Core:** Ferrite rod, Ø8–10 mm × 100–120 mm
- **Wire:** 32–36 AWG enameled magnet wire

## Materials Required

- Ferrite rod core (Ø8–10 mm × 100–120 mm)
- Magnet wire (32–36 AWG, enameled copper)
- Coil form/bobbin (optional, or wind directly on core)
- Epoxy or varnish for securing windings
- LCR meter for testing (helpful but not essential)

## Winding Procedure

### Step 1: Prepare Core

1. Clean ferrite rod surface
2. If using bobbin, mount on core
3. Mark winding start position

### Step 2: Wind Coil

**For ~1.5 mH inductance on Ø10 mm × 100 mm ferrite:**

- **Wire:** 34 AWG (0.16 mm diameter)
- **Turns:** ~200–300 turns (start with 250)
- **Winding:** Single layer, close-wound
- **Length:** ~40–50 mm of core length

**Calculation notes:**
- Inductance ≈ (turns² × core factor) / length
- Core factor depends on ferrite permeability (typically 1000–3000)
- Adjust turns based on measured inductance

### Step 3: Secure Windings

1. Apply thin layer of epoxy or varnish
2. Allow to cure completely
3. Trim wire ends, leave ~10 cm for connections
4. Strip insulation from ends (carefully)

### Step 4: Test Inductance

1. Measure inductance with LCR meter
2. Measure Q factor if possible
3. Measure DC resistance
4. Adjust if needed (add/remove turns)

## TX Coil Specifications

- **Inductance:** 1–2 mH
- **Q Factor:** >20
- **DC Resistance:** <10 Ω (typical)
- **Current Rating:** 50–100 mA RMS
- **Orientation:** Mount orthogonal to RX coil

## RX Coil Specifications

- **Inductance:** 1–2 mH (match TX if possible)
- **Q Factor:** >20
- **DC Resistance:** <10 Ω (typical)
- **Sensitivity:** Optimize for low-noise reception
- **Orientation:** Mount orthogonal to TX coil

## Winding Variations

### Higher Inductance (2–3 mH)
- More turns (300–400)
- Thicker wire (32 AWG) if space allows
- Longer winding length

### Lower Inductance (0.5–1 mH)
- Fewer turns (100–150)
- Shorter winding length
- May need higher frequency operation

### Multi-Layer Winding
- If single layer insufficient
- Wind in layers with insulation between
- Note: May reduce Q factor

## Quality Control

### Before Installation
- [ ] Inductance within spec (1–2 mH)
- [ ] Q factor acceptable (>20)
- [ ] DC resistance reasonable (<10 Ω)
- [ ] Windings secure (no loose turns)
- [ ] Wire ends properly prepared

### Testing
- Measure inductance at operating frequencies if possible
- Check for shorts (coil to core)
- Verify wire continuity
- Test with known metal target if possible

## Troubleshooting

### Low Inductance
- **Cause:** Too few turns
- **Fix:** Add more turns

### Low Q Factor
- **Cause:** Poor core, loose windings, or losses
- **Fix:** Check core quality, secure windings, use better wire

### High Resistance
- **Cause:** Wire too thin or too long
- **Fix:** Use thicker wire or shorter leads

### Inconsistent Results
- **Cause:** Winding variations between coils
- **Fix:** Standardize procedure, document each coil

## Coil Constants

Record for each coil:
- Probe ID: _______
- Coil Type: TX / RX
- Inductance: _______ mH
- Q Factor: _______
- DC Resistance: _______ Ω
- Turns: _______
- Wire Gauge: _______
- Date: _______

## Notes

- Inductance may vary slightly with frequency (measure at operating frequency if possible)
- Q factor indicates coil efficiency (higher is better)
- Match TX and RX coils if possible for consistent performance
- Keep winding procedure consistent across all probes

