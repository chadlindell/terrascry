# Coil Winding Recipe

## Overview

This document provides detailed specifications for winding the MIT TX and RX coils used in the HIRT probe system. These coils are optimized for broadband operation across the 2-50 kHz frequency range.

## Target Specifications

| Parameter | Target | Tolerance | Verification |
|-----------|--------|-----------|--------------|
| Inductance | 1.5 mH | ±10% | LCR meter @ 1 kHz |
| Q Factor @ 2 kHz | ≥25 | Minimum | LCR meter |
| Q Factor @ 10 kHz | ≥30 | Minimum | LCR meter |
| Q Factor @ 50 kHz | ≥20 | Minimum | LCR meter |
| DC Resistance | <8 Ω | Maximum | DMM |
| Self-Resonant Freq | >200 kHz | Minimum | VNA sweep |

## Core Material Selection

**Recommended:** NiZn Ferrite (Fair-Rite 61 material or equivalent)
- Permeability: μᵣ = 125-250
- Frequency range: Optimized for 1-100 MHz
- Dimensions: Ø8 mm × 100 mm

**Why NiZn over MnZn?**
- MnZn ferrite (μᵣ = 1000-3000) has higher permeability but becomes lossy above 10-20 kHz
- NiZn maintains low core loss across the full 2-50 kHz operating range
- Requires more turns (~300 vs ~250) but achieves better high-frequency Q

## Materials Required

- Ferrite rod core: NiZn, Ø8 mm × 100 mm (Fair-Rite 61 or equivalent)
- Magnet wire: 34 AWG enameled copper (primary); 36 AWG acceptable
- Coil form/bobbin (optional, or wind directly on core)
- Epoxy or varnish for securing windings
- LCR meter for testing (required for quality control)

## Winding Procedure

### Step 1: Prepare Core

1. Inspect ferrite rod for cracks or chips (reject if damaged)
2. Clean ferrite rod surface with isopropyl alcohol
3. If using bobbin, mount on core with slight friction fit
4. Mark winding start position 25 mm from one end

### Step 2: Wind Coil

**Optimized specification for ~1.5 mH on NiZn Ø8 mm × 100 mm ferrite:**

| Parameter | Specification |
|-----------|---------------|
| Wire | 34 AWG (0.16 mm diameter) enameled copper |
| Turns | 280-320 turns (start with 300) |
| Winding style | Single-layer, close-wound (bank winding) |
| Winding length | ~50 mm of core length |
| Start position | 25 mm from end (center winding on core) |
| Tension | Light, consistent (~50g pull) |

**Why single-layer bank winding?**
- Minimizes inter-turn capacitance → maximizes self-resonant frequency
- Achieves highest Q factor
- More predictable inductance

**Calculation notes:**
- Inductance ≈ (N² × μᵣ × μ₀ × A) / l
- For NiZn (μᵣ ≈ 125): ~300 turns yields 1.5 mH
- Adjust turns ±10% based on measured inductance

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

| Parameter | Specification | Notes |
|-----------|---------------|-------|
| Inductance | 1.5 mH ±10% | Measure at 1 kHz |
| Q Factor @ 10 kHz | ≥30 | Primary performance metric |
| Q Factor @ 2-50 kHz | ≥20 | Across full operating range |
| DC Resistance | <8 Ω | Lower is better |
| Self-Resonant Freq | >200 kHz | Measure with VNA |
| Current Rating | 50-100 mA RMS | Driver-limited |
| Orientation | Orthogonal to RX | 90° ±5° for <-40 dB coupling |

## RX Coil Specifications

| Parameter | Specification | Notes |
|-----------|---------------|-------|
| Inductance | 1.5 mH ±10% | Match TX within 5% if possible |
| Q Factor @ 10 kHz | ≥30 | Higher Q = better sensitivity |
| Q Factor @ 2-50 kHz | ≥20 | Across full operating range |
| DC Resistance | <8 Ω | Lower is better |
| Self-Resonant Freq | >200 kHz | Must exceed operating range |
| Orientation | Orthogonal to TX | Minimizes direct coupling |

## TX-RX Coupling Requirements

| Parameter | Target | Measurement Method |
|-----------|--------|-------------------|
| Coupling coefficient | <-40 dB | VNA S21 at 10 kHz |
| Orthogonality | 90° ±5° | Physical alignment |
| Optimal angle | Determined per probe | Sweep angle, find minimum |

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

### Acceptance Criteria (All Must Pass)

| Test | Specification | Pass/Fail |
|------|---------------|-----------|
| Inductance @ 1 kHz | 1.35-1.65 mH | [ ] |
| Q Factor @ 10 kHz | ≥30 | [ ] |
| Q Factor @ 2 kHz | ≥25 | [ ] |
| Q Factor @ 50 kHz | ≥20 | [ ] |
| DC Resistance | <8 Ω | [ ] |
| Self-Resonant Freq | >200 kHz | [ ] |
| Coil-to-Core Isolation | >10 MΩ | [ ] |
| Wire Continuity | <1 Ω variation | [ ] |
| Visual: Windings Secure | No loose turns | [ ] |
| Visual: Terminations | Clean, tinned | [ ] |

### Multi-Frequency Characterization

Record measurements at each operating frequency:

| Frequency | Inductance (mH) | Q Factor | Impedance (Ω) |
|-----------|-----------------|----------|---------------|
| 2 kHz | ______ | ______ | ______ |
| 5 kHz | ______ | ______ | ______ |
| 10 kHz | ______ | ______ | ______ |
| 20 kHz | ______ | ______ | ______ |
| 50 kHz | ______ | ______ | ______ |

### Testing Procedure
1. Measure inductance at 1 kHz (reference frequency)
2. Sweep Q factor at 2, 5, 10, 20, 50 kHz
3. Check for shorts (coil to core, >10 MΩ)
4. Verify wire continuity (end-to-end resistance)
5. If VNA available: sweep 100 kHz - 1 MHz to find SRF
6. Functional test with known metal target if available

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

| Field | Value |
|-------|-------|
| Probe ID | _______ |
| Coil Type | TX / RX |
| Core Material | MnZn / NiZn |
| Core Dimensions | Ø_____ × _____ mm |
| Wire Gauge | _____ AWG |
| Turns | _______ |
| Winding Length | _____ mm |
| Inductance @ 1 kHz | _______ mH |
| Q Factor @ 10 kHz | _______ |
| DC Resistance | _______ Ω |
| Self-Resonant Freq | _______ kHz |
| Winder Initials | _______ |
| Date | _______ |
| QC Pass/Fail | _______ |

## Notes

- **Frequency dependence:** Inductance typically decreases 5-10% from 1 kHz to 50 kHz due to eddy currents in core
- **Q factor interpretation:** Higher Q = more efficient energy storage, better sensitivity
- **Matching:** TX and RX inductance should match within 5% for optimal performance
- **Temperature effects:** Ferrite permeability decreases ~0.1%/°C; inductance may drift with temperature
- **Core material:** NiZn recommended for broadband 2-50 kHz; MnZn acceptable for low-frequency-only operation

