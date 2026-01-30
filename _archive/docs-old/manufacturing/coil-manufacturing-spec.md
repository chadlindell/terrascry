# MIT Coil Manufacturing Specification

**Document Number:** MFG-COIL-001
**Revision:** 1.0
**Date:** 2026-01-27
**Status:** Released

---

## 1. Scope

This specification defines the manufacturing requirements for MIT (Magnetic Induction Tomography) TX and RX coils used in the HIRT probe system. These coils are optimized for broadband operation across the 2-50 kHz frequency range.

## 2. Applicable Documents

- `docs/field-guide/coil-winding-recipe.md` - Winding procedure
- `hardware/schematics/electronics/mit-circuit.md` - Circuit integration
- `docs/hirt-whitepaper/sections/09-calibration.qmd` - Calibration procedures

## 3. Materials

### 3.1 Ferrite Core

| Parameter | Specification | Acceptable Alternatives |
|-----------|---------------|------------------------|
| Material | NiZn ferrite | Fair-Rite 61, Ferroxcube 4C65 |
| Permeability (μᵣ) | 125 ±25% | 100-200 acceptable |
| Diameter | 8.0 mm ±0.5 mm | 6-10 mm with adjusted turns |
| Length | 100 mm ±5 mm | 80-120 mm acceptable |
| Surface | Smooth, no cracks | Reject if chipped |

**Preferred Part Numbers:**
- Fair-Rite: 5961001801
- Alternative: Ferrite rod Ø8×100mm NiZn material

**Incoming Inspection:**
- Visual check for cracks, chips, surface defects
- Dimensional verification (caliper measurement)
- Permeability verification not required if from approved supplier

### 3.2 Magnet Wire

| Parameter | Specification | Notes |
|-----------|---------------|-------|
| Gauge | 34 AWG (0.160 mm) | 36 AWG acceptable |
| Insulation | Polyurethane enamel | Class 155 or better |
| Conductor | Copper, annealed | Solid, not stranded |
| Minimum Length | 15 meters per coil | ~300 turns on Ø8mm core |

**Preferred Part Numbers:**
- Remington Industries: 34SNSP.25 (1/4 lb spool)
- Alternative: Any 34 AWG magnet wire, MW-35C or equivalent

**Incoming Inspection:**
- Visual check for kinks, oxidation
- Continuity check (no breaks)
- Insulation integrity (no bare spots)

### 3.3 Consumables

| Material | Purpose | Notes |
|----------|---------|-------|
| Isopropyl alcohol | Core cleaning | 90%+ purity |
| Epoxy | Winding fixation | 2-part, slow cure preferred |
| Kapton tape | Temporary fixation | Optional, for winding aid |
| Heat shrink | Lead protection | 2:1 ratio, 3mm diameter |

## 4. Equipment Required

### 4.1 Winding Equipment

| Equipment | Purpose | Specification |
|-----------|---------|---------------|
| Winding fixture | Core holding | Non-metallic, adjustable |
| Turn counter | Turn counting | Mechanical or electronic |
| Tensioner | Wire tension | ~50g pull, adjustable |

**Note:** Hand winding is acceptable for prototype and low-volume production.

### 4.2 Test Equipment

| Equipment | Purpose | Specification |
|-----------|---------|---------------|
| LCR Meter | L, Q, Z measurement | 1 kHz - 100 kHz range |
| Digital Multimeter | DC resistance | 4-digit minimum |
| VNA (optional) | SRF measurement | 100 kHz - 10 MHz |

**Calibration:** All test equipment must be calibrated within 12 months.

## 5. Manufacturing Procedure

### 5.1 Core Preparation

1. Remove core from packaging, handle with clean gloves
2. Inspect for cracks, chips, or surface defects (reject if damaged)
3. Clean surface with isopropyl alcohol, allow to dry completely
4. Mark winding start position: 25 mm from one end

### 5.2 Winding Setup

1. Mount core in winding fixture, ensure secure grip
2. Thread wire through tensioner, set to ~50g pull
3. Anchor wire at start position with tape or fixture clamp
4. Zero turn counter

### 5.3 Winding Execution

**Target: 300 turns, single-layer, bank wound**

1. Wind first turn, verify tight against core surface
2. Continue winding, maintaining:
   - Consistent tension (50g nominal)
   - Close-wound spacing (turns touching)
   - Single layer only (no overlap)
3. Every 50 turns: verify counter, check tension
4. Target winding length: ~50 mm (centered on core)
5. At 300 turns: anchor wire temporarily

### 5.4 Winding Verification

Before finalizing:

1. Measure inductance at 1 kHz
   - Target: 1.5 mH ±10% (1.35-1.65 mH)
   - If low: add turns in increments of 10
   - If high: remove turns in increments of 10
2. Measure DC resistance
   - Target: <8 Ω
   - If high: check for wire damage
3. Record final turn count

### 5.5 Termination

1. Leave 100 mm wire tails at each end
2. Secure final turn with small drop of epoxy
3. Allow epoxy to cure per manufacturer instructions
4. Strip wire ends: 5 mm of insulation removed
5. Tin stripped ends with solder
6. Apply heat shrink over solder joints (optional)

### 5.6 Fixation

1. Apply thin coat of epoxy over entire winding
2. Rotate coil during cure to prevent drips
3. Allow full cure time (typically 24 hours)
4. Inspect for coverage and uniformity

## 6. Quality Control

### 6.1 In-Process Inspection

| Checkpoint | Criteria | Action if Fail |
|------------|----------|----------------|
| After core prep | No visible defects | Reject core |
| Every 50 turns | Counter matches, tension OK | Correct and continue |
| After winding | L = 1.5 mH ±10% | Add/remove turns |
| After termination | Continuity OK | Repair or reject |

### 6.2 Final Acceptance Testing

All coils must pass the following tests:

| Test | Method | Pass Criteria |
|------|--------|---------------|
| Inductance @ 1 kHz | LCR meter | 1.35-1.65 mH |
| Q Factor @ 2 kHz | LCR meter | ≥25 |
| Q Factor @ 10 kHz | LCR meter | ≥30 |
| Q Factor @ 50 kHz | LCR meter | ≥20 |
| DC Resistance | DMM | <8 Ω |
| Coil-Core Isolation | DMM (megohm) | >10 MΩ |
| Visual Inspection | Manual | No defects |

### 6.3 Optional Advanced Testing

If VNA available:

| Test | Method | Pass Criteria |
|------|--------|---------------|
| Self-Resonant Freq | VNA sweep | >200 kHz |
| Impedance @ 10 kHz | VNA | 94 Ω ±20% |

### 6.4 Rejection Criteria

Reject coil if any of the following:
- Inductance outside 1.35-1.65 mH range
- Any Q factor below minimum threshold
- DC resistance exceeds 8 Ω
- Visible damage to winding or core
- Coil-core short detected (<10 MΩ)

## 7. Labeling and Traceability

### 7.1 Coil Identification

Each coil shall be labeled with:
- Coil type: TX or RX
- Serial number: YYMMDD-NNN (date + sequence)
- Inductance value (measured)
- Q factor @ 10 kHz (measured)

### 7.2 Documentation

Maintain records for each coil:
- Manufacturing date
- Operator initials
- Core lot number (if available)
- Wire lot number (if available)
- All test measurements
- Pass/Fail status

### 7.3 Traceability Matrix

| Coil S/N | Core Lot | Wire Lot | Build Date | Operator | L (mH) | Q@10k | Status |
|----------|----------|----------|------------|----------|--------|-------|--------|
| ________ | ________ | ________ | __________ | ________ | ______ | _____ | ______ |

## 8. Storage and Handling

### 8.1 Storage Requirements

- Temperature: 15-30°C
- Humidity: <60% RH
- Keep away from strong magnetic fields
- Store in ESD-safe packaging if integrated with electronics

### 8.2 Handling Precautions

- Handle by core ends, not windings
- Avoid dropping (ferrite is brittle)
- Do not flex wire leads excessively
- Keep clean and dry

## 9. Revision History

| Rev | Date | Author | Changes |
|-----|------|--------|---------|
| 1.0 | 2026-01-27 | HIRT Team | Initial release |

---

**END OF SPECIFICATION**
