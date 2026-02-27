# Probe Assembly Drawing

## Overview

This document provides detailed assembly instructions, exploded views, and specifications for the HIRT modular micro-probe system. The design uses a bottom-to-top assembly sequence with M12x1.75 threaded connections throughout.

## Design Philosophy

The probe assembly follows the "archaeologist brain first, engineer brain second" principle - prioritizing minimal ground disturbance while maintaining robust measurement capability. The 16mm OD modular design achieves approximately 10x less soil disturbance compared to traditional 25mm designs.

## Complete Probe Stack Architecture

```
┌─────────────────────────────────┐
│  SURFACE JUNCTION BOX           │ ◄── Terminal block, cable gland
│  (25mm dia. x 35mm height)      │     No active electronics
└───────────┬─────────────────────┘
            │ M12x1.75 Internal Thread
            ▼
┌───────────────────────────────────┐
│  TOP ROD SEGMENT                  │
│  (Fiberglass, 100 cm length)      │ ◄── Hollow center for wiring
│                                   │     16mm OD x 12-13mm ID
│  [TX Coil integrated in segment]  │
└───────────┬───────────────────────┘
            │ Male Insert (epoxied)
            │ M12x1.75 Thread
            ▼
┌───────────────────────────────────┐
│  ERT RING MODULE (Optional)       │ ◄── 3D Printed sensor module
│  Ring B @ 1.5m from tip           │     Stainless steel ring
└───────────┬───────────────────────┘
            │ Thread connection
            ▼
┌───────────────────────────────────┐
│  MIDDLE ROD SEGMENT               │
│  (Fiberglass, 50 cm length)       │ ◄── Spacer segment
└───────────┬───────────────────────┘
            │ Male Insert
            │ M12x1.75 Thread
            ▼
┌───────────────────────────────────┐
│  RX COIL SENSOR MODULE            │ ◄── 3D Printed sensor body
│  (Ferrite core + windings)        │     200-400 turns, 1-2 mH
└───────────┬───────────────────────┘
            │ Thread connection
            ▼
┌───────────────────────────────────┐
│  BOTTOM ROD SEGMENT               │
│  (Fiberglass, 50 cm length)       │
│                                   │
│  [ERT Ring A @ 0.5m from tip]     │ ◄── First electrode ring
└───────────┬───────────────────────┘
            │ Male Insert
            │ M12x1.75 Thread
            ▼
┌───────────────────────────────────┐
│  PROBE TIP                        │ ◄── 3D Printed, tapered end
│  (Tapered nose for insertion)     │     M12x1.75 Female thread
└───────────────────────────────────┘
```

## Exploded View Components

### 1. Probe Tip (Bottom Terminator)
- **Function:** Provides tapered entry for soil insertion
- **Thread:** M12x1.75 female internal thread
- **Length:** 40-60mm including taper
- **Material:** PETG or ASA, 100% infill
- **Notes:** Print with tip pointing up, no supports needed

### 2. Bottom Rod Segment with ERT Ring A
- **Length:** 50 cm (standard spacing segment)
- **Material:** Fiberglass G10 tube, 16mm OD x 12-13mm ID
- **Features:**
  - Male insert epoxied into top end
  - ERT Ring A at 0.5m from probe tip
  - Central wiring channel (6mm clear path)

### 3. RX Coil Sensor Module
- **Function:** Receive coil for MIT sensing
- **Thread:** Female M12x1.75 (top), epoxied to rod (bottom)
- **Length:** 120-150mm body
- **Coil Specs:**
  - Ferrite core: 8mm dia x 100mm, NiZn or MnZn
  - Wire: 34-38 AWG enameled copper
  - Turns: 200-400 (target 1.5 mH)
  - Q Factor: >20 @ 10 kHz
- **Assembly:** Coil wound before epoxy bonding to rod

### 4. Middle Rod Segment (Spacer)
- **Length:** 50 cm or 100 cm (variable spacing)
- **Material:** Fiberglass G10 tube
- **Features:**
  - Male inserts epoxied at both ends
  - Pure spacer (no sensors)
  - Hollow for wiring pass-through

### 5. ERT Ring Module (Ring B - Optional)
- **Position:** 1.5m from tip (standard configuration)
- **Thread:** Female M12x1.75 (top and bottom)
- **Length:** 50mm collar
- **Ring Specs:**
  - Material: Stainless steel 316L or copper
  - Width: 3-5mm
  - Thickness: 0.5-1mm
  - Diameter: 16mm (flush with rod OD)

### 6. Top Rod Segment with TX Coil
- **Length:** 100 cm (adjustable based on total depth)
- **Features:**
  - TX coil integrated (same specs as RX)
  - Coil orthogonal to RX (90° rotation)
  - Male insert epoxied at bottom
  - Female insert epoxied at top (for junction box)

### 7. Surface Junction Box
- **Function:** Surface termination, passive connections only
- **Dimensions:** 25mm dia. x 35mm height
- **Thread:** Internal M12x1.75 (bottom connection)
- **Features:**
  - Terminal block (8-position)
  - Cable gland (PG11 or equivalent)
  - Weatherproof gasket at base thread
  - No active electronics

## Assembly Sequence

### Pre-Assembly: Component Preparation

1. **Thread Cutting (Critical Step)**
   - Print all parts in "Blanks" mode if using tap/die approach
   - Male inserts: Print at 12.2mm OD, cut with M12x1.75 Die
   - Female threads: Print at 10.5mm hole, cut with M12x1.75 Tap
   - Use cutting oil, back out frequently to clear chips
   - Test fit all threads dry before epoxy

2. **Coil Winding**
   - Wind RX and TX coils on ferrite cores
   - Target: 300-400 turns, 1.5 mH inductance
   - Measure L and Q before securing
   - Apply thin CA glue or nail polish to fix windings
   - Strip and tin wire ends (5mm exposed)

3. **Rod Segment Preparation**
   - Cut fiberglass tubes to exact lengths (±2mm)
   - Deburr cut ends with sandpaper
   - Clean inner bore with IPA
   - Mark insertion depth for male/female inserts (20mm)

### Field Assembly Procedure

**Location:** Assemble on-site, bottom-to-top sequence

**Step 1: Probe Tip Installation**
1. Thread probe tip onto bottom rod segment male insert
2. Hand-tighten until shoulder seats (firm finger pressure)
3. **Torque:** 2-3 Nm (hand-tight, no wrench required)
4. Verify alignment (no cross-threading visible)

**Step 2: First Electrode Ring (Ring A)**
1. Slide wiring for Ring A through rod center channel
2. Position ring at 0.5m mark from tip
3. Verify ring sits flush with rod OD
4. Confirm electrical continuity with multimeter

**Step 3: RX Coil Module Connection**
1. Thread RX coil wires through next rod segment
2. Align RX coil module with bottom segment
3. Screw connection hand-tight (2-3 Nm)
4. Verify coil orientation (mark for reference)
5. Test inductance: should read 1.2-1.8 mH

**Step 4: Middle Spacer Segment**
1. Thread all wires through spacer segment center
2. Connect spacer to RX module
3. Hand-tighten M12 thread
4. Keep wires loose (no tension)

**Step 5: Optional Ring B Module**
1. If using second ERT ring, thread wires through
2. Position at 1.5m mark (measured from tip)
3. Connect and hand-tighten
4. Test ring isolation: Ring A to Ring B should read >1 MΩ

**Step 6: Top Rod Segment with TX Coil**
1. Thread all wires through top segment
2. Align TX coil 90° rotated from RX (orthogonal)
3. Connect to upper module/segment
4. Hand-tighten thread connection
5. Verify TX coil inductance (1.2-1.8 mH)

**Step 7: Junction Box Termination**
1. Strip all wire ends (8mm exposed)
2. Connect to terminal block:
   - Terminals 1-2: TX coil (+/-)
   - Terminals 3-4: RX coil (+/-)
   - Terminals 5-6: ERT Ring A (+/-)
   - Terminals 7-8: ERT Ring B (+/-)
3. Thread junction box onto top rod female insert
4. Hand-tighten until gasket compresses
5. Connect cable gland to external harness

**Step 8: Functional Verification**
1. Measure coil inductances (should be unchanged)
2. Test ERT ring isolation (all pairs >1 MΩ)
3. Verify no shorts to ground
4. Perform bench metal detection test if available

## Torque Specifications

| Connection | Torque | Tool | Notes |
|------------|--------|------|-------|
| Probe tip to rod | 2-3 Nm | Hand | Finger-tight, no wrench |
| Rod-to-module | 2-3 Nm | Hand | Do not overtighten |
| Junction box | 3-4 Nm | Hand + gasket compression | Seal engagement |
| ERT ring modules | 2-3 Nm | Hand | Avoid crushing printed parts |

**Critical:** Do NOT use wrenches or pliers on printed threads. The M12x1.75 threads are sized for hand assembly only. Over-torquing will crack the PETG/ASA material.

## Alignment Requirements

### Thread Engagement
- **Minimum engagement:** 10mm (approximately 5-6 turns)
- **Maximum engagement:** 15mm (full thread depth)
- **Alignment tolerance:** <1° (visual inspection)
- **Cross-threading check:** First 2 turns should spin freely

### Coil Orientation
- **RX-TX angle:** 90° ± 10° (orthogonal to minimize direct coupling)
- **Marking method:** Apply colored tape or paint stripe to each coil module
- **Field check:** Use compass or reference marks during assembly

### ERT Ring Position Tolerances
- **Ring A position:** 0.5m ± 2cm from tip
- **Ring B position:** 1.5m ± 2cm from tip
- **Ring spacing:** Minimum 0.3m between rings
- **Ring flush tolerance:** ±0.5mm from rod OD

## Wiring Path Details

### Central Channel Routing
- **Channel diameter:** 6mm clear path through all components
- **Wire bundle:** Maximum 8 conductors (4 twisted pairs)
- **Wire gauge:** 26-28 AWG for coils, 24 AWG for ERT
- **Shielding:** Recommended for coil leads, required for ERT

### Strain Relief Strategy
- **No tension points:** Leave 10-20cm slack in each segment
- **Tie-down locations:** Use cable ties at junction box only
- **Bend radius:** Minimum 25mm for wire bundle
- **Connector transitions:** Heat-shrink tubing at terminal block

## Waterproofing Details

### Thread Sealing
- **O-ring location:** Shoulder of each male insert
- **O-ring size:** 12mm ID x 2mm cord (Buna-N or Viton)
- **Lubrication:** Silicone grease on O-ring before assembly
- **Seal test:** Visual inspection, no epoxy required for field joints

### Junction Box Sealing
- **Base gasket:** Flat rubber washer, 16mm ID x 2mm thick
- **Cable gland:** PG11 with compression nut
- **Sealant:** Optional silicone around cable exit
- **IP Rating:** Target IP54 (splash resistant)

### ERT Ring Isolation
- **Insulating collar:** 3D printed PETG, printed solid
- **Ring groove:** 0.2mm interference fit for friction hold
- **Edge sealing:** Epoxy or silicone at ring edges
- **Waterproofing test:** Submerge 1m depth, 5 minutes, verify >1 MΩ isolation

## Quality Control Checklist

### Pre-Insertion Inspection
- [ ] All threads engage smoothly (no cross-threading)
- [ ] Coil inductances: 1.2-1.8 mH (both TX and RX)
- [ ] Coil Q factors: >15 @ 10 kHz
- [ ] ERT ring isolation: >1 MΩ between all rings
- [ ] ERT ring-to-ground: >1 MΩ
- [ ] Visual alignment: No visible kinks or misalignment
- [ ] Total probe length: Measured and recorded
- [ ] Junction box sealed: Cable gland tightened

### Post-Insertion Verification
- [ ] Probe vertical (plumb check)
- [ ] Junction box accessible at surface
- [ ] Cable routing clear of foot traffic
- [ ] Re-test coil L and Q (should be unchanged)
- [ ] Re-test ERT isolation (may decrease slightly, must be >100 kΩ)
- [ ] Baseline measurement recorded

## Disassembly and Storage

### Field Disassembly
1. Disconnect cable at junction box
2. Unscrew junction box (save O-ring)
3. Extract probe from hole (steady vertical pull)
4. Unscrew segments in reverse order (bottom-to-top)
5. Clean threads with brush (remove soil)
6. Dry all components before storage

### Storage Recommendations
- **Environment:** Dry, room temperature
- **Orientation:** Vertical or horizontal with support
- **Protection:** Wrap coils in anti-static bag
- **Thread preservation:** Light coat of silicone grease on threads
- **Inspection interval:** Every 6 months, check for cracks or corrosion

## Troubleshooting Common Assembly Issues

### Thread Won't Engage
- **Cause:** Cross-threading or debris in threads
- **Fix:** Back out completely, clean threads, align carefully, retry
- **Prevention:** Test fit all threads dry before field assembly

### Coil Reading Changed After Assembly
- **Cause:** Wire pinched or shorted turn
- **Fix:** Disassemble, inspect coil, check for damage
- **Prevention:** Leave wire slack during assembly

### ERT Rings Shorted to Each Other
- **Cause:** Moisture bridging or insulation failure
- **Fix:** Dry probe, re-seal ring edges with epoxy
- **Prevention:** Ensure rings are fully dry before waterproofing

### Junction Box Leaking
- **Cause:** Missing O-ring or under-torqued cable gland
- **Fix:** Replace O-ring, apply silicone grease, re-tighten
- **Prevention:** Visual inspection of seals before insertion

## Component Sourcing Reference

| Component | Specification | Supplier Example |
|-----------|---------------|------------------|
| Fiberglass tube | 16mm OD x 12mm ID, G10 | McMaster-Carr 8659K41 |
| M12x1.75 Tap | HSS, plug style | Irwin Hanson 3018 |
| M12x1.75 Die | HSS, hex die nut | Irwin Hanson 3019 |
| Ferrite rod | 8mm x 100mm, NiZn | Fair-Rite 2661100202 |
| Magnet wire | 34 AWG, enameled | Belden 8051 |
| O-rings | 12mm ID x 2mm cord, Buna-N | Apple Rubber 237 |
| Cable gland | PG11, IP68 | Heyco M3191 |

## Manufacturing Time Estimates

| Task | Time | Notes |
|------|------|-------|
| 3D print all parts | 8-12 hours | Overnight print recommended |
| Thread cutting | 1-2 hours | All male/female threads |
| Coil winding (2x) | 2-3 hours | 300-400 turns each |
| Rod segment prep | 30 min | Cut, deburr, clean |
| Epoxy curing | 4-8 hours | Male/female inserts into rods |
| ERT ring mounting | 1 hour | Per ring |
| Junction box wiring | 30 min | Terminal block connections |
| **Total:** | **18-28 hours** | Plus 4-8 hours curing time |

**Field assembly time:** 15-20 minutes per probe (experienced operator)

## References

1. HIRT Whitepaper Section 5: Mechanical Design
2. HIRT Whitepaper Section 7: Assembly and Wiring
3. Hardware CAD Files: `/hardware/cad/openscad/modular_flush_connector.scad`
4. Manufacturing Guide: `/hardware/cad/docs/manufacturing-guide.md`
5. Rod Specifications: `/hardware/schematics/mechanical/rod-specifications.md`
