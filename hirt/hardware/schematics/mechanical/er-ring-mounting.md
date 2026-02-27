# ERT Ring Mounting Specification

## Overview

This document provides detailed specifications for mounting Electrical Resistivity Tomography (ERT) ring electrodes on the HIRT micro-probe system. The design uses narrow stainless steel ring bands flush-mounted on 3D-printed insulating collars to maintain the 16mm OD profile while ensuring reliable soil contact.

## Design Philosophy

ERT rings must achieve two competing goals:
1. **Excellent soil contact** - Maximizing contact area for low-resistance current injection
2. **Complete electrical isolation** - Preventing cross-talk between rings and maintaining >1 MΩ inter-ring isolation

The flush-mount collar design solves this by embedding conductive rings in precision grooves on printed insulation, sealed with epoxy for waterproofing.

## Ring Specifications

### Material Selection

| Material | Resistivity (µΩ·cm) | Corrosion Resistance | Cost | Recommended Use |
|----------|---------------------|----------------------|------|-----------------|
| **Stainless steel 316L** | 74 | Excellent (marine grade) | Moderate | Primary choice |
| Copper (bare) | 1.7 | Poor (oxidizes rapidly) | Low | Not recommended |
| Copper (tin-plated) | ~2 | Good | Low | Budget alternative |
| Brass | 7 | Moderate | Low | Acceptable |

**Primary recommendation:** Stainless steel 316L offers the best balance of corrosion resistance, mechanical strength, and electrical conductivity for long-term field use.

### Dimensional Specifications

| Parameter | Specification | Tolerance | Notes |
|-----------|---------------|-----------|-------|
| **Outer diameter** | 16.0 mm | ±0.2 mm | Must match rod OD |
| **Width (axial)** | 3-5 mm | ±0.3 mm | Wider = better contact |
| **Thickness (radial)** | 0.5-1.0 mm | ±0.1 mm | Thin for flexibility |
| **Surface finish** | Brushed or bead-blasted | - | Increases contact area |

### Standard Ring Quantities

| Probe Configuration | Ring Count | Positions from Tip |
|---------------------|------------|--------------------|
| Minimal (2-point) | 2 rings | 0.5m, 1.5m |
| Standard (3-point) | 3 rings | 0.5m, 1.5m, 2.5m |
| Extended (4-point) | 4 rings | 0.5m, 1.0m, 2.0m, 3.0m |

**Note:** Minimum 2 rings required per probe for crosshole ERT measurements (current source + voltage sense).

## Mounting Position Standards

### Standard Configuration (Most Common)

```
     SURFACE (0.0 m)
         ▲
         │
         │
         │ 1.5 m spacing
         │
    [RING B @ 1.5m]  ◄── Upper electrode
         │
         │ 1.0 m spacing
         │
    [RING A @ 0.5m]  ◄── Lower electrode (primary)
         │
         │ 0.5 m depth
         │
    [PROBE TIP]
```

### Extended Configuration (Deep Targets)

```
     SURFACE (0.0 m)
         ▲
         │ 0.5 m
    [RING C @ 2.5m]  ◄── Deep electrode
         │ 1.0 m
    [RING B @ 1.5m]  ◄── Mid electrode
         │ 1.0 m
    [RING A @ 0.5m]  ◄── Shallow electrode
         │ 0.5 m
    [PROBE TIP]
```

### Spacing Requirements

| Parameter | Minimum | Recommended | Maximum |
|-----------|---------|-------------|---------|
| Ring-to-tip distance | 0.3 m | 0.5 m | 1.0 m |
| Inter-ring spacing | 0.3 m | 1.0 m | 2.0 m |
| Top ring to surface | 0.3 m | 0.5 m | 1.0 m |

**Critical:** Minimum 0.3m spacing required to maintain >100 kΩ isolation in moist soil.

## Insulating Collar Design

### 3D Printed Collar Specifications

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Material** | PETG or ASA | Required: high insulation resistance |
| **Length** | 50 mm | Sufficient for wiring routing |
| **OD** | 16.0 mm | Matches rod OD (flush profile) |
| **ID (center bore)** | 6.0 mm | For wiring pass-through |
| **Ring groove depth** | 0.6-0.8 mm | For 0.5-1mm thick ring |
| **Ring groove width** | 5.0 mm | Matches ring axial width |
| **Print infill** | 100% | Critical for insulation |

### Groove Geometry (Cross-Section)

```
    ←───── 16 mm OD ─────→
    ┌─────────────────────┐
    │                     │ ◄── Collar body (PETG)
    │  ╔═════════════╗    │
    │  ║   RING      ║    │ ◄── Stainless steel ring (0.6mm proud)
    │  ╚═════════════╝    │     (ensures soil contact)
    │                     │
    │      [6mm bore]     │ ◄── Central wiring channel
    └─────────────────────┘

    Ring groove: 0.6-0.8 mm deep
    Ring proud of surface: 0.2-0.4 mm (interference fit)
```

**Design intent:** Ring sits slightly proud (0.2-0.4mm) of the collar surface to ensure direct soil contact, even if collar has surface irregularities from printing.

## Mounting Procedure

### Pre-Assembly: Ring Preparation

**Step 1: Source Ring Material**
- Option A: Purchase pre-formed rings (316L stainless, 16mm ID)
- Option B: Cut from stainless tube stock (16mm OD tube, 3-5mm width)
- Option C: Wrap stainless steel shim stock (0.5mm thick) around collar

**Step 2: Surface Preparation**
1. Deburr ring edges with fine file (remove sharp edges)
2. Clean with IPA (remove oils, contaminants)
3. Optional: Bead-blast or sand surface (increases contact area)
4. Final clean with IPA on lint-free cloth

**Step 3: Collar Printing**
1. Print collar in PETG or ASA, 100% infill
2. Verify groove dimensions with calipers
3. Clean groove with compressed air (remove support material)
4. Test-fit ring (should press in with light force)

### Assembly: Ring Installation

**Step 1: Dry-Fit Test**
1. Slide ring onto collar groove (should be snug)
2. Verify ring sits proud of collar surface (0.2-0.4mm)
3. Check that ring is perpendicular to collar axis (not tilted)
4. Measure ring-to-end distance for position verification

**Step 2: Electrical Connection**
1. Solder 26 AWG stranded wire to ring inner surface
2. Solder joint location: 180° opposite from wiring exit point
3. Solder joint length: 3-5mm (minimize profile)
4. Use silver-bearing solder (higher corrosion resistance)
5. Clean flux residue with IPA

**Step 3: Wire Routing**
1. Route wire through collar center bore
2. Leave 5-10cm slack inside collar (strain relief)
3. Secure wire to collar wall with small dot of CA glue (single point)
4. **Do not** create tension on solder joint

**Step 4: Epoxy Sealing (Critical)**
1. Mix two-part epoxy (5-minute or 30-minute cure)
2. Apply thin bead around ring edges (both top and bottom)
3. Work epoxy into ring-collar interface with toothpick
4. Wipe excess flush with collar surface
5. Cure per manufacturer instructions (room temperature)

**Step 5: Waterproofing Enhancement**
1. After epoxy cure, apply silicone conformal coating over ring edges
2. Optional: Heat-shrink tubing over entire collar (clear, 3:1 ratio)
3. Avoid covering the ring contact surface (must touch soil)

### Assembly: Collar Integration into Probe

**Step 6: Thread Collar into Probe Stack**
1. Verify female M12x1.75 threads at collar top (if printed)
2. Thread collar onto rod segment male insert
3. Hand-tighten (2-3 Nm torque)
4. Verify ring position with tape measure (from probe tip)

**Step 7: Wiring Continuity to Junction Box**
1. Thread ring wire through all upper rod segments
2. Label wire at junction box (Ring A, Ring B, etc.)
3. Connect to designated terminal block positions
4. Record wire color codes in probe documentation

## Electrical Connection Details

### Internal Wiring Specifications

| Parameter | Specification | Notes |
|-----------|---------------|-------|
| Wire gauge | 24-26 AWG stranded | Flexible for routing |
| Wire type | PTFE or silicone insulation | High voltage isolation |
| Shielding | Optional (grounded at hub only) | Reduces noise pickup |
| Color coding | Unique per ring | Simplifies troubleshooting |

### Solder Joint Requirements

| Property | Requirement | Test Method |
|----------|-------------|-------------|
| Pull strength | >2 N (minimum) | Gentle tug test |
| Contact resistance | <0.5 Ω | 4-wire ohmmeter |
| Insulation breakdown | >500 V | Megger test |
| Corrosion resistance | No oxidation after 24h salt spray | Visual inspection |

**Best practice:** Use silver-bearing solder (Sn96.5/Ag3.5) for marine-grade corrosion resistance.

### Terminal Block Connection at Junction Box

```
Junction Box Terminal Block (8-position):

1  [TX+]       5  [Ring A+]
2  [TX-]       6  [Ring A-]
3  [RX+]       7  [Ring B+]
4  [RX-]       8  [Ring B-]

Note: Ring terminals may be swapped (polarity
is arbitrary for bipolar AC excitation).
```

**Wiring convention:** Each ring connects to a dedicated terminal pair. Multiple rings never share a common return (full isolation required).

## Waterproofing and Sealing

### Sealing Strategy Overview

The waterproofing approach uses a **defense-in-depth** strategy:

1. **Primary seal:** Epoxy at ring-collar interface (permanent)
2. **Secondary seal:** Silicone conformal coating over edges
3. **Tertiary seal:** Optional heat-shrink over entire collar (except ring surface)
4. **Thread seal:** O-ring at collar M12 connection

### Epoxy Selection

| Epoxy Type | Cure Time | Max Temp | Water Resistance | Cost | Recommended |
|------------|-----------|----------|------------------|------|-------------|
| **J-B Weld Marine** | 24 hours | 150°C | Excellent | Low | ✓ Primary choice |
| Loctite E-30CL | 30 minutes | 120°C | Good | Moderate | Fast prototyping |
| West System 105/205 | 9 hours | 80°C | Excellent | High | Production builds |

**Application note:** Apply epoxy in thin layer (0.3-0.5mm). Thick layers create stress points and may crack.

### Silicone Coating Application

1. After epoxy fully cures, apply silicone conformal coating
2. Use syringe or fine brush for precision
3. Cover epoxy edges and solder joint only
4. **Do not** coat the ring contact surface (must remain conductive)
5. Allow 12-24 hours cure before field deployment

### Heat-Shrink Protection (Optional)

For added protection, especially in abrasive soils:

1. Select 3:1 ratio heat-shrink tubing, clear, 18-20mm diameter
2. Slide over collar before ring installation
3. After epoxy cure, shrink tubing over collar
4. Cut window exposing ring contact surface (3-5mm width)
5. Seal cut edges with silicone

## Isolation Requirements and Testing

### Target Isolation Specifications

| Test Configuration | Minimum | Target | Maximum Current Leakage |
|--------------------|---------|--------|-------------------------|
| Ring A to Ring B (air) | 10 MΩ | 100 MΩ | <50 nA @ 5V |
| Ring A to Ring B (moist soil) | 100 kΩ | 1 MΩ | <5 µA @ 5V |
| Ring to ground (air) | 10 MΩ | 100 MΩ | <50 nA @ 5V |
| Ring to probe rod (any) | 10 MΩ | 100 MΩ | <50 nA @ 5V |

### Isolation Test Procedures

**Test 1: Pre-Deployment Dry Test**
1. Measure ring-to-ring resistance with megohmmeter
2. Apply 50V test voltage (if available)
3. All ring pairs must read >10 MΩ
4. Record baseline values

**Test 2: Moisture Simulation Test**
1. Spray light mist of water on probe surface
2. Wait 5 minutes for surface wetting
3. Re-measure ring isolation (should drop to 1-5 MΩ)
4. Dry probe, verify return to >10 MΩ

**Test 3: Post-Deployment Field Test**
1. After probe insertion, measure ring isolation in-situ
2. Typical moist soil: 100 kΩ - 1 MΩ
3. Typical saturated soil: 10 kΩ - 100 kΩ
4. If <10 kΩ, suspect waterproofing failure

### Failure Modes and Diagnostics

| Symptom | Likely Cause | Remediation |
|---------|--------------|-------------|
| Isolation <1 MΩ (air) | Epoxy void or crack | Re-seal with additional epoxy |
| Isolation <10 kΩ (soil) | Moisture bridging rings | Improve sealing, extend collar length |
| Fluctuating readings | Loose solder joint | Re-solder connection |
| High contact resistance | Oxidized ring surface | Clean ring, re-polish surface |

## Ring Spacing and Positioning

### Theoretical Basis for Spacing

The inter-ring spacing affects the **sensitivity pattern** and **depth of investigation** for ERT measurements. Closer rings provide higher resolution but shallower penetration.

**Rule of thumb:** Depth of investigation ≈ 0.5 × inter-ring spacing

| Ring Spacing | Depth of Investigation | Resolution | Application |
|--------------|------------------------|------------|-------------|
| 0.3-0.5 m | 0.15-0.25 m | High | Shallow targets (<1m) |
| 1.0 m | 0.5 m | Medium | Standard archaeology (1-2m) |
| 2.0 m | 1.0 m | Low | Deep targets (>2m) |

### Position Measurement and Verification

**During Assembly:**
1. Measure ring position from probe tip with tape measure
2. Mark desired position on rod segment before epoxy
3. Install collar at marked position
4. Verify with calipers: ±2 cm tolerance acceptable

**Field Verification:**
1. Before insertion, measure visible ring spacing
2. After insertion, verify top ring depth from surface
3. Record actual positions in field notes (may differ from design)

## Quality Control and Inspection

### Pre-Deployment Inspection Checklist

- [ ] Ring flush with collar OD (±0.5mm)
- [ ] Ring proud of collar surface (0.2-0.4mm)
- [ ] Solder joint secure (pull test >2N)
- [ ] Contact resistance <0.5 Ω (4-wire measurement)
- [ ] Ring-to-ring isolation >10 MΩ (air, dry)
- [ ] Ring-to-ground isolation >10 MΩ
- [ ] Epoxy seal continuous (no voids visible)
- [ ] Silicone coating covers edges
- [ ] Wire routing secure (no tension on joint)
- [ ] Ring position verified (tape measure)

### Post-Deployment Inspection Checklist

- [ ] Ring-to-ring isolation >100 kΩ (in soil)
- [ ] Contact resistance <5 Ω (check for soil contact)
- [ ] No visible cracks in collar or epoxy
- [ ] Wire connections intact at junction box
- [ ] Baseline ERT measurement recorded

### Long-Term Maintenance

**After each field use:**
1. Clean rings with soft brush and water (remove soil)
2. Inspect epoxy seals for cracks or delamination
3. Re-test isolation (should return to >10 MΩ after drying)
4. Re-coat with silicone if seals show wear

**Storage:**
1. Store probes vertically or with support
2. Protect rings from impact (bubble wrap)
3. Inspect every 6 months for corrosion
4. Re-polish ring surfaces if oxidation visible

## Troubleshooting Common Issues

### Issue: Low Isolation Between Rings (<1 MΩ in air)

**Diagnosis:**
1. Visually inspect epoxy seals for cracks or voids
2. Check for moisture or condensation inside collar
3. Measure insulation resistance of collar material alone

**Fix:**
1. Dry probe completely (oven at 50°C, 2 hours)
2. Re-seal ring edges with fresh epoxy (thin layer)
3. Allow full cure before re-testing

### Issue: High Contact Resistance (>10 Ω in soil)

**Diagnosis:**
1. Check ring surface for oxidation or contamination
2. Verify ring is proud of collar (not recessed)
3. Test in known conductive medium (salt water)

**Fix:**
1. Remove collar from probe
2. Polish ring surface with fine sandpaper (600 grit)
3. Clean with IPA, re-install

### Issue: Wire Pulled Loose from Ring

**Diagnosis:**
1. Check solder joint (may be cold joint or contaminated)
2. Look for strain or tension on wire

**Fix:**
1. Remove collar from probe
2. Re-solder wire with proper flux and temperature
3. Add strain relief (glue wire to collar wall)
4. Re-seal with epoxy

### Issue: Ring Fell Out of Groove

**Diagnosis:**
1. Check groove dimensions (should be 0.2mm interference fit)
2. Verify epoxy was applied (may have been missed)

**Fix:**
1. Reprint collar with tighter groove (reduce by 0.1mm)
2. Re-install ring with proper epoxy sealing
3. Consider adding mechanical retention (split-ring groove)

## Advanced Options

### Option 1: Flush-Mount Spring-Loaded Contacts

For maximum reliability in poor-contact soils, use spring-loaded pogo pins to maintain pressure on the ring:

- **Components:** 2-4 pogo pins per ring (equally spaced)
- **Installation:** Drill radial holes in collar, press-fit pogo pins
- **Advantage:** Maintains contact even if ring-soil gap develops
- **Disadvantage:** Adds complexity, more failure modes

### Option 2: Conductive Fabric Wrap

For very soft or muddy soils, replace rigid ring with conductive fabric:

- **Material:** Silver-plated nylon fabric (Shieldex)
- **Installation:** Wrap around collar, seal edges with conductive epoxy
- **Advantage:** Conformal contact, large surface area
- **Disadvantage:** Lower durability, harder to waterproof

### Option 3: Multi-Ring Arrays (4+ rings per probe)

For high-resolution ERT imaging, use 4 or more rings per probe:

- **Spacing:** 0.5m intervals (4 rings = 2m probe)
- **Wiring:** Requires 8-16 terminal positions at junction box
- **Advantage:** Much higher vertical resolution
- **Disadvantage:** Complex wiring, higher cost

## References

1. HIRT Whitepaper Section 5: Mechanical Design
2. HIRT Whitepaper Section 6: Electronics and Circuits (ERT frontend)
3. HIRT Whitepaper Section 10: Field Operations (ERT measurement protocols)
4. Hardware CAD Files: `/hardware/cad/openscad/modular_flush_connector.scad`
5. Loke, M.H. (2004). Tutorial: 2-D and 3-D electrical imaging surveys. Geotomo Software.
6. ASTM G48. Standard Test Methods for Pitting and Crevice Corrosion Resistance of Stainless Steels.
