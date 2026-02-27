# Probe Head Technical Drawing

## Overview

The HIRT probe head (sensor module) is the active sensing element containing both MIT-3D inductive coils and ERT electrodes. This document provides detailed cross-sectional views, component placement, dimensional specifications, and material callouts for the sensor module assembly.

**Design Philosophy:** The probe head is a passive assembly containing only coils and electrodes. All active electronics remain at the surface in the junction box and central hub, simplifying the downhole design and improving reliability.

## Quick Reference

| Parameter | Value | Notes |
|-----------|-------|-------|
| Outer Diameter | 16 mm | Matches fiberglass rod |
| Body Length | 70 mm | Standard sensor module |
| Wall Thickness | 2.0 mm | At thread regions |
| Thread Standard | M12×1.75 | Both ends (female) |
| Wire Channel | 6 mm | Center pass-through |
| Material | PETG or ASA | 3D printed |

## Component Architecture

The sensor module uses a dual-female threaded design, acting as a coupling between male rod caps epoxied to fiberglass segments. This allows sensors to be positioned at any depth by selecting appropriate rod segment lengths.

### Assembly Stack (Bottom to Top)

```
[Probe Tip - Female Thread Top]
         |
    [Male Rod Cap]
         |
  [50cm Fiberglass Rod]
         |
    [Male Rod Cap]
         |
[SENSOR MODULE - Dual Female]  ← This document
         |
    [Male Rod Cap]
         |
  [100cm Fiberglass Rod]
         |
    [Male Rod Cap]
         |
   [Junction Cap]
```

## Cross-Section View A-A: Axial (Lengthwise)

This cross-section reveals the internal arrangement along the probe axis, showing thread engagement, coil placement, and wiring channels.

```
        ┌──────────────────────────────────────────┐
        │                                          │
        │  SENSOR MODULE (Dual Female)             │
        │  Total Length: 70mm                      │
        │                                          │
    ────┴──────────────────────────────────────────┴────

    ╔══════════════════════════════════════════════════╗
    ║                                                  ║
    ║  ┌──────┐                            ┌──────┐  ║
    ║  │      │  Female Thread M12×1.75    │      │  ║
    ║  │  ┌───┴───┬────────────────────┬───┴───┐  │  ║
    ║  │  │ ▒▒▒▒▒ │                    │ ▒▒▒▒▒ │  │  ║
    ║  │  │ ▒▒▒▒▒ │   Center Cavity    │ ▒▒▒▒▒ │  │  ║ ← Wall
    ║  │  │ ▒▒▒▒▒ │      12mm ID        │ ▒▒▒▒▒ │  │  ║   2mm
    ║  │  │ ▒▒▒▒▒ │                    │ ▒▒▒▒▒ │  │  ║
    ║  │  └───┬───┴────────────────────┴───┬───┘  │  ║
    ║  │      │                            │      │  ║
    ║  └──────┘                            └──────┘  ║
    ║                                                  ║
    ║    ├─15mm─┤──────── 40mm ────────┤─15mm─┤     ║
    ║    Thread    MIT Coil Zone         Thread       ║
    ║                                                  ║
    ╚══════════════════════════════════════════════════╝

    Dimensions:
    - Total Length: 70mm
    - Thread Depth (each end): 15mm
    - Coil Zone Length: 40mm
    - Outer Diameter: 16mm (flush with rod)
    - Inner Cavity Diameter: 12mm
    - Wire Channel: 6mm (center)
```

### Key Features (Axial View)

1. **Female Threads (Both Ends):** M12×1.75 internal threads, depth 15mm, receive male rod caps
2. **Central Cavity:** 12mm ID × 40mm length houses ferrite core and coil assembly
3. **Wire Channel:** 6mm diameter axial hole runs entire length for electrical connections
4. **Wall Thickness:** 2mm minimum at threaded regions for structural integrity

## Cross-Section View B-B: Radial (Perpendicular)

This cross-section cuts perpendicular to the probe axis through the coil zone, showing the arrangement of ferrite core, coil windings, and radial wire entry holes.

```
    Looking down the probe axis (from above)

    ┌────────────────────────────────────┐
    │                                    │
    │    16mm OD (Outer Diameter)        │
    │         ┌───────────┐              │
    │         │           │              │
    │    ┌────┴───┐   ┌───┴────┐        │
    │    │  ┌─────┴───┴─────┐  │        │
    │    │  │ ╔═══════════╗ │  │  ←2mm wall
    │    │  │ ║ ████████  ║ │  │        │
    │    │  │ ║ ████████  ║ │  │        │
    │    │  │ ║ ████████  ║ │  │  ← Ferrite Core
    │    │  │ ║ ████████  ║ │  │    (8mm dia)
    │    │  │ ║ ████████  ║ │  │        │
    │    │  │ ╚═══════════╝ │  │        │
    │    │  │  Coil Wire    │  │        │
    │    │  │  ~0.3mm thick │  │        │
    │    │  └───────────────┘  │        │
    │    │         ●            │  ← Wire Entry
    │    │      6mm Wire        │    Hole (1.5mm)
    │    │      Channel         │        │
    │    └─────────────────────┘        │
    │                                    │
    │    ┌─── ERT Ring Groove ───┐      │
    │    │   (if on this section) │      │
    └────┴────────────────────────┴──────┘

    Component Layers (Outside to Inside):
    1. Outer Shell: 16mm OD PETG/ASA wall
    2. Coil Windings: 300 turns of 34 AWG wire (~0.3mm layer)
    3. Ferrite Core: 8mm diameter NiZn rod
    4. Wire Channel: 6mm center hole for lead wires
```

### Key Features (Radial View)

1. **Ferrite Core:** 8mm diameter × 100mm NiZn ferrite (Fair-Rite 61 material, μᵣ=125)
2. **Coil Winding:** 300 turns of 34 AWG enameled copper wire, single-layer bank wound
3. **Radial Wire Holes:** 1.5mm diameter holes allow coil lead wires to exit to center channel
4. **ERT Ring Groove:** 4mm wide × 1mm deep groove for stainless steel electrode ring (optional, positioned at specific depths)

## Component Details

### 1. Ferrite Core and Coil Assembly

The MIT-3D sensing element consists of a ferrite rod core with precision-wound copper coil.

**Ferrite Core Specifications:**
| Parameter | Value | Source |
|-----------|-------|--------|
| Material | NiZn Ferrite (Fair-Rite 61) | Fair-Rite 5961001801 |
| Permeability | μᵣ = 125-250 | Optimized for 2-50 kHz |
| Diameter | 8 mm | Fits in 12mm cavity |
| Length | 100 mm | Spans coil zone + margin |
| Core Loss | Low at 2-50 kHz | NiZn advantage |

**Coil Winding Specifications:**
| Parameter | Value | Notes |
|-----------|-------|-------|
| Wire Gauge | 34 AWG (0.16mm dia) | Enameled copper |
| Turns | 300 (280-320 range) | Target 1.5 mH |
| Winding Style | Single-layer, close-wound | Bank winding |
| Winding Length | ~50mm of core | Centered on 100mm core |
| Inductance Target | 1.5 mH ±10% | Measured at 1 kHz |
| Q Factor | ≥30 @ 10 kHz | Quality metric |
| DC Resistance | <8 Ω | Lower is better |

**Coil Positioning:**
- TX Coil: Typically in upper sensor module (near surface)
- RX Coil: Typically in lower sensor module (deeper)
- Orthogonal Orientation: TX and RX coils rotated 90° to minimize direct coupling (<-40 dB)

### 2. ERT Ring Electrode Assembly

Electrical resistivity electrodes are implemented as flush-mounted stainless steel rings embedded in grooves on the sensor body exterior.

**ERT Ring Specifications:**
| Parameter | Value | Notes |
|-----------|-------|-------|
| Material | 316L Stainless Steel | Corrosion resistant |
| Alternative | Copper tape | Lower cost, less durable |
| Width | 4 mm | Groove width |
| Thickness | 0.5-1.0 mm | Thin band |
| Diameter | 16 mm OD | Flush with body |
| Depth Below Surface | 1 mm | Groove depth |
| Quantity per Module | 0-1 | Depends on position |

**ERT Ring Positions (Standard Configuration):**
| Position | Distance from Tip | Module Type |
|----------|-------------------|-------------|
| Ring A (Upper) | 0.5 m | Upper sensor module |
| Ring B (Mid) | 1.5 m | Middle sensor module |
| Ring C (Deep) | 2.5-3.0 m | Lower sensor (optional) |

**Wire Connection:**
- Spring-loaded contact pins or soldered connections
- Leads route through radial wire holes to center channel
- Twisted pair recommended for noise immunity

### 3. Threaded Connections

**Thread Specifications:**
| Parameter | Value | Notes |
|-----------|-------|-------|
| Thread Standard | ISO M12×1.75 | Metric coarse |
| Thread Type | Internal (Female) | Both ends |
| Thread Depth | 15 mm | Adequate engagement |
| Thread Class | Tap-finished preferred | Or printed direct |
| Clearance | 0.2mm on diameter | For assembly |

**Manufacturing Methods:**
1. **Tap-Finished (Recommended):**
   - Print female hole at 10.5mm diameter
   - Cut threads with M12×1.75 plug tap
   - Superior strength and fit

2. **Direct Print:**
   - Print threads at 10.4mm minor diameter
   - Functional but lower strength
   - Requires 0.12mm layer height

### 4. Wire Channel and Cable Management

**Center Wire Channel:**
| Parameter | Value | Purpose |
|-----------|-------|---------|
| Diameter | 6 mm | Through-hole |
| Length | Full 70mm | Continuous |
| Contents | 2-6 conductors | MIT + ERT leads |
| Wire Gauge | 26-30 AWG | Thin flexible |

**Radial Wire Entry Holes:**
| Parameter | Value | Purpose |
|-----------|-------|---------|
| Diameter | 1.5 mm | Coil/ring leads |
| Quantity | 2-4 per module | As needed |
| Position | At coil ends | Near ferrite |
| Angle | 90° to axis | Radial |

## Dimensional Drawing

### Overall Dimensions (Sensor Module)

```
                70mm Total Length
    ←──────────────────────────────────────→

    ╔═══════════════════════════════════════╗  ↑
    ║ ┌───────────────────────────────────┐ ║  │
    ║ │   Female Thread M12×1.75          │ ║  │ 16mm
    ║ │   Depth: 15mm                     │ ║  │ OD
    ║ └───────────────────────────────────┘ ║  ↓
    ║                                       ║
    ║     ┌─────────────────────────┐       ║  ↑
    ║     │  MIT Coil Zone          │       ║  │
    ║     │  40mm Length            │       ║  │ 12mm
    ║     │  Contains:              │       ║  │ Cavity
    ║     │  - 8mm Ferrite Core     │       ║  │
    ║     │  - 300T Coil Winding    │       ║  │
    ║     │  - 6mm Wire Channel     │       ║  │
    ║     └─────────────────────────┘       ║  ↓
    ║                                       ║
    ║ ┌───────────────────────────────────┐ ║
    ║ │   Female Thread M12×1.75          │ ║
    ║ │   Depth: 15mm                     │ ║
    ║ └───────────────────────────────────┘ ║
    ╚═══════════════════════════════════════╝

    ├──15mm──┤────── 40mm ──────┤──15mm──┤
    Thread    Sensor Zone         Thread
```

### Critical Dimensions Table

| Dimension | Nominal | Tolerance | Notes |
|-----------|---------|-----------|-------|
| Total Length | 70 mm | ±0.5 mm | Printed dimension |
| Outer Diameter | 16.0 mm | ±0.1 mm | Must match rod |
| Inner Cavity Diameter | 12.0 mm | ±0.2 mm | Clearance for core |
| Thread Depth | 15 mm | ±0.3 mm | Each end |
| Thread Major Dia (Female) | 12.0 mm | ±0.1 mm | ISO standard |
| Wire Channel Dia | 6.0 mm | ±0.2 mm | Wiring clearance |
| Wall Thickness (Min) | 2.0 mm | +0.2/-0.0 mm | Structural |
| Coil Zone Length | 40 mm | ±1.0 mm | Flexible |
| ERT Groove Width | 4.0 mm | ±0.2 mm | If present |
| ERT Groove Depth | 1.0 mm | ±0.1 mm | Flush mount |

## Material Callouts

### Primary Structure

**Body Material: PETG or ASA (3D Printed)**

| Property | PETG | ASA | Selection Criteria |
|----------|------|-----|-------------------|
| Impact Strength | Good | Excellent | Field durability |
| UV Resistance | Poor | Excellent | Outdoor exposure |
| Print Difficulty | Easy | Moderate | Manufacturing |
| Temperature Range | -20 to 70°C | -20 to 90°C | Operating conditions |
| Water Resistance | Excellent | Excellent | Both suitable |
| Cost | Lower | Higher | Budget consideration |

**Recommendation:** ASA for field instruments, PETG for lab prototypes

### Electromagnetic Components

**Ferrite Core:**
- Material: NiZn Ferrite (Fair-Rite 61 or equivalent)
- Part Number: Fair-Rite 5961001801 (example)
- Supplier: Fair-Rite, Mouser Electronics

**Coil Wire:**
- Material: Enameled copper magnet wire
- Gauge: 34 AWG (0.16mm diameter)
- Insulation: Polyurethane or polyester enamel
- Part Number: Remington Industries 8081 (example)

**ERT Electrode:**
- Material: 316L stainless steel or copper
- Form: Thin strip or tape
- Thickness: 0.5-1.0mm
- Supplier: McMaster-Carr, local metal shop

### Sealing and Assembly

**O-Rings (Optional, for weatherproofing threads):**
- Size: AS568-014 (approx 12mm ID)
- Material: Buna-N or Viton
- Quantity: 1-2 per module

**Epoxy/Potting:**
- Type: 2-part epoxy or polyurethane
- Application: Secure coil, seal wire entries
- Examples: Loctite Marine Epoxy, 3M Scotch-Weld

## Manufacturing Notes

### 3D Printing Requirements

**Critical Print Settings:**
| Setting | Value | Importance |
|---------|-------|------------|
| Material | PETG or ASA | Required |
| Layer Height | 0.12 mm | Critical for threads |
| Infill Density | 100% (Solid) | Structural strength |
| Wall Loops | 6 minimum | Thread regions |
| Supports | DISABLED | Use scaffolding |
| Print Speed (Outer) | 50 mm/s | Quality over speed |

**Why 100% Infill?**
- Threads under mechanical stress
- Prevents collapse during tapping
- Improves impact resistance
- Negligible material cost increase

### Thread Post-Processing (Recommended)

**Tap-Finished Method:**
1. Print female hole at 10.5mm diameter (undersized)
2. Clean any print artifacts with drill bit or reamer
3. Apply cutting oil to M12×1.75 plug tap
4. Cut threads slowly, backing out frequently to clear chips
5. Test fit with male rod cap before final assembly

**Why Post-Process Threads?**
- 3x stronger than printed threads
- Better dimensional accuracy
- Smoother assembly/disassembly
- Reduces risk of cracking

### Coil Assembly Procedure

**Step-by-Step:**
1. Wind coil on ferrite core per `/docs/field-guide/coil-winding-recipe.md`
2. Verify inductance (1.35-1.65 mH) before installation
3. Thread coil lead wires through radial holes to center channel
4. Insert ferrite/coil assembly into sensor cavity
5. Secure with epoxy at core ends (do not pot entire coil)
6. Route lead wires to threaded ends for extraction
7. Test continuity and inductance after installation

### ERT Ring Installation

**Step-by-Step:**
1. Cut stainless steel strip to 50mm length × 4mm width
2. Form into ring (16mm OD)
3. Press-fit into exterior groove
4. Solder connection wire to ring (if using copper)
5. Route connection through radial hole to center channel
6. Seal with thin epoxy layer (maintain conductivity at surface)

## Assembly Integration

### Connection to Rod System

The sensor module threads onto male rod caps epoxied into fiberglass tube ends.

**Assembly Sequence:**
1. Prepare fiberglass rod segments (50cm or 100cm lengths)
2. Epoxy male rod caps into tube ends with M12×1.75 external threads
3. Thread sensor module onto bottom male cap (hand-tight)
4. Route electrical leads through center channel
5. Thread next rod segment onto top of sensor module
6. Continue stacking to desired probe depth

### Electrical Connections

**Wiring Through Center Channel:**
- MIT TX Coil: 2 conductors (coil leads)
- MIT RX Coil: 2 conductors (coil leads)
- ERT Rings: 1 conductor each (2-3 rings typical)
- Total: 6-8 conductors typical (26-30 AWG)

**Termination at Junction Box:**
- All leads terminate at surface junction box
- Phoenix terminal blocks recommended
- Color coding: TX (red/black), RX (blue/white), ERT (green series)

## Related Documentation

### CAD Files
- **Source:** `/hardware/cad/openscad/modular_flush_connector.scad`
- **Exported STL (Dual Sensor):** `sensor_body_dual_4x.stl`
- **Exported STL (Test):** `sensor_body_single_test.stl`
- **Tap-Ready Variant:** `sensor_body_dual_4x_tapready.stl`

### Assembly Guides
- **Mechanical Assembly:** `/hardware/schematics/mechanical/probe-assembly.md`
- **Rod Specifications:** `/hardware/schematics/mechanical/rod-specifications.md`
- **ERT Ring Mounting:** `/hardware/schematics/mechanical/er-ring-mounting.md`

### Technical Specifications
- **Coil Winding:** `/docs/field-guide/coil-winding-recipe.md`
- **Probe BOM:** `/hardware/bom/probe-bom.md`
- **Mechanical Design Theory:** `/docs/hirt-whitepaper/sections/05-mechanical-design.qmd`

### Manufacturing
- **Manufacturing Guide:** `/hardware/cad/docs/manufacturing-guide.md`
- **CAD Quickstart:** `/hardware/cad/QUICKSTART.md`

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-28 | HIRT Team | Initial technical drawing documentation, expanded from placeholder |

## Notes

1. **Probe Head vs. Probe Tip:** This document describes the *sensor module* (active sensing element). The *probe tip* is the pointed bottom terminator, documented separately.

2. **Modularity:** Sensor modules can be positioned at any depth by varying rod segment lengths. Standard configurations use 50cm and 100cm segments.

3. **Passive Design:** No active electronics in probe head. All signal conditioning and digitization occurs at surface. This simplifies waterproofing and improves reliability.

4. **Field Serviceability:** Threaded connections allow sensor modules to be replaced in the field without rebuilding entire probe.

5. **Manufacturing Flexibility:** Design supports both direct 3D printing and tap/die post-processing. Tap-finished threads recommended for production use.

6. **Coil Orientation:** TX and RX coils should be oriented orthogonally (90° rotation) to minimize direct coupling. Optimal angle can be determined experimentally by minimizing baseline signal.

7. **Waterproofing:** O-ring seals at threaded joints provide basic weather resistance. For submersion, additional sealing compound recommended at wire entry points.
