# Assembly Drawings

## Overview

This document provides visual assembly guidance for the HIRT modular probe system using 3D CAD references and exploded view descriptions. The drawings complement the detailed manufacturing procedures found in the [Manufacturing Guide](/development/projects/active/HIRT/hardware/cad/docs/manufacturing-guide.md) and [Manufacturing Notes](/development/projects/active/HIRT/hardware/cad/docs/manufacturing-notes.md).

## Design Architecture

The HIRT probe system uses a **modular flush connector architecture** with independent sensor bodies:

- **Fiberglass Segments**: 16mm OD × 12mm ID tubes (50cm or 100cm lengths) with epoxied male caps
- **Sensor Bodies**: Independent 70mm units with dual female sockets containing MIT coils and ERT electrodes
- **Thread Standard**: M12×1.75 for all connections
- **Center Channel**: 6mm diameter for wiring throughout

## CAD File Reference

All components are parametrically defined in:
- **Source**: `/development/projects/active/HIRT/hardware/cad/openscad/modular_flush_connector.scad`
- **Exports**: `/development/projects/active/HIRT/hardware/cad/stl/`

### Available Export Formats

**Production STL Files** (print-ready with threads):
- `male_rod_cap_4x.stl` - 4× male rod caps for fiberglass ends
- `sensor_body_dual_4x.stl` - 4× dual-socket sensor bodies
- `probe_tip_4x.stl` - 4× probe tips (bottom terminators)
- `junction_cap_4x.stl` - 4× junction caps (top terminators)
- `probe_system_complete.stl` - Full assembled visualization

**Tap-Ready Blanks** (for die/tap threading):
- `sensor_body_dual_4x_tapready.stl` - Smooth bores for tapping M12 threads
- `probe_tip_4x_tapready.stl` - Smooth bores for tapping
- `junction_cap_4x_tapready.stl` - Smooth bores for tapping

## Component Identification

### 1. Male Rod Cap
**Function**: Epoxies into fiberglass tube ends, provides external M12 male thread
**Location**: Both ends of each fiberglass segment
**Part Number**: Per BOM - Custom 3D printed
**Key Features**:
- 20mm insertion depth into fiberglass ID
- Stop flange at rod end (2mm thick)
- 15mm external M12×1.75 thread
- 6mm center wire channel
- O-ring groove at thread base

**Quantity per 3m probe**: 4-6 pieces (depends on segment count)

### 2. Sensor Body Dual (Independent Unit)
**Function**: Houses MIT coil and ERT electrode, connects rod segments
**Location**: Between fiberglass sections
**Part Number**: Per BOM - Custom 3D printed
**Key Features**:
- 70mm total length, 16mm OD (flush with rod)
- Female M12×1.75 sockets on BOTH ends (25mm depth each)
- ERT ring groove (4mm wide × 1mm deep) at 1/4 position
- MIT coil winding zone (15mm long) at 3/4 position
- Three radial wire entry holes (1.5mm) for sensor connections
- Orientation flat indicator for alignment
- O-ring grooves at both thread interfaces

**Quantity per 3m probe**: 2-3 bodies (typical: 0.5m and 1.5m depths)

### 3. Probe Tip
**Function**: Bottom terminator with tapered insertion point
**Location**: Bottom of probe stack
**Part Number**: Per BOM - Custom 3D printed
**Key Features**:
- 30mm total length
- Tapered cone (15mm, from 8mm to 16mm diameter)
- Single female M12×1.75 socket at top (13mm depth)
- Blind center channel (stops before tip)
- Wire termination cavity for cable ends

**Quantity per probe**: 1 piece

### 4. Junction Cap
**Function**: Top terminator with cable exit and weather sealing
**Location**: Top of probe stack (at surface)
**Part Number**: Per BOM - Custom 3D printed
**Key Features**:
- 35mm total length
- 16mm OD body section
- 20mm OD cable gland boss
- Female M12×1.75 socket at bottom (25mm depth)
- 12.5mm cable gland thread hole (M12 or PG7 compatible)
- Wire routing chamber for splices
- Weather seal grooves

**Quantity per probe**: 1 piece

## Assembly Sequence Overview

### Step 1: Prepare Fiberglass Segments
1. Cut fiberglass tubes to length (50cm or 100cm)
2. Sand ends flat and clean with IPA
3. Apply epoxy to male rod cap insertion sections
4. Insert caps into tube ends, ensuring center wire channel alignment
5. Cure overnight, then thread O-rings onto male threads

**Reference**: Manufacturing Notes, "Rod Threading" section

### Step 2: Wind MIT Coils
1. Wrap 300 turns of 34 AWG magnet wire in coil winding zone
2. Bank-wound single layer around 15mm zone on sensor body
3. Route wire ends through radial entry holes to center channel
4. Target inductance: 1.5 mH ±10%, Q ≥30 @ 10 kHz

**Reference**: `/development/projects/active/HIRT/docs/field-guide/coil-winding-recipe.md`

### Step 3: Install ERT Electrodes
1. Wrap stainless steel or copper tape in ERT ring groove
2. Overlap ends 3mm and secure with rivet or conductive epoxy
3. Route connection wire through radial hole to center channel
4. Cover joint with heat-shrink for strain relief

**Reference**: Manufacturing Notes, "ERT Ring Collar" section

### Step 4: Stack Probe Components
**Bottom to Top Configuration (typical 3m probe)**:

```
[Junction Cap] ──────────── Surface level
      |
   (thread)
      |
[Male Rod Cap] ═════════════ 50cm fiberglass segment
      |
   (thread)
      |
[Sensor Body] ─────────────── 0.5m depth (ERT + MIT)
      |
   (thread)
      |
[Male Rod Cap] ═════════════ 100cm fiberglass segment
      |
   (thread)
      |
[Sensor Body] ─────────────── 1.5m depth (ERT + MIT)
      |
   (thread)
      |
[Male Rod Cap] ═════════════ 100cm fiberglass segment
      |
   (thread)
      |
[Probe Tip] ─────────────── 3.0m depth
```

### Step 5: Route Internal Wiring
1. Feed sensor wires through center 6mm channel as each section connects
2. Leave service loops at each sensor body
3. Terminate all wires in junction cap routing chamber
4. Connect to cable gland and external cable

### Step 6: Final Assembly
1. Thread all connections hand-tight
2. Apply thread sealant to weatherproof joints
3. Install cable gland with weather seal
4. Test continuity of all sensor connections
5. Label probe with depth markers

**Reference**: Manufacturing Guide, "Assembly Sequence" section (lines 176-185)

## Critical Interfaces

### Thread Engagement
- **Hand-tighten only** - no tools required
- Minimum engagement: 10mm (≥6 full threads)
- Apply Loctite 567 or equivalent thread sealant
- O-ring must compress but not extrude

### Wire Routing
- **Never pull wires taut** - leave 50mm service loop per section
- Use tape flags at junctions for identification
- Secure with wire ties at entry/exit points only
- Test continuity before final assembly

### Waterproofing
- O-rings required at all threaded connections
- Thread sealant on every male thread
- Cable gland must be IP65 rated minimum
- Junction cap weather seal verified by dunk test

## Exploded View Descriptions

### Complete Probe Stack (Top to Bottom)
**View in CAD**: Open `/development/projects/active/HIRT/hardware/cad/stl/probe_system_complete.stl` in FreeCAD or Fusion 360

**Layer Spacing** (for visualization):
- Junction cap: Z=0
- Gap: 5mm
- Male cap (top segment): Z=-40mm
- Fiberglass segment (50cm): implicit connection
- Sensor body 1: Z=-600mm
- Male cap (mid segment): Z=-680mm
- Fiberglass segment (100cm): implicit connection
- Sensor body 2: Z=-1780mm
- Male cap (bottom segment): Z=-1860mm
- Fiberglass segment (100cm): implicit connection
- Probe tip: Z=-3060mm

**To generate exploded view in OpenSCAD**:
Edit `modular_flush_connector.scad`, set `part = "complete_system"`, and adjust `translate()` Z-offsets in lines 996-1005.

### Sensor Body Detail View
**Features to highlight**:
- Top female socket with O-ring groove (0-25mm)
- ERT ring groove at Z=32mm (1/4 position in 20mm sensor zone)
- MIT coil winding zone at Z=48mm (3/4 position in 20mm sensor zone)
- Three radial wire entry holes (spaced 60° apart)
- Orientation flat indicator on cylinder surface
- Bottom female socket with O-ring groove (45-70mm)
- 6mm center channel throughout

**View in CAD**: Extract single sensor body from `sensor_body_dual_4x.stl` using section plane

### Thread Engagement Cross-Section
**To visualize in FreeCAD**:
1. Import `male_rod_cap_4x.stl` and `sensor_body_dual_4x.stl`
2. Position male thread engaged in female socket (10mm overlap)
3. Create vertical section plane through center axis
4. Dimensions to verify:
   - Thread major diameter: 12.0mm
   - Thread pitch: 1.75mm
   - Engagement depth: 10-12mm
   - O-ring groove clearance: 1.5mm depth
   - Center wire channel: 6mm continuous

## Drawing Standards

### Dimensioning Conventions
- All dimensions in millimeters (mm)
- Thread callouts: M12×1.75-6H (internal) / 6g (external)
- Tolerances: ±0.2mm unless noted (FDM typical)
- O-ring grooves: AS568-014 standard (or metric equivalent)

### Assembly Notes and Callouts

**CRITICAL**:
- Print with **100% solid infill** (see Manufacturing Notes lines 12-13)
- Threads may crush if printed with low infill
- **Recommended**: Use tap-ready blanks and cut threads with M12×1.75 tap/die for superior strength

**WATERPROOFING**:
- All threaded joints require O-rings AND thread sealant
- Junction cap must pass 5-minute submersion test
- Cable gland rated IP65 minimum

**FIELD ASSEMBLY**:
- Hand-tighten only (finger-tight + ¼ turn)
- Over-torquing will strip printed threads
- Verify wire continuity before final tightening

## Related Documentation

### Manufacturing Procedures
- [Manufacturing Guide](/development/projects/active/HIRT/hardware/cad/docs/manufacturing-guide.md) - Complete 3D printing and CNC instructions
- [Manufacturing Notes](/development/projects/active/HIRT/hardware/cad/docs/manufacturing-notes.md) - Component-specific recommendations and at-home methods

### Field Operations
- [Quick Reference](/development/projects/active/HIRT/docs/field-guide/quick-reference.md) - Grid spacing and deployment layouts
- [Coil Winding Recipe](/development/projects/active/HIRT/docs/field-guide/coil-winding-recipe.md) - MIT coil fabrication procedure

### Bill of Materials
- [Probe BOM](/development/projects/active/HIRT/hardware/bom/probe-bom.md) - Component procurement details, part numbers, costs

### CAD Source
- [OpenSCAD Source](/development/projects/active/HIRT/hardware/cad/openscad/modular_flush_connector.scad) - Parametric model (1019 lines, fully commented)

## Visualization Tools

### Opening STL Files
**FreeCAD** (Free, Linux/Windows/Mac):
```bash
freecad /development/projects/active/HIRT/hardware/cad/stl/probe_system_complete.stl
```

**OpenSCAD Preview** (Parametric editing):
```bash
openscad /development/projects/active/HIRT/hardware/cad/openscad/modular_flush_connector.scad
```

**PrusaSlicer** (Print preparation):
```bash
prusa-slicer /development/projects/active/HIRT/hardware/cad/stl/sensor_body_dual_4x.stl
```

### Generating Custom Views
To export specific assembly states:
1. Edit `modular_flush_connector.scad`, set `part` variable (line 109)
2. Adjust component positions in render logic (lines 948-1018)
3. Export STL: `openscad -o output.stl modular_flush_connector.scad`

Available `part` options:
- `"complete_system"` - Assembled stack visualization
- `"male_cap_single"` - Individual male cap
- `"sensor_body_single"` - Individual sensor body
- `"probe_tip_single"` - Individual tip
- `"junction_cap_single"` - Individual junction cap
- `"mixed_system_array"` - Complete print plate (2 caps, 2 sensors, 1 tip, 1 junction)

## Measurement and Inspection

### Using Section Planes in FreeCAD
For internal inspection and dimension verification:
1. Open any STL file in FreeCAD
2. Create a section plane: Menu → Part → Cross-sections
3. Position plane through axis or features of interest
4. Use Measure tool to verify critical dimensions

### Key Dimensions to Verify
**Male Rod Cap**:
- Insertion depth: 20mm ±0.3mm
- Thread length: 15mm ±0.5mm
- Center hole: 6mm ±0.2mm
- O-ring groove depth: 1.5mm ±0.1mm

**Sensor Body**:
- Total length: 70mm ±0.3mm
- OD: 16mm ±0.2mm (must match rod)
- Female socket depth: 25mm ±0.5mm each end
- ERT groove width: 4mm ±0.2mm
- MIT zone length: 15mm ±1mm

**Thread Fit**:
- Hand-tight engagement: 10-12mm
- No wobble when assembled
- O-ring visible compression (not extrusion)

---

**Status**: Assembly drawings reference complete. All CAD files current as of 2026-01-28.

**Next Steps**:
- Generate 2D engineering drawings (DXF) for critical interfaces
- Create photo documentation during prototype assembly
- Add field deployment grid overlay drawings (see quick-reference.md)
