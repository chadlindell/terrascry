# CAD Files for HIRT Micro-Probe System

## Overview

This directory contains CAD files for manufacturing HIRT **micro-probe** components (12mm OD design). Files are provided in multiple formats for different manufacturing methods:

**Design Philosophy:** Minimal intrusion, archaeology-friendly. Rod diameter: 10-16mm (target: 12mm).

- **OpenSCAD** (`.scad`) - Parametric CAD source files
- **STL** (`.stl`) - 3D printing format
- **STEP** (`.step`) - CNC machining format (to be exported from OpenSCAD)
- **Docs** (`docs/`) - Manufacturing guides, including `base-hub-breakout.md` for the centralized harness/backplane

## Directory Structure

```
cad/
├── openscad/          # OpenSCAD source files (.scad)
├── stl/              # 3D printable STL files
├── step/             # CNC-ready STEP files
└── docs/             # Manufacturing documentation
```

## Parts Available

### 1. Probe Tip (Nose Cone) - BOTTOM
- **File:** `openscad/micro_probe_tip.scad`
- **Output:** `stl/micro_probe_tip.stl`
- **Manufacturing:** 3D Print (PETG) or CNC (Delrin)
- **Dimensions:** 25mm long, 12mm base, 4mm point
- **Threading:** Internal M12×1.5 (screws onto rod bottom)
- **Features:** Tapered for easy insertion, points DOWN
- **Assembly:** Screws onto rod bottom (external M12×1.5 thread)

### 2. Micro-Rod Coupler - MIDDLE
- **File:** `openscad/micro_rod_coupler.scad`
- **Output:** `stl/micro_rod_coupler.stl`
- **Manufacturing:** CNC (Nylon/Delrin) preferred, or 3D Print + tap
- **Dimensions:** 45mm long × 18mm OD
- **Threading:** Internal M12×1.5 both ends (screws onto rod sections)
- **Features:** O-ring grooves, smooth center for grip
- **Assembly:** Joins rod sections (modular design)

### 3. ERT Ring Collar - MIDDLE
- **File:** `openscad/micro_ert_ring_collar.scad`
- **Output:** `stl/micro_ert_ring_collar.stl`
- **Manufacturing:** 3D Print (PETG)
- **Dimensions:** 5mm wide × 12mm ID
- **Threading:** None (slides onto rod, bonds with epoxy)
- **Features:** Narrow band for 3-5mm ERT rings, wire channel
- **Assembly:** Slides onto rod at 0.5m, 1.5m, 2.5m positions

### 4. Surface Junction Box - TOP
- **File:** `openscad/micro_probe_head.scad`
- **Output:** `stl/micro_probe_head.stl`
- **Manufacturing:** 3D Print (PETG) preferred
- **Dimensions:** Ø25mm × 35mm
- **Threading:** Internal M12×1.5 (screws onto rod top)
- **Features:** Terminal block mount, cable strain relief, weatherproof
- **Assembly:** Screws onto rod TOP (surface, not inserted)
- **Note:** Electronics stay at surface, probe is passive

### 5. Complete Assembly View
- **File:** `openscad/micro_probe_assembly.scad`
- **Purpose:** Visualization of complete modular assembly
- **Note:** Export individual parts separately for manufacturing

### Legacy Parts (25mm Design - Deprecated)
- `probe_head.scad` - Old 30mm design (not recommended)
- `rod_coupler.scad` - Old 30mm design (not recommended)
- `ert_ring_collar.scad` - Old design (not recommended)

## Manufacturing Methods

### 3D Printing

**Recommended Settings:**
- **Layer Height:** 0.2 mm for structural parts (0.15 mm for threaded features)
- **Infill:** 100% solid infill (critical for skinny load-bearing parts)
- **Wall Count:** 3-4 perimeters
- **Support:** Minimal (parts designed to print without supports)
- **Material:** PETG (preferred) or ABS
- **Bed Adhesion:** 6-10 mm brim or small sacrificial raft + glue stick/hairspray on PEI
- **Stabilization:** Enable Z-hop and slow first layers (15-20 mm/s) so tall slender sections stay anchored

**Post-Processing:**
- Remove support material
- Clean threads with tap if needed
- Test fit before final assembly
- Flood threads/tips with thin epoxy if additional abrasion resistance is needed

See `docs/manufacturing-notes.md` for part-by-part print/tap/epoxy instructions tailored to home printers.

### CNC Machining

**For couplers and precision parts:**
- Export STEP files from OpenSCAD
- Use CAM software to generate toolpaths
- Material: Nylon, Delrin, or aluminum
- Threads: Cut with appropriate tap/die

## Using OpenSCAD Files

1. **Install OpenSCAD:**
   - Download from: https://openscad.org/
   - Available for Windows, Mac, Linux

2. **Open File:**
   - Launch OpenSCAD
   - Open `.scad` file from `openscad/` directory

3. **Customize (if needed):**
   - Adjust parameters at top of file
   - Modify dimensions as needed

4. **Render:**
   - Press F5 (Preview) or F6 (Render)
   - Wait for rendering to complete

5. **Export:**
   - **STL:** File → Export → Export as STL
   - **STEP:** Requires additional plugin or FreeCAD conversion

## Converting to STEP (for CNC)

### Method 1: Using FreeCAD
1. Open FreeCAD
2. Import STL file
3. Convert mesh to solid
4. Export as STEP

### Method 2: Using OpenSCAD with STEP export plugin
1. Install STEP export plugin
2. Use `export_stl()` or export menu

## Dummy Prototype Parts

For testing and prototyping, simplified versions are available:
- `probe_head_dummy.scad` - Simplified probe head for fit testing
- `coupler_dummy.scad` - Basic coupler for thread testing

## Thread Specifications

**All Threads: M12×1.5 Metric**
- **Type:** Standard metric thread (M12×1.5)
- **Pitch:** 1.5mm
- **Engagement:** 12-15mm
- **Rod:** External threads on both ends (cut with M12×1.5 die)
- **Tip:** Internal thread (tapped M12×1.5)
- **Junction Box:** Internal thread (tapped M12×1.5)
- **Coupler:** Internal threads both ends (tapped M12×1.5)

**Note:** Threads must be precise for reliable connection. Use appropriate taps/dies. Test fit before production.

## Quality Control

Before using manufactured parts:
- [ ] Verify dimensions with calipers
- [ ] Test thread fit with actual rod
- [ ] Check O-ring groove dimensions
- [ ] Verify wall thickness
- [ ] Test waterproofing (for probe head)

## Modular Assembly

**Assembly Order (bottom to top):**
1. **Tip** - Screws onto rod bottom (pointing down for insertion)
2. **Rod Section 1** - 1.5m section with external threads
3. **ERT Collar + Ring** - Slides onto rod at 0.5m position
4. **Coupler** - Screws onto rod sections (if multi-section)
5. **Rod Section 2** - Additional 1.5m section
6. **ERT Collar + Ring** - At 1.5m from tip
7. **Junction Box** - Screws onto rod top (at surface)

**Key Points:**
- All parts screw together (modular design)
- Tip points DOWN for insertion
- Junction box at TOP (surface, not inserted)
- Rod sections join with couplers
- ERT collars slide on and bond with epoxy

## Manufacturing Methods

See [Manufacturing Notes](docs/manufacturing-notes.md) for detailed recommendations:
- **Tip:** 3D Print (PETG) or CNC (Delrin)
- **Coupler:** CNC (Nylon) preferred for precision threads
- **Junction Box:** 3D Print (PETG) for weatherproof enclosure
- **ERT Collar:** 3D Print (PETG) - simple part

## Notes

- All dimensions are in millimeters
- Threads are simplified in models - use taps/dies for actual threads
- Rod requires external M12×1.5 threads on both ends
- All threaded connections should use thread sealant
- Test print small parts first before full production
- See manufacturing notes for method recommendations

## Support

For issues or questions:
- Check OpenSCAD file comments for parameter descriptions
- Review manufacturing documentation in `docs/` directory
- Test with dummy parts first

---

*Last Updated: 2024-12-19*

