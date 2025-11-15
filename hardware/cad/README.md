# CAD Files for HIRT Probe System

## Overview

This directory contains CAD files for manufacturing HIRT probe components. Files are provided in multiple formats for different manufacturing methods:

- **OpenSCAD** (`.scad`) - Parametric CAD source files
- **STL** (`.stl`) - 3D printing format
- **STEP** (`.step`) - CNC machining format (to be exported from OpenSCAD)

## Directory Structure

```
cad/
├── openscad/          # OpenSCAD source files (.scad)
├── stl/              # 3D printable STL files
├── step/             # CNC-ready STEP files
└── docs/             # Manufacturing documentation
```

## Parts Available

### 1. Probe Head (Nose Capsule)
- **File:** `openscad/probe_head.scad`
- **Output:** `stl/probe_head.stl`
- **Material:** PETG or ABS (3D printing)
- **Dimensions:** Ø30mm × 100mm
- **Features:** Threaded connection, cable gland mount, removable cap

### 2. Threaded Coupler
- **File:** `openscad/rod_coupler.scad`
- **Output:** `stl/rod_coupler.stl`
- **Material:** Glass-filled nylon or PETG
- **Dimensions:** 75mm long × 30mm OD
- **Features:** M/F threads, O-ring grooves

### 3. ERT Ring Mounting Collar
- **File:** `openscad/ert_ring_collar.scad`
- **Output:** `stl/ert_ring_collar.stl`
- **Material:** PETG or ABS
- **Dimensions:** 12mm wide × 25mm ID
- **Features:** Insulating collar for ERT ring mounting

## Manufacturing Methods

### 3D Printing

**Recommended Settings:**
- **Layer Height:** 0.2-0.3mm
- **Infill:** 40-50% (structural parts)
- **Wall Count:** 3-4 perimeters
- **Support:** Minimal (parts designed to print without supports)
- **Material:** PETG (preferred) or ABS

**Post-Processing:**
- Remove support material
- Clean threads with tap if needed
- Test fit before final assembly

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

**Rod Thread:**
- **Type:** Custom ACME-style or 1" NPT
- **Pitch:** 4 TPI (ACME) or 11.5 TPI (NPT)
- **Engagement:** 25-30mm

**Note:** Threads may need adjustment based on actual rod threading. Test fit before production.

## Quality Control

Before using manufactured parts:
- [ ] Verify dimensions with calipers
- [ ] Test thread fit with actual rod
- [ ] Check O-ring groove dimensions
- [ ] Verify wall thickness
- [ ] Test waterproofing (for probe head)

## Notes

- All dimensions are in millimeters
- Threads are modeled but may need post-processing
- Some parts may require supports during 3D printing
- Test print small parts first before full production

## Support

For issues or questions:
- Check OpenSCAD file comments for parameter descriptions
- Review manufacturing documentation in `docs/` directory
- Test with dummy parts first

---

*Last Updated: 2024-12-19*

