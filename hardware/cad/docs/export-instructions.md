# Export Instructions - CAD Files to Manufacturing Formats

## Overview

This guide explains how to export OpenSCAD files to formats needed for manufacturing (STL for 3D printing, STEP for CNC).

## Exporting STL Files (3D Printing)

### Method 1: Using OpenSCAD GUI

1. **Open OpenSCAD**
   - Launch OpenSCAD application
   - File → Open → Select `.scad` file

2. **Render the Part**
   - Press **F6** (Render) or use Design → Render
   - Wait for rendering to complete (may take 30-60 seconds)

3. **Export STL**
   - File → Export → Export as STL
   - Choose save location (save to `stl/` directory)
   - File will be saved as `.stl`

4. **Verify**
   - Open STL in slicer (Cura, PrusaSlicer, etc.)
   - Check dimensions
   - Verify part looks correct

### Method 2: Using Command Line

```bash
# Install OpenSCAD command-line tools
# Then run:

openscad -o stl/probe_head.stl openscad/probe_head.scad
openscad -o stl/rod_coupler.stl openscad/rod_coupler.scad
openscad -o stl/ert_ring_collar.stl openscad/ert_ring_collar.scad
```

### Batch Export Script

Create a script to export all parts:

**Linux/Mac (`export_stl.sh`):**
```bash
#!/bin/bash
mkdir -p stl
openscad -o stl/probe_head.stl openscad/probe_head.scad
openscad -o stl/rod_coupler.stl openscad/rod_coupler.scad
openscad -o stl/ert_ring_collar.stl openscad/ert_ring_collar.scad
openscad -o stl/probe_head_dummy.stl openscad/probe_head_dummy.scad
echo "STL files exported to stl/ directory"
```

**Windows (`export_stl.bat`):**
```batch
@echo off
mkdir stl
openscad -o stl\probe_head.stl openscad\probe_head.scad
openscad -o stl\rod_coupler.stl openscad\rod_coupler.scad
openscad -o stl\ert_ring_collar.stl openscad\ert_ring_collar.scad
openscad -o stl\probe_head_dummy.stl openscad\probe_head_dummy.scad
echo STL files exported to stl\ directory
```

## Exporting STEP Files (CNC Machining)

OpenSCAD doesn't directly export STEP files. Use one of these methods:

### Method 1: OpenSCAD → STL → FreeCAD → STEP

1. **Export STL from OpenSCAD** (see above)

2. **Import into FreeCAD**
   - Open FreeCAD
   - File → Open → Select `.stl` file
   - Select "Mesh" workbench

3. **Convert to Solid**
   - Select the mesh
   - Part → Create shape from mesh
   - Part → Convert to solid
   - Select the solid

4. **Export STEP**
   - File → Export
   - Choose format: STEP (*.step, *.stp)
   - Save to `step/` directory

### Method 2: Using FreeCAD Python Script

Create a script to automate conversion:

```python
import FreeCAD
import Mesh
import Part

# Import STL
mesh = Mesh.Mesh("stl/probe_head.stl")
shape = Part.Shape()
shape.makeShapeFromMesh(mesh.Topology, 0.1)  # 0.1 is tolerance
solid = Part.makeSolid(shape)

# Export STEP
solid.exportStep("step/probe_head.step")
```

### Method 3: Using Online Converters

1. Export STL from OpenSCAD
2. Use online converter (e.g., CloudConvert)
3. Convert STL → STEP
4. Download and save to `step/` directory

**Note:** Online converters may lose precision. Use FreeCAD for best results.

## Verifying Exported Files

### STL Verification

1. **Open in Slicer**
   - Import STL into Cura, PrusaSlicer, etc.
   - Check dimensions match specifications
   - Verify part looks correct

2. **Check File Size**
   - STL files should be reasonable size (100KB - 5MB typically)
   - Very large files may indicate issues

3. **Visual Inspection**
   - Rotate part in 3D viewer
   - Check for missing geometry
   - Verify threads are present

### STEP Verification

1. **Open in CAD Software**
   - Import STEP into FreeCAD, SolidWorks, etc.
   - Check dimensions
   - Verify geometry

2. **Check for Errors**
   - Look for import warnings
   - Verify surfaces are closed
   - Check for gaps or holes

## Recommended Export Settings

### OpenSCAD Render Settings

- **$fn (facet number):** 100 (for smooth curves)
- **Resolution:** High (for quality)
- **Render time:** Allow 30-60 seconds per part

### STL Export Settings

- **Format:** ASCII or Binary (Binary is smaller)
- **Precision:** High (0.001mm typical)
- **No compression needed**

### STEP Export Settings

- **Format:** STEP AP214 (preferred) or AP203
- **Units:** Millimeters
- **Precision:** High

## File Organization

After exporting, organize files:

```
cad/
├── openscad/          # Source files
├── stl/              # 3D printing files
│   ├── probe_head.stl
│   ├── rod_coupler.stl
│   └── ert_ring_collar.stl
└── step/             # CNC files
    ├── probe_head.step
    ├── rod_coupler.step
    └── ert_ring_collar.step
```

## Troubleshooting

### Export Fails

**Problem:** OpenSCAD won't export
- **Solution:** Make sure part is fully rendered (F6)
- **Check:** No errors in console

### STL File is Empty

**Problem:** STL file is 0 bytes or empty
- **Solution:** Re-render part, check for errors
- **Check:** Make sure part is visible in preview

### STEP Import Errors

**Problem:** STEP file won't import into CAM software
- **Solution:** Try different STEP format (AP203 vs AP214)
- **Alternative:** Export from different CAD software

### Threads Missing

**Problem:** Threads don't appear in exported file
- **Solution:** Threads are simplified in OpenSCAD
- **Note:** Use tap/die for actual threads after printing/machining

## Next Steps

After exporting:
1. Verify files in appropriate software
2. Test print/machine sample parts
3. Adjust CAD files if needed
4. Export final production files

---

*For manufacturing instructions, see [Manufacturing Guide](manufacturing-guide.md).*

