# Manufacturing Guide - HIRT CAD Parts

## Overview

This guide provides instructions for manufacturing HIRT probe components from the provided CAD files.

## Available Parts

### 1. Probe Tip (Nose Cone) - BOTTOM

**File:** `openscad/micro_probe_tip.scad`

**3D Printing:**
- **Material:** PETG (preferred) or ABS
- **Layer Height:** 0.2-0.3mm
- **Infill:** 40-50%
- **Supports:** Minimal (may need supports for point)
- **Orientation:** Print standing on end (point up)
- **Post-Processing:**
  - Clean threads with M12×1.5 tap
  - Test fit with rod
  - Verify sharp point

**CNC Machining:**
- Export as STEP file
- Material: Delrin, Nylon, or Aluminum
- Threads: Cut with M12×1.5 tap
- Surface finish: Smooth

### 2. Rod Coupler - MIDDLE

**File:** `openscad/micro_rod_coupler.scad`

**3D Printing:**
- **Material:** Glass-filled nylon (preferred) or PETG
- **Layer Height:** 0.2mm (for thread quality)
- **Infill:** 50-60% (structural part)
- **Supports:** None needed
- **Orientation:** Print horizontally (lying down)
- **Post-Processing:**
  - Clean internal threads with tap
  - Test fit with rod sections
  - Install O-rings in grooves

**CNC Machining:**
- Export as STEP file
- Material: Nylon, Delrin, or Aluminum
- Threads: Cut with appropriate tap/die
- O-ring grooves: Machine to exact dimensions

### 3. ERT Ring Collar - MIDDLE

**File:** `openscad/micro_ert_ring_collar.scad`

**3D Printing:**
- **Material:** PETG or ABS
- **Layer Height:** 0.2-0.3mm
- **Infill:** 30-40%
- **Supports:** None needed
- **Orientation:** Print standing on end
- **Post-Processing:**
  - Test fit on rod (should be snug)
  - Verify ring groove dimensions

### 4. Surface Junction Box - TOP

**File:** `openscad/micro_probe_head.scad`

**3D Printing:**
- **Material:** PETG (preferred for weatherproof)
- **Layer Height:** 0.2-0.3mm
- **Infill:** 40-50%
- **Supports:** Minimal
- **Orientation:** Print standing on end
- **Post-Processing:**
  - Clean threads with M12×1.5 tap
  - Test fit with rod
  - Verify cable gland fit
  - Test weatherproofing

**CNC Machining:**
- Export as STEP file
- Material: Delrin or Nylon
- Threads: Cut with M12×1.5 tap
- Surface finish: Smooth for weatherproofing

## Thread Specifications

### Rod Threads

The coupler uses internal threads that match the rod threads. Actual thread specifications depend on your rod threading method:

**Option 1: 1" NPT (National Pipe Thread)**
- Threads per inch: 11.5
- Taper: 1:16
- Use 1" NPT tap for coupler

**Option 2: Custom ACME Thread**
- Threads per inch: 4
- Straight thread (no taper)
- Use custom tap or thread mill

**Option 3: Metric Thread**
- M25×2.5 (25mm diameter, 2.5mm pitch)
- Straight thread
- Use M25×2.5 tap

**Note:** The OpenSCAD files use simplified thread models. You will need to:
1. Test print a small sample
2. Adjust thread dimensions if needed
3. Use appropriate tap for final threads

## 3D Printing Tips

### Material Selection

**PETG (Recommended):**
- Good strength and durability
- Waterproof
- Easy to print
- Good for probe head and collars

**ABS:**
- Strong and durable
- Requires heated bed
- May warp
- Good for probe head

**Glass-Filled Nylon:**
- Highest strength
- Best for couplers
- Requires high-temperature printing
- May need annealing

### Print Settings

**General:**
- Use 0.2-0.3mm layer height for good surface finish
- 3-4 perimeters for strength
- 40-50% infill for structural parts
- Print slowly (30-50mm/s) for quality

**Threads:**
- Print at 0.2mm layer height for thread quality
- Use 100% infill for threaded sections
- Consider printing threads separately if needed

**Supports:**
- Minimize supports where possible
- Use tree supports for complex overhangs
- Remove supports carefully to avoid damage

### Post-Processing

1. **Remove Supports:**
   - Carefully remove all support material
   - Use flush cutters or knife
   - Sand if needed

2. **Clean Threads:**
   - Use appropriate tap for internal threads
   - Use die for external threads
   - Test fit frequently

3. **Surface Finishing:**
   - Sand smooth surfaces if needed
   - Clean with isopropyl alcohol
   - Test fit all parts

## CNC Machining

### Exporting STEP Files

1. **From OpenSCAD:**
   - Render the part (F6)
   - Export as STL
   - Import STL into FreeCAD
   - Convert mesh to solid
   - Export as STEP

2. **From FreeCAD:**
   - Import STL
   - Part → Create shape from mesh
   - Part → Convert to solid
   - File → Export → STEP

### CAM Setup

1. **Import STEP file** into CAM software
2. **Set material** (Nylon, Delrin, Aluminum)
3. **Generate toolpaths:**
   - Roughing pass
   - Finishing pass
   - Thread cutting (if applicable)
4. **Generate G-code**
5. **Machine part**

### Thread Cutting

For CNC-machined threads:
- Use thread mill or tap
- Follow thread specifications
- Test fit with rod
- Adjust if needed

## Quality Control

### Dimensional Checks

Before using manufactured parts:
- [ ] Measure outer diameter (should be 30mm for probe head)
- [ ] Measure inner diameter (should match rod OD)
- [ ] Measure length (should match specifications)
- [ ] Check wall thickness
- [ ] Verify thread fit with rod

### Functional Tests

- [ ] Test thread engagement (should engage smoothly)
- [ ] Test O-ring fit in grooves
- [ ] Test cable gland fit
- [ ] Test waterproofing (for probe head)
- [ ] Test ERT ring fit on collar

### Fit Testing

1. **Probe Head:**
   - Test fit with electronics PCB
   - Test fit with coils
   - Test cable gland installation
   - Test cap fit

2. **Coupler:**
   - Test fit with rod sections
   - Test O-ring installation
   - Test thread engagement
   - Verify smooth center section

3. **ERT Collar:**
   - Test fit on rod
   - Test ERT ring fit
   - Verify wire channel

## Troubleshooting

### Thread Issues

**Problem:** Threads too tight
- **Solution:** Adjust thread diameter in OpenSCAD, re-export
- **Alternative:** Use larger tap size

**Problem:** Threads too loose
- **Solution:** Reduce thread diameter, use smaller tap
- **Alternative:** Add thread sealant

### Print Quality Issues

**Problem:** Poor surface finish
- **Solution:** Reduce layer height, slow print speed
- **Alternative:** Post-process with sanding

**Problem:** Warping
- **Solution:** Use heated bed, add brim/raft
- **Alternative:** Use different material

### Fit Issues

**Problem:** Parts don't fit together
- **Solution:** Check dimensions, adjust tolerances
- **Alternative:** Test with dummy parts first

## Dummy Prototype Parts

For initial testing, use dummy parts:
- `probe_head_dummy.scad` - Simplified probe head
- Test fit before printing full version
- Verify dimensions and clearances

## Next Steps

After manufacturing:
1. Test fit all parts
2. Assemble test probe
3. Verify dimensions
4. Adjust CAD files if needed
5. Manufacture production parts

---

*For assembly instructions, see [Assembly Guide](../../../build/assembly-guide-detailed.md).*

