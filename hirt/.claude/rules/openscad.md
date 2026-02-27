---
paths:
  - "hardware/cad/openscad/*.scad"
---

# OpenSCAD Design Rules

## Parameters
- All dimensions in mm
- Use descriptive parameter names: `rod_outer_diameter` not `d1`
- Group related parameters with comments

## Modules
- One component per module
- Export modules: `*_stl()` suffix for printable parts
- Test modules: `*_test()` suffix for verification

## Threading
- M12x1.75 standard for probe connections
- Use thread library for accuracy
- Include `_tapready` variants for tap/die finishing

## Output
- STL files to `hardware/cad/stl/`
- Use `_4x` suffix for print-plate arrangements
