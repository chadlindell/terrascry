# Assembly Drawings

## Overview

This document will contain assembly drawings for the complete probe system.

## Status

Latest exports live in `hardware/cad/stl/` and can be opened directly in FreeCAD, Fusion 360, or PrusaSlicer for visualization/measurement. See below for file references.

## Planned Content

### Probe Assembly
- **3D Assembly:** `hardware/cad/openscad/micro_probe_assembly.scad` (parametric)  
  - Exported STL: `hardware/cad/stl/micro_probe_assembly.stl` (3 m probe, tip → junction box)
- Depicts:
  - Tip, dual 1.5 m rod sections, M12×1.5 coupler
  - Three ERT collars (0.5 m / 1.5 m / 2.5 m offsets)
  - Surface junction box with cable gland
- Use section planes in CAD to pull dimensions for prototype manufacturing.

### Base Hub Assembly
- Reference: `hardware/cad/docs/base-hub-breakout.md` (block layout + harness pinout)
- Pending exports:
  - Backplane PCB STEP (`hardware/cad/step/backplane_160x120.step`) – *in progress*
  - Front-panel DXF (`hardware/cad/step/base_hub_panel.dxf`) – *in progress*
- TODO: add exploded enclosure drawing once PCB and panel files are generated.

### Field Deployment
- See `docs/field-guide/quick-reference.md` for grid spacing callouts.
- Planned additions:
  - Top-down 10×10 m grid DXF
  - Cable routing overlay with hub placement

## Drawing Standards

- Use standard engineering drawing conventions
- Include dimensions and tolerances
- Show assembly sequence
- Indicate critical interfaces
- Include notes and callouts

