# Changelog

All notable changes to the HIRT project will be documented in this file.

## [Unreleased]

### 2024-12-19 - Micro-Probe Design Implementation

#### Design Changes
- **Rod diameter:** Reduced from 25mm to 12mm OD (~10× less disturbance)
- **Electronics:** Moved from embedded in probes to central surface hub
- **Probe type:** Changed from "smart probe" to passive probe design
- **Cost reduction:** ~$1,650-2,250 savings per system

#### CAD Files
- Added micro-probe CAD files (OpenSCAD):
  - `micro_probe_tip.scad` - Tapered tip (12mm, M12×1.5 thread)
  - `micro_rod_coupler.scad` - Rod coupler (M12×1.5 threads)
  - `micro_probe_head.scad` - Surface junction box
  - `micro_ert_ring_collar.scad` - ERT ring collar
  - `micro_probe_assembly.scad` - Assembly visualization
- Deprecated old 25mm design files (moved to `deprecated/` folder)
- Fixed thread specifications (correct tap drill sizes: 10.5mm for M12×1.5)

#### Documentation Updates
- Updated system architecture for passive probes + central hub
- Updated BOM files for micro-probe design
- Updated assembly guides for new design
- Updated electronics documentation for surface electronics
- Added connection methods documentation
- Added threading and wall thickness specifications
- Consolidated root documentation (removed redundant update/change files)

#### Files Added
- `hardware/cad/CONNECTION_METHODS.md` - Connection methods guide
- `hardware/cad/THREADING_AND_WALL_THICKNESS.md` - Threading specifications
- `hardware/schematics/mechanical/micro-probe-design.md` - Design overview
- `CHANGELOG.md` - This file

#### Files Removed
- `DESIGN_CHANGE_MICRO_PROBE.md` - Consolidated into CHANGELOG
- `DESIGN_UPDATE_SUMMARY.md` - Consolidated into CHANGELOG
- `PROJECT_REVIEW.md` - Consolidated into CHANGELOG

---

## [0.9] - 2024-11-XX

### Initial Release
- Complete documentation structure
- White paper (19 sections)
- BOM files and order sheets
- Field guides and checklists
- Build instructions
