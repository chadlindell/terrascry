# Quick Start - Micro-Probe CAD Files

## Get Started in 5 Minutes

### Step 1: Install OpenSCAD

**Download:** https://openscad.org/
- Available for Windows, Mac, Linux
- Free and open-source

### Step 2: Open a Part

1. Launch OpenSCAD
2. File → Open → `openscad/micro_probe_tip.scad` (start with tip - simplest)
3. Press **F5** to preview
4. Press **F6** to render

### Step 3: Export for 3D Printing

1. After rendering (F6), go to File → Export → Export as STL
2. Save to `stl/` directory
3. Open STL in your slicer (Cura, PrusaSlicer, etc.)
4. Print!

### Step 4: Test Print

Start with the **probe tip** (`micro_probe_tip.scad`):
- Simple part (fast to print)
- Good for testing dimensions
- Verify 12mm base fits your rod

## Available Parts (Micro-Probe Design)

### Essential Parts
- **`micro_probe_tip.scad`** - Tapered nose cone (25mm long)
  - Best for: Testing insertion, verifying dimensions
  - Print time: ~30 minutes
  - Material: PETG

- **`micro_probe_head.scad`** - Surface junction box (25mm × 35mm)
  - Best for: Terminal block mounting, cable connections
  - Print time: ~1-2 hours
  - Material: PETG

- **`micro_rod_coupler.scad`** - Threaded coupler (45mm × 18mm)
  - Best for: Joining rod sections
  - Print time: ~1 hour
  - Material: Glass-filled nylon or PETG

- **`micro_ert_ring_collar.scad`** - ERT ring collar (5mm × 12mm ID)
  - Best for: Mounting narrow ERT rings
  - Print time: ~20 minutes
  - Material: PETG

## Recommended Print Settings

**Material:** PETG
- **Nozzle:** 0.4mm
- **Layer Height:** 0.2-0.3mm
- **Infill:** 40-50%
- **Temperature:** 230-240°C (check your filament)
- **Bed:** 70-80°C
- **Speed:** 40-50mm/s

**Note:** Micro-probe parts are smaller and faster to print than the old 25mm design!

## Design Notes

**Important:** This is the **micro-probe design** (12mm OD):
- Much smaller than old 25mm design
- Passive probes (no electronics in probe)
- Electronics stay at surface
- Minimal intrusion for archaeology

## Next Steps

1. Print probe tip first (test fit with 12mm rod)
2. Print junction box (test terminal block fit)
3. Print coupler (test thread fit)
4. Assemble test probe
5. Test insertion

## Need Help?

- See [README.md](README.md) for detailed information
- See [Manufacturing Guide](docs/manufacturing-guide.md) for production tips
- See [Export Instructions](docs/export-instructions.md) for file conversion
- See [Design Change Document](../../DESIGN_CHANGE_MICRO_PROBE.md) for design rationale

---

**Happy Printing!**
