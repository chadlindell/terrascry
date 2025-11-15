# Quick Start - CAD Files for Prototyping

## Get Started in 5 Minutes

### Step 1: Install OpenSCAD

**Download:** https://openscad.org/
- Available for Windows, Mac, Linux
- Free and open-source

### Step 2: Open a Part

1. Launch OpenSCAD
2. File → Open → `openscad/probe_head_dummy.scad`
3. Press **F5** to preview
4. Press **F6** to render

### Step 3: Export for 3D Printing

1. After rendering (F6), go to File → Export → Export as STL
2. Save to `stl/` directory
3. Open STL in your slicer (Cura, PrusaSlicer, etc.)
4. Print!

### Step 4: Test Print

Start with the **dummy probe head** (`probe_head_dummy.scad`):
- No threads (faster to print)
- Good for fit testing
- Verify dimensions before printing full version

## Available Parts

### For Quick Testing
- **`probe_head_dummy.scad`** - Simplified probe head (no threads)
  - Best for: Fit testing, size verification
  - Print time: ~2-3 hours
  - Material: PETG

### For Production
- **`probe_head.scad`** - Full probe head with threads
- **`rod_coupler.scad`** - Threaded coupler for rod sections
- **`ert_ring_collar.scad`** - ERT ring mounting collar

## Recommended Print Settings

**Material:** PETG
- **Nozzle:** 0.4mm
- **Layer Height:** 0.2-0.3mm
- **Infill:** 40-50%
- **Temperature:** 230-240°C (check your filament)
- **Bed:** 70-80°C
- **Speed:** 40-50mm/s

## Next Steps

1. Print dummy parts first
2. Test fit with actual components
3. Adjust dimensions if needed (edit `.scad` file)
4. Print production parts

## Need Help?

- See [README.md](README.md) for detailed information
- See [Manufacturing Guide](docs/manufacturing-guide.md) for production tips
- See [Export Instructions](docs/export-instructions.md) for file conversion

---

**Happy Printing!**

