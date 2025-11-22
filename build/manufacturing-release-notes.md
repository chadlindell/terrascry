# Manufacturing Release Report: 16mm Modular Probe

**Date:** 2024-12-19
**Design Status:** RELEASED FOR PRINTING

## 1. Final Design Summary
The HIRT probe has been successfully migrated to a **16mm Modular Flush-Connector** architecture optimized for FDM manufacturing on Bambu Lab A1 Mini printers.

### Key Specs
- **Rod Standard:** 16mm OD / 12mm ID Fiberglass Tube
- **Connector Type:** Flush Modular Inserts (Epoxied)
- **Thread Standard:** M12×1.75 ISO (Modified "Chunky" Profile for Printability)
- **Wiring:** Central 6mm hollow conduit (Confirmed Clear)

## 2. Manufacturing Settings (Bambu A1 Mini)
- **Material:** PETG or ASA (Required for impact/UV)
- **Layer Height:** 0.12mm (Fine) - *Critical for threads*
- **Infill:** 100% (Solid) - *Critical for strength*
- **Walls:** 6 Loops
- **Supports:** DISABLED (Use built-in scaffolding)
- **Brim:** DISABLED (Use built-in Super Brim)
- **Speed:** 50mm/s Outer Wall

## 3. File Manifest
- **Source:** `hardware/cad/openscad/modular_flush_connector.scad`
- **Print File:** `hardware/cad/stl/modular_mixed_array_4x.stl`
    - Contains: 2x Male Plugs + 2x Female Sensors
    - Features: Super Brim + Non-Intersecting Rigid Scaffolding

## 4. Assembly & Post-Processing
1.  **Clean:** Snip scaffolding with flush cutters.
2.  **Thread:** "Chase" threads with M12×1.75 Tap/Die if tight.
3.  **Bond:** Epoxy inserts into 16mm rod segments.
4.  **Seal:** Install O-ring at thread shoulder.
5.  **Wire:** Run cable through center before screwing shut.

## 5. Known Constraints
- **Female Thread Thickness:** Optimized to 50/50 ratio to prevent thin walls.
- **Tap Clearance:** Female hole depth increased to 25mm to prevent tap bottoming out.

**Project Phase: Manufacturing**
