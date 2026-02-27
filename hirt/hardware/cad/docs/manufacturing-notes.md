# Manufacturing Notes - Micro-Probe Parts

## Overview

This document provides manufacturing method recommendations and notes for each micro-probe component.

## Home 3D Printing Setup

- **Bed prep:** Clean PEI or glass with 90% IPA, then apply a thin glue-stick film or hairspray where each part will sit. Always add a 6-10 mm brim or a 0.4 mm sacrificial raft so tall, skinny parts stay locked down.
- **Stabilizers:** For components taller than 25 mm, add a 5×5 mm stabilizer tower connected with a single-layer tab. Break the tab off after printing.
- **First layers:** Run the first two layers at 15-20 mm/s with 110% extrusion width and keep part cooling off until layer three.
- **Infill:** Set infill density to **100% solid** with a simple grid or lines pattern. Combine with 5 perimeters and 8 top/bottom layers so thin threads do not crush while tapping.
- **Temperatures:** PETG nozzle 245 °C / bed 80 °C (adjust ±5 °C per filament brand). Disable z-hop only on the first layer to avoid knocking parts loose.
- **Post-print hardening:** While parts are still warm, wick thin epoxy or CA into threads/tips to toughen the surface before tapping. Allow full cure (overnight) before final assembly.

## Manufacturing Methods

### 3D Printing
- **Best for:** Complex shapes, prototypes, low-volume production
- **Materials:** PETG (preferred), ABS, Nylon
- **Post-processing:** Thread tapping required
- **Tolerances:** ±0.2mm typical

### CNC Machining
- **Best for:** Precision threads, high-volume production, strength-critical parts
- **Materials:** Nylon, Delrin, Aluminum
- **Tolerances:** ±0.05mm typical
- **Threads:** Can be cut directly

## Component Manufacturing Guide

### 1. Probe Tip (Nose Cone)

**File:** `micro_probe_tip.scad`

**Recommended Method:**
- **3D Printing (PETG)** - Good for prototypes and low volume
- **CNC Machining (Delrin)** - Better for production, smoother finish

**Threading:**
- Internal M12×1.5 thread
- Use M12×1.5 tap after printing
- Or cut threads directly if CNC machined

**Notes:**
- Tip must be sharp for insertion
- Solid tip section for strength
- Thread sealant recommended for waterproofing

**Orientation:**
- Mounts at BOTTOM of rod (pointing down for insertion)
- Screws onto rod external thread

#### At-Home Manufacturing Steps
1. **Slice:** Orient vertically with tip pointing upward. Enable an 8 mm brim, 100% infill, 5 perimeters, 0.2 mm layers, and a slim stabilizer tower.
2. **Print:** PETG at 245 °C / 80 °C bed. Keep cooling <30% after layer three. Do not remove until the bed cools below 35 °C.
3. **Deburr:** Remove brim, sand the base flat, and clean the bore with a dowel wrapped in 400-grit paper.
4. **Tap:** Clamp carefully and run an M12×1.5 tap through the bore using cutting oil. Back out every half turn to clear chips.
5. **Seal:** Flood the threads with thin epoxy (e.g., West System 105/205 or Loctite EA 9462) while spinning the tap once more. Let cure 12 hours.
6. **QA:** Test-fit on a scrap rod, ensure no wobble, then bag and label by probe batch.

---

### 2. Rod Coupler

**File:** `micro_rod_coupler.scad`

**Recommended Method:**
- **CNC Machining (Nylon or Delrin)** - Preferred for precision threads
- **3D Printing (PETG) + Tapping** - Acceptable for prototypes

**Threading:**
- Internal M12×1.5 threads on both ends
- Use M12×1.5 tap after printing
- Or cut threads directly if CNC machined
- Threads must be precise for reliable connection

**Notes:**
- Smooth center section for grip during assembly
- O-ring grooves for sealing
- Critical part - threads must be accurate

**Assembly:**
- Screws onto rod external threads from both sides
- Joins 1.5m sections to reach 3m depth

#### At-Home Manufacturing Steps
1. **Slice:** Lay the coupler on its side so both thread bores point left/right. Enable dual 10 mm brims, tree supports only under the chamfers, 100% infill, 0.2 mm layers, 6 perimeters.
2. **Print:** PETG 245 °C / 80 °C bed, 20 mm/s outer walls. Pause mid-print if you intend to embed brass inserts for anti-rotation screws.
3. **Cleanup:** Remove supports while the part is warm, then chase O-ring grooves with a plastic scraper.
4. **Tap:** Use an M12×1.5 bottoming tap from each side until it hits the center relief. Ensure taps stay colinear.
5. **Epoxy soak:** Brush thin epoxy onto both thread bores and O-ring grooves. Hang vertically to cure.
6. **QC:** Thread sacrificial rods into both ends, verify smooth engagement, and check that O-rings seat without pinch points.

---

### 3. Surface Junction Box

**File:** `micro_probe_head.scad`

**Recommended Method:**
- **3D Printing (PETG)** - Preferred (weatherproof, good for enclosures)
- **CNC Machining (Delrin)** - Alternative

**Threading:**
- Internal M12×1.5 thread (bottom, screws onto rod)
- Internal thread for cap (top)
- Use appropriate taps after printing/machining

**Notes:**
- Must be weatherproof (IP65 rating)
- Terminal block mounts inside
- Cable gland for cable entry
- Cap with O-ring groove for sealing

**Assembly:**
- Mounts at TOP of rod (surface, not inserted)
- Screws onto rod external thread
- Cap screws on top for access

#### At-Home Manufacturing Steps
1. **Slice:** Print body upright (rod thread down). Use a 10 mm brim, 100% infill, 0.2 mm layers, 5 perimeters, and add a stabilizer tower opposite the cable gland. Slice the cap separately, also upright.
2. **Print:** PETG 245 °C / 80 °C bed. Pause at the insert ledge if installing heat-set inserts for terminal blocks.
3. **Tap/Chase:** Run an M12×1.5 tap into the base, chase the cap threads with the matching die, and ream the gland hole to the gland’s datasheet dimension.
4. **Seal:** Brush thin epoxy over all threads and the O-ring groove, then install the O-ring after cure.
5. **Dry Fit Hardware:** Install cable gland, strain relief, and terminal block to confirm clearances. Drill pilot holes for mounting screws as needed.
6. **Water Test:** Assemble cap, plug ports, and dunk halfway in water while applying light vacuum with a syringe for 5 minutes. Reject if bubbles appear.

---

### 4. ERT Ring Collar

**File:** `micro_ert_ring_collar.scad`

**Recommended Method:**
- **3D Printing (PETG)** - Preferred (simple part, low cost)
- **CNC Machining (Delrin)** - Alternative

**Threading:**
- No threads - slides onto rod
- Snug fit (12.0-12.2mm ID)

**Notes:**
- Non-conductive material required
- Ring groove for ERT ring mounting
- Wire channel for routing
- Bonds to rod with epoxy

**Assembly:**
- Slides onto rod at specified position
- Multiple collars per probe (2-3 typical)
- Positions: 0.5m, 1.5m, 2.5m from tip

#### At-Home Manufacturing Steps
1. **Slice:** Nest collars flat in batches. Use a 5 mm brim, 0.16 mm layers, 100% infill, and enable “monotonic” top layers for smooth ring seats.
2. **Print:** PETG 240 °C / 80 °C bed. Let the bed cool to room temperature so the 12 mm bore stays round.
3. **Deburr:** Knock down brim and lightly chamfer the inner bore with a countersink. Clean the wire channel with a 1/16" drill bit if needed.
4. **Dry Fit:** Slide onto a 12.00 mm test mandrel; adjust with a reamer if the fit is too tight.
5. **Epoxy Install:** Before final assembly, scuff rod surface, apply epoxy, slide collar into place, rotate 360° to wet the joint, and clamp until cured.
6. **Ring Attachment:** Wrap 0.5 mm stainless strip in the groove, overlap 3 mm, and secure with a miniature screw, rivet, or conductive epoxy. Cover joint with heat-shrink for strain relief.

---

## Rod Threading

**Rod Sections:**
- External M12×1.5 threads on both ends
- Cut with M12×1.5 die
- Thread length: 15-20mm
- Thread sealant recommended

**Note:** Rod threading is critical - must be precise for reliable connections.

---

## Assembly Sequence

1. **Thread rod sections** (external M12×1.5 on both ends)
2. **Mount tip** (screw onto rod bottom, point down)
3. **Mount ERT collars** (slide onto rod, bond with epoxy)
4. **Join sections** (screw into coupler, apply sealant)
5. **Mount junction box** (screw onto rod top)

---

## Quality Control

### Threads
- [ ] Threads engage smoothly
- [ ] No binding or excessive play
- [ ] Thread sealant applied
- [ ] O-rings installed (coupler)

### Fit
- [ ] Tip fits rod securely
- [ ] Junction box fits rod securely
- [ ] ERT collars fit rod (snug but not tight)
- [ ] Coupler fits rod sections

### Waterproofing
- [ ] Thread sealant on all threaded connections
- [ ] O-rings in coupler grooves
- [ ] Junction box cap seals properly

---

## Manufacturing Recommendations Summary

| Part | Method | Material | Threading | Notes |
|------|--------|----------|-----------|-------|
| Tip | 3D Print or CNC | PETG/Delrin | Tap M12×1.5 | Sharp point critical |
| Coupler | CNC (preferred) | Nylon/Delrin | Tap or cut M12×1.5 | Precision threads critical |
| Junction Box | 3D Print | PETG | Tap M12×1.5 | Weatherproof required |
| ERT Collar | 3D Print | PETG | None (slip fit) | Simple part, low cost |

---

*For detailed assembly instructions, see [Assembly Guide](../../../build/assembly-guide-detailed.md).*

