# Rod Specifications - Modular Micro-Probe Design

## Overview

**Design Philosophy:** A modular, segmented probe system where sensors are integrated into 3D-printed couplers between standard fiberglass rod sections.

**Core Changes:**
1.  **Increased Diameter:** Moved to **16 mm (5/8") OD** to enable structural 3D-printed threads and internal wiring.
2.  **Modular Assembly:** Probes are built by stacking **Rod Segments** and **Sensor Modules**.
3.  **Flush Design:** All connectors and modules match the rod OD (16 mm) for a smooth, snag-free profile.

## Stacked Assembly Visualization

```
      [TOP CAP]
         │
         │ Thread
         ▼
┌───────────────────────┐
│ Female Insert (Top)   │
│ (Epoxied into rod)    │
└──────────┬────────────┘
           │
   [ROD SEGMENT 1]
   (Fiberglass Tube)
           │
┌──────────┴────────────┐
│ Male Insert Plug      │
│ (Epoxied into rod)    │
└──────────┬────────────┘
           │ Thread (M12)
           ▼
┌───────────────────────┐
│ SENSOR MODULE         │ ◄── 3D Printed Part
│ (Housing Coils/Rings) │
│                       │
│ Top: Female Thread    │
│ Btm: Epoxied to Rod 2 │
└──────────┬────────────┘
           │
   [ROD SEGMENT 2]
   (Fiberglass Tube)
           │
┌──────────┴────────────┐
│ Male Insert Plug      │
└──────────┬────────────┘
           │
           ▼
      [PROBE TIP]
```

## Dimensions

### Fiberglass Rod Segments
*   **Outer Diameter (OD):** 16 mm (approx. 5/8") - *Increased from 12mm for strength*
*   **Inner Diameter (ID):** 12–13 mm (standard pultruded tube)
*   **Wall Thickness:** ~1.5–2.0 mm
*   **Segment Lengths:**
    *   **Spacer Segments:** 50 cm, 100 cm (defines sensor spacing)
    *   **Top/Bottom Segments:** Variable
*   **Material:** Fiberglass (non-conductive, RF transparent)

## Connector Architecture: "Flush-Mount Modular Inserts"

The system uses a 2-part connector system that is permanently epoxied into the ends of rod segments to create a screw-together stack.

### 1. Male Insert (Thread Side)
*   **Function:** Provides the male thread for the joint.
*   **Geometry:**
    *   **Flange:** Inserts into fiberglass rod ID (epoxied).
    *   **Thread:** M12×1.75 or M14×2.0 (printed or cut).
    *   **Shoulder:** Matches Rod OD (16 mm) for flush fit.
*   **Wiring:** Hollow center for wire pass-through.

### 2. Female Insert / Sensor Module (Socket Side)
*   **Function:** Receives the male thread AND houses the sensors (Coils/Electrodes).
*   **Geometry:**
    *   **Flange:** Inserts into fiberglass rod ID (epoxied).
    *   **Body:** Extended section matching Rod OD (16 mm).
    *   **Thread:** Female internal thread.
*   **Sensor Integration:**
    *   **MIT Coils:** Wound directly onto the printed module body (or on a ferrite core embedded within).
    *   **ERT Rings:** Conductive bands (copper/steel) mounted in grooves on the module body.

## Wiring Path
*   **Center Channel:** All wiring runs up the center of the hollow fiberglass rods and through the hollow center of the connectors.
*   **Assembly:** Wires are threaded through as segments are screwed together.

## Advantages of New Design
1.  **Strength:** 16 mm OD allows for robust M12/M14 threads (unlike fragile M10 on 12mm rods).
2.  **Modularity:** Sensor spacing is determined by rod segment length. Need sensors every 50cm? Use 50cm rods.
3.  **Manufacturability:** Sensors are built into the printed parts, not glued onto the rod surface.
4.  **Smooth Profile:** Flush connections prevent snagging during insertion/extraction.

## Trade-offs
*   **Slightly Larger Hole:** 16 mm rod requires ~18-20 mm hole (vs 14 mm for 12 mm rod). Still significantly smaller than original 25-50 mm designs.
*   **Assembly Time:** Requires screwing segments together on-site (or transporting long assemblies).

## Manufacturing Notes
*   **3D Printing:** PETG or ASA recommended for impact resistance and UV stability.
*   **Threads:** Print threads horizontally if possible, or use high wall count/100% infill for vertical printed threads.
*   **Sealing:** O-rings should be included at the thread shoulder to seal the internal wiring channel from groundwater.
