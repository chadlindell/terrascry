# Assembly Guide - Modular Micro-Probe System (16mm)

## Overview

This guide covers the assembly of the **16mm Modular Flush-Probe** system. Unlike previous versions where components were glued onto a single long rod, this system is built by stacking **Fiberglass Rod Segments** and **3D-Printed Sensor Modules**.

## Parts List

### Printed Parts (PETG/ASA)
1.  **Male Insert Plug:** The threaded male screw end.
2.  **Sensor Module (Female):** The sensor body with female threads.
3.  **Probe Tip:** The pointed nose cone.
4.  **Top Cap:** The cable exit/handle.

### Hardware
1.  **Fiberglass Tube:** 16mm OD x 12mm ID, cut to lengths (50cm, 100cm).
2.  **Epoxy:** 2-part structural epoxy (e.g., Loctite Marine, JB Weld).
3.  **O-rings:** Size to fit M12 thread shoulder (e.g., 10mm ID x 1.5mm).

## Step 1: Prepare Rod Segments

1.  **Cut Tubing:** Cut your 16mm fiberglass tubing to the desired segment lengths (e.g., 50cm for sensor spacing).
2.  **Clean Ends:** Sand the inner diameter (ID) of the tube ends to ensure a good glue bond. Clean with alcohol.

## Step 2: Install Inserts

**For a standard extension segment:**
1.  **Bottom End:** Epoxy a **Male Insert Plug** into one end. Ensure the shoulder sits flush against the tube cut.
2.  **Top End:** Epoxy a **Female Insert** (or another Male Insert depending on gender convention) into the other end.
    *   *Convention Recommendation:* Rods have Male threads at bottom, Female at top.

**Curing:** Let epoxy cure fully (24h) before stressing the threads.

## Step 3: Build Sensor Modules

The Sensor Module is a standalone 3D printed part that acts as a coupler between rods.

1.  **Wind Coils:** If this module contains an MIT coil, wind the magnet wire into the recessed channel on the printed body. Secure with varnish/superglue.
2.  **Install Electrodes:** If this module contains ERT rings, wrap the conductive tape/wire into the electrode grooves.
3.  **Pre-wire:** Solder short pigtails to the sensors.

## Step 4: Field Assembly

1.  **Start at Bottom:** Take the **Probe Tip** (or bottom rod with tip attached).
2.  **Thread Wires:**
    *   Take your main cable harness.
    *   Thread it through the **first Rod Segment**.
    *   Screw the rod segment onto the tip.
3.  **Add Sensor:**
    *   Thread cable through a **Sensor Module**.
    *   Connect the sensor pigtails to the main harness (solder/crimp or micro-connectors).
    *   Screw the Sensor Module onto the rod segment. *Don't forget the O-ring!*
4.  **Repeat:** Continue adding Rod Segments and Sensor Modules until the full depth is reached.
5.  **Top Cap:** Screw on the Top Cap and secure the cable strain relief.

## Critical Checks
*   **Sealing:** Ensure O-rings are present at every joint. Water entering the center channel can ruin the probe.
*   **Flush Fit:** Run your hand over joints. They should be smooth to prevent snagging on soil during insertion.
