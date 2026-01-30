# 7. Assembly and Wiring

## Overview

This section provides comprehensive step-by-step assembly instructions for the HIRT modular micro-probe system (16mm). The modular design allows probes to be built by stacking fiberglass rod segments and 3D-printed sensor modules.

---

## Table of Contents

1. [Parts List](#parts-list)
2. [Tools Required](#tools-required)
3. [Preparation Steps](#preparation-steps)
4. [Rod Segment Preparation](#rod-segment-preparation)
5. [Sensor Module Assembly](#sensor-module-assembly)
6. [Field Assembly Procedure](#field-assembly-procedure)
7. [Zone Wiring Architecture](#zone-wiring-architecture)
8. [Quality Checks](#quality-checks)
9. [Troubleshooting](#troubleshooting)

---

## Parts List

### Printed Parts (PETG/ASA)

| Part | Qty per Probe | Notes |
|------|---------------|-------|
| Male Insert Plug | 2 | Threaded male screw end |
| Sensor Module (Female) | 2-3 | Sensor body with female threads |
| Probe Tip | 1 | Pointed nose cone |
| Top Cap | 1 | Cable exit/handle |

### Hardware

| Part | Qty per Probe | Notes |
|------|---------------|-------|
| Fiberglass Tube | 2-3 sections | 16mm OD x 12mm ID |
| Epoxy | As needed | 2-part structural (Loctite Marine, JB Weld) |
| O-rings | 4-6 | Size for M12 thread shoulder (10mm ID x 1.5mm) |
| Ferrite Cores | 1-2 | For MIT coils (6-8mm x 40-80mm) |
| Magnet Wire | 10-20m | 30-34 AWG for coil winding |
| ERT Ring Material | 2-3 bands | Stainless steel or copper tape |
| Cable | 3-5m | Multi-conductor shielded |

---

## Tools Required

### Essential Tools

| Tool | Purpose |
|------|---------|
| Hacksaw or Tube Cutter | Cutting fiberglass tubing |
| 400-grit Sandpaper | Surface preparation |
| M12x1.75 Tap | Threading female parts |
| M12x1.75 Die | Threading male parts |
| Tap Handle | For tap operation |
| Mixing Cups | For epoxy |
| Nitrile Gloves | Epoxy handling |
| Soldering Iron | Wire connections |
| Multimeter | Testing continuity |
| Calipers | Measuring dimensions |

### Recommended Additional Tools

| Tool | Purpose |
|------|---------|
| Bench Vise | Holding parts during tapping |
| Thread Cutting Oil | Lubrication for tap/die |
| LCR Meter | Coil testing |
| Heat Gun | Heat shrink tubing |
| Wire Strippers | Cable preparation |

---

## Preparation Steps

### Step 1: Inspect All Parts

Before assembly, verify:
- [ ] All printed parts present and undamaged
- [ ] No visible layer separation or cracks in prints
- [ ] Fiberglass tubing is straight and correct length
- [ ] All hardware components available
- [ ] Epoxy not expired

### Step 2: Prepare Workspace

- Clean, well-lit, well-ventilated area
- Protect work surface from epoxy
- Organize parts by assembly stage
- Have cleaning supplies ready (IPA, paper towels)

### Step 3: Test-Fit Parts (Dry Run)

Before committing with epoxy:
1. Test thread engagement on all threaded parts
2. Verify tube sections fit inserts
3. Check O-ring sizing
4. Identify any parts needing adjustment

---

## Rod Segment Preparation

### Step 1: Cut Tubing

1. Mark fiberglass tubing at desired lengths:
   - 50 cm segments for sensor spacing
   - 100 cm segments for longer runs

2. Cut with hacksaw or tube cutter
   - Use steady, even strokes
   - Rotate tube to keep cut square

3. Deburr cut ends
   - Remove fiberglass fibers with file
   - Sand smooth with 400-grit

### Step 2: Prepare Tube Ends

1. Sand inner diameter (ID) of tube ends
   - Creates better bonding surface
   - Use rolled sandpaper or dowel with sandpaper

2. Clean with isopropyl alcohol
   - Remove dust and oils
   - Allow to dry completely

### Step 3: Install Inserts

**For a standard extension segment:**

1. **Bottom End:** Apply epoxy to flange of Male Insert Plug
   - Apply thin, even coat to flange
   - Insert into tube end
   - Ensure shoulder sits flush against tube cut
   - Wipe excess epoxy

2. **Top End:** Apply epoxy to Female Insert flange
   - Same process as bottom end
   - Verify proper orientation (female at top)

**Convention:** Rods have Male threads at bottom, Female at top

3. **Cure:** Let epoxy cure fully (24 hours) before stressing threads

---

## Sensor Module Assembly

### MIT Coil Integration

If the sensor module contains MIT coils:

1. **Select Ferrite Core**
   - Match core to recess in printed module
   - 6-8mm diameter x 40-80mm length typical

2. **Wind Coil**
   - Clean ferrite surface
   - Apply thin layer of varnish to winding area
   - Wind 150-200 turns of 30-34 AWG wire
   - Maintain even tension, layer neatly
   - Leave 10-15cm leads at each end

3. **Secure Coil**
   - Apply varnish or thin epoxy over windings
   - Allow to cure
   - Verify inductance (target: 1-2 mH)

4. **Install in Module**
   - If external: glue into recessed channel
   - If internal: epoxy in place within module
   - Route wire leads through center channel

### ERT Ring Integration

If the sensor module contains ERT electrodes:

1. **Prepare Ring Material**
   - Cut stainless steel or copper tape to length
   - Width: 3-5 mm
   - Length: circumference + 3mm overlap

2. **Install Ring**
   - Clean electrode groove on printed module
   - Wrap conductive material into groove
   - Overlap ends by 3mm
   - Secure with miniature screw, rivet, or conductive epoxy

3. **Solder Lead Wire**
   - Strip and tin wire end
   - Solder to ring (use flux)
   - Cover joint with heat shrink for strain relief

4. **Test Isolation**
   - Verify >1 M-ohm between rings
   - Verify >1 M-ohm to ground/rod

---

## Field Assembly Procedure

### Assembly Sequence

**Start from Bottom (Tip) and work up:**

### Step 1: Prepare Probe Tip

1. Inspect probe tip for damage
2. Verify thread engagement
3. Set aside (will attach first)

### Step 2: Thread Cable Through First Segment

1. Take main cable harness
2. Thread through first rod segment (bottom to top)
3. Leave adequate working length at bottom for tip connection

### Step 3: Attach First Segment to Tip

1. Apply thread sealant to tip threads
2. Screw rod segment onto tip
3. Tighten firmly (hand tight + 1/4 turn)
4. Verify no gaps at joint

### Step 4: Add First Sensor Module

1. Thread cable through sensor module
2. Connect sensor pigtails to main harness:
   - Solder connections, or
   - Use micro-connectors
3. **Install O-ring** on thread shoulder
4. Screw sensor module onto rod segment
5. Tighten to compress O-ring (do not over-tighten)

### Step 5: Continue Stacking

Repeat for each additional segment:
1. Thread cable through next rod segment
2. Screw onto previous sensor module
3. Add next sensor module
4. Continue until full depth reached

### Step 6: Install Top Cap

1. Thread cable through top cap
2. Screw top cap onto final rod segment
3. Secure cable strain relief
4. Test all connections with multimeter

---

## Zone Wiring Architecture (Scalability Solution)

To manage the cabling complexity of 20-50 passive probes (where each probe has 8-12 conductors), we use a **Zone Wiring Strategy**.

### The Problem:
Connecting 50 probes directly to the Main Hub requires ~600 conductors entering one box. This is mechanically impossible and an EMI nightmare.

### The Solution: "Breakout Zones"
We divide the probe array into "Zones" of 4 probes each.
- **Probes 1-4** connect to **Zone Box A** (Small passive hub on the ground).
- **Zone Box A** connects to **Main Hub** via a single high-density **Trunk Cable**.

### System Topology

```
[Probe 1] --\
[Probe 2] ---+--> [Zone Box A] =====(Trunk Cable A)=====\
[Probe 3] ---|                                          |
[Probe 4] --/                                           |
                                                        |
[Probe 5] --\                                           +===> [MAIN HUB]
[Probe 6] ---+--> [Zone Box B] =====(Trunk Cable B)=====/
...           |                                         |
              |                                         |
              ... (Up to 8-12 Zones) .................../
```

### Components

1.  **Zone Box:**
    *   Small IP65 enclosure (100x100mm).
    *   **Inputs:** 4x Cable Glands (from Probes).
    *   **Output:** 1x DB25 or Centronics connector (to Trunk).
    *   **Function:** Passive pass-through. Aggregates signals.

2.  **Trunk Cable:**
    *   High-quality shielded multi-core cable (e.g., 25-conductor or 37-conductor).
    *   Length: 10-20 meters (shielded to protect analog signals).
    *   Carries signals for 4 probes simultaneously back to the Main Hub.

### Advantages
*   **Modular:** Need more probes? Add another Zone Box.
*   **Clean:** Only 5-6 Trunk Cables enter the Main Hub instead of 20+.
*   **Field-Ready:** Deploy Zone Boxes near their probe clusters; run only Trunks to the tent/vehicle.

---

## Wiring Diagram

### Probe Internal Routing

- Use **thin multi-core shielded cable** (2-4 mm diameter)
- Route inside rod (center or along wall)

### Connector Pinout Reference (Probe Side)

**12-pin Phoenix Contact (1757248) or similar:**

| Pin | Signal | Wire Color (Suggested) |
|-----|--------|------------------------|
| 1 | TX+ | Red |
| 2 | TX- | Black |
| 3 | RX+ | White |
| 4 | RX- | Green |
| 5 | Guard | Yellow |
| 6 | Ring A | Blue |
| 7 | Ring B | Orange |
| 8 | Ring C | Brown |
| 9 | ID Sense | Purple |
| 10 | Spare+ | Gray |
| 11 | Spare- | Pink |
| 12 | Shield | Bare/Drain |

*These 12 pins are multiplied by 4 in the Zone Box to feed the Trunk Connector.*

---

## Quality Checks

### During Assembly

Check at each joint:
- [ ] O-ring properly seated
- [ ] No gaps between components
- [ ] Threads fully engaged
- [ ] Cable has slack (not stretched)
- [ ] Joints are flush (smooth to touch)

### After Complete Assembly

**Mechanical Checks:**
- [ ] Total length correct
- [ ] All joints tight
- [ ] Probe straight (no bends)
- [ ] Cable secure at strain relief

**Electrical Checks:**
- [ ] All coil leads continuous (no opens)
- [ ] Coil inductance in spec (1-2 mH)
- [ ] ERT ring isolation >1 M-ohm
- [ ] No shorts between conductors
- [ ] Shield continuity verified

**Seal Check:**
- [ ] Run hand over all joints - should be smooth
- [ ] No visible gaps at O-ring locations
- [ ] Cable gland tight

---

## Waterproofing

### Probe (Downhole)

- **Coil potting:** Thin epoxy layer over coils
- **Ring sealing:** Epoxy seal around ring edges
- **Cable routing:** Sealed inside rod
- **Rod joints:** Thread sealant on couplers

### Surface Junction Box

- **Weatherproof enclosure:** IP65 rated
- **Cable glands:** Seal cable entries
- **Terminal block:** Protected from moisture

---

## Troubleshooting

### Thread Issues

**Problem:** Threads too tight, won't engage
- **Solution:** Chase threads with tap/die
- **Prevention:** Print at correct tolerance, post-process

**Problem:** Threads too loose, won't tighten
- **Solution:** Apply thread sealant (Teflon tape or liquid)
- **Alternative:** Replace part if severely oversized

**Problem:** Threads stripped during tightening
- **Solution:** Replace part
- **Prevention:** Don't over-tighten, hand-tight + 1/4 turn max

### Fit Issues

**Problem:** Insert won't fit in tube
- **Solution:** Sand insert flange slightly
- **Check:** Tube ID may be undersized

**Problem:** Insert loose in tube
- **Solution:** Use more epoxy, shim if needed
- **Check:** Tube ID may be oversized

### Electrical Issues

**Problem:** Open circuit in coil
- **Solution:** Check for broken wire, resolder connections
- **Check:** Wire may be damaged during assembly

**Problem:** Short between rings
- **Solution:** Check for bridging material, clean
- **Check:** Conductive debris in gaps

**Problem:** Low coil Q factor
- **Solution:** Rewrap coil more neatly, use better wire
- **Check:** Shorted turns from damaged insulation

---

## Assembly Tips

### General Tips

1. **Work clean:** Fiberglass dust and epoxy don't mix well
2. **Test often:** Verify continuity at each stage
3. **Don't rush:** Allow full cure time for epoxy
4. **Label everything:** Mark probe ID on each segment
5. **Document:** Record any deviations or issues

### Epoxy Tips

1. Mix thoroughly (2+ minutes)
2. Apply thin coats - too much is messy
3. Work in well-ventilated area
4. Clean up drips immediately
5. Allow full cure before stressing joints

### Thread Tips

1. Use cutting oil with tap/die
2. Back out tap every half-turn to clear chips
3. Don't force - if stuck, back out and clear
4. Test fit with mating part before committing
5. Apply light lubricant before assembly

---

## Post-Assembly

After complete assembly:
1. Complete [Testing Procedures](08-testing-verification.md)
2. Complete [Calibration Procedures](09-calibration.md)
3. Label and register probe
4. Store properly until deployment

---

## Base Hub Assembly Notes

### Stack Overview

```
+-----------------------------+
|   Weatherproof enclosure    |  Bud NBF-32016 or similar
+-----------------------------+
|  Front panel (Trunk Ports)  |  DB25/Centronics Connectors (High Density)
+-----------------------------+
|  Backplane PCB (160x120mm)  |  DDS, TX/RX, ERT, MCU
+-----------------------------+
|  Harness strain relief bar  |  Internal ribbon cables
+-----------------------------+
|  Power shelf (battery/fuse) |  LiFePO4, fuse block
+-----------------------------+
```

### Backplane Zones

| Zone | Function | Key Parts |
|------|----------|-----------|
| DDS/TX Driver | Sweep generation | AD9833, OPA454 |
| RX Front End | Signal conditioning | AD620, INA128 |
| ERT Source | Current injection | OPA177, REF5025 |
| Lock-In/ADC | Digitization | ADS1256 |
| Control/Sync | Scheduling, logging | ESP32, USB-UART |
| Power | Regulation | DC-DC, LDO |

### Front Panel Layout

1. **Zone Inputs:** 6x DB25 Female Connectors (Support up to 24 probes)
2. **Control:** USB-C (Data), SMA (GPS Antenna)
3. **Power:** Anderson Powerpole (Battery Input), Main Switch

---

*For testing procedures, see Section 8: Testing and Verification. For calibration, see Section 9: Calibration.*
