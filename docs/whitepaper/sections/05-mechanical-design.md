# 5. Mechanical Design

## Overview

This section consolidates all mechanical design specifications for the HIRT probe system, including the micro-probe architecture, rod specifications, component drawings, and manufacturing procedures for 3D printed parts.

---

## Design Philosophy

**"Archaeologist brain first, engineer brain second"**

- **Goal:** Smallest possible hole
- **Reality:** Need enough physics (coil area, electrode contact) for good signal
- **Constraint:** Many thin, gentle holes rather than few big ones
- **Result:** 10x less disturbance than conventional designs

### Core Design Principles

1. **No big electronics at the tip**
   - PCBs, ADCs, MCUs stay at surface
   - Only passive components downhole

2. **Probes are mostly passive**
   - Downhole: coils and electrodes + thin wiring
   - Everything "smart" happens in junction box above ground

3. **Rod diameter target: 10-16 mm OD**
   - 10-12 mm = like a skinny tent pole / hiking pole
   - Hole size: 12-20 mm (much less destructive than 50 mm)

### Micro-Probe Architecture

**Passive probes with surface electronics.** Rod diameter: 16 mm OD for optimal strength-to-intrusion ratio.

```
Top view (schematic):

[soil] ================
       |  o      o    | <-- ring electrodes (ERT)
       |    ||||||    | <-- slim ferrite + coil along rod
       |     ROD      |
       ================

Side view:

Surface ----------------
         |
         | Junction Box
         | (electronics)
         |
         |
         | Rod (16mm OD)
         |
         |  [Coil] <-- ferrite core
         |
         |  [Ring] <-- ERT electrode
         |
         |  [Ring] <-- ERT electrode
         |
         |
         +-- Tip (tapered)
```

### Disturbance Comparison

**Micro-Probe (16 mm rod, 18-20 mm hole):**
- Cross-section: ~2.5-3.0 cm^2
- 3 m depth: ~0.75-1.0 liters per hole
- 25 probes: ~20-25 liters total

**Old Design (25 mm rod, 50 mm hole):**
- Cross-section: ~19.6 cm^2
- 3 m depth: ~6 liters per hole
- 25 probes: ~150 liters total

**Reduction:** ~7-10x less disturbance

---

## Key Dimensions

| Component | Dimension | Notes |
|-----------|-----------|-------|
| Rod OD | 16 mm | Increased from 12mm for strength |
| Rod ID | 12-13 mm | Standard pultruded tube |
| Wall Thickness | 1.5-2.0 mm | |
| Segment Length | 50, 100 cm | Defines sensor spacing |
| Hole Size | 18-20 mm | For 16mm rod |
| Total Length | 2.5-3.0 m | Multiple segments |

---

## Rod Segments and Couplers

### Rod Material Selection

| Material | Pros | Cons | Recommendation |
|----------|------|------|----------------|
| **Fiberglass** | High strength, non-conductive, durable, lightweight | Moderate cost | **Preferred** |
| Carbon-fiber | Very high strength, lighter | Conductive, expensive | Premium option |
| PVC | Lower cost | Lower strength | Budget/shallow only |
| Metal | Strong | Interferes with measurements | **Avoid** |

### Fiberglass Rod Specifications

| Parameter | Specification |
|-----------|---------------|
| Outer Diameter (OD) | 16 mm (approx. 5/8") |
| Inner Diameter (ID) | 12-13 mm |
| Wall Thickness | ~1.5-2.0 mm |
| Material | Fiberglass (non-conductive, RF transparent) |
| Segment Lengths | 50 cm, 100 cm (spacers) |

### Modular Design Overview

The system uses a 2-part connector system permanently epoxied into rod ends to create a screw-together stack.

### Stacked Assembly Visualization

```
      [TOP CAP]
         |
         | Thread
         v
+-------------------------+
| Female Insert (Top)     |
| (Epoxied into rod)      |
+-----------+-------------+
            |
   [ROD SEGMENT 1]
   (Fiberglass Tube)
            |
+-----------+-------------+
| Male Insert Plug        |
| (Epoxied into rod)      |
+-----------+-------------+
            | Thread (M12)
            v
+-------------------------+
| SENSOR MODULE           | <-- 3D Printed Part
| (Housing Coils/Rings)   |
|                         |
| Top: Female Thread      |
| Btm: Epoxied to Rod 2   |
+-----------+-------------+
            |
   [ROD SEGMENT 2]
   (Fiberglass Tube)
            |
+-----------+-------------+
| Male Insert Plug        |
+-----------+-------------+
            |
            v
      [PROBE TIP]
```

### Connector Architecture: Flush-Mount Modular Inserts

**1. Male Insert (Thread Side):**
- Function: Provides the male thread for the joint
- Flange: Inserts into fiberglass rod ID (epoxied)
- Thread: M12x1.75 (printed oversize at 12.2mm for Die cutting)
- Shoulder: Matches Rod OD (16 mm) for flush fit
- Wiring: Hollow center for wire pass-through

**2. Female Insert / Sensor Module (Socket Side):**
- Function: Receives male thread AND houses sensors
- Flange: Inserts into fiberglass rod ID (epoxied)
- Body: Extended section matching Rod OD (16 mm)
- Thread: Female internal thread (printed undersize at 10.5mm for Tapping)
- Sensor Integration:
  - MIT Coils: Wound onto module body or embedded ferrite
  - ERT Rings: Conductive bands in grooves on module body

### Thread Specifications

**All Threads: M12x1.75 ISO**

| Parameter | Value |
|-----------|-------|
| Type | Standard metric thread |
| Pitch | 1.75 mm |
| Major Diameter | 12.0 mm |
| Engagement | 12-15 mm |
| Modification | "Chunky" profile for printability |

---

## Coil Mounting and Ferrite Cores

### Ferrite Core Selection

| Parameter | Specification |
|-----------|---------------|
| Diameter | 6-8 mm |
| Length | 40-80 mm |
| Material | MnZn ferrite |
| Quantity | 1-2 per probe (TX/RX) |

### Coil Winding

| Parameter | Specification |
|-----------|---------------|
| Wire Gauge | 34-38 AWG (fine wire) |
| Turns | 200-400 turns |
| Target Inductance | 1-2 mH |
| Target Q Factor | >20 |

### Mounting Procedure

1. **Glue coils along rod** (not in bulky head)
2. Position at desired depth (typically 0.5-1.0 m from tip)
3. **Orientation:** Orthogonal (90 degree separation) to reduce direct coupling
4. Pot with thin epoxy layer to keep OD small
5. **Result:** Coil adds only ~1-2 mm to rod OD

### Wire Connections

- Connect coil wires to thin multi-core cable
- Route cable inside rod (center or along wall)
- Connect to surface junction box (not to electronics in probe)

---

## ERT Ring Electrodes

### Ring Specifications

| Parameter | Value |
|-----------|-------|
| Material | Stainless steel or copper |
| Width | 3-5 mm (narrow bands) |
| Thickness | 0.5-1 mm |
| Diameter | Match rod OD (16 mm) |
| Quantity | 2-3 rings per probe |

### Mounting Positions

| Position | Distance from Tip |
|----------|-------------------|
| Ring A (Upper) | 0.5 m |
| Ring B (Mid) | 1.5 m |
| Ring C (Deep) | 2.5-3.0 m (optional) |
| Minimum Spacing | 0.3 m between rings |

### Mounting Method

1. Use **3D-printed insulating collars** (5 mm wide x 12 mm ID)
2. Bond rings to collars with epoxy
3. Mount collars on rod at specified positions
4. Ensure rings are **flush with rod surface** (minimal OD increase)
5. Run **thin twisted pair** inside rod wall to surface

### Electrical Isolation

- Ensure rings are **electrically isolated** from each other
- Prevent moisture-bridging between rings
- Test with multimeter before deployment (should be >1 M-ohm)

### Waterproofing

- Epoxy seal at ring edges
- Heat-shrink over connections
- Ensure no moisture bridging between rings
- Test insulation resistance (>1 M-ohm)

---

## Junction Box Design

### Surface Junction Box

| Dimension | Value |
|-----------|-------|
| Diameter | 25 mm |
| Height | 35 mm |
| Thread | Internal M12x1.5 (bottom) |
| Features | Terminal block mount, cable gland |

```
    +--------+
    | Cable  |   <-- Cable gland
    | Gland  |
    +--------+
    |        |
    |Terminal|   <-- Terminal block inside
    | Block  |
    |        |
    +--------+
    | Thread |   <-- Internal M12x1.5
    +--------+
       25mm
```

### Key Design Points

- **3D-printed or purchased enclosure**
- Material: PETG or ABS, weatherproof
- Mount at top of rod
- **No electronics in junction box** (just connections)
- All electronics in central hub at surface
- This keeps probe lightweight and simple

### Cable Connections

- Install **terminal block** for coil and electrode leads
- Organize connections clearly
- Label probe ID
- Connect probe cable to **multi-probe harness**
- Route to central electronics hub
- Provide **strain relief** at junction box
- Seal cable entry (cable gland or epoxy)

---

## 3D Printed Components

### CAD File Organization

```
hardware/cad/
|-- openscad/          # OpenSCAD source files (.scad)
|-- stl/               # 3D printable STL files
|-- step/              # CNC-ready STEP files
+-- docs/              # Manufacturing documentation
```

### File Formats

| Format | Purpose | Software |
|--------|---------|----------|
| .scad | Parametric source | OpenSCAD |
| .stl | 3D printing | Any slicer |
| .step | CNC machining | CAM software |
| .3mf | Print-ready | Bambu/Prusa |

### Available Parts

#### 1. Probe Tip (Nose Cone)

**File:** `openscad/micro_probe_tip.scad`
**Position:** BOTTOM (pointing down for insertion)

| Dimension | Value |
|-----------|-------|
| Total Length | 25 mm |
| Base Diameter | 12 mm |
| Point Diameter | 4 mm |
| Thread | Internal M12x1.5 |

```
        /\
       /  \
      /    \     <-- 4mm point
     /      \
    |        |
    |        |   <-- 25mm body
    |        |
    +--------+
    | Thread |   <-- Internal M12x1.5
    +--------+
       12mm
```

#### 2. Rod Coupler

**File:** `openscad/micro_rod_coupler.scad`
**Position:** MIDDLE (joins rod sections)

| Dimension | Value |
|-----------|-------|
| Total Length | 45 mm |
| Outer Diameter | 18 mm |
| Thread | Internal M12x1.5 both ends |
| Features | O-ring grooves, smooth center |

#### 3. Surface Junction Box (Probe Head)

**File:** `openscad/micro_probe_head.scad`
**Position:** TOP (at surface)

| Dimension | Value |
|-----------|-------|
| Diameter | 25 mm |
| Height | 35 mm |
| Thread | Internal M12x1.5 (bottom) |
| Features | Terminal block mount, cable gland |

#### 4. ERT Ring Collar

**File:** `openscad/micro_ert_ring_collar.scad`
**Position:** MIDDLE (slides onto rod)

| Dimension | Value |
|-----------|-------|
| Width | 5 mm |
| Inner Diameter | 12.0-12.2 mm |
| Features | Ring groove, wire channel |
| Mounting | Epoxy bond to rod |

#### 5. Modular Connector System (16mm)

**Files:**
- `modular_flush_connector.scad` - Source
- `modular_mixed_array_4x.stl` - Print-ready (2x Male + 2x Female)

| Specification | Value |
|---------------|-------|
| Rod Standard | 16mm OD / 12mm ID |
| Thread | M12x1.75 ISO |
| Wiring Channel | 6mm hollow center |

---

## Manufacturing Procedures

### 3D Printing Guide

#### Recommended Printer Settings

**For Bambu Lab A1 Mini (or similar FDM):**

| Setting | Value | Notes |
|---------|-------|-------|
| Material | PETG or ASA | Required for impact/UV |
| Layer Height | 0.12mm | Critical for threads |
| Infill | 100% (Solid) | Critical for strength |
| Walls | 6 Loops | Solid threaded regions |
| Supports | DISABLED | Use built-in scaffolding |
| Brim | Use Super Brim | Built into print file |
| Speed | 50mm/s outer wall | Quality over speed |

#### Material Selection

| Material | Best For | Notes |
|----------|----------|-------|
| PETG | General use | Waterproof, good strength |
| ASA | UV exposure | Better UV resistance |
| ABS | Strong parts | Requires heated bed |
| Nylon (GF) | Couplers | Highest strength |

#### Print Orientation

| Part | Orientation | Notes |
|------|-------------|-------|
| Probe Tip | Vertical, tip up | Minimize supports |
| Coupler | Horizontal | Both threads accessible |
| Junction Box | Vertical, thread down | Best thread quality |
| ERT Collar | Flat | Fastest, best bore |

#### Bed Adhesion

- Clean PEI or glass with 90% IPA
- Apply thin glue-stick film or hairspray
- Always add 6-10mm brim
- For tall parts, use stabilizer towers

### Post-Processing

#### Thread Cutting

**Critical:** 3D printed threads require post-processing.

**For Male Threads (M12x1.75):**
1. Print oversize at 12.2mm
2. Use M12x1.75 Die
3. Apply cutting oil
4. Cut slowly, back out frequently
5. Test fit with female part

**For Female Threads (M12x1.75):**
1. Print undersize at 10.5mm hole
2. Use M12x1.75 Tap
3. Apply cutting oil
4. Cut slowly, clear chips frequently
5. Test fit with male part

#### Epoxy Hardening

While parts are still warm:
1. Wick thin epoxy or CA into threads/tips
2. Rotate tap/die once more to spread
3. Allow full cure (overnight)
4. This toughens surface for tapping

#### General Cleanup

1. Remove support material carefully
2. Sand flat surfaces if needed
3. Clean threads with tap/die
4. Remove brim with flush cutters
5. Test fit all mating parts

### CNC Machining Guide

#### When to Use CNC

| Part | Recommended | Notes |
|------|-------------|-------|
| Probe Tip | 3D Print OK | CNC for production |
| Coupler | CNC Preferred | Precision threads critical |
| Junction Box | 3D Print OK | PETG for weatherproof |
| ERT Collar | 3D Print OK | Simple part |

#### Material Selection for CNC

| Material | Application | Notes |
|----------|-------------|-------|
| Delrin | Tips, couplers | Easy to machine |
| Nylon | Structural parts | Good strength |
| Aluminum | Production parts | Most durable |

### Quality Control for Manufactured Parts

#### Dimensional Checks

Before using manufactured parts:
- [ ] Measure outer diameter
- [ ] Measure inner diameter
- [ ] Measure length
- [ ] Check wall thickness
- [ ] Verify thread fit with rod

#### Functional Tests

- [ ] Thread engagement (should be smooth)
- [ ] O-ring fit in grooves
- [ ] Cable gland fit
- [ ] Waterproofing (for junction box)
- [ ] ERT ring fit on collar

#### Manufacturing Defects

| Defect | Cause | Solution |
|--------|-------|----------|
| Threads too tight | Under-extrusion | Adjust flow, use tap |
| Threads too loose | Over-extrusion | Reprint, adjust tolerance |
| Layer separation | Poor adhesion | Increase temp, clean bed |
| Warping | Temperature | Use brim, adjust bed temp |
| Weak threads | Low infill | Use 100% infill |

---

## Insertion Methods

### Method 1: Hand Auger
- 10-20 mm hand auger
- Create hole, insert probe
- Backfill if needed

### Method 2: Pilot Rod
- 8-10 mm steel pilot rod
- Drive to depth
- Wiggle to 12-14 mm
- Remove pilot rod
- Insert probe
- **Never leave metal in place** for measurements

### Method 3: Direct Push (Sand)
- In sandy loam, may push probe directly
- Requires robust tip
- Minimal disturbance

### Method 4: Water-Jet (Sand)
- Small water lance to fluidize sand
- 12-15 mm column
- Slide probe in
- Water drains, sand squeezes back
- Excellent contact, low disturbance

---

## Assembly Sequence

### Order (Bottom to Top)

1. **Tip** - Screws onto rod bottom (pointing down)
2. **Rod Section 1** - 1.5m section with external threads
3. **ERT Collar + Ring** - Slides onto rod at 0.5m position
4. **Coupler** - Screws onto rod sections
5. **Rod Section 2** - Additional 1.5m section
6. **ERT Collar + Ring** - At 1.5m from tip
7. **ERT Collar + Ring** - At 2.5m from tip (if needed)
8. **Junction Box** - Screws onto rod top (at surface)

### Key Points

- All parts screw together (modular design)
- Tip points DOWN for insertion
- Junction box at TOP (surface, not inserted)
- Rod sections join with couplers
- ERT collars slide on and bond with epoxy
- Wires run through hollow center

---

## Advantages of Micro-Probe Design

1. **Strength:** 16 mm OD allows for robust M12 threads
2. **Modularity:** Sensor spacing determined by rod segment length
3. **Manufacturability:** Sensors built into printed parts, not glued onto rod
4. **Smooth Profile:** Flush connections prevent snagging during insertion/extraction
5. **Field Serviceable:** Replace individual segments without rebuilding entire probe
6. **Simpler Assembly:** No electronics in probe
7. **Lighter Weight:** ~50-100 g per meter (vs 200-250 g)
8. **Easier Insertion:** Smaller diameter, less force needed
9. **Lower Cost:** ~$40-60 per probe (vs $130-180)
10. **Better Reliability:** Passive probes more robust
11. **Minimal Intrusion:** ~10x less disturbance than 25mm design

---

*For assembly procedures, see Section 7: Assembly and Wiring. For electronics integration, see Section 6: Electronics and Circuits.*
