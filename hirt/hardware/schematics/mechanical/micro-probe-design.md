# Micro-Probe Design - Architecture Overview

## Design Philosophy

**"Archaeologist brain first, engineer brain second"**

- **Goal:** Smallest possible hole
- **Reality:** Need enough physics (coil area, electrode contact) for good signal
- **Constraint:** Many thin, gentle holes rather than few big ones

## Core Design Principles

1. **No big electronics at the tip**
   - PCBs, ADCs, MCUs → **stay at surface**
   - Only passive components downhole

2. **Probes are mostly passive**
   - Downhole: coils and electrodes + thin wiring
   - Everything "smart" happens in junction box above ground

3. **Rod diameter target: 10-16 mm OD**
   - 10-12 mm = like a skinny tent pole / hiking pole
   - Hole size: 12-18 mm (much less destructive than 50 mm)

## Mechanical Design

### Rod Structure

**Material:** Fiberglass or carbon-fiber tube/rod
- **OD:** 10-16 mm (target: 12 mm)
- **Wall Thickness:** 1-2 mm
- **Length:** 1.5 m segments, threaded joiners to reach 3 m
- **Weight:** ~50-100 g per meter (much lighter than 25mm design)

### Downhole Hardware

**Coils:**
- 1-2 ferrite cores with coils epoxied into recesses or along rod
- Ferrite: 6-8 mm diameter × 40-80 mm long
- Wound with fine wire (34-38 AWG)
- Glued along rod (not around it in a bulge)
- Pot with thin epoxy layer to keep OD small

**Electrodes:**
- 2-3 narrow ring electrodes (stainless/copper bands)
- Width: 3-5 mm (narrow bands)
- Bonded around rod with epoxy
- Connected by thin twisted pair inside rod wall

**Cable:**
- Thin multi-core cable inside rod
- Connects coil & rings to surface junction box
- No electronics in probe (passive only)

### Surface Hardware

**Junction Box (per probe or shared):**
- Small IP-rated box at top of rod
- Terminal block for coil and electrode leads
- Optional small buffer amp
- Multi-probe cable harness to central electronics

**Central Electronics:**
- MIT driver/receiver (rack or Pelican case)
- ERT current source + voltmeter
- Data acquisition system
- Power distribution

## Cross-Section View

```
Top view (schematic):

[soil] ████████████████
       │  o      o      │ ← ring electrodes (ERT)
       │    ||||||      │ ← slim ferrite + coil along rod
       │     ROD        │
       ████████████████

Side view:

Surface ────────────────
         │
         │ Junction Box
         │ (electronics)
         │
         │
         │ Rod (12mm OD)
         │
         │  [Coil] ← ferrite core
         │
         │  [Ring] ← ERT electrode
         │
         │  [Ring] ← ERT electrode
         │
         │
         └─ Tip (tapered)
```

## Performance Trade-offs

### Smaller Radius → Smaller Coil Area

**Compensation Strategies:**
- More turns on coil (fine wire, many turns)
- Lower frequency (2-10 kHz for deeper penetration)
- Longer integration time (lock-in detection can average more)
- Careful noise control

**Result:** Sacrifice some SNR to gain low intrusion. Acceptable with longer dwell times and careful design.

### ERT Contact

**Advantages:**
- Narrow rings (3-5 mm) barely increase OD
- With slurry/water in hole, excellent contact
- No need for big metal shoes
- Better contact than dry insertion

## Array Layout

**Field Geometry (unchanged):**
- 10×10 m section, 2 m spacing
- Can go denser: 1-1.5 m spacing
- Each probe: pencil-thick rod (12 mm)

**Visual Impact:**
- Looks like tent stakes, not construction
- Minimal visual disturbance
- Easy to backfill after removal

## Disturbance Comparison

### Micro-Probe (12 mm rod, 14-16 mm hole)
- Cross-section: ~1.5-2.0 cm²
- 3 m depth: ~0.5 liters per hole
- 25 probes: ~12-15 liters total

### Old Design (25 mm rod, 50 mm hole)
- Cross-section: ~19.6 cm²
- 3 m depth: ~6 liters per hole
- 25 probes: ~150 liters total

**Reduction:** ~10× less disturbance

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

## Novel Solutions

### Collapsible Sleeve Probe
- Drive thin pilot rod (8-10 mm) to depth
- Withdraw, leaving narrow path
- Insert slightly larger composite probe (12-14 mm)
- Compressible foam/rubber around electrodes for contact
- Removable, leaves almost no permanent void

### Cluster of Micro-Spikes
- Instead of one 50 mm probe, use 3-4 × 6-8 mm spikes
- Small cluster (local sub-array)
- Very small disturbance
- Flexible geometry

## 3D Printing Requirements

**For Micro-Probe:**
- **Tapered nose cone** (to help insertion)
- **Small top cap/enclosure** (for terminal block at surface)
- **No large electronics pod** (electronics stay at surface)

**Example Design:**
- 12 mm OD fiberglass rod
- 40 mm long, 8 mm diameter ferrite glued along rod at 0.8 m depth
- 6 mm wide stainless ring at 0.5 m, another at 1.5 m
- Small 3D-printed cap at top for cable strain relief and labeling

## Advantages of Micro-Probe Design

1. **Minimal Intrusion**
   - ~10× less disturbance than 25mm design
   - Acceptable for sensitive archaeological contexts

2. **Easy Insertion**
   - Lightweight, easy to handle
   - Can use hand auger or pilot rod
   - Minimal force required

3. **Better Contact**
   - Slurry/water in hole improves ERT contact
   - Narrow rings work well with good contact

4. **Flexible Deployment**
   - Can go denser spacing (1-1.5 m)
   - Easy to remove and backfill
   - Minimal visual impact

5. **Simplified Electronics**
   - Centralized electronics (easier maintenance)
   - Passive probes (more reliable)
   - Easier troubleshooting

## Trade-offs

1. **Smaller Coil Area**
   - Lower signal strength
   - Compensate with more turns, lower frequency, longer integration

2. **Centralized Electronics**
   - More cables to surface
   - Central failure point (mitigate with redundancy)

3. **Thinner Rod**
   - Less robust (but adequate for field use)
   - May need careful handling

---

*This design keeps the science of HIRT while making the mechanics feel like archaeology, not construction.*

