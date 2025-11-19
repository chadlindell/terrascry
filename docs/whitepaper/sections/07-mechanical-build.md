# 7. Mechanical Build (step‑by‑step) - Micro-Probe Design

## Design Philosophy

**Micro-Probe Architecture:** Passive probes with surface electronics. Rod diameter: 10–16 mm (target: 12 mm) for minimal intrusion.

## 7.1 Rods & Couplers

### Step 1: Choose Rod Material
- **Fiberglass** (preferred): High strength, non-conductive, durable, lightweight
- **Carbon-fiber** (premium): Very high strength, lighter than fiberglass
- **PVC** (budget): Lower cost, adequate for shallow deployments
- **Avoid metal** for the sensor rod (interferes with measurements)

### Step 2: Cut to Length
- Standard sections: **1.5 m** each (or 1.0 m for flexibility)
- **Rod diameter:** 12 mm OD (target), range 10–16 mm
- **Wall thickness:** 1–2 mm
- Prepare **M/F threaded couplers** (M12×1.5 thread)
  - Glass‑filled nylon (preferred)
  - 3D‑printed PETG (alternative)

### Step 3: Pilot Rod
- Keep a **separate steel pilot rod** (8–10 mm diameter) for making holes
- **Never leave metal in place** for measurements
- Remove pilot rod before inserting sensor rod
- **Hole size:** 12–16 mm after pilot rod + wiggle

## 7.2 Probe Tip (Nose Cone)

### Step 1: Tapered Tip
- **3D‑printed or machined nose cone**
  - Base diameter: 12 mm (matches rod OD)
  - Tip diameter: 3–5 mm (sharp point)
  - Length: 20–30 mm
  - Material: PETG or ABS

### Step 2: Mounting
- Thread or press-fit to rod end
- Secure with epoxy if needed
- **Purpose:** Easy insertion into soil

## 7.3 MIT Coils (Passive, Along Rod)

### Step 1: Prepare Ferrite Cores
- **Ferrite cores:** 6–8 mm diameter × 40–80 mm long
- Two cores needed: one for TX, one for RX
- Source from Mouser or specialty suppliers

### Step 2: Wind Coils
- Wind **TX coil** (target **1–2 mH**) on ferrite rod
  - Use fine wire: 34–38 AWG
  - Many turns (200–400) to compensate for smaller size
  - Secure windings with epoxy or varnish
- Wind **RX coil** similarly
- Measure inductance and Q factor

### Step 3: Mount Coils on Rod
- **Glue coils along rod** (not in bulky head)
- Position at desired depth (typically 0.5–1.0 m from tip)
- **Orientation:** Orthogonal (90° separation) to reduce direct coupling
- Pot with thin epoxy layer to keep OD small
- **Result:** Coil adds only ~1–2 mm to rod OD

### Step 4: Wire Connections
- Connect coil wires to thin multi-core cable
- Route cable inside rod (center or along wall)
- Connect to surface junction box (not to electronics in probe)

## 7.4 ERT Rings (Narrow Bands)

### Step 1: Ring Positions
- Standard positions: **0.5 m** and **1.5 m** from tip
- Add **2.5–3.0 m** ring for longer rods (3 m deployment)

### Step 2: Prepare Rings
- Cut **narrow bands** (3–5 mm wide) from stainless steel or copper foil
- Thickness: 0.5–1 mm
- **Material:** 304SS or copper sheet

### Step 3: Mount Rings
- Use **3D‑printed insulating collars** (5 mm wide × 12 mm ID)
- Bond rings to collars with epoxy
- Mount collars on rod at specified positions
- Ensure rings are **flush with rod surface** (minimal OD increase)
- Run **thin twisted pair** inside rod wall to surface

### Step 4: Isolation
- Ensure rings are **electrically isolated** from each other
- Prevent moisture‑bridging between rings
- Test with multimeter before deployment (should be >1 MΩ)

## 7.5 Surface Junction Box

### Step 1: Junction Box
- **3D‑printed or purchased enclosure**
  - Diameter: Ø 25 mm
  - Height: 30–40 mm
  - Material: PETG or ABS, weatherproof
  - Mount at top of rod

### Step 2: Terminal Block
- Install **terminal block** for coil and electrode leads
- Organize connections clearly
- Label probe ID

### Step 3: Cable Connections
- Connect probe cable to **multi-probe harness**
- Route to central electronics hub
- Provide **strain relief** at junction box
- Seal cable entry (cable gland or epoxy)

### Step 4: No Electronics
- **No electronics in junction box** (just connections)
- All electronics in central hub at surface
- This keeps probe lightweight and simple

## 7.6 Cable Routing

### Step 1: Internal Cable
- Use **thin multi-core shielded cable** (2–4 mm diameter)
- Route inside rod (center or along wall)
- Cable connects:
  - TX coil → central TX driver
  - RX coil → central RX amplifier
  - ERT rings → central ERT system

### Step 2: Cable Management
- Secure cable inside rod (strain reliefs every 30–50 cm)
- Exit at surface junction box
- Connect to multi-probe harness
- Avoid sharp bends

## 7.7 Assembly Sequence

1. **Prepare rod sections** (cut, thread if needed)
2. **Mount probe tip** (tapered nose cone)
3. **Wind and mount coils** (glue along rod)
4. **Mount ERT rings** (narrow bands on collars)
5. **Route cable** (inside rod)
6. **Install surface junction box** (terminal block, connections)
7. **Connect to central hub** (multi-probe harness)
8. **Test and label** (verify connections, label probe ID)

## 7.8 Waterproofing

### Probe (Downhole)
- **Coil potting:** Thin epoxy layer over coils
- **Ring sealing:** Epoxy seal around ring edges
- **Cable routing:** Sealed inside rod
- **Rod joints:** Thread sealant on couplers

### Surface Junction Box
- **Weatherproof enclosure:** IP65 rated
- **Cable glands:** Seal cable entries
- **Terminal block:** Protected from moisture

## 7.9 Quality Control

Before field deployment:
- [ ] Verify rod dimensions (12 mm OD)
- [ ] Test coil inductance and Q factor
- [ ] Confirm ERT ring isolation (>1 MΩ)
- [ ] Verify all connections at junction box
- [ ] Test cable continuity
- [ ] Check waterproofing integrity
- [ ] Verify probe ID labeling
- [ ] Test connection to central hub

## 7.10 Advantages of Micro-Probe Design

1. **Simpler Assembly:** No electronics in probe
2. **Lighter Weight:** ~50–100 g per meter (vs 200–250 g)
3. **Easier Insertion:** Smaller diameter, less force needed
4. **Lower Cost:** ~$40–60 per probe (vs $130–180)
5. **Better Reliability:** Passive probes more robust
6. **Minimal Intrusion:** ~10× less disturbance than 25mm design

---

*For electronics assembly, see Section 8 (Central Electronics Hub).*
