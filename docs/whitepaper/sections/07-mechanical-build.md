# 7. Mechanical Build (step‑by‑step)

## 7.1 Rods & Couplers

### Step 1: Choose Rod Material
- **Fiberglass** (preferred): High strength, non-conductive, durable
- **PVC** (rigid): Lower cost, adequate for shallow deployments
- **Avoid metal** for the sensor rod (interferes with measurements)

### Step 2: Cut to Length
- Standard sections: **1.0 m** each
- Prepare **M/F threaded couplers**
  - Glass‑filled nylon (preferred)
  - 3D‑printed PETG with brass inserts (alternative)

### Step 3: Pilot Rod
- Keep a **separate steel pilot rod** (8–12 mm diameter) for making holes
- **Never leave metal in place** for measurements
- Remove pilot rod before inserting sensor rod

## 7.2 Probe Head (Nose Capsule)

### Step 1: Housing
- **PVC or 3D‑printed capsule**
  - Diameter: Ø 25–32 mm
  - Length: 80–120 mm
  - Removable cap for service access

### Step 2: Mount Coils
- Wind **TX coil** (target **1–2 mH**) on ferrite rod
- Wind **RX coil** similarly
- Glue inside capsule at **orthogonal orientation** or slightly separated
- **Purpose:** Reduce direct coupling between TX and RX

### Step 3: Electronics Pod
- Small PCB with:
  - DDS sine generator
  - TX driver amplifier
  - RX low-noise amplifier (LNA)
  - ADC/MCU
- Pot with **neutral cure silicone** (serviceable if needed)

### Step 4: Cable Pass‑through
- Feed shielded cable up the rod
- Seal with **cable gland + epoxy**
- Label probe ID clearly

## 7.3 ERT Rings

### Step 1: Ring Positions
- Standard positions: **0.5 m** and **1.5 m** from tip
- Add **2.5–3.0 m** ring for longer rods (3 m deployment)

### Step 2: Mount Rings
- Wrap thin stainless strip or copper band (10–12 mm wide) around insulating collar
- Run a **twisted pair** inside rod wall
- Seal joints with epoxy and heat‑shrink

### Step 3: Isolation
- Ensure rings are **electrically isolated** from each other
- Prevent moisture‑bridging between rings
- Test with multimeter before deployment

## 7.4 Waterproofing & Strain Relief

### Waterproofing
- Conformal coat on PCBs
- Potting in capsule (neutral cure silicone)
- Heat‑shrink over all junctions
- Test in water before field use

### Strain Relief
- Tether cable to rod with strain clamps every 30–50 cm
- Use cable glands at entry/exit points
- Avoid sharp bends in cable routing

## Quality Control

Before field deployment:
- [ ] Verify all electrical connections
- [ ] Test coil inductance and Q factor
- [ ] Confirm ERT ring isolation
- [ ] Check waterproofing integrity
- [ ] Verify probe ID labeling

