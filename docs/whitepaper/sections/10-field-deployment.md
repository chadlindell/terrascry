# 10. Field Deployment (sectional survey)

## 10.1 Grid Layout

### Section Size
- **Standard:** ~**10 × 10 m** (manageable by 2–3 people)
- **Small:** 8 × 8 m (woods burials)
- **Large:** 15 × 15 m (crater sites, with more personnel)

### Node Spacing
- **Standard:** **1.5–2.0 m** node spacing
- **Tight:** 1.0–1.5 m over anomalies (higher resolution)
- **Wide:** 2.5–3.0 m for reconnaissance (faster coverage)

### Probes per Section
- **Standard:** **20–24** probes (e.g., 4×5, 4×6 grid)
- **Small:** 12–16 probes (e.g., 3×4, 4×4 grid)
- **Large:** 30–36 probes (e.g., 5×6, 6×6 grid)

### Insertion Depth
- **Woods:** Insert to **1.5 m**
- **Crater:** Insert to **3.0 m**
- Use **pilot rod** if needed
- **Remove pilot** before inserting sensor rod

## 10.2 "Set Once; Measure Many" Workflow

### Step 1: Install All Probes
- Deploy all probes for the section
- Mark with flags (numbered)
- Record GPS/total‑station coordinates
- Record rod depth for each probe
- Verify probe IDs match records

### Step 2: Background Scan
- Perform a **short MIT & ERT scan** outside the suspected zone
- Establishes baseline/control measurements
- Helps identify site-wide variations

### Step 3: MIT Sweep
- For each probe **P**: set P=TX
- All other probes record (RX) at **3–5 frequencies**
  - Typical: 2, 5, 10, 20, 50 kHz
- Log amplitude and phase for each TX→RX pair
- Complete sweep before moving probes

### Step 4: ERT Patterns
- Inject current across **long baselines**:
  - Corner‑to‑corner
  - Edge‑to‑edge
  - Center‑to‑edge
- All probes log voltages simultaneously
- **Reverse polarity** periodically (every 1–2 s)
- Use multiple injection pairs for redundancy

### Step 5: Quality Control
- Repeat 5–10% of TX→RX pairs
- Verify **reciprocity** (A→B ≈ B→A)
- Check for outliers or inconsistent readings
- Document any issues

### Step 6: Extract and Move
- Extract probes carefully
- Shift to **next section**
- Leave **one column overlap** for continuity if possible
- Maintain coordinate system across sections

## 10.3 Minimal‑Intrusion Variants

### Rim‑Only Deployment
- Place a **ring of probes** around suspected crater edge
- Add a few probes **angled inward**
- Reduces ground disturbance in sensitive areas
- Still provides good coverage with proper geometry

### Shallow Mode
- Insert to **≤1 m** depth
- Use **wider spacing** (2–3 m)
- Rely on **deeper current/field paths**:
  - Lower frequencies (2–5 kHz for MIT)
  - Longer offsets for ERT
- Suitable for very sensitive sites

## Field Logging

### Essential Data to Record
- **Probe locations:** GPS coordinates or total station
- **Insertion depths:** Actual depth for each probe
- **Soil conditions:** Moisture, type, compaction
- **Weather:** Temperature, recent precipitation
- **Time stamps:** For all measurements
- **Notes:** Any disturbances, anomalies, issues

### Data Organization
- One file per section
- Consistent naming convention
- Backup data frequently
- Keep paper log as backup

## Time Estimates

### Per Section (10×10 m, 20 probes)
- **Setup:** 30–60 minutes (probe insertion)
- **MIT sweep:** 30–45 minutes (all frequencies)
- **ERT patterns:** 15–30 minutes
- **QC checks:** 10–15 minutes
- **Extraction:** 15–30 minutes
- **Total:** ~2–3 hours per section

*Times vary with team size, soil conditions, and measurement density.*

