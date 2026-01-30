# 10. Field Operations

## Overview

This section provides comprehensive procedures for deploying and operating the HIRT (Hybrid Impedance-Resistivity Tomography) system in the field, from pre-deployment planning through data backup. For quick reference, see Section 22: Quick Reference Card.

---

## 10.1 Pre-Deployment Planning

### Site Assessment (Day Before)

1. **Review site conditions:**
   - Soil type and moisture
   - Presence of utilities or obstructions
   - Access routes and staging areas

2. **Check equipment:**
   - All probes tested and calibrated
   - Base hub charged (12V battery >80%)
   - Cables verified for continuity
   - Data logger/tablet charged and configured

3. **Verify permits and permissions:**
   - Site access authorization
   - Excavation permits (if required)
   - Archaeological survey approval

### Equipment Checklist

**Essential Items:**
- [ ] Probes (20-24 typical) - tested
- [ ] Base hub/control unit - charged
- [ ] Connection cables (3-5 m each)
- [ ] Fiberglass survey stakes
- [ ] Measuring tape (30 m minimum)
- [ ] Field tablet or laptop
- [ ] Tool kit (wrenches, screwdrivers, multimeter)
- [ ] Ground truth equipment (soil corer, if permitted)

**Optional Items:**
- [ ] GPS unit
- [ ] Weather station
- [ ] Backup battery
- [ ] Shade canopy
- [ ] First aid kit

---

## 10.2 Site Assessment and Grid Design

### Section Size

- **Standard:** ~**10 x 10 m** (manageable by 2-3 people)
- **Small:** 8 x 8 m (woods burials)
- **Large:** 15 x 15 m (crater sites, with more personnel)

### Node Spacing

- **Standard:** **1.5-2.0 m** node spacing
- **Tight:** 1.0-1.5 m over anomalies (higher resolution)
- **Wide:** 2.5-3.0 m for reconnaissance (faster coverage)

### Probes per Section

- **Standard:** **20-24** probes (e.g., 4x5, 4x6 grid)
- **Small:** 12-16 probes (e.g., 3x4, 4x4 grid)
- **Large:** 30-36 probes (e.g., 5x6, 6x6 grid)

### Standard 10x10 m Grid Layout (2 m spacing)

```
    0m    2m    4m    6m    8m   10m
  0m  P01   P02   P03   P04   P05   P06
  2m  P07   P08   P09   P10   P11   P12
  4m  P13   P14   P15   P16   P17   P18
  6m  P19   P20   P21   P22   P23   P24
  8m  (P25) (P26) (P27) (P28) (P29) (P30)
```

### Grid Setup Steps

1. **Establish baseline:**
   - Set reference point (0,0) with survey stake
   - Align baseline with site grid (N-S or as specified)
   - Mark corners with bright flags

2. **Mark probe positions:**
   - Measure 2 m spacing along baseline
   - Measure perpendicular rows
   - Mark each position with small stake or flag

3. **Verify geometry:**
   - Cross-measure diagonals (should match within 5 cm)
   - Record any deviations in field notes

---

## 10.3 Probe Installation

### Insertion Depth

- **Woods:** Insert to **1.5 m**
- **Crater:** Insert to **3.0 m**
- Use **pilot rod** if needed
- **Remove pilot** before inserting sensor rod

### Standard Insertion Procedure

**For Each Probe:**

1. **Prepare pilot hole (if needed):**
   - Use hand auger or pilot rod
   - Target depth: same as probe insertion depth (2-3 m)
   - Remove pilot tool, note any obstructions

2. **Insert probe:**
   - Align probe with hole
   - Push gently, using steady pressure
   - Do not force - if stuck, remove and re-clear hole

3. **Achieve proper depth:**
   - Target: tip at 2.5-3 m depth
   - Junction box ~10 cm above surface
   - Verify probe is vertical (use spirit level if needed)

4. **Secure probe:**
   - Ensure junction box is stable
   - Connect cable
   - Apply strain relief

5. **Record position:**
   - Note probe ID and grid position
   - Record actual depth
   - Note any insertion difficulties

### Insertion Methods by Soil Type

**Sandy/Loose Soil:**
- May push probe directly
- Use water jet if available
- Watch for collapse of hole walls

**Clay/Compact Soil:**
- Pre-drill pilot hole
- May need to enlarge hole slightly
- Allow settling before measurement

**Rocky/Mixed Soil:**
- Use auger with care
- Note rock contact locations
- May need to relocate probe slightly

### Cable Routing

1. **Route cables radially from base hub:**
   - Place base hub at grid center or edge (convenient location)
   - Route cables along grid lines to avoid tangling
   - Use cable clips or ties to secure

2. **Allow slack:**
   - Leave 0.5 m slack at each probe connection
   - Coil excess cable at base hub

---

## 10.4 Coil Winding Recipe (Field Winding)

If field winding of replacement coils is needed, follow these specifications.

### Coil Specifications

| Parameter | TX Coil | RX Coil |
|-----------|---------|---------|
| Inductance | 1-2 mH | 1-2 mH |
| Q Factor | >20 | >20 |
| DC Resistance | <10 ohm | <10 ohm |
| Core | Ferrite | Ferrite |
| Wire | 28-32 AWG | 28-32 AWG |
| Turns | 100-300 | 100-300 |

### Materials Required

**Ferrite Cores:**
- **Material:** MnZn ferrite (high permeability)
- **Shape:** Cylindrical rod or toroid
- **Size (Micro-Probe):** 6-8 mm diameter x 40-80 mm long
- **Permeability:** 1000-3000 (typical)

**Magnet Wire:**
- **Type:** Enameled copper (polyurethane or polyester)
- **Gauge:** 28-32 AWG (0.32-0.20 mm diameter)

**Auxiliary Materials:**
- Varnish or epoxy (for securing windings)
- Heat-shrink tubing (for lead protection)
- Kapton tape (for insulation layers)
- Solder (rosin core, 60/40 or lead-free)

### Hand Winding Procedure (Micro-Probe)

**For ferrite rod cores (6-8 mm x 40-80 mm):**

1. **Prepare the core:**
   - Clean ferrite rod with isopropyl alcohol
   - Apply thin layer of varnish or tape over winding area
   - Mark start and end positions

2. **Attach start lead:**
   - Leave 10-15 cm lead wire
   - Secure to core end with tape or small drop of glue
   - Route lead away from winding area

3. **Wind turns:**
   - Wind evenly along core length
   - Maintain consistent tension (not too tight)
   - Layer windings if needed (for high inductance)
   - Count turns carefully (use tally counter)

   **Target Turns (Micro-Probe):**
   - For 1 mH: ~150-200 turns on 6 mm x 50 mm ferrite
   - For 2 mH: ~250-300 turns on 6 mm x 50 mm ferrite
   - Adjust based on actual core permeability

4. **Secure windings:**
   - Apply varnish over windings (thin coat)
   - Allow to dry/cure
   - Apply second coat if needed

5. **Attach end lead:**
   - Leave 10-15 cm lead wire
   - Secure with tape or glue
   - Twist leads together loosely (for noise rejection)

6. **Test coil:**
   - Measure inductance with LCR meter
   - Measure DC resistance
   - Calculate Q factor: Q = (2 * pi * f * L) / R

### Field Coil Quality Check

- [ ] Inductance within spec (1-2 mH)
- [ ] Q factor > 20
- [ ] DC resistance < 10 ohms
- [ ] No visible shorts or damage
- [ ] Windings secure (no movement)
- [ ] Leads properly terminated

---

## 10.5 System Setup and Power-Up

### Equipment Preparation

**Probe Preparation:**
1. **Visual inspection:**
   - Check probe head seal
   - Verify ERT ring contacts are clean
   - Inspect cable connections

2. **Functional test:**
   - Power each probe briefly
   - Verify LED indicators
   - Test communication with base hub

3. **Labeling:**
   - Ensure all probes are clearly labeled (P01, P02, etc.)
   - Match labels with deployment map

**Base Hub Preparation:**
1. **Power check:**
   - Verify battery voltage (>12.0 V)
   - Connect charger if time permits
   - Check fuse integrity

2. **Software configuration:**
   - Load site configuration file
   - Set measurement parameters
   - Configure data logging

3. **Communication test:**
   - Connect to field tablet
   - Verify data transfer
   - Test sync signal

### Power-Up Sequence

1. **Connect all cables:**
   - Verify each probe is connected
   - Check cable routing (no kinks or tension)

2. **Power on base hub:**
   - Turn main power switch ON
   - Wait for initialization (10-15 seconds)
   - Verify power LED is solid green

3. **Verify probe communication:**
   - Run diagnostic scan
   - All probes should report status
   - Note any non-responsive probes

4. **Initialize measurement system:**
   - Start field software
   - Load site configuration
   - Verify probe array geometry in software

---

## 10.6 MIT Measurement Protocol

### Pre-Measurement Checks

1. **System self-test:**
   - Run built-in self-test
   - Verify all probes pass
   - Note any warnings

2. **Baseline measurement:**
   - Take one measurement cycle with no targets
   - Record as baseline reference
   - Store in data file

### MIT Survey Procedure

For each probe **P**: set P=TX, all other probes record (RX) at **3-5 frequencies**.

**Frequency Selection:**
- Typical: 2, 5, 10, 20, 50 kHz
- Lower frequencies (2-5 kHz) for deeper penetration
- Higher frequencies (20-50 kHz) for near-surface detail

**For Each Measurement:**
1. Log amplitude and phase for each TX-RX pair
2. Monitor for errors or outliers
3. Complete full sweep before moving probes

**Timing:**
- Full matrix: all TX-RX pairs
- Typical: 10-15 minutes per complete scan
- Multi-frequency sweep: 30-45 minutes (all frequencies)

---

## 10.7 ERT Measurement Protocol

### ERT Survey Procedure

Inject current across **long baselines**:
- Corner-to-corner
- Edge-to-edge
- Center-to-edge

**For Each Measurement:**
1. All probes log voltages simultaneously
2. **Reverse polarity** periodically (every 1-2 s)
3. Use multiple injection pairs for redundancy

**Configuration:**
- Set current level for ERT (0.5-2 mA)
- Set integration time (1-5 seconds per measurement)
- Wenner or dipole-dipole configuration

**Timing:**
- Typical: 5-10 minutes per configuration
- Multiple current levels for noise assessment

### Combined Survey

- Interleaved MIT and ERT measurements
- Provides complementary data
- Typical: 15-20 minutes per scan

---

## 10.8 "Set Once, Measure Many" Workflow

### Step 1: Install All Probes

- Deploy all probes for the section
- Mark with flags (numbered)
- Record GPS/total-station coordinates
- Record rod depth for each probe
- Verify probe IDs match records

### Step 2: Background Scan

- Perform a **short MIT & ERT scan** outside the suspected zone
- Establishes baseline/control measurements
- Helps identify site-wide variations

### Step 3: MIT Sweep

- Complete full TX-RX matrix at all frequencies
- Complete sweep before moving probes

### Step 4: ERT Patterns

- Complete all injection patterns
- Multiple baselines for redundancy

### Step 5: Quality Control

- Repeat 5-10% of TX-RX pairs
- Verify **reciprocity** (A-B approximately equals B-A)
- Check for outliers or inconsistent readings
- Document any issues

### Step 6: Extract and Move

- Extract probes carefully
- Shift to **next section**
- Leave **one column overlap** for continuity if possible
- Maintain coordinate system across sections

---

## 10.9 Quality Checks

### Data Quality Verification

1. **Check for outliers:**
   - Values >3 std from mean
   - Signal levels outside expected range

2. **Verify reciprocity:**
   - A-B should approximately equal B-A
   - Poor reciprocity indicates coupling or calibration issues

3. **Spatial consistency:**
   - Smooth spatial variations expected
   - Abrupt changes may indicate problems

4. **Baseline stability:**
   - Compare to initial baseline
   - Significant drift indicates issues

### Good Data Indicators

- Consistent reciprocity
- Smooth spatial variations
- Expected depth sensitivity
- Stable baseline measurements

### Problematic Data Indicators

- Poor reciprocity (check coupling, calibration)
- Noisy/spiky readings (check connections, shielding)
- No depth sensitivity (check spacing, frequency)
- Inconsistent repeats (check timebase, connectors)

---

## 10.10 Shutdown and Data Backup

### Safe Shutdown Procedure

1. **Complete final measurement:**
   - Take final data set
   - Verify data is saved

2. **Power down:**
   - Stop measurement software
   - Power off base hub
   - Disconnect cables from probes

3. **Extract probes:**
   - Pull gently with straight vertical motion
   - Do not twist excessively
   - Note any probes that are stuck

4. **Site restoration:**
   - Fill probe holes (as required by permit)
   - Remove all markers and equipment
   - Photograph site condition

### Probe Extraction Tips

**Stuck Probe:**
- Work probe back and forth gently
- Add water around probe (lubrication)
- Use extraction handle if available
- Do not use excessive force (may damage probe)

**Collapsed Hole:**
- Probe may be gripped by soil
- Wait for water to soften soil
- Consider leaving probe if damage risk is high

### Data Backup Procedure

1. **Download data:**
   - Connect tablet/laptop to base hub
   - Download all raw data files
   - Verify file sizes (not empty)

2. **Backup to multiple locations:**
   - Copy to laptop hard drive
   - Copy to USB drive
   - Upload to cloud (if connectivity available)

3. **Data organization:**
   - Use consistent naming: `SITE_DATE_SCAN#.dat`
   - Create folder for each site/day
   - Include field notes file

4. **Verify backups:**
   - Open files in viewer software
   - Check data is complete
   - Note any missing or corrupted files

---

## 10.11 Minimal-Intrusion Variants

### Rim-Only Deployment

- Place a **ring of probes** around suspected crater edge
- Add a few probes **angled inward**
- Reduces ground disturbance in sensitive areas
- Still provides good coverage with proper geometry

### Shallow Mode

- Insert to **<=1 m** depth
- Use **wider spacing** (2-3 m)
- Rely on **deeper current/field paths**:
  - Lower frequencies (2-5 kHz for MIT)
  - Longer offsets for ERT
- Suitable for very sensitive sites

---

## 10.12 Field Logging

### Essential Data to Record

- **Probe locations:** GPS coordinates or total station
- **Insertion depths:** Actual depth for each probe
- **Soil conditions:** Moisture, type, compaction
- **Weather:** Temperature, recent precipitation
- **Time stamps:** For all measurements
- **Notes:** Any disturbances, anomalies, issues

### Field Notes Template

Record for each survey:
- Date, time, weather conditions
- Site name and location
- Grid layout and probe positions
- Equipment used (probe IDs)
- Any issues or anomalies
- Photos taken (reference numbers)

### Data Organization

- One file per section
- Consistent naming convention
- Backup data frequently
- Keep paper log as backup

---

## 10.13 Time Estimates

### Per Section (10x10 m, 20 probes)

| Activity | Time |
|----------|------|
| Setup (probe insertion) | 30-60 minutes |
| MIT sweep (all frequencies) | 30-45 minutes |
| ERT patterns | 15-30 minutes |
| QC checks | 10-15 minutes |
| Extraction | 15-30 minutes |
| **Total** | **~2-3 hours per section** |

*Times vary with team size, soil conditions, and measurement density.*

---

## Safety Reminders

- Always inform someone of your field location
- Stay hydrated and take breaks
- Be aware of wildlife and site hazards
- Follow all site-specific safety rules
- Keep first aid kit accessible

---

*For quick reference, see Section 22: Quick Reference Card. For troubleshooting details, see Section 13: Troubleshooting.*
