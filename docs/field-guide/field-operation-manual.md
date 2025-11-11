# Field Operation Manual - HIRT System

## Overview

Complete field operation manual for deploying and operating the HIRT probe array system in archaeological and forensic contexts. This manual covers all aspects of field deployment, data collection, and quality control.

## Table of Contents

1. Pre-Deployment Preparation
2. Site Setup and Grid Layout
3. Probe Deployment
4. Data Collection Procedures
5. Quality Control in Field
6. Troubleshooting Field Issues
7. Data Management
8. Post-Deployment Procedures
9. Safety and Emergency Procedures

---

## 1. Pre-Deployment Preparation

### 1.1 Equipment Check

**Before leaving for field:**

- [ ] **Probes:** All probes calibrated and tested (see [Testing Procedures](../../build/testing-procedures.md))
- [ ] **Base Hub:** Fully tested, battery charged
- [ ] **Cables:** All cables tested for continuity
- [ ] **Tools:** Pilot rods, driver, extraction tools, flags, measuring tape
- [ ] **Documentation:** Field logs, calibration sheets, site maps
- [ ] **Spares:** 2-4 spare probes, spare cables, spare fuses
- [ ] **Power:** Battery fully charged, charger packed
- [ ] **Data Storage:** Tablets/computers charged, storage media ready

### 1.2 Site Preparation

**Before deployment:**

- [ ] **Permits:** All permits and permissions obtained
- [ ] **UXO Clearance:** EOD clearance confirmed (if applicable)
- [ ] **Site Access:** Access routes confirmed
- [ ] **Safety:** Safety briefing completed, emergency contacts documented
- [ ] **Weather:** Weather conditions checked, appropriate gear packed
- [ ] **Site Survey:** Preliminary site survey completed, target areas identified

### 1.3 Team Briefing

**Conduct team briefing:**

- [ ] Review site plan and objectives
- [ ] Assign roles and responsibilities
- [ ] Review safety procedures
- [ ] Review data collection procedures
- [ ] Establish communication protocols
- [ ] Set work schedule and breaks

---

## 2. Site Setup and Grid Layout

### 2.1 Site Assessment

**Evaluate site conditions:**

1. **Soil Type:**
   - Sandy, loamy, clay, mixed
   - Moisture content (dry, moist, wet)
   - Compaction (loose, compact, hard)
   - Document: Soil type = _____

2. **Surface Conditions:**
   - Vegetation (grass, bare, forest)
   - Slope (flat, gentle, steep)
   - Obstacles (rocks, roots, debris)
   - Document: Surface conditions noted

3. **Access:**
   - Vehicle access to site
   - Walking distance to deployment area
   - Equipment transport requirements
   - Document: Access OK

4. **Target Information:**
   - Expected target depth: _____ m
   - Expected target size: _____ m
   - Target type: _____ (metal, disturbed soil, void)
   - Document: Target information recorded

### 2.2 Grid Layout Procedure

**Standard 10×10 m Grid (20 probes, 2 m spacing):**

1. **Establish Reference Point:**
   - Mark reference point (corner of grid)
   - Record GPS coordinates or total station position
   - Document: Reference point: _____ (lat/long or coordinates)

2. **Lay Out Grid:**
   - Use measuring tape to establish grid
   - Mark probe positions with flags
   - Verify spacing (2 m)
   - Check squareness (measure diagonals)
   - Document: Grid laid out, 10×10 m

3. **Mark Probe Positions:**
   - Number flags: P01, P02, ..., P20
   - Ensure flags are visible and secure
   - Document: All positions marked

4. **Record Coordinates:**
   - Measure coordinates for each probe position
   - Use GPS or total station
   - Record in field log
   - Document: Coordinates recorded

**Alternative Grids:**

- **8×8 m (Woods):** 12-16 probes, 1-1.5 m spacing
- **15×15 m (Crater):** 28-36 probes, 2 m spacing
- **Custom:** Adapt to site conditions

### 2.3 Base Hub Placement

**Select base hub location:**

- **Criteria:**
  - Accessible but out of way
  - Protected from weather (if possible)
  - Close enough for cable management
  - Safe from traffic/disturbance

- **Setup:**
  - Place enclosure on stable surface
  - Connect battery
  - Connect data logger/tablet
  - Verify power and communication
  - Document: Base hub location: _____

---

## 3. Probe Deployment

### 3.1 Pre-Insertion Checks

**Before inserting each probe:**

- [ ] Probe ID matches flag number
- [ ] Probe is calibrated and tested
- [ ] Cable is untangled and ready
- [ ] Depth target confirmed
- [ ] Insertion method selected (pilot rod or direct)

### 3.2 Insertion Procedure

**Method 1: Direct Insertion (Soft Soil)**

1. **Position Probe:**
   - Align probe with flag position
   - Ensure vertical orientation
   - Check for obstacles

2. **Insert Probe:**
   - Push probe into ground
   - Rotate while pushing (if needed)
   - Insert to target depth
   - Verify depth using graduations

3. **Secure Position:**
   - Ensure probe is stable
   - Verify vertical orientation
   - Check cable routing

**Method 2: Pilot Rod Method (Hard Soil)**

1. **Create Pilot Hole:**
   - Insert pilot rod at flag position
   - Drive to target depth (use driver if needed)
   - Remove pilot rod carefully

2. **Insert Probe:**
   - Insert sensor probe into pilot hole
   - Push to bottom of hole
   - Verify depth
   - Backfill if needed (for good contact)

3. **Secure Position:**
   - Ensure probe is stable
   - Verify good soil contact
   - Check cable routing

### 3.3 Depth Verification

**Verify insertion depth:**

1. **Measure Depth:**
   - Use depth graduations on rod
   - Measure from surface to tip
   - Record actual depth

2. **Document:**
   - Probe ID: _____
   - Target Depth: _____ m
   - Actual Depth: _____ m
   - Notes: _____

3. **Verify Consistency:**
   - All probes should be at same depth (±0.1 m)
   - Document any variations

### 3.4 Cable Management

**Route cables to base hub:**

1. **Organize Cables:**
   - Route cables along grid edges (if possible)
   - Avoid crossing paths
   - Use cable ties to organize
   - Keep cables off ground (if wet)

2. **Connect to Base Hub:**
   - Connect power connector
   - Connect data connector
   - Verify connections are secure
   - Test communication

3. **Strain Relief:**
   - Ensure cables have slack
   - Use strain reliefs at connections
   - Protect from damage

### 3.5 Post-Deployment Verification

**After all probes deployed:**

- [ ] All probes inserted to correct depth
- [ ] All cables connected
- [ ] All probes communicating
- [ ] Power system stable
- [ ] Sync signal distributed
- [ ] Coordinates recorded
- [ ] Field log updated

---

## 4. Data Collection Procedures

### 4.1 Pre-Measurement Setup

**Before starting measurements:**

1. **System Check:**
   - Verify all probes powered
   - Test communication with all probes
   - Verify sync signal
   - Check data logger ready

2. **Background Measurement** (Optional but Recommended):
   - Perform quick scan outside target area
   - Establishes baseline
   - Helps identify site-wide variations
   - Document: Background scan complete

3. **Configure Measurement Parameters:**
   - MIT frequencies: 2, 5, 10, 20, 50 kHz (or selected subset)
   - ERT current: 1 mA (or as needed)
   - Measurement sequence: Plan order
   - Document: Parameters configured

### 4.2 MIT Measurement Sequence

**Complete MIT sweep:**

1. **For each probe P (P01 to P20):**
   - Set probe P as TX (transmitter)
   - Configure TX frequency (start with 5 kHz)
   - Enable TX output
   - Verify TX current: **Expected:** 10-50 mA

2. **For each other probe R (all except P):**
   - Configure probe R as RX (receiver)
   - Measure amplitude and phase
   - Record measurement
   - Verify signal quality

3. **Repeat for all frequencies:**
   - 2 kHz (deep penetration)
   - 5 kHz (standard)
   - 10 kHz (standard)
   - 20 kHz (near-surface)
   - 50 kHz (very near-surface)

4. **Document:**
   - TX probe ID
   - RX probe ID
   - Frequency
   - Amplitude
   - Phase
   - Timestamp
   - Notes

**Total Measurements:**
- 20 TX probes × 19 RX probes × 5 frequencies = 1,900 MIT measurements

**Time Estimate:** 30-45 minutes per section

### 4.3 ERT Measurement Sequence

**Complete ERT patterns:**

**Pattern 1: Corner-to-Corner (Long Baseline)**
1. Set P01 as current injection + (positive)
2. Set P20 as current injection - (negative)
3. Inject current: 1 mA
4. Measure voltage at all other probes (P02-P19)
5. Reverse polarity
6. Repeat measurements
7. Document: Pattern 1 complete

**Pattern 2: Edge-to-Edge**
1. Set P05 as current injection +
2. Set P16 as current injection -
3. Inject current: 1 mA
4. Measure voltage at all other probes
5. Reverse polarity
6. Repeat measurements
7. Document: Pattern 2 complete

**Pattern 3: Center-to-Edge**
1. Set P13 (center) as current injection +
2. Set P01 (corner) as current injection -
3. Inject current: 1 mA
4. Measure voltage at all other probes
5. Reverse polarity
6. Repeat measurements
7. Document: Pattern 3 complete

**Additional Patterns** (if time permits):
- Pattern 4: Other corner-to-corner pairs
- Pattern 5: Other edge-to-edge pairs

**Total Measurements:**
- 3-5 patterns × 2 polarities × ~18 measurements = 108-180 ERT measurements

**Time Estimate:** 15-30 minutes per section

### 4.4 Quality Control Measurements

**During data collection:**

1. **Reciprocity Checks:**
   - Select 5-10% of TX→RX pairs
   - Measure A→B
   - Measure B→A
   - Compare: **Expected:** Within 5%
   - Document: Reciprocity verified

2. **Repeat Measurements:**
   - Repeat 5-10% of measurements
   - Verify consistency: **Expected:** Within noise floor
   - Document: Repeatability verified

3. **Real-Time Monitoring:**
   - Monitor data quality during collection
   - Check for outliers or errors
   - Verify signal levels
   - Document: Data quality monitored

### 4.5 Data Recording

**Record all measurements:**

1. **MIT Data Format:**
   ```
   timestamp, section_id, tx_probe_id, rx_probe_id, freq_hz, amp, phase_deg, tx_current_mA, notes
   ```

2. **ERT Data Format:**
   ```
   timestamp, section_id, inject_pos_probe_id, inject_neg_probe_id, sense_probe_id, volt_mV, current_mA, polarity, notes
   ```

3. **Metadata:**
   - Site name/location
   - Survey date/time
   - Team members
   - Weather conditions
   - Soil conditions
   - Probe depths
   - Grid coordinates

4. **File Organization:**
   - One file per section
   - Consistent naming: `MIT_S01_2024-03-15.csv`
   - Backup frequently
   - Keep paper log as backup

---

## 5. Quality Control in Field

### 5.1 Real-Time QC Checks

**During measurements:**

- [ ] Monitor signal levels (not too high, not too low)
- [ ] Check for saturation (ADC maxed out)
- [ ] Verify communication (all probes responding)
- [ ] Check sync signal (timing consistent)
- [ ] Monitor power (battery voltage stable)

### 5.2 Data Quality Indicators

**Good Data:**
- Consistent reciprocity (A→B ≈ B→A)
- Smooth spatial variations
- Expected depth sensitivity
- Stable baseline measurements
- No outliers or spikes

**Problematic Data:**
- Poor reciprocity (check coupling, calibration)
- Noisy/spiky readings (check connections, shielding)
- No depth sensitivity (check spacing, frequency)
- Inconsistent repeats (check timebase, connectors)
- Saturation (reduce TX level or increase separation)

### 5.3 Field Troubleshooting

**If data quality issues:**

1. **Identify Problem:**
   - Check specific measurements
   - Identify pattern (all probes, specific probe, specific frequency)
   - Note symptoms

2. **Quick Fixes:**
   - Re-seat connectors
   - Check cable connections
   - Verify power
   - Check sync signal
   - Reduce TX level (if saturation)

3. **Document:**
   - Problem description
   - Fix applied
   - Result
   - Notes for future

---

## 6. Troubleshooting Field Issues

### 6.1 Probe Not Responding

**Symptoms:** No communication with probe

**Troubleshooting:**
1. Check power connection
2. Verify cable connection
3. Check probe power LED (if present)
4. Test with multimeter (check power at probe)
5. Try different cable
6. Check base hub communication

**Fix:**
- Re-seat connections
- Replace cable if damaged
- Check probe power supply
- Replace probe if necessary

### 6.2 Poor Data Quality

**Symptoms:** Noisy data, inconsistent measurements

**Troubleshooting:**
1. Check all connections
2. Verify shielding
3. Check for ground loops
4. Verify sync signal
5. Check power supply quality
6. Look for interference sources

**Fix:**
- Improve connections
- Add shielding
- Fix ground loops
- Verify sync distribution
- Improve power supply filtering

### 6.3 ERT Not Working

**Symptoms:** No ERT measurements, unstable voltages

**Troubleshooting:**
1. Check ring connections
2. Verify ring isolation
3. Check current source
4. Verify soil contact
5. Check mux operation

**Fix:**
- Improve ring contact (pre-moisten hole)
- Check ring isolation
- Verify current source
- Ensure good soil contact

### 6.4 Power Issues

**Symptoms:** Low battery, probes not powering

**Troubleshooting:**
1. Check battery voltage
2. Verify fuse
3. Check power distribution
4. Measure current draw
5. Check for shorts

**Fix:**
- Charge or replace battery
- Replace fuse
- Fix power distribution
- Reduce current draw
- Fix shorts

---

## 7. Data Management

### 7.1 Data Backup

**During survey:**

- Backup data frequently (every hour or after each section)
- Use multiple storage media (tablet, USB drive, cloud if available)
- Keep paper log as backup
- Verify backups are complete

### 7.2 Data Organization

**File Structure:**
```
data/
├── 2024-03-15/
│   ├── MIT_S01_2024-03-15.csv
│   ├── ERT_S01_2024-03-15.csv
│   ├── MIT_S02_2024-03-15.csv
│   └── ERT_S02_2024-03-15.csv
├── probe_registry.csv
└── field_log_2024-03-15.txt
```

### 7.3 Data Verification

**After each section:**

- Verify data completeness
- Check for missing measurements
- Verify file integrity
- Check data format
- Document any issues

---

## 8. Post-Deployment Procedures

### 8.1 Probe Extraction

**After measurements complete:**

1. **Disconnect Cables:**
   - Disconnect power
   - Disconnect data
   - Organize cables

2. **Extract Probes:**
   - Use extraction tool if needed
   - Pull probes straight up
   - Clean probes (remove soil)
   - Inspect for damage

3. **Store Probes:**
   - Clean thoroughly
   - Inspect for damage
   - Store in protective cases
   - Document any issues

### 8.2 Equipment Packing

**Pack equipment:**

- [ ] All probes accounted for
- [ ] Cables organized and packed
- [ ] Base hub packed
- [ ] Tools packed
- [ ] Documentation secured
- [ ] Spares accounted for

### 8.3 Site Cleanup

**Leave site clean:**

- [ ] Remove all flags and markers
- [ ] Fill any holes (if required)
- [ ] Remove all equipment
- [ ] Leave site as found (or better)
- [ ] Document site condition

### 8.4 Post-Survey Documentation

**Complete documentation:**

- [ ] Field log completed
- [ ] Data backed up and verified
- [ ] Photos taken (if applicable)
- [ ] Site notes completed
- [ ] Equipment issues documented
- [ ] Lessons learned recorded

---

## 9. Safety and Emergency Procedures

### 9.1 Safety Protocols

**Always:**

- Follow UXO/EOD protocols (if applicable)
- Use appropriate PPE
- Maintain safe working distances
- Follow site-specific safety rules
- Have emergency contacts available

### 9.2 Emergency Procedures

**If emergency:**

1. **Stop work immediately**
2. **Assess situation**
3. **Call emergency services** (if needed)
4. **Evacuate if necessary**
5. **Document incident**
6. **Report to supervisor**

### 9.3 Emergency Contacts

**Document before deployment:**

- Site supervisor: _____
- EOD contact: _____
- Emergency services: _____
- Medical: _____

---

## Appendices

### Appendix A: Field Log Template

**Date:** _____
**Site:** _____
**Section:** _____
**Team:** _____
**Weather:** _____
**Soil:** _____

**Probes Deployed:** _____
**Depths:** _____
**Issues:** _____

**Notes:**
_________________________________________________________________
_________________________________________________________________

### Appendix B: Quick Reference

See [Quick Reference](quick-reference.md) for one-page field reference.

### Appendix C: Troubleshooting Quick Guide

See [Troubleshooting](../whitepaper/sections/13-troubleshooting.md) for detailed troubleshooting guide.

---

*For calibration procedures, see [Calibration Guide](../../build/calibration-procedures.md)*
*For testing procedures, see [Testing Guide](../../build/testing-procedures.md)*

