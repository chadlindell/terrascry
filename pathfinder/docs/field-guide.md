# Pathfinder Field Operations Guide

A comprehensive guide for planning, executing, and interpreting magnetic gradiometer surveys with the Pathfinder system. Read this before your first field deployment. Print the Survey Log Template (Section 7) and the Troubleshooting table (Section 6) as pocket references.

---

## 1. Pre-Survey Planning

A successful survey starts the day before you go to the field. Rushing setup leads to missed anomalies, wasted time, and data that cannot be interpreted.

### 1.1 Site Assessment

Visit the site or study it on satellite imagery before survey day. Evaluate:

- **Terrain**: Flat, sloped, wooded, overgrown? Pathfinder works on any walkable terrain, but dense brush slows traverse pace and may snag the crossbar.
- **Access**: Where will you park? How far is the walk-in? Plan your route to avoid carrying the assembled trapeze through narrow gates or dense vegetation.
- **Known hazards**: Livestock, unstable ground, live utilities, restricted areas. Obtain landowner permission and any required permits before arrival.
- **Expected targets**: What are you looking for? The answer determines traverse spacing, coverage area, and how aggressively you interpret weak anomalies.
  - Ferrous debris (nails, shell casings): Strong signal, small footprint -- tight spacing helps.
  - Archaeological features (kilns, hearths): Moderate signal, broader footprint -- standard spacing works.
  - Disturbed soil (graves, pits): Very weak signal at Pathfinder's detection limit -- consider whether HIRT or a commercial gradiometer is more appropriate for this target class.
- **Magnetic environment**: Note proximity to power lines, fences, buildings with steel frames, railway lines, or buried utilities. These produce interference that can mask real targets or generate false positives.

### 1.2 Coverage Planning

Pathfinder's coverage rate depends on three factors: swath width, walking pace, and traverse spacing.

**Key numbers for the standard 4-pair configuration:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| Sensor pairs | 4 | 50 cm horizontal spacing |
| Swath width | 2.0 m | Outer sensor to outer sensor |
| Effective swath | ~1.5 m | With recommended overlap |
| Walking pace | ~1 m/s | Beeper at 1 Hz guides cadence |
| Sample spacing | ~1 m along-track | At 10 Hz sample rate and 1 m/s pace |
| GPS sample rate | 1 Hz | NEO-6M default |

**Estimating survey time for a rectangular area:**

1. Determine the number of traverse lines: divide the cross-track dimension by the line spacing (typically 1.0 m for 50% swath overlap, or 1.5 m for minimal overlap).
2. Multiply the number of lines by the along-track length to get total traverse distance.
3. Divide total distance by walking speed (1 m/s) and add ~15 seconds per line for turns.
4. Add setup time (10-15 minutes) and a buffer for GPS issues or re-walks.

**Example: 50 x 50 m area with 50% overlap (1.0 m line spacing)**

| Step | Calculation | Result |
|------|-------------|--------|
| Traverse lines | 50 m / 1.0 m | 50 lines |
| Total distance | 50 lines x 50 m | 2,500 m |
| Walking time | 2,500 m / 1 m/s | ~42 minutes |
| Turns | 50 x 15 s | ~12 minutes |
| Setup | -- | ~15 minutes |
| **Total** | | **~70 minutes** |

For initial reconnaissance where you want to cover ground quickly, increase line spacing to 2.0 m (no overlap). This cuts survey time roughly in half at the cost of leaving gaps between swaths. Any anomaly detected during the fast pass can be re-surveyed at tighter spacing.

### 1.3 Equipment Checklist

Pack the following before leaving for the field:

**Essential:**

- [ ] Pathfinder instrument (crossbar, drop tubes, sensors, electronics enclosure)
- [ ] Harness with bungee suspension
- [ ] Charged battery (check voltage >7.0 V)
- [ ] Spare battery
- [ ] SD card (FAT32 formatted, Class 10, capacity verified)
- [ ] Spare SD card
- [ ] Field notebook and pencil (not pen -- pens fail in rain and cold)
- [ ] Measuring tape (30 m minimum)
- [ ] Survey stakes or flags (for marking traverse start/end)

**Recommended:**

- [ ] Handheld GPS or phone with GPS for recording site reference point
- [ ] Multi-tool or small screwdriver set
- [ ] Spare carabiners (2x)
- [ ] Zip ties (10x)
- [ ] Duct tape
- [ ] Electrical tape
- [ ] Spare bungee cord (1 m)
- [ ] Laptop with Python and `visualize_data.py` (for on-site data QC)
- [ ] USB SD card reader
- [ ] Sunscreen, water, snacks

**Optional (for extended or high-precision surveys):**

- [ ] RTK GPS module (ZED-F9P) for sub-centimeter positioning
- [ ] Serial/USB cable for real-time debug monitoring
- [ ] Weather station or thermometer
- [ ] Camera for site documentation

### 1.4 Weather Considerations

Pathfinder is designed for all-weather operation (IP65 electronics enclosure), but environmental conditions affect both the operator and the data.

| Condition | Effect | Mitigation |
|-----------|--------|------------|
| Rain | Water ingress risk at cable glands; wet sensor cables can introduce noise | Verify cable gland seals before survey; use drip loops; dry equipment after survey |
| Cold (<5 C) | Battery capacity drops 20-40%; GPS may be slower to lock; fluxgate sensor offset drifts | Keep battery warm (insulate or body-heat pocket); allow extra warm-up time; take baseline reading at field temperature |
| Heat (>35 C) | Electronics may overheat in direct sun; operator fatigue increases | Shade the electronics enclosure; take frequent breaks; carry extra water |
| Wind | Crossbar sways, introducing tilt error; bungee suspension amplifies oscillation | Brace crossbar lightly with hands; shorten bungee cords; avoid survey in strong gusts |
| Thunderstorms | **Do not survey.** The carbon fiber crossbar and open terrain create a lightning risk. | Abort and seek shelter immediately. |

---

## 2. Site Setup

### 2.1 Establish the Site Grid

Even though GPS provides absolute coordinates, a physical grid on the ground keeps traverse lines parallel, ensures complete coverage, and provides a backup reference if GPS data has gaps.

1. **Choose a reference point.** Pick a permanent or semi-permanent feature (fence post, large rock, building corner) and record its GPS coordinates and a written description in your field notebook.
2. **Lay out the baseline.** Stretch a measuring tape along one edge of the survey area. This becomes your primary axis. Secure both ends with stakes.
3. **Mark traverse start and end points.** Place stakes or flags at your chosen line spacing (1.0 m or 2.0 m) along the baseline. If the area is large, mark every 5th stake with a different color for easy counting.
4. **Establish the cross-axis** at each end of the baseline so you know where to turn. A second tape or line of stakes perpendicular to the baseline defines the survey boundary.

For rough reconnaissance on open ground, you may skip the full grid and rely on GPS track logging alone. In this case, walk parallel lines by sighting on a distant landmark, and accept that line spacing will be approximate.

### 2.2 Power-On Sequence

Follow this sequence every time. Rushing past GPS lock is the most common cause of data without position.

1. **Insert the SD card** (FAT32, confirmed space available).
2. **Power on.** The status LED starts fast blinking (100 ms) during initialization.
3. **Watch the LED transition:**
   - Fast blink (100 ms) = Error. Check SD card. If the LED stays fast, the SD card is not detected. Remove, reinsert, try a different card.
   - Medium blink (500 ms) = Waiting for GPS lock. Normal on cold start; wait up to 2 minutes.
   - **Slow blink (2 s) = GPS locked and logging.** The system is ready.
4. **Listen for the beeper.** One beep per second confirms the pace timer is running (handheld/backpack platforms only; the drone platform disables the beeper).
5. **Check serial debug output** (optional). Connect a laptop via USB and open a serial terminal at 115200 baud. You should see:
   ```
   Pathfinder Gradiometer v1.4.0 (2026-02-18)
   Platform: HANDHELD
   Sensor pairs: 4
   ADS1115 #1 (0x48) OK
   ADS1115 #2 (0x49) OK
   GPS initialized at 9600 baud
   SD card ready
   Created log file: PATH0001.CSV
   Initialization complete. Starting acquisition...
   Sample rate: 10 Hz
   ```
   Verify both ADCs report OK and the log file was created.

### 2.3 Test Sample Acquisition

Before committing to the full survey, walk a short test line (20-30 m) and verify that data is being collected:

1. Walk a straight line at beeper pace.
2. Stop. Power off. Remove SD card.
3. Insert card into laptop. Open the newest `PATHXXXX.CSV` file.
4. Verify it contains data rows (not just the header).
5. Check that `lat` and `lon` columns contain non-zero values (GPS was locked).
6. Optionally, run `python visualize_data.py PATHXXXX.CSV --stats-only` for a quick summary.
7. Reinsert the SD card, power on, wait for slow blink, and begin the survey.

This takes 5 minutes and can save an entire day of surveying with a bad SD card or loose cable.

---

## 3. Survey Execution

### 3.1 Walking Technique

Data quality depends more on consistent operator technique than on instrument specifications. Practice these habits until they become automatic.

**Pace:** Match the beeper cadence. One beep per second produces approximately one meter per stride. Resist the urge to walk faster -- rushing creates irregular sample spacing and increases crossbar tilt.

**Crossbar level:** Tilt introduces error proportional to Earth's field multiplied by sin(tilt angle). At typical mid-latitude field strengths (~50,000 nT), a 5-degree tilt produces ~4,400 nT of apparent field change, which is enormous relative to the anomalies we are trying to detect. The gradiometer configuration cancels most of this (both sensors tilt together), but any difference in the sensors' orientation sensitivity creates a residual. Keep the crossbar as level as possible. The bungee suspension helps, but you must avoid sudden lateral movements.

**Consistent height:** The harness sets the bottom sensors at 15-20 cm above ground (Target). Adjust the shoulder straps on level ground before beginning. If terrain varies significantly (e.g., crossing a ditch), note the location in your field notebook rather than trying to adjust mid-traverse.

**Avoid metal on the operator:** Before your first traverse, remove or relocate:

- Belt buckles (move to pocket or use a plastic buckle)
- Keys (leave in vehicle or bag at edge of survey area)
- Mobile phone (leave at baseline or carry in a rear pocket, not hip pocket near sensors)
- Watch (especially steel case or band)
- Steel-toe boots (if possible, wear non-ferrous footwear)

Test by standing still and rotating slowly. If the gradient values on the serial debug output shift as you rotate, you are carrying something magnetic. Remove items until the values stabilize.

**Walking direction:** Walk in a consistent direction (all lines north-to-south, for example) for simplest data processing. Bidirectional (zigzag) surveying is more time-efficient because you eliminate the walk-back between lines, but you must record which lines were walked in which direction so that heading-dependent errors can be corrected in post-processing. Note direction changes in your field notebook or mark them with flags.

### 3.2 Traverse Patterns

Choose a pattern based on survey objectives and site geometry.

**Parallel lines (standard)**

The default pattern for systematic area coverage. Walk parallel lines with consistent spacing.

```
Start -->-->-->-->-->-->-->--> End
       |
       v  (step over by line spacing)
       |
      <--<--<--<--<--<--<--<-- Turn
       |
       v
       |
      -->-->-->-->-->-->-->--> Continue
```

- **50% swath overlap (1.0 m line spacing):** Every point on the ground is covered by at least one sensor pair. Provides redundancy and the best spatial resolution. Use for detailed surveys.
- **Minimal overlap (1.5-2.0 m line spacing):** Faster coverage. Some narrow gaps between swaths. Acceptable for initial reconnaissance where you plan to re-survey anomalies.

**Zigzag (bidirectional)**

Walk alternating directions to eliminate the dead walk-back. More efficient for large areas. Record direction changes in field notes or toggle a flag at each turn.

**Radial pattern**

For investigating a known point anomaly (e.g., a feature detected during reconnaissance). Walk lines radiating outward from the anomaly center, like spokes of a wheel. This gives high sample density near the target and sparser coverage at the edges.

### 3.3 Data Quality Monitoring

Check data quality periodically during the survey rather than discovering problems after you leave the site.

**Serial debug output** (if connected): Every 10 samples, the firmware prints a summary line:

```
S:1230 GPS:51.234567,18.345678 G1:225 G2:218 G3:210 G4:223
```

Watch for:

| Indicator | Normal | Problem | Action |
|-----------|--------|---------|--------|
| GPS coordinates | Non-zero, changing as you walk | `GPS:--` or coordinates stuck | Wait for lock; check sky view |
| Gradient values (G1-G4) | Baseline near zero (after calibration), spikes at anomalies | All near zero constantly | Check sensor cables; verify ADC addresses |
| Gradient values | Reasonable range (-500 to +500 typical) | Values hitting 32000 or -32000 | ADC saturated -- too close to large metal; move away |
| `SD_ERR:` | Absent | Count increasing | SD card issue; finish current line, then swap card |
| `SAT:` | Absent or low | Count climbing rapidly | Near large ferrous source; note location and move on |

**Beeper cadence:** If the beeper stops or becomes erratic, the system may have frozen. Power cycle and check the SD card.

**Physical checks:** Every 10-15 lines, verify that sensor cables have not snagged or pulled loose from the crossbar, that the SD card LED (if visible) blinks during writes, and that the battery voltage remains above 7.0 V.

---

## 4. Data Download and Verification

Do not leave the site until you have verified that usable data was collected. An extra 10 minutes of checking in the field can save a return trip.

### 4.1 Download Procedure

1. Power off the Pathfinder.
2. Remove the SD card carefully (do not force -- it uses a push-to-release mechanism on most modules).
3. Insert into a laptop via USB card reader.
4. Copy **all** `PATHXXXX.CSV` files from the card to your laptop. Each power cycle creates a new file (PATH0001, PATH0002, etc.).
5. Do not delete files from the SD card until you have verified the copies.

### 4.2 Quick Verification

Run the visualization tool on each file:

```bash
python visualize_data.py PATH0001.CSV
```

This produces three plots. Check each one:

**Time series plot:**
- Gradient traces should show a stable baseline with distinct spikes at anomaly locations.
- Look for large gaps in the timeline (= missed samples, possible SD write stalls).
- Look for sudden jumps in all channels simultaneously (= operator bumped the system or walked past a large metal object like a fence).

**GPS track plot (requires `--map` flag):**
- The track should match the pattern you walked (parallel lines, zigzag, etc.).
- Missing track segments indicate GPS dropouts.
- If the entire track is at (0, 0), GPS was never locked.

**Statistics output:**
- GPS lock percentage should be >90%. Below 80% means significant position data is missing.
- Gradient mean should be near zero for each pair (if your calibration offsets are applied).
- Gradient standard deviation indicates the noise level and signal strength.

### 4.3 Backup

Before leaving the site:

1. Verify the laptop copy opens and contains data.
2. Copy to a second medium (USB drive, cloud upload via phone hotspot, or email a compressed archive to yourself).
3. Reinsert the SD card into the Pathfinder (or store separately in a labeled bag).
4. Record in your field notebook which files correspond to which survey areas.

---

## 5. Data Interpretation Guide

This section provides practical guidance for interpreting Pathfinder gradient data. Pathfinder is a screening tool -- it identifies areas of interest for follow-up with detailed systems like HIRT, GPR, or commercial gradiometers. Interpretation at the screening stage is about classification and prioritization, not definitive identification.

### 5.1 What Gradient Anomalies Look Like

The gradiometer measures the vertical gradient of the magnetic field -- the difference between the bottom sensor (closer to the ground and the target) and the top sensor (farther away, measuring background). Anomalies appear as departures from the baseline gradient.

| Target Type | Gradient Strength | Signature Shape | Detection Depth (Modeled) |
|-------------|-------------------|-----------------|---------------------------|
| **Ferrous metal** (iron, steel -- nails, tools, ordnance, pipes) | Strong: 100-1,000+ nT | Sharp, dipolar (positive and negative lobes) | 20-100 cm for small objects; 1-2 m for large objects |
| **Fired clay** (kilns, hearths, brick) | Moderate: 20-100 nT | Broader anomaly, less sharp than metal | 30-60 cm |
| **Disturbed soil** (graves, pits, ditches) | Very weak: 5-20 nT | Broad, diffuse | 20-50 cm; **at Pathfinder's detection limit** |
| **Geological features** (igneous intrusions, mineralisation) | Variable: 10-500 nT | Very broad, regional trends across multiple traverse lines | Extends to depth |
| **Modern interference** (fences, utilities, vehicles) | Strong: 100-10,000+ nT | Linear (fences, pipes) or very strong point (vehicles, manholes) | Surface or shallow |

**Note:** Gradient strength values above are approximate and depend on target size, depth, magnetic susceptibility, and soil conditions. All detection depths are **(Modeled)** based on published fluxgate gradiometer performance for similar sensor configurations.

### 5.2 Anomaly Classification Quick Reference

Use this table to make initial field classifications. These are starting points for prioritization, not definitive identifications.

| Gradient (nT) | Spatial Extent | Shape | Likely Source | Priority |
|----------------|----------------|-------|---------------|----------|
| >500 | Point-like (<2 m) | Sharp dipole | Ferrous metal near surface | HIGH -- investigate or avoid (UXO context) |
| 100-500 | Point to small area | Dipolar | Buried metal, fired feature | HIGH -- mark for follow-up |
| 20-100 | Small area (2-5 m) | Broad, asymmetric | Fired clay, large pit fill, deep metal | MEDIUM -- characterize with detailed survey |
| 5-20 | Area (>3 m) | Diffuse | Disturbed soil, subtle archaeology | LOW -- may be at detection limit; consider confirmation with HIRT or GPR |
| <5 | Regional | Gradual trend | Geological variation, instrument drift | BACKGROUND -- usually not actionable |

### 5.3 Depth Estimation

For isolated dipole sources (small metal objects, discrete archaeological features), a rough depth estimate can be made from the anomaly width:

**Rule of thumb: Anomaly half-width is approximately equal to source depth.**

The "half-width" is the horizontal distance from the anomaly peak to the point where the signal drops to half its peak value. For a compact magnetic dipole source, this distance roughly equals the depth below the sensor. Since the bottom sensor is 15-20 cm above ground (Target), add the sensor height to get depth below the ground surface.

**Example:** An anomaly peaks at traverse position 25.0 m and drops to half its peak value at positions 24.5 m and 25.5 m. The half-width is 0.5 m. Estimated depth below sensor: ~0.5 m. Estimated depth below ground: ~0.5 m + 0.15 m (sensor height) = ~0.65 m.

**Limitations:**
- This rule applies only to compact, isolated dipole sources. Extended features (ditches, walls) do not follow this relationship.
- Multiple overlapping anomalies invalidate the half-width measurement.
- The rule provides order-of-magnitude guidance, not precision. Expect errors of 30-50%.
- For precise depth estimation, use HIRT or another system with tomographic capability.

### 5.4 False Positives

Not every anomaly indicates a buried target of interest. Common false positive sources:

| Source | How to Recognize | Prevention |
|--------|-----------------|------------|
| Surface debris (nails, wire, cans) | Very strong, very narrow spike; only on one traverse line | Walk the survey area first and remove visible surface metal |
| Nearby vehicles | Extremely strong anomaly (>10,000 nT) that appears on multiple lines near the same location; decreases as you walk away | Park vehicles at least 20 m from survey area (farther for trucks) |
| Operator metal | Consistent offset or drift correlated with operator position; changes if operator rotates | Remove all metal before surveying; test by rotating in place |
| Fences | Linear anomaly aligned with known fence line; repeating pattern at post intervals | Map known fences on your site plan; exclude data within 3-5 m of fences |
| Underground utilities | Linear anomaly, often strong, following a straight line | Consult utility maps; mark known utility corridors on site plan |
| Geology | Broad, gentle gradients spanning the entire survey area | Usually removed by regional trend subtraction in post-processing |

---

## 6. Troubleshooting

### 6.1 Quick Reference Table

Consult this table first when something goes wrong in the field.

| Symptom | Likely Cause | Solution |
|---------|-------------|----------|
| LED stays fast blink (100 ms) after startup | SD card not detected | Remove SD card, reinsert firmly. Verify FAT32 format. Try a different card. |
| LED stays medium blink (500 ms) for >5 minutes | No GPS satellite lock | Move to open area with clear sky view. Check GPS antenna connection. Wait -- cold starts can take 2-3 minutes. If still no lock after 5 minutes, try a different location (buildings and trees block satellite signals). |
| No LED activity at all | No power | Check battery voltage (should be >7.0 V). Check power switch. Check regulator output (should be 5.0 V +/- 0.1 V). |
| SD write errors increasing (`SD_ERR:` in serial output) | SD card full, failing, or loose | Check remaining card space. Try a different card. Reseat the SD card connection. Power cycle. |
| All gradient values near zero | Sensor cables disconnected or ADC not communicating | Check all sensor cable connections at the crossbar and at the electronics enclosure. Verify both ADS1115 modules detected at startup (check serial output). Run an I2C scanner sketch if needed. |
| Gradient values very noisy (large random fluctuations) | Electromagnetic interference or poor shielding | Move away from power lines, vehicles, radio transmitters. Check that sensor cable shields are grounded at the enclosure end only. Verify battery voltage (low voltage increases ADC noise). |
| One pair reads differently from the others | Cable fault, sensor alignment, or ADC channel issue | Swap the suspect pair's cable with a known-good cable. Check that the sensor pair is vertically aligned (top and bottom sensors parallel). Check the ADC channel wiring. |
| Beeper not sounding | Beeper disabled in firmware, wiring issue, or buzzer failure | Verify `ENABLE_BEEPER` is set to 1 in `config.h`. Check buzzer wiring (pin D9 to buzzer positive, buzzer negative to GND). Test buzzer with a multimeter or direct 5 V. |
| Serial output garbled or unreadable | Baud rate mismatch | Set your terminal to 115200 baud, 8N1. Verify `SERIAL_BAUD` in `config.h` matches. |
| GPS coordinates present but track looks wrong | GPS multipath or position jumps | Common near buildings and under tree canopy. Filter in post-processing by removing samples with HDOP >5 (requires `GPS_LOG_QUALITY=1` in config). |
| Gradients spike at every step | Operator carrying magnetic material | Remove belt buckle, keys, phone, watch, steel-toe boots. Test by standing still. |
| Firmware version or pair count mismatch | Wrong build uploaded | Rebuild with correct `NUM_SENSOR_PAIRS` and platform flags. Verify with serial output at startup. |

### 6.2 Detailed Procedures

#### No GPS Lock After 5 Minutes

GPS modules require clear line-of-sight to at least 4 satellites. The NEO-6M typically acquires a fix in 30-60 seconds with good sky view (cold start), but can take longer in challenging conditions.

**Step-by-step diagnosis:**

1. Confirm the GPS module power LED is on (visible through the enclosure vent or cable gland window).
2. Move to the most open area available -- away from buildings, vehicles, dense tree canopy, and cliff faces.
3. Orient the GPS antenna toward the sky (it should face upward, not be shadowed by the crossbar or enclosure).
4. Wait 3 full minutes without moving.
5. If still no lock, check the serial debug output for NMEA sentences. If you see `$GPGGA` sentences with all zeros in the position fields, the GPS is receiving data but cannot compute a fix -- this is a sky-view problem. If you see no NMEA data at all, the GPS module is not communicating -- check wiring (GPS TX to Arduino pin D4) and baud rate (default 9600 for NEO-6M).
6. As a last resort, power cycle the entire system. Some GPS modules enter a bad state on rare occasions.

**Note:** Pathfinder logs data even without GPS lock -- timestamps and gradient values are still recorded. You can survey without GPS if you maintain a careful physical grid and record your position from field notes. The data will lack automatic georeferencing but remains scientifically usable.

#### SD Card Problems

The firmware automatically increments file numbers (PATH0001.CSV, PATH0002.CSV, etc.) and attempts to reopen the file after 5 consecutive write errors.

- **Card full:** The system creates files until the card runs out of space. A 2 GB card holds approximately 100 hours of survey data at 10 Hz with 4 pairs. Check free space before each survey day.
- **Card failing:** Repeated `SD_ERR` counts that do not reset after file reopen suggest a hardware fault. Swap to a new card immediately. Cheap SD cards have higher failure rates -- use name-brand Class 10 cards.
- **Card formatting:** Pathfinder requires FAT32. Cards larger than 32 GB are often pre-formatted as exFAT, which the SdFat library does not support. Reformat as FAT32 using your computer's disk utility.

#### High Noise

If gradient values fluctuate by more than 50 ADC counts when the system is stationary in a quiet location, investigate:

1. Battery voltage -- below 7.0 V, the regulator may drop out of regulation, increasing ADC noise.
2. Cable shielding -- sensor cable shields must be connected to ground at the electronics enclosure end only. Double-grounding creates a ground loop that injects 50/60 Hz noise.
3. I2C cable length -- keep the wires between the Arduino and the ADS1115 modules short (<30 cm). Long I2C runs pick up interference.
4. Nearby EMI sources -- power lines, cell towers, and running vehicle engines all generate magnetic noise. Survey at least 20 m from these sources.

---

## 7. Survey Log Template

Print this template and fill out one copy per survey session. Keep it with your SD card and field notebook.

```
================================================================================
PATHFINDER SURVEY LOG
================================================================================

Date: _______________     Operator: _________________________________

Site name: __________________________________________________________________

Site description / address: _________________________________________________

____________________________________________________________________________

GPS reference point:  Lat: _________________  Lon: _________________

Description of reference point: _____________________________________________

____________________________________________________________________________

Weather:  ☐ Clear  ☐ Overcast  ☐ Light rain  ☐ Windy  ☐ Cold (<5 C)  ☐ Hot (>35 C)

Temperature: _________ C (approx.)

================================================================================
EQUIPMENT
================================================================================

Instrument:    Pathfinder v_______     Platform: ☐ Handheld  ☐ Backpack  ☐ Drone

Sensor pairs:  _______                Sample rate: _______ Hz

GPS module:    ☐ NEO-6M (standard)    ☐ NEO-M9N    ☐ ZED-F9P (RTK)

Battery voltage at start: _______ V   Battery voltage at end: _______ V

SD card label/ID: _______________

================================================================================
SURVEY PARAMETERS
================================================================================

Survey area dimensions: _______ m x _______ m

Traverse direction: ☐ N-S  ☐ E-W  ☐ Other: ___________

Traverse line spacing: _______ m

Walking direction: ☐ Unidirectional  ☐ Bidirectional (zigzag)

Total number of traverse lines: _______

================================================================================
TIMING
================================================================================

Time start: _______________     Time end: _______________

GPS lock achieved at: _______________

Survey start time: _______________     Survey end time: _______________

SD files created: PATH________ to PATH________

================================================================================
CALIBRATION
================================================================================

Baseline reading taken:  ☐ Yes  ☐ No

Gradient baseline values (static, from test line or calibration):

  Pair 1: _______   Pair 2: _______   Pair 3: _______   Pair 4: _______

Last full calibration date: _______________

================================================================================
FIELD OBSERVATIONS
================================================================================

Known metal/interference near survey area:

____________________________________________________________________________

____________________________________________________________________________

Anomalies noted during survey (location, strength, description):

____________________________________________________________________________

____________________________________________________________________________

____________________________________________________________________________

____________________________________________________________________________

Equipment issues or unusual behavior:

____________________________________________________________________________

____________________________________________________________________________

General notes:

____________________________________________________________________________

____________________________________________________________________________

____________________________________________________________________________

____________________________________________________________________________

================================================================================
POST-SURVEY CHECKLIST
================================================================================

☐ Data downloaded from SD card
☐ Data verified (visualize_data.py or manual CSV check)
☐ Data backed up to second medium
☐ SD card relabeled and stored
☐ Equipment powered off and packed
☐ Site restored (stakes removed if required)
☐ This log form completed

Signature: __________________________     Date: _______________

================================================================================
```

---

## 8. Post-Processing Workflow Summary

This section provides a brief overview of what to do with your data after the survey. For detailed post-processing instructions, see the `firmware/tools/README.md` and `firmware/CALIBRATION.md` documents.

### 8.1 Basic Processing Steps

1. **Transfer** all CSV files from the SD card to your working directory.
2. **Visualize** each file:
   ```bash
   python visualize_data.py PATH0001.CSV          # Time series
   python visualize_data.py PATH0001.CSV --map     # Spatial map
   python visualize_data.py PATH0001.CSV --hist    # Gradient distributions
   python visualize_data.py PATH0001.CSV --stats-only  # Statistics only
   ```
3. **Apply calibration offsets** from your most recent calibration record (see `CALIBRATION.md`).
4. **Filter** GPS dropouts: remove rows where `lat` and `lon` are both zero.
5. **Grid** the point data to a regular grid using kriging or inverse-distance weighting. Standard GIS tools (QGIS, ArcGIS) and Python libraries (scipy, verde) can do this.
6. **Visualize** the gridded gradient map. Color-scale the gradient values to highlight anomalies.
7. **Mark anomalies** on a site plan with GPS coordinates and estimated gradient strength.
8. **Plan follow-up** using HIRT, GPR, or a commercial gradiometer for anomalies requiring detailed characterization.

### 8.2 Export to GIS

Pathfinder CSV data can be imported directly into QGIS or ArcGIS as delimited text with X=lon, Y=lat fields. For programmatic export, see the GeoJSON example in `firmware/tools/README.md`.

---

## 9. Reference Information

### 9.1 CSV Data Format

Each power cycle creates a new file: `PATH0001.CSV`, `PATH0002.CSV`, etc. The file begins with metadata comments and a header row.

**Standard format (4 pairs, no GPS quality logging):**

```csv
# Pathfinder v1.4.0 (2026-02-18) pairs=4
# epoch_offset=0 (relative ms, no RTC)
timestamp,lat,lon,g1_top,g1_bot,g1_grad,g2_top,g2_bot,g2_grad,g3_top,g3_bot,g3_grad,g4_top,g4_bot,g4_grad
12345,51.2345670,18.3456780,15234,15456,222,15189,15401,212,15150,15380,230,15210,15440,230
```

| Column | Description | Units |
|--------|-------------|-------|
| `timestamp` | Milliseconds since boot (or ISO 8601 UTC if RTC enabled) | ms or ISO string |
| `lat`, `lon` | GPS coordinates (0 if no fix) | Decimal degrees, WGS84 |
| `gN_top` | Top sensor raw ADC reading | 16-bit signed ADC counts |
| `gN_bot` | Bottom sensor raw ADC reading | 16-bit signed ADC counts |
| `gN_grad` | Gradient = bottom minus top | ADC counts |

With `GPS_LOG_QUALITY=1` (recommended for RTK or drone platforms), three additional columns appear after `lon`: `fix_quality`, `hdop`, and `altitude`.

### 9.2 Status LED Quick Reference

| LED Pattern | Meaning | System State |
|-------------|---------|-------------|
| Very fast blink (50 ms) | Critical error: no ADCs found | ADC-only diagnostics |
| Fast blink (100 ms) | SD card error | Not logging; fix SD card |
| Medium blink (500 ms) | No GPS fix | Logging data without position |
| **Slow blink (2 s)** | **Normal operation** | **GPS locked, logging to SD** |

### 9.3 Firmware Configuration Quick Reference

Key parameters in `firmware/include/config.h`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_SENSOR_PAIRS` | 4 (handheld), 2 (drone) | Number of gradiometer pairs |
| `SAMPLE_RATE_HZ` | 10 (handheld), 20 (drone) | Samples per second |
| `ENABLE_BEEPER` | 1 (handheld), 0 (drone) | Pace beeper on/off |
| `GPS_BAUD` | 9600 | GPS serial baud rate (115200 for ZED-F9P) |
| `GPS_LOG_QUALITY` | 0 (handheld), 1 (drone) | Log fix quality, HDOP, altitude |
| `SERIAL_DEBUG` | 1 | Serial debug output on/off |
| `SERIAL_BAUD` | 115200 | Debug serial baud rate |
| `SD_FLUSH_INTERVAL` | 10 | Flush SD every N samples |
| `ADC_SATURATION_THRESHOLD` | 32000 | ADC saturation warning level |
| `ENABLE_WATCHDOG` | 0 | Hardware watchdog (test first) |
| `ENABLE_RTC` | 0 | DS3231 real-time clock support |

### 9.4 Related Documentation

| Document | Location | Content |
|----------|----------|---------|
| Quick Start | `firmware/QUICK_START.md` | Pre-flight checklist and basic operation |
| Calibration | `firmware/CALIBRATION.md` | 5-level calibration procedures |
| Wiring | `firmware/WIRING.md` | Complete hardware assembly guide |
| Firmware README | `firmware/README.md` | Build, upload, and library information |
| Design Concept | `docs/design-concept.md` | System architecture and physics |
| Platform Variants | `docs/platform-variants.md` | Handheld, backpack, and drone configurations |
| Frame Design | `hardware/cad/frame-design.md` | Physical harness and crossbar construction |
| Data Tools | `firmware/tools/README.md` | Python visualization and export utilities |

---

*Pathfinder Field Operations Guide -- v1.0*
*For firmware version 1.4.0 and later*
*All detection depths and coverage rates are **(Target)** or **(Modeled)** unless noted otherwise.*
