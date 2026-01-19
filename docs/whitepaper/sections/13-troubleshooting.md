# 13. Troubleshooting

## Overview

This section provides diagnostic approaches and solutions for common issues encountered during HIRT field operations.

---

## 13.1 Diagnostic Approach

### System Check Procedure

1. **Power check:** Verify all probes have power (LED indicators)
2. **Communication check:** Ping each probe, verify response
3. **Sync check:** Verify timebase is distributed correctly
4. **Calibration check:** Run quick calibration on one probe
5. **Coupling test:** Test TX->RX on two probes in air

### Quick Reference Table

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| **RX saturation on nearby probes** | TX too strong; direct coupling | Lower TX level; increase TX-RX separation; rotate coils |
| **Noisy MIT data** | Poor shielding; ground loop | Use twisted/shielded pairs; single-point ground; increase integration time |
| **Unstable ERT voltages** | Poor ring contact / dry sand | Pre-moisten hole; saline gel; ensure solid ring-soil contact |
| **No depth sensitivity** | Spacing too small; too high freq | Add longer offsets; include 2-5 kHz MIT; use longer ERT baselines |
| **Inconsistent repeats** | Loose connectors; timebase drift | Re-seat connectors; verify sync; redo calibration/QC pass |
| **Probe not responding** | Cable issue or hub port | Check cable connection; try different port; swap cable |
| **Base hub not powering on** | Battery or fuse issue | Check battery voltage; check fuse; verify power switch |

---

## 13.2 MIT Troubleshooting

### RX Saturation on Nearby Probes

**Symptoms:**
- Amplitude readings maxed out on nearby probes
- Phase readings erratic or stuck
- Only affects probes close to TX

**Causes:**
- TX output too high for close-range measurements
- Direct magnetic coupling between TX and RX coils
- Insufficient separation or poor coil orientation

**Solutions:**
1. **Reduce TX level:** Lower DDS output or TX driver gain
2. **Increase separation:** Use wider probe spacing for close pairs
3. **Rotate coils:** Ensure TX and RX coils are orthogonal
4. **Add attenuation:** Use lower gain on RX for nearby probes
5. **Skip close pairs:** Don't measure TX->RX pairs < 0.5 m apart

### Noisy MIT Data

**Symptoms:**
- High noise floor
- Erratic amplitude/phase readings
- Inconsistent measurements

**Causes:**
- Poor cable shielding
- Ground loops (multiple ground paths)
- EMI from nearby sources
- Insufficient integration time

**Solutions:**
1. **Check shielding:** Verify all cables are properly shielded
2. **Single-point ground:** Ensure only one ground connection
3. **Twist pairs:** Use twisted-pair cables for signal lines
4. **Increase integration:** Longer measurement time reduces noise
5. **Check EMI sources:** Move away from power lines, vehicles, radios
6. **Verify connections:** Check all connectors are tight and clean

### No Depth Sensitivity (MIT)

**Symptoms:**
- Measurements don't change with depth
- All readings similar regardless of target depth
- No response to known deep targets

**Causes:**
- Frequency too high (limited penetration)
- Probe spacing too small
- Targets too deep for configuration

**Solutions:**
1. **Lower frequencies:** Use 2-5 kHz for MIT (deeper penetration)
2. **Add longer offsets:** Include TX->RX pairs with >3 m spacing
3. **Deeper probes:** Insert probes deeper if possible
4. **Check targets:** Verify expected target depths are realistic

---

## 13.3 ERT Troubleshooting

### Unstable ERT Voltages

**Symptoms:**
- Voltage readings drift or jump
- Inconsistent measurements
- Poor contact indicated

**Causes:**
- Dry soil preventing good electrical contact
- Poor ring-to-soil contact
- Loose connections
- Polarization effects

**Solutions:**
1. **Pre-moisten hole:** Add water to improve contact
2. **Use saline gel:** Apply conductive gel around rings
3. **Check ring contact:** Ensure rings are flush with soil
4. **Reverse polarity:** Use AC or periodic polarity reversal
5. **Check connections:** Verify all wiring is secure
6. **Increase current:** Slightly higher current may improve SNR

### ERT Contact Problems

**Symptoms:**
- High contact resistance
- Erratic voltage readings
- Some electrodes not responding

**Causes:**
- Dry soil conditions
- Poor ring-soil interface
- Corroded or dirty contacts

**Solutions:**
1. **Verify ring contact with soil**
2. **Add water around probe** (improve contact)
3. **Check ring-to-cable connection**
4. **Clean rings if accessible**
5. **Consider relocating probe** if contact remains poor

### No Depth Sensitivity (ERT)

**Symptoms:**
- All measurements show surface effects only
- No response to known deep features
- Uniform readings across array

**Causes:**
- ERT baselines too short
- All measurements near-surface only

**Solutions:**
1. **Longer ERT baselines:** Corner-to-corner, edge-to-edge injections
2. **Deeper probes:** Insert probes deeper if possible
3. **Wider spacing:** Increase array dimensions

---

## 13.4 Power Issues

### Probe Not Responding

**Symptoms:**
- No LED indicator
- No communication with base hub
- Probe not detected in diagnostic scan

**Causes:**
- Cable disconnection or damage
- Power issue at probe
- Base hub port malfunction

**Solutions:**
1. **Check cable connections** (both ends)
2. **Test with multimeter** (continuity)
3. **Try different port on base hub**
4. **Swap to spare cable**
5. **Restart probe** (disconnect/reconnect power)
6. **Replace with spare probe** if available

### Base Hub Not Powering On

**Symptoms:**
- No power LED
- No response to power switch
- System completely dead

**Causes:**
- Depleted battery
- Blown fuse
- Power switch failure

**Solutions:**
1. **Check battery voltage** (should be >12.0 V)
2. **Check fuse** (replace if blown)
3. **Verify power switch** operation
4. **Check internal connections** if accessible
5. **Use backup power supply** if available

---

## 13.5 Communication Issues

### Lost Probe Communication

**Symptoms:**
- Probe was working, now unresponsive
- Intermittent connection
- Partial data received

**Causes:**
- Cable damage or disconnection
- Connector corrosion
- Interference

**Solutions:**
1. **Check cable connections** (both ends)
2. **Inspect cable for damage** (kinks, cuts)
3. **Clean connectors** with contact cleaner
4. **Swap to known-good cable**
5. **Check for EMI sources** nearby
6. **Restart probe and base hub**

### Sync Problems

**Symptoms:**
- Timing errors in data
- Inconsistent measurements between probes
- Data corruption

**Causes:**
- Timebase distribution failure
- Clock drift
- Cable issues on sync line

**Solutions:**
1. **Verify sync signal** at each probe
2. **Check sync cable** for damage
3. **Restart measurement system**
4. **Re-initialize sync** before continuing

---

## 13.6 Environmental Factors

### Temperature Effects

**Symptoms:**
- Readings drift over time
- Morning vs. afternoon differences
- Inconsistent calibration

**Causes:**
- Electronics temperature sensitivity
- Soil property changes with temperature

**Solutions:**
1. **Allow system to stabilize** (10-15 min warmup)
2. **Note temperature variations** in field log
3. **Apply temperature compensation** if available
4. **Take baseline measurements** at current temperature
5. **Shield equipment from direct sun** if possible

### Weather Impacts

**Symptoms:**
- Data quality degradation
- Increased noise
- Contact problems

**Causes:**
- Rain affecting connections
- Wind causing cable movement
- Humidity affecting electronics

**Solutions:**
1. **Protect connections** from moisture
2. **Secure cables** against wind movement
3. **Use weather covers** for sensitive equipment
4. **Postpone measurements** in severe conditions

---

## 13.7 Field Repairs

### Field Repair Kit

Keep these items handy:
- Spare cables (2-3)
- Spare probes (2-4 recommended)
- Electrical tape
- Multimeter
- Small screwdriver set
- Contact cleaner spray
- Zip ties
- Heat-shrink tubing
- Solder and iron (battery powered, optional)
- Notebook and pencil (for notes)
- Calibration sheet

### Emergency Cable Repair

If a cable is damaged in the field:
1. **Locate the break** (continuity test)
2. **Cut out damaged section** if possible
3. **Strip and splice** wires
4. **Insulate with tape** or heat-shrink
5. **Test before use**

*Note: Field-repaired cables should be replaced at first opportunity.*

### Connector Cleaning

1. **Apply contact cleaner** to connector
2. **Wipe with clean cloth**
3. **Allow to dry** before reconnecting
4. **Apply thin layer of dielectric grease** (optional, for moisture protection)

---

## 13.8 When to Abort Survey

### Conditions Requiring Survey Abort

**Safety Issues:**
- Severe weather (lightning, high winds)
- Site hazards discovered
- Equipment malfunction creating hazard

**Data Quality Issues:**
- More than 25% of probes non-functional
- Persistent noise making data unusable
- Unable to achieve ground contact on majority of probes
- Critical equipment failure (base hub, data logger)

**Practical Issues:**
- Battery depletion with no backup
- Time constraints preventing quality data collection
- Site access revoked

### Before Aborting

1. **Document all issues** in field log
2. **Save all data collected** (even partial)
3. **Note probe positions** for potential return
4. **Photograph site conditions**
5. **Extract probes safely** if time permits
6. **Backup data immediately**

### Partial Survey Options

If full abort is not necessary:
- **Reduce survey area** to functional probes
- **Simplify measurement protocol** (fewer frequencies, fewer patterns)
- **Focus on priority targets** only
- **Document limitations** for data interpretation

---

## 13.9 Prevention

### Pre-Field Checks

- [ ] All connectors tight and clean
- [ ] Cables tested for continuity
- [ ] Calibration up to date
- [ ] Spare parts available
- [ ] Field diagnostic kit packed
- [ ] Batteries charged
- [ ] Weather forecast reviewed

### During Survey

- Monitor data quality in real-time
- Check reciprocity periodically
- Note any anomalies immediately
- Keep spare probes ready
- Protect equipment from weather

### Post-Survey

- Inspect all equipment
- Note any issues for repair
- Update calibration records
- Clean and store properly
- Recharge all batteries
- Restock field repair kit

---

*For detailed operating procedures, see Section 10: Field Operations. For data format specifications, see Section 11: Data Recording.*
