# Pathfinder Calibration Guide

Procedures for calibrating and validating the Pathfinder gradiometer system.

## Overview

Calibration establishes baseline performance and corrects for:
1. **Sensor offset**: Each sensor's zero-field reading
2. **Gain variation**: Sensitivity differences between sensors
3. **Gradient baseline**: Top-bottom pair matching
4. **Temperature drift**: Environmental effects

## Pre-Calibration Checklist

- [ ] All sensors mechanically mounted and secure
- [ ] System powered and logging for 10+ minutes (thermal stabilization)
- [ ] Magnetically clean environment (no metal objects within 5m)
- [ ] GPS locked (for position logging)
- [ ] Notebook ready for recording values

## Level 1: Basic Offset Calibration

**Purpose**: Establish zero-field baseline for each sensor

**Location**: Magnetically quiet area (park, field, away from buildings/vehicles)

### Procedure

1. **Thermal Stabilization**
   - Power on system
   - Let run for 15 minutes before measurements
   - This allows electronics to reach stable temperature

2. **Static Measurement**
   - Place gradiometer on ground in level position
   - Ensure no metal objects nearby (>5m clear zone)
   - Record 60 seconds of data (600 samples at 10 Hz)
   - Do NOT move system during measurement

3. **Download and Analyze**
   ```bash
   python tools/visualize_data.py PATH_CALIB.CSV --stats-only
   ```

4. **Record Baseline Values**

   Note the **mean** value for each channel from statistics output:

   | Sensor | Mean ADC | Notes |
   |--------|----------|-------|
   | Pair 1 Top | _______ | |
   | Pair 1 Bot | _______ | |
   | Pair 2 Top | _______ | |
   | Pair 2 Bot | _______ | |
   | Pair 3 Top | _______ | |
   | Pair 3 Bot | _______ | |
   | Pair 4 Top | _______ | |
   | Pair 4 Bot | _______ | |

5. **Check Gradient Baseline**

   Each pair's gradient should be **near zero** (typically <50 ADC counts):

   | Pair | Gradient Mean | Status |
   |------|---------------|--------|
   | 1 | _______ | □ Pass (<50) |
   | 2 | _______ | □ Pass (<50) |
   | 3 | _______ | □ Pass (<50) |
   | 4 | _______ | □ Pass (<50) |

   **If gradient >100**: Sensors not properly balanced. Check:
   - Sensors at same height
   - Sensors parallel to ground
   - No metal near one sensor but not the other

## Level 2: Noise Floor Measurement

**Purpose**: Verify system meets target noise specifications

### Procedure

1. **Indoor Test** (electromagnetically shielded if possible)
   - Record 60 seconds stationary
   - Calculate standard deviation of gradient

2. **Noise Metrics**

   From statistics output, note **Std Dev** for gradients:

   | Pair | Gradient Std Dev | Target | Status |
   |------|------------------|--------|--------|
   | 1 | _______ | <20 | □ Pass |
   | 2 | _______ | <20 | □ Pass |
   | 3 | _______ | <20 | □ Pass |
   | 4 | _______ | <20 | □ Pass |

   **Target**: Gradient noise <20 ADC counts RMS

   **If noise >50**: Investigate:
   - Poor power supply filtering
   - Ground loops
   - Electromagnetic interference (RFI)
   - Loose sensor connections
   - Sensor cable shielding issues

## Level 3: Dynamic Response Test

**Purpose**: Verify system tracks magnetic changes correctly

### Test 1: Ferrous Object Response

1. **Baseline Measurement**
   - Record 10 seconds with no object present
   - Note mean gradient value

2. **Object Introduction**
   - Place small ferrous object (wrench, hammer) 50 cm below bottom sensor
   - Record 10 seconds
   - Note gradient change

3. **Object Removal**
   - Remove object
   - Record 10 seconds
   - Verify gradient returns to baseline

**Expected Results**:
- Clear gradient increase when object present (100-500 ADC counts)
- Return to baseline within ±10 counts after removal
- All pairs show similar sensitivity

### Test 2: Background Rejection

**Purpose**: Verify gradiometer cancels uniform fields

1. **Rotate System**
   - Hold gradiometer and slowly rotate 360° (one full turn over 60 seconds)
   - Record data throughout
   - Gradient should remain stable

**Expected Results**:
- Gradient variation <50 ADC counts during rotation
- No systematic drift with orientation

**If fails**:
- Sensors not vertically aligned
- Different sensor gains
- Sensor mounting not rigid

## Level 4: Inter-Pair Consistency

**Purpose**: Verify all 4 pairs respond similarly

### Procedure

1. **Controlled Anomaly**
   - Place calibrated ferrous target at known position
   - Record reading from all pairs
   - Compare response amplitude

2. **Consistency Check**

   | Pair | Gradient Response | Normalized |
   |------|------------------|------------|
   | 1 | _______ | 100% (reference) |
   | 2 | _______ | ____% |
   | 3 | _______ | ____% |
   | 4 | _______ | ____% |

   **Target**: All pairs within ±20% of reference

**If variation >30%**:
- Check sensor type/model consistency
- Verify sensor power supply voltage
- Inspect sensor mounting (different heights?)
- Check ADC wiring (swapped channels?)

## Level 5: GPS Positional Accuracy

**Purpose**: Verify GPS coordinates align with physical position

### Procedure

1. **Static Position Test**
   - Place system at known GPS coordinates (e.g., survey marker)
   - Record 5 minutes of data
   - Calculate GPS scatter

2. **GPS Metrics**

   From visualize_data.py output:
   - **Lat scatter**: ____________ degrees (target: <0.00001° = ~1m)
   - **Lon scatter**: ____________ degrees (target: <0.00001° = ~1m)
   - **Fix percentage**: _________% (target: >95%)

**If fix <90%**:
- Improve antenna placement (clear sky view)
- Check antenna connection
- Wait longer for satellite acquisition
- Consider external active antenna

## Calibration Data Recording

### Template

```
PATHFINDER CALIBRATION RECORD
Date: _______________
Location: _______________
Temperature: _______°C
Humidity: _______%

SENSOR BASELINES (ADC counts):
Pair 1: Top=______ Bot=______ Grad=______
Pair 2: Top=______ Bot=______ Grad=______
Pair 3: Top=______ Bot=______ Grad=______
Pair 4: Top=______ Bot=______ Grad=______

NOISE FLOOR (ADC counts RMS):
Pair 1: ______
Pair 2: ______
Pair 3: ______
Pair 4: ______

DYNAMIC TESTS:
Ferrous Response: □ Pass □ Fail
Rotation Test: □ Pass □ Fail
Inter-Pair Consistency: □ Pass □ Fail

GPS PERFORMANCE:
Fix Rate: ______%
Position Scatter: ______ meters

NOTES:
_________________________________
_________________________________
_________________________________

Calibrated by: _______________
```

## Applying Calibration in Post-Processing

After calibration, apply corrections to survey data:

### Python Example

```python
import pandas as pd

# Load survey data
df = pd.read_csv('PATH0001.CSV')

# Calibration offsets from Level 1
offsets = {
    'g1_grad': 25,   # Example values
    'g2_grad': 18,
    'g3_grad': -12,
    'g4_grad': 22,
}

# Apply offset correction
for col, offset in offsets.items():
    df[f'{col}_cal'] = df[col] - offset

# Now df['g1_grad_cal'] etc. are offset-corrected gradients
```

### Gain Correction (Advanced)

If sensors show different sensitivities (Level 4), apply gain correction:

```python
# Gain factors from Level 4 (normalized to pair 1)
gains = {
    'g1_grad': 1.00,
    'g2_grad': 0.95,  # 5% less sensitive
    'g3_grad': 1.08,  # 8% more sensitive
    'g4_grad': 1.02,
}

# Apply gain correction
for col, gain in gains.items():
    df[f'{col}_cal'] = df[f'{col}_cal'] / gain
```

## Recalibration Schedule

- **Daily**: Quick baseline check (5 min static measurement)
- **Weekly**: Full Level 1-2 calibration
- **Monthly**: Complete Level 1-5 validation
- **After repair**: Full calibration sequence
- **Season change**: Recheck offsets (temperature effects)

## Field Calibration Procedure

Quick baseline check before each survey:

1. Power on, wait 5 minutes (thermal stabilization)
2. Record 30 seconds static in survey area
3. Check gradient means <100 ADC counts
4. If >100, investigate:
   - System on level ground?
   - Metal objects nearby?
   - Sensor mount loosened during transport?

## Troubleshooting Calibration Issues

### Large Gradient Offsets (>200 counts)

**Possible Causes**:
1. Sensors at different heights
2. Metal contamination near one sensor
3. Sensor gain mismatch
4. Poor mounting (vibration during measurement)

**Solutions**:
- Verify mechanical alignment with level
- Measure sensor separation (should be 50 cm ±1 cm)
- Check for metal screws, bolts in mounting
- Rigidly secure sensors to prevent vibration

### High Noise (>50 counts RMS)

**Possible Causes**:
1. Power supply noise
2. Ground loops
3. Electromagnetic interference
4. Sensor cable issues

**Solutions**:
- Add capacitors (10µF, 0.1µF) at sensor power pins
- Ensure single-point ground (no loops)
- Shield sensor cables, ground at one end only
- Move away from power lines, WiFi, cell towers
- Lower ADC data rate (128 → 64 SPS)

### GPS Poor Fix Rate

**Possible Causes**:
1. Obstructed sky view
2. Metallic objects nearby
3. Antenna placement
4. Atmospheric conditions

**Solutions**:
- Move to open area
- Raise antenna higher
- Use external active antenna
- Wait longer for satellite acquisition (2+ minutes)

## Validation Targets

Use known targets to verify system performance:

### Buried Pipe Test

1. Bury steel pipe (10 cm diameter, 1m length) at 30 cm depth
2. Survey over pipe location
3. Verify clear gradient anomaly (>200 counts)
4. Compare to theoretical model

### Reference Anomaly

Create permanent calibration site:
- Bury ferrous object at known location
- Record GPS coordinates
- Use for periodic validation checks

## Advanced: Temperature Compensation

If operating in varying temperatures:

1. **Measure Temperature Effect**
   - Record data while temperature changes (morning to noon)
   - Plot gradient vs. temperature
   - Calculate drift coefficient (ADC counts / °C)

2. **Apply Compensation**
   - Add temperature sensor (DS18B20)
   - Correct readings in post-processing:
   ```python
   drift_coeff = 2.5  # ADC counts per °C
   temp_ref = 20.0    # Reference calibration temp
   df['g1_grad_temp_comp'] = df['g1_grad'] - drift_coeff * (temp - temp_ref)
   ```

## Support

For calibration questions:
- See `README.md` for hardware troubleshooting
- See `WIRING.md` for connection issues
- Project repository for community support
