# Pathfinder-to-HIRT Handoff Protocol

## Overview

Pathfinder and HIRT form a two-stage geophysical workflow. Pathfinder screens large areas quickly using magnetic gradiometry, walking at 1 m/s to cover thousands of square meters per hour. When Pathfinder identifies anomalies of interest, HIRT deploys subsurface probes at those locations for detailed 3D tomographic imaging using MIT-3D (electromagnetic induction) and ERT-Lite (electrical resistivity).

This division of labor matters because HIRT deployment is labor-intensive---inserting 20-24 probes into the ground, routing cables, and running multi-frequency measurement sweeps takes 2-3 hours per 10x10 m section. Without prior screening, an operator must guess where to deploy, risking hours of effort on empty ground. Pathfinder eliminates that guesswork by identifying exactly which areas contain anomalies worth investigating.

This document provides the operational procedure for transitioning from a completed Pathfinder survey to a targeted HIRT deployment.

## Prerequisites

Before beginning the handoff workflow, confirm the following are in place:

- **Completed Pathfinder survey** with GPS-tagged anomaly data exported as CSV from the SD card
- **HIRT system assembled and calibrated** per the HIRT Technical Manual (see `HIRT/docs/field-guide/deployment.qmd`)
- **GeoSim coordinate tools installed** for coordinate transformations:
  ```bash
  pip install -e /path/to/GeoSim
  ```
- **Python environment** with `pandas` and `matplotlib` available (for Pathfinder visualization tools)
- **Field equipment** for grid stakeout: measuring tape (30 m minimum), survey stakes, ground marker paint or flags, compass or GPS unit

## Step 1: Pathfinder Survey Completion

### Verify Data Quality

After completing the Pathfinder walking survey, verify the collected data before leaving the site.

1. **Remove the SD card** and copy the CSV file (e.g., `PATH0001.CSV`) to a laptop.

2. **Run the visualization tool** to check data completeness:
   ```bash
   python firmware/tools/visualize_data.py PATH0001.CSV --stats-only
   ```
   This prints a summary including GPS lock percentage, gradient statistics for each sensor pair, survey duration, and sample rate. Confirm:
   - GPS lock rate exceeds 80% (ideally >90%)
   - All four gradient channels show reasonable mean and standard deviation values
   - No channels report near-zero standard deviation (indicates a dead sensor)
   - No channels show persistent saturation (values at +/- 32000 ADC counts)

3. **Generate a spatial map** to visually confirm coverage:
   ```bash
   python firmware/tools/visualize_data.py PATH0001.CSV --map
   ```
   The map overlays gradient values on GPS positions. Look for gaps in coverage (missed swaths), GPS drift (track wandering off the actual path), and obvious anomaly clusters.

4. **Check GPS fix quality** if logged (firmware config `GPS_LOG_QUALITY=1`). When the `fix_quality` and `hdop` columns are present, filter for samples with poor HDOP (>5.0) and consider whether those readings affect anomaly positions.

### Record Survey Metadata

Document the following in a field notebook and in a digital metadata file alongside the CSV:

| Field | Example | Purpose |
|-------|---------|---------|
| Survey date/time | 2026-03-15 09:30-10:45 | Correlation with HIRT deployment |
| Operator | J. Smith | Accountability, technique notes |
| Site name/ID | Clearing-North-7 | Cross-reference across datasets |
| Weather | Overcast, 12 C, dry | Soil moisture affects HIRT decisions |
| GPS module | NEO-6M / ZED-F9P | Determines position accuracy budget |
| Firmware version | 1.4.0 | Data format compatibility |
| Sensor pair count | 4 | Swath width calculation |
| Walking direction | N-S traverses, 1.5 m line spacing | Coverage pattern |
| Known landmarks | Fence post at SW corner, trail at N edge | Ground truth for GPS validation |

## Step 2: Anomaly Classification and HIRT Deployment Decision

### Classification Table

Use the Pathfinder gradient map to classify each identified anomaly. The following table provides guidelines for prioritizing HIRT follow-up based on anomaly characteristics:

| Anomaly Characteristic | Gradient Strength (nT) | Spatial Extent | HIRT Priority | Rationale |
|------------------------|------------------------|----------------|---------------|-----------|
| Strong, compact dipole | > 200 | < 2 m diameter | **HIGH** | Likely discrete metallic object; HIRT provides depth, size, and material information |
| Broad, diffuse anomaly | 50-200 | 2-10 m | **MEDIUM** | Possible disturbed ground, filled feature, or deep target; HIRT distinguishes fill from intact soil |
| Linear feature | Any | Extended/continuous | **LOW** | Likely utility pipe, fence line, or geological contact; identity usually clear from shape alone |
| Weak, isolated | < 50 | < 1 m | **LOW** | May be sensor noise, small shallow fragment, or geological variation; revisit only if context demands it |

### Decision Factors

Beyond the classification table, several factors influence whether a particular anomaly warrants HIRT deployment:

**Target depth estimate.** Deeper targets produce weaker magnetic gradients because the field falls off as 1/r^3. A moderate-strength anomaly (50-100 nT) at an estimated depth of 1-2 m may represent a large, significant target that HIRT's crosshole geometry is uniquely suited to image. Shallower targets that produce strong gradients may be identifiable through simpler means (hand excavation, metal detector).

**Soil conditions.** Clay-rich soils have high electrical conductivity, which makes HIRT's MIT-3D channel particularly effective---the inductive response is stronger in conductive ground. Sandy, dry soils favor the ERT-Lite channel. Knowing the soil type helps predict which HIRT modality provides the most useful data.

**Available time and equipment.** A full HIRT deployment takes 2-3 hours per section. If time is limited, prioritize the highest-ranked anomalies. If equipment must be transported long distances, cluster nearby anomalies into a single grid to maximize the return on deployment effort.

**Safety considerations.** At sites with UXO (unexploded ordnance) risk, Pathfinder screening is a critical precursor to any intrusive work. HIRT probe insertion at UXO sites requires additional safety protocols (see `HIRT/docs/field-guide/deployment.qmd`, Section "Pre-Push UXO Assessment"). The Pathfinder data provides the magnetic anomaly map that informs standoff distances and exclusion zones.

**Scientific or forensic objectives.** Archaeological surveys may require imaging every anomaly above a threshold, regardless of HIRT priority level. Forensic searches prioritize anomalies consistent with disturbed ground signatures. Match the deployment decision to the investigation's objectives.

### Decision Record

For each anomaly, record the deployment decision in a table:

| Anomaly ID | Lat | Lon | Gradient (nT) | Extent (m) | Classification | HIRT Decision | Notes |
|------------|-----|-----|----------------|------------|----------------|---------------|-------|
| A1 | 51.23462 | 18.34571 | 340 | 1.2 | Strong dipole | DEPLOY | Highest priority |
| A2 | 51.23448 | 18.34590 | 85 | 4.5 | Broad diffuse | DEPLOY | Possible pit feature |
| A3 | 51.23470 | 18.34555 | 120 | 15+ | Linear | SKIP | Follows known fence line |

## Step 3: Coordinate Transformation

Pathfinder records positions in WGS84 geographic coordinates (latitude/longitude in decimal degrees). HIRT operates in a local Cartesian grid (meters) centered on a physical reference point. Bridging these coordinate systems is essential for placing the HIRT probe grid on top of the Pathfinder anomaly locations.

### Establish the Grid Origin

Select a grid origin point that serves as the (0, 0) reference for the HIRT local grid. Good choices include:

- The center of the strongest anomaly cluster
- A prominent, recoverable landmark (fence post, survey monument, large tree)
- An arbitrary point marked with a driven survey stake and a GPS reading

The origin must be physically marked on the ground with a stake, flag, or paint and recorded with the best available GPS fix. Take multiple GPS readings and average them if using standard consumer GPS (NEO-6M). With RTK GPS (ZED-F9P), a single fix suffices.

**Record the origin coordinates prominently** in:
- The field notebook (written by hand)
- A digital metadata file alongside the survey data
- Optionally, a photo of the GPS screen at the origin stake

### Transform Anomaly Positions to Grid Coordinates

GeoSim provides coordinate transformation utilities for converting GPS positions to a local grid. Once GeoSim is installed, use the following approach:

```python
import numpy as np

# Define grid origin from GPS reading at origin stake
origin_lat = 51.23462   # degrees, WGS84
origin_lon = 18.34571   # degrees, WGS84

# Anomaly positions from Pathfinder CSV
anomaly_lat = np.array([51.23462, 51.23448, 51.23470])
anomaly_lon = np.array([18.34571, 18.34590, 18.34555])

# Approximate conversion: degrees to meters at this latitude
# 1 degree latitude ~ 111,320 m everywhere
# 1 degree longitude ~ 111,320 * cos(latitude) m
lat_scale = 111320.0  # m/degree
lon_scale = 111320.0 * np.cos(np.radians(origin_lat))  # m/degree

# Convert to local grid (meters from origin)
grid_x = (anomaly_lon - origin_lon) * lon_scale   # East direction
grid_y = (anomaly_lat - origin_lat) * lat_scale    # North direction

for i, (x, y) in enumerate(zip(grid_x, grid_y)):
    print(f"Anomaly {i+1}: grid_x = {x:.1f} m, grid_y = {y:.1f} m")
```

This flat-Earth approximation introduces less than 0.01% error at survey scales (<1 km) and is adequate for HIRT grid placement. For larger areas or higher precision, use a proper geodetic library (e.g., `pyproj`).

### Verify the Transform

Sanity-check the transformation by converting a known landmark position (not the origin) and verifying that the calculated grid distance matches a tape measure distance on the ground. Discrepancies exceeding 2 m indicate a GPS averaging or datum issue that must be resolved before proceeding.

## Step 4: HIRT Probe Grid Design

With anomaly positions converted to grid coordinates, design the HIRT probe array to cover the target area.

### Grid Layout Guidelines

| Parameter | Guideline | Rationale |
|-----------|-----------|-----------|
| Grid center | Strongest anomaly position | Maximizes sensitivity at the primary target |
| Grid extent | Extend 2-3 m beyond anomaly boundary in all directions | HIRT needs measurements from undisturbed ground surrounding the target to constrain the inversion |
| Probe spacing | 1.0 m, 1.5 m, or 2.0 m | Closer spacing improves resolution but requires more probes; see selection guide below |
| Minimum probes per side | 4 | Fewer than 4 probes per side provides insufficient measurement diversity for tomographic reconstruction |
| Total probe count | 12-24 typical | Balances resolution against deployment time |

### Spacing Selection Guide

| Target Depth Estimate | Recommended Spacing | Probe Count (10 m grid) |
|-----------------------|---------------------|-------------------------|
| < 1 m (shallow) | 1.0 m | 25-36 |
| 1-2 m (moderate) | 1.5 m | 16-25 |
| > 2 m (deep) | 2.0 m | 12-16 |

Deeper targets produce smoother anomaly fields and benefit less from closely-spaced probes. Shallow targets require finer spacing to resolve their lateral extent.

### Account for GPS Positioning Error

The accuracy budget (see table below) determines how much the HIRT grid must be expanded beyond the nominal anomaly position to ensure the target falls within the probe array.

With standard consumer GPS (NEO-6M, +/-2-5 m accuracy), expand the grid by at least 3 m in every direction beyond what the anomaly map suggests. With RTK GPS (ZED-F9P, +/-0.02 m accuracy), the Pathfinder anomaly positions are precise enough to center the grid directly.

### Example Grid Design

Suppose Pathfinder identifies a strong dipole anomaly (A1) at grid position (5.2, 3.8) with an estimated extent of 1.5 m diameter, and the survey used standard NEO-6M GPS.

1. **Center the grid** at (5, 4) -- rounded for stakeout convenience.
2. **Anomaly radius**: 0.75 m. Add 2.5 m buffer for undisturbed ground, plus 4 m for GPS uncertainty.
3. **Required coverage radius**: ~7 m from center.
4. **Grid design**: 8x8 m grid at 2.0 m spacing = 5x5 = 25 probes.

```
Grid: 8m x 8m, centered at (5, 4)
Probe spacing: 2.0 m
Probes: 5 x 5 = 25

        1.0   3.0   5.0   7.0   9.0    (grid_x, meters)
 8.0     o     o     o     o     o
 6.0     o     o     o     o     o
 4.0     o     o    [A1]   o     o      <- anomaly near center
 2.0     o     o     o     o     o
 0.0     o     o     o     o     o
```

With RTK GPS, the same anomaly could be covered by a tighter 4x4 m grid at 1.0 m spacing (25 probes) since GPS uncertainty is negligible.

## Step 5: Field Stakeout

Transfer the designed grid from paper to the ground.

### Stakeout Procedure

1. **Locate the grid origin stake** placed during Step 3. If time has passed since the Pathfinder survey, verify the stake is undisturbed by checking the GPS reading.

2. **Establish the grid baseline.** Using a compass or GPS bearing, lay a measuring tape from the origin in the primary grid direction (typically North). Mark the tape at each probe interval.

3. **Mark probe positions.** From each baseline mark, measure perpendicular offsets to place stakes at every grid node. Use a second tape held at 90 degrees, verified by the 3-4-5 triangle method for right angles.

4. **Drive survey stakes** at each probe position. Use the same numbered stakes and color-coded zone system described in the HIRT deployment guide (`HIRT/docs/field-guide/deployment.qmd`, "Cable Management and Field Logistics").

5. **Verify geometry.** Cross-measure the diagonal of the grid. For a square grid of side length L, the diagonal should equal L * sqrt(2). Accept deviations under 10 cm for standard surveys, under 5 cm for high-resolution work.

### Improving Accuracy

| Method | Equipment | Position Accuracy | Stakeout Time |
|--------|-----------|-------------------|---------------|
| Tape measure from origin | 30 m tape, compass | +/-10-20 cm | 30-60 min |
| Total station | Optical survey instrument | +/-1-2 cm | 45-90 min |
| RTK GPS with stakeout mode | ZED-F9P or equivalent | +/-2-5 cm | 20-40 min |

For forensic or high-precision archaeological surveys, total station or RTK GPS stakeout is recommended. For general screening where the probe grid is intentionally oversized, tape measure stakeout is adequate.

### Documentation

Before beginning probe insertion:

1. Photograph the staked grid from at least two angles, with a north arrow and scale reference visible.
2. Record the grid origin coordinates, grid dimensions, probe spacing, and any offset probe positions in the survey metadata file.
3. Sketch the grid layout in the field notebook, noting any probes relocated due to obstructions (trees, rocks, standing water).

## Step 6: Data Correlation

After HIRT data collection, correlating the HIRT results with the original Pathfinder anomaly map provides powerful diagnostic capability.

### Shared Coordinate Reference

Both datasets reference the same grid origin established in Step 3. The Pathfinder anomaly positions, transformed to grid coordinates in Step 3, overlay directly onto the HIRT inversion results.

To produce a combined visualization:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load Pathfinder anomaly positions (in grid coordinates from Step 3)
pf_anomalies = pd.DataFrame({
    'grid_x': [5.2, -3.1, 8.4],
    'grid_y': [3.8, 6.2, 1.0],
    'gradient_nT': [340, 85, 120],
    'label': ['A1', 'A2', 'A3']
})

# HIRT inversion results would be loaded from the inversion output
# For illustration, plot Pathfinder anomalies on the HIRT grid
fig, ax = plt.subplots(figsize=(8, 8))
scatter = ax.scatter(
    pf_anomalies['grid_x'], pf_anomalies['grid_y'],
    c=pf_anomalies['gradient_nT'], cmap='RdYlBu_r',
    s=200, edgecolors='black', zorder=5
)
for _, row in pf_anomalies.iterrows():
    ax.annotate(row['label'], (row['grid_x'], row['grid_y']),
                textcoords="offset points", xytext=(10, 10))

plt.colorbar(scatter, label='Pathfinder gradient (nT)')
ax.set_xlabel('Grid X (m, East)')
ax.set_ylabel('Grid Y (m, North)')
ax.set_aspect('equal')
ax.set_title('Pathfinder Anomalies on HIRT Grid')
plt.show()
```

### Using Pathfinder Data as Prior Information

The Pathfinder anomaly map constrains HIRT inversion in two ways:

1. **Spatial prior.** Anomaly locations from Pathfinder indicate where conductivity or susceptibility contrasts are expected. The inversion algorithm can use these positions as initial model seeds, reducing the number of iterations needed for convergence and improving solution uniqueness.

2. **Depth prior.** The magnetic gradient magnitude provides a rough depth estimate through the "half-width rule" (depth approximately equals the half-width of the anomaly at half-maximum amplitude). This estimate, while approximate, constrains the vertical extent of the initial model.

### Metadata Linkage

Record all coordinate transformations and cross-references in the survey metadata:

| Field | Value | Purpose |
|-------|-------|---------|
| Grid origin (lat, lon) | 51.23462, 18.34571 | Common reference point |
| Grid origin (physical) | Steel stake with orange flag, 2 m south of large oak | Recovery on future visits |
| Pathfinder CSV file | PATH0001.CSV | Source data |
| HIRT data directory | HIRT_20260315_Clearing7/ | Inversion input |
| Coordinate transform | Flat-Earth, origin = (51.23462, 18.34571) | Reproducibility |
| Grid alignment | X = East, Y = North | Convention consistency |

## Accuracy Budget

The total positional uncertainty in locating a Pathfinder anomaly on the HIRT grid accumulates through multiple error sources:

| Source | Error (m) | Notes |
|--------|-----------|-------|
| Pathfinder GPS (NEO-6M) | +/-2-5 | Standard consumer GPS, single-frequency, no correction |
| Pathfinder GPS (ZED-F9P RTK) | +/-0.02 | RTK-corrected, dual-frequency |
| Grid origin GPS reading | +/-2-5 or +/-0.02 | Same GPS unit, averaged reading helps |
| Grid origin stakeout | +/-0.1 | Tape measure from GPS point to physical stake |
| Probe placement from stakes | +/-0.05 | Measured from survey stakes |
| **Total (standard GPS)** | **+/-2.1-5.1** | **Expand HIRT grid by this margin** |
| **Total (RTK GPS)** | **+/-0.12** | **Direct probe placement feasible** |

The standard GPS error dominates the budget. Two strategies mitigate this:

1. **Oversize the HIRT grid** to account for the GPS uncertainty envelope. A 3-5 m expansion in every direction ensures the target falls within the probe array even with worst-case GPS error.
2. **Upgrade to RTK GPS.** The ZED-F9P module costs approximately $200 and reduces positioning error by two orders of magnitude. With RTK, the HIRT grid can be tightly targeted to the anomaly, saving deployment time and probes.

## Example Workflow

This section walks through a concrete example from start to finish.

### Scenario

A team surveys a 50x50 m clearing in a forest, searching for a buried concrete foundation (archaeological context). The Pathfinder system uses a standard NEO-6M GPS module.

### Pathfinder Survey (Day 1 Morning)

1. The operator walks N-S traverses at 1.5 m line spacing, covering the 50x50 m area in approximately 33 traverses.
2. At a walking speed of 1 m/s and 10 Hz sample rate, each 50 m traverse takes 50 seconds and produces 500 data points.
3. Total survey time: approximately 35 minutes including turns.
4. Total data: approximately 16,500 samples across 33 traverses.

### Data Review (Day 1 Midday)

1. The operator runs `visualize_data.py PATH0001.CSV --map` and identifies three anomaly clusters:
   - **A1**: Strong dipole (340 nT), compact (1.2 m), at the center of the clearing. Priority: HIGH.
   - **A2**: Broad diffuse anomaly (85 nT), 4.5 m extent, near the east edge. Priority: MEDIUM.
   - **A3**: Linear feature (120 nT) running N-S along the west boundary. Priority: LOW (identified as an old fence line from historical maps).

2. Decision: Deploy HIRT at A1 (highest priority) and A2 (if time permits). Skip A3.

### Coordinate Transformation (Day 1 Midday)

1. The operator places a survey stake at the midpoint between A1 and A2, takes a GPS reading (51.23455, 18.34580), and records this as the grid origin.
2. Using the flat-Earth transformation:
   - A1: grid_x = -6.0 m, grid_y = +7.8 m
   - A2: grid_x = +6.7 m, grid_y = -7.8 m

### HIRT Grid Design (Day 1 Afternoon)

1. For A1 (strong compact anomaly, estimated depth 0.5-1.0 m):
   - Grid: 10x10 m at 2.0 m spacing, centered at (-6, 8).
   - 6x6 = 36 probes? No---the standard GPS uncertainty (+/-2-5 m) already accounts for most of the needed coverage.
   - Final design: 10x10 m at 2.0 m spacing = 6x6 = 36 probes, which exceeds typical capacity. Scale back to 8x8 m at 2.0 m = 5x5 = 25 probes.

### Field Stakeout and HIRT Deployment (Day 2)

1. Return to the site. Locate the origin stake.
2. Measure 6 m west and 8 m north from the origin to find the center of the A1 grid.
3. Stake out a 5x5 probe grid at 2.0 m spacing, extending 4 m in each direction from center.
4. Insert probes, connect cables, and run HIRT measurement sequence per the HIRT field guide.
5. Estimated deployment and measurement time: 3-4 hours for the 25-probe grid.

### Data Correlation (Day 2 or Later)

1. Overlay the Pathfinder gradient map on the HIRT inversion result.
2. The HIRT tomogram reveals a conductive anomaly at 0.8 m depth centered at (-5.5, 8.2) in grid coordinates---consistent with the Pathfinder dipole location after accounting for GPS uncertainty.
3. The anomaly's conductivity signature (high on ERT-Lite, moderate on MIT-3D) is consistent with a concrete foundation with embedded rebar.

## Troubleshooting

### GPS Coordinates Do Not Match Expected Location

**Symptoms:** Anomaly positions in the Pathfinder CSV place features at impossible locations (in a river, 50 m from the survey area, etc.).

**Likely causes:**
- GPS had not acquired a stable fix at survey start. Check the first few rows of the CSV for (0, 0) or rapidly changing coordinates.
- Datum mismatch. Pathfinder logs in WGS84. If comparing to a map in a different datum (e.g., local grid, UTM without zone), coordinates will be offset.
- Clock/timestamp confusion affecting GPS position interpolation.

**Resolution:** Filter the CSV to rows with GPS lock (lat != 0 and lon != 0). If using firmware with `GPS_LOG_QUALITY=1`, filter to rows with HDOP < 3.0. Re-plot the filtered data.

### Anomaly Not Found at Expected Position (GPS Drift)

**Symptoms:** The HIRT grid is deployed at the Pathfinder-indicated position but the HIRT data shows no anomaly.

**Likely causes:**
- GPS drift placed the Pathfinder anomaly 3-5 m from its true position (common with NEO-6M).
- The anomaly is outside the HIRT grid extent.

**Resolution:** Expand the HIRT grid by 3-5 m in the direction of suspected drift. If possible, re-survey the immediate area with Pathfinder to re-locate the anomaly with fresh GPS readings (averaging multiple passes improves position accuracy). Consider upgrading to RTK GPS for future surveys.

### Multiple Overlapping Anomalies

**Symptoms:** Pathfinder shows a complex anomaly pattern that could be one large feature or several small features close together.

**Resolution:** Design the HIRT grid to span the entire anomaly complex rather than targeting individual peaks. HIRT's tomographic inversion separates overlapping anomalies that appear merged in the Pathfinder gradient map. Use the High Resolution grid configuration (1.0 m spacing) if the complex is smaller than 6x6 m.

### Grid Does Not Fit Terrain (Obstacles, Slopes)

**Symptoms:** Trees, boulders, standing water, or steep slopes prevent placing probes at designed grid positions.

**Resolution:** HIRT inversion uses actual probe positions, not idealized grid positions. Relocate individual probes up to 30 cm from their designed position and record the actual position. Skip positions entirely if obstructions prevent any placement, and document the skipped positions. For slopes exceeding 50 cm elevation change across the grid, record surface elevation at each probe position for topographic correction. See `HIRT/docs/field-guide/deployment.qmd`, "Handling Field Obstructions" for detailed guidance.

### Time Gap Between Pathfinder and HIRT Surveys

**Symptoms:** Days or weeks pass between the Pathfinder screening and HIRT deployment. The origin stake may have been disturbed.

**Resolution:** Always use a durable origin marker (steel stake driven flush with ground, marked with GPS coordinates written on flagging tape). On return, verify the stake position with a fresh GPS reading. If the stake is missing, re-establish the origin using GPS and the recorded coordinates. Accept that the re-established origin introduces an additional +/-2-5 m uncertainty (standard GPS) or +/-0.02 m (RTK GPS).

## References

- **Pathfinder data format and firmware configuration:** `Pathfinder/firmware/SUMMARY.md` and `Pathfinder/firmware/include/config.h`
- **Pathfinder data visualization tools:** `Pathfinder/firmware/tools/README.md`
- **Pathfinder design concept and sensor specifications:** `Pathfinder/docs/design-concept.md`
- **HIRT deployment procedures:** `HIRT/docs/field-guide/deployment.qmd`
- **HIRT grid configurations and probe insertion:** `HIRT/docs/field-guide/deployment.qmd`, Sections "Site Assessment and Grid Design" and "Probe Insertion Methods"
- **GeoSim coordinate conventions:** X=East, Y=North, Z=Up (SI units throughout). See `GeoSim/CLAUDE.md`
- **GeoSim Pathfinder sensor model:** `GeoSim/geosim/sensors/pathfinder.py`
