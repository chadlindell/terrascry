# 12. Data Interpretation

## Overview

This section provides guidance on interpreting HIRT field data, including depth of investigation, lateral resolution, and what each measurement method detects.

---

## 12.1 Depth of Investigation

### Factors Affecting Depth

- **Probe depth:** Deeper insertion -> deeper sensitivity
- **Probe spacing:** Wider spacing -> deeper investigation
- **Soil conductivity:** Lower conductivity -> deeper penetration
- **Frequency (MIT):** Lower frequency -> deeper sensitivity
- **Current geometry (ERT):** Longer baselines -> deeper investigation

### Depth Claims with Confidence Levels

Based on physics analysis (see Section 04), depth claims are qualified as follows:

| Configuration | MIT Depth | ERT Depth | Combined Claim | Confidence |
|---------------|-----------|-----------|----------------|------------|
| 1.5m probes, 1.5m spacing | 1.5-2.5m | 2-3m | **2-3m** | HIGH |
| 1.5m probes, 2.0m spacing | 2-3m | 2-3m | **2-3m** | HIGH |
| 3.0m probes, 2.0m spacing | 3-4m | 3-5m | **3-5m** | MEDIUM |
| 3.0m probes, 2.5m spacing | 3-4m | 4-6m | **4-6m** | LOW |

**Important:** The commonly cited "3-6m" depth range represents **favorable conditions only**. For most field conditions, expect **2-4m typical depth**, with up to 5-6m achievable in optimal soil conditions with longer probes and wider spacing.

### Rule-of-Thumb Estimates

| Probe Depth | Probe Spacing | Expected Depth Range | Confidence |
|-------------|---------------|---------------------|------------|
| 1.5 m       | 1.0-1.5 m    | ~1.5-2.5 m         | HIGH |
| 1.5 m       | 2.0 m        | ~2-3 m             | HIGH |
| 3.0 m       | 1.5-2.0 m    | ~3-4 m             | MEDIUM |
| 3.0 m       | 2.5-3.0 m    | ~4-5 m (up to 6m)  | LOW |

*Note: Actual depth depends on soil properties and target characteristics. "LOW confidence" means achievable under favorable conditions but not guaranteed.*

---

## 12.2 Lateral Resolution

### Approximate Resolution

- **Lateral resolution** approximately equals **0.5-1.5 x spacing**
- Tighter spacing -> finer resolution
- Wider spacing -> coarser resolution but faster coverage

### Examples

| Spacing | Approximate Resolution |
|---------|----------------------|
| 1.0 m   | 0.5-1.5 m           |
| 1.5 m   | 0.75-2.25 m         |
| 2.0 m   | 1.0-3.0 m           |

---

## 12.3 HIRT vs. Surface Methods: Resolution Comparison

HIRT's crosshole geometry provides **2-5x better resolution** than surface methods at depths greater than 2m. This advantage increases with depth.

### Quantified Resolution Comparison

| Method | Lateral Resolution | Depth Resolution | Performance at 3m Depth |
|--------|-------------------|------------------|------------------------|
| Surface Magnetometry | 1-2m | Poor (no discrimination) | ~2m lateral, no depth info |
| GPR (in sand) | 0.3-0.5m | 0.05-0.1m | Degrades to 1m+ |
| GPR (in clay) | Severely limited | Limited | Often fails entirely |
| Surface ERT (Wenner) | ~1x spacing | ~0.5x spacing | ~2-3m |
| EM31/CMD | 1-2m | Poor | ~2m |
| **HIRT (1.5m spacing)** | **0.75-1.5m** | **0.5-0.75m** | **~1m lateral, 0.5m vertical** |
| **HIRT (2m spacing)** | **1-2m** | **0.5-1m** | **~1.5m lateral, 0.75m vertical** |

### Why HIRT Achieves Better Resolution

1. **Direct ray paths:** Surface methods must send energy down and back up. HIRT rays pass horizontally through the target volume at depth.

2. **No surface clutter:** Near-surface heterogeneity, topography, and cultural noise degrade surface measurements. HIRT measurements occur below these interference sources.

3. **True 3D sampling:** Surface methods create 2D maps with pseudo-depth. HIRT's multiple ray angles enable genuine tomographic reconstruction.

4. **Better depth discrimination:** A target at 3m depth looks nearly identical to one at 4m from surface. HIRT geometry can clearly distinguish them.

### Resolution by Application

| Application | HIRT Resolution | Best Surface Alternative | HIRT Advantage |
|-------------|-----------------|-------------------------|----------------|
| WWII crash site (2-4m) | 1-1.5m lateral | Magnetometry: 2m | **2x better** |
| Shallow burial (0.5-1.5m) | 0.5-1m lateral | GPR: 0.3-0.5m | GPR wins shallow |
| Deep crater (4-6m) | 1.5-2m lateral | Surface ERT: 3m+ | **2x better** |
| Conductive clay | 1-1.5m lateral | GPR: fails | **HIRT only option** |

### Key Insight

HIRT is NOT meant to replace surface methods for initial screening. Its value is in **targeted, high-resolution follow-up** after surface methods identify areas of interest. For depths >2m and in conductive soils, HIRT provides resolution that surface methods cannot match.

---

## 12.4 What Each Method Detects

### MIT (Magneto-Inductive Tomography)

**Highlights:**
- **Metal objects** (aluminum, steel, iron)
- **Conductive regions** (saline water, clay layers)
- **Eddy current anomalies** (metallic wreckage)

**Characteristics:**
- Strong response to conductive metals
- Phase lag indicates conductivity
- Amplitude attenuation indicates size/distance
- Frequency-dependent response (higher freq = near-surface)

### ERT (Electrical Resistivity)

**Highlights:**
- **Disturbed fill** (different compaction/moisture)
- **Moisture variations** (wet vs. dry zones)
- **Crater walls** (boundary between fill and native soil)
- **Possible grave shafts** (disturbed, often moister soil)
- **Voids** (air-filled spaces)

**Characteristics:**
- High resistivity: dry soil, voids, air
- Low resistivity: wet soil, clay, saline water
- Contrasts indicate boundaries
- Depth slices show layering

---

## 12.5 Combined Interpretation

### Complementary Information

- **MIT** finds metallic targets (aircraft parts, artifacts)
- **ERT** finds disturbed zones (fill, graves, voids)
- **Together:** More complete picture of subsurface

### Example Scenarios

**Bomb Crater:**
- MIT: Metal parts near base (aluminum/steel)
- ERT: Fill bowl geometry, wet pockets, crater walls

**Woods Burial:**
- MIT: Metallic artifacts (buckles, dog tags, small clusters)
- ERT: Grave shaft (disturbed, moister soil)

**Aircraft Wreckage:**
- MIT: Large conductive masses (engine, landing gear)
- ERT: Disturbed ground, fuel contamination (if present)

---

## 12.6 Data Quality Indicators

### Good Data

- Consistent reciprocity (A->B approximately equals B->A)
- Smooth spatial variations
- Expected depth sensitivity
- Stable baseline measurements

### Problematic Data

- Poor reciprocity (check coupling, calibration)
- Noisy/spiky readings (check connections, shielding)
- No depth sensitivity (check spacing, frequency)
- Inconsistent repeats (check timebase, connectors)

---

## 12.7 Field Expectations

### Typical Anomaly Sizes

- **Large metal:** 1-3 m (engine, large parts) -> strong MIT response
- **Small metal:** 0.1-0.5 m (artifacts) -> weaker MIT, may need tight spacing
- **Grave shaft:** 0.5-1.5 m wide -> ERT contrast
- **Crater fill:** 10-15 m diameter -> ERT shows boundaries

### Detection Limits

- **MIT:** Can detect ~0.1 m metal at 1-2 m depth (depending on size)
- **ERT:** Can resolve ~0.5 m features at 1-2 m depth
- **Depth:** Practical limit **2-4m typical** with 3m probes (up to 5-6m in favorable conditions; see Section 04 for physics analysis)

---

## 12.8 Next Steps After Data Collection

1. **QA/QC:** Check data quality, reciprocity
2. **Inversion:** Reconstruct 3D models (software step)
3. **Fusion:** Combine MIT and ERT results
4. **Visualization:** Generate depth slices, 3D isosurfaces
5. **Interpretation:** Correlate anomalies with site context
