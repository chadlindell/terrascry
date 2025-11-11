# 12. Interpreting Depth & Resolution (field expectations)

## Depth of Investigation

### Factors Affecting Depth

- **Probe depth:** Deeper insertion → deeper sensitivity
- **Probe spacing:** Wider spacing → deeper investigation
- **Soil conductivity:** Lower conductivity → deeper penetration
- **Frequency (MIT):** Lower frequency → deeper sensitivity
- **Current geometry (ERT):** Longer baselines → deeper investigation

### Rule-of-Thumb Estimates

| Probe Depth | Probe Spacing | Expected Depth Range |
|-------------|---------------|---------------------|
| 1.5 m       | 1.0–1.5 m    | ~1.5–2.5 m         |
| 1.5 m       | 2.0 m        | ~2–3 m             |
| 3.0 m       | 1.5–2.0 m    | ~3–5 m             |
| 3.0 m       | 2.5–3.0 m    | ~4–6 m             |

*Note: Actual depth depends on soil properties and target characteristics.*

## Lateral Resolution

### Approximate Resolution

- **Lateral resolution** ≈ **0.5–1.5 × spacing**
- Tighter spacing → finer resolution
- Wider spacing → coarser resolution but faster coverage

### Examples

| Spacing | Approximate Resolution |
|---------|----------------------|
| 1.0 m   | 0.5–1.5 m           |
| 1.5 m   | 0.75–2.25 m         |
| 2.0 m   | 1.0–3.0 m           |

## What Each Method Detects

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

## Combined Interpretation

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

## Data Quality Indicators

### Good Data
- Consistent reciprocity (A→B ≈ B→A)
- Smooth spatial variations
- Expected depth sensitivity
- Stable baseline measurements

### Problematic Data
- Poor reciprocity (check coupling, calibration)
- Noisy/spiky readings (check connections, shielding)
- No depth sensitivity (check spacing, frequency)
- Inconsistent repeats (check timebase, connectors)

## Field Expectations

### Typical Anomaly Sizes

- **Large metal:** 1–3 m (engine, large parts) → strong MIT response
- **Small metal:** 0.1–0.5 m (artifacts) → weaker MIT, may need tight spacing
- **Grave shaft:** 0.5–1.5 m wide → ERT contrast
- **Crater fill:** 10–15 m diameter → ERT shows boundaries

### Detection Limits

- **MIT:** Can detect ~0.1 m metal at 1–2 m depth (depending on size)
- **ERT:** Can resolve ~0.5 m features at 1–2 m depth
- **Depth:** Practical limit ~3–6 m with 3 m probes

## Next Steps

After data collection:
1. **QA/QC:** Check data quality, reciprocity
2. **Inversion:** Reconstruct 3D models (software step)
3. **Fusion:** Combine MIT and ERT results
4. **Visualization:** Generate depth slices, 3D isosurfaces
5. **Interpretation:** Correlate anomalies with site context

