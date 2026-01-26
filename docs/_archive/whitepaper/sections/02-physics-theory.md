# 2. Physics Theory (Practical, Field-Level)

## 2.1 MIT-3D (Low-Frequency EM)

### Operating Principle

- **TX coil** drives a stable sine wave at **2-50 kHz**
- **RX coils** measure magnetic field amplitude/phase
- Conductive objects (metal) create **eddy currents** -> cause **attenuation & phase lag** along TX->RX paths

### Frequency Selection

- **Lower frequency** -> **deeper penetration**
- **Higher frequency** -> **sharper sensitivity** near the probes
- Typical range: **2-50 kHz** (choose 3-5 points)

### Measurement Geometry

With many TX->RX pairs around/through the site, the dataset samples the volume like a **CT scan**:
- Each TX->RX pair provides a path-integrated measurement
- Multiple paths through the same volume provide redundancy
- Inversion algorithms reconstruct 3D conductivity distribution

## 2.2 ERT-Lite (Galvanic)

### Operating Principle

- Two electrodes **inject constant current**
- Voltage measured at other electrodes yields **apparent resistivity**
- **Disturbed fill, moisture, compacted layers, voids** produce detectable contrasts

### Current Specifications

- Small, safe currents: **0.5-2 mA**
- **Polarity reversal** periodically to minimize polarization artifacts
- Low-freq AC option: **8-16 Hz** to reduce polarization

### Measurement Patterns

- **Long baselines** (corner-to-corner, edge-to-edge, center-to-edge)
- Multiple injection pairs for redundancy
- All probes log voltages simultaneously

## 2.3 Electromagnetic Skin Depth

Electromagnetic skin depth (delta) defines how deeply alternating EM fields penetrate conductive media:

**Formula:** delta = sqrt(2/(omega*mu*sigma)) where omega = 2*pi*f, mu = 4*pi*10^-7 H/m

### Skin Depth vs. Frequency and Conductivity

| **Conductivity** | **2 kHz** | **5 kHz** | **10 kHz** | **20 kHz** | **50 kHz** |
|---|---|---|---|---|---|
| 0.01 S/m (dry sand) | 112 m | 71 m | 50 m | 35 m | 22.5 m |
| 0.1 S/m (moist sand) | 35.6 m | 22.5 m | 15.9 m | 11.2 m | 7.1 m |
| 0.5 S/m (wet clay) | 15.9 m | 10.1 m | 7.1 m | 5.0 m | 3.2 m |
| **1.0 S/m (saturated clay)** | **11.2 m** | **7.1 m** | **5.0 m** | **3.6 m** | **2.3 m** |

### Key Insight

Skin depth alone does **NOT** limit MIT depth. **Coil coupling geometry (proportional to 1/r^3)** dominates in near-field conditions. Practical MIT depth is approximately 1-2x probe spacing, regardless of skin depth in most field conditions.

## 2.4 Target-Dependent Frequency Selection

| Target Depth | Recommended Frequencies | Integration Time |
|--------------|-------------------------|------------------|
| 0.5-1.5m (shallow) | 20-50 kHz | 1-3 sec |
| 1.5-2.5m (mid-range) | 10-20 kHz | 3-5 sec |
| 2.5-4m (deep) | 2-10 kHz | 5-15 sec |
| >4m (very deep) | 2-5 kHz | 10-30 sec |

**Guidance:**
- Higher frequencies provide sharper near-surface resolution
- Lower frequencies provide better depth penetration
- Longer integration times improve SNR at any frequency

## 2.5 ERT Geometric Factor K

The geometric factor K converts measured voltage/current ratios to apparent resistivity:

**Formula:** rho_a = K * (V / I)

### For HIRT Ring Electrodes (Borehole Geometry)

```
K approximately equal to pi * L   (where L = distance between current electrodes)
```

| Configuration | L (electrode separation) | K |
|---------------|--------------------------|---|
| A(0.5m) -> B(1.5m) | 1.0 m | 3.14 ohm-m |
| A(0.5m) -> C(2.5m) | 2.0 m | 6.28 ohm-m |
| A(0.5m) -> D(3.0m) | 2.5 m | 7.85 ohm-m |

### Depth of Investigation (DOI)

- **Rule:** DOI approximately 1.5x maximum electrode separation
- **1.5m probes** (rings at 0.5m, 1.5m): DOI = **2-3m**
- **3.0m probes** (rings at 0.5m, 1.5m, 2.5m): DOI = **3-5m** (edge cases to 6m)

## 2.6 Depth of Investigation Summary

### Revised Depth Claims with Confidence Levels

| Configuration | MIT Depth | ERT Depth | Combined Claim |
|---------------|-----------|-----------|----------------|
| 1.5m probes, 2m spacing | 2-3m | 2-3m | **2-3m (HIGH confidence)** |
| 3.0m probes, 2m spacing | 3-4m | 3-5m | **3-5m (MEDIUM confidence)** |
| Edge cases (conductive soil) | 2-3m | 4-6m | **Up to 6m (LOW confidence)** |

### Rule-of-Thumb Estimates

- With rods inserted **~3 m**: **sensitivity volume** typically extends **~3-5 m** deep (up to 6m in favorable conditions)
- With **~1.5 m rods** (woods): expect **~2-3 m** effective sensitivity
- Actual depth depends on:
  - Soil conductivity
  - Probe spacing
  - Measurement frequency (for MIT)
  - Current injection geometry (for ERT)

## 2.7 Why Crosshole Geometry Beats Surface Methods

HIRT's borehole/crosshole tomography provides **fundamental physics advantages** over surface geophysical methods for targets deeper than ~1.5m.

### The Physics Explanation

#### 1. Ray Path Geometry

**Surface methods:** Sensitivity decreases as 1/r^2 to 1/r^4 with depth. Energy must travel down, interact with target, and return to surface--doubling the path length and attenuation.

**Crosshole methods:** Rays pass **directly through** the target volume. Energy travels horizontally between probes at depth, with sensitivity concentrated where targets are located.

```
Surface Method:              Crosshole Method:

   [sensors]                    Probe    Probe
   =========                      |        |
       | ^                        |--ray---|
       | ^                        |--->----|
     [target]                   [target]
       weak                      strong
     coupling                   coupling
```

#### 2. No Surface Interference

Surface methods suffer from:
- Near-surface heterogeneity (fill, roots, utilities)
- Topographic effects
- Cultural noise (fences, buildings, vehicles)

Crosshole measurements occur **below** these interference sources.

#### 3. True Volumetric 3D Sampling

**Surface methods:** Create 2D maps with "pseudo-depth" estimated from diffraction patterns or multi-coil separation. Depth discrimination is inherently poor.

**Crosshole methods:** Multiple ray paths at different angles through the same volume enable true 3D tomographic reconstruction--similar to medical CT scanning.

#### 4. Superior Depth Discrimination

| Scenario | Surface Method | Crosshole Method |
|----------|---------------|------------------|
| Target at 3m vs 4m depth | Nearly identical response | Clearly distinguishable |
| Two targets at same depth | Merged anomaly | Resolved if spacing > 0.5x probe spacing |
| Target size estimation | Ambiguous (depth/size trade-off) | Better constrained |

### Quantified Resolution Advantage

Research on crosshole ERT and EM tomography demonstrates **2-5x better resolution** than surface methods at depths >2m:

| Method | Lateral Resolution | Depth Resolution | At 3m Depth |
|--------|-------------------|------------------|-------------|
| Surface Magnetometry | 1-2m | Poor (no discrimination) | ~2m lateral |
| GPR | 0.3-0.5m shallow | 0.05-0.1m shallow | Degrades to 1m+ in clay |
| Surface ERT (Wenner) | ~1x spacing | ~0.5x spacing | ~2-3m |
| EM31/CMD | 1-2m | Poor | ~2m |
| **HIRT Crosshole** | **0.5-1x spacing** | **0.3-0.5x spacing** | **0.75-1.5m** |

### When Crosshole Wins vs. Loses

**HIRT crosshole geometry is SUPERIOR for:**
- Targets deeper than 1.5-2m
- 3D localization requirements
- Conductive soils where GPR fails
- Distinguishing multiple targets at similar depths
- Non-ferrous (aluminum) target detection

**Surface methods remain SUPERIOR for:**
- Rapid large-area screening (10x faster)
- Shallow targets (<1m) where GPR resolution excels
- Purely ferrous targets (magnetometry)
- Initial site characterization before targeted investigation

### Optimal Workflow

The physics supports a **two-stage approach**:

1. **Surface screening** (magnetometry, GPR, EM31): Identify anomalies quickly over large areas
2. **HIRT crosshole follow-up**: Characterize identified anomalies with superior 3D resolution

This leverages the strengths of both approaches while minimizing deployment time.

