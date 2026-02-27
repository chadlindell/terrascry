# Joint Inversion Concept: Above Ground + Below Ground

## Overview

The joint inversion concept combines surface survey data from Pathfinder with subsurface tomographic data from HIRT into a unified 3D geophysical model. Pathfinder provides the "above ground" view (magnetic anomalies, surface conductivity, thermal anomalies, topography), while HIRT provides the "below ground" view (crosshole resistivity and inductive tomography). By jointly inverting both datasets, we constrain the solution space far more than either instrument alone.

## The Multi-Physics Fusion Approach

### Data Streams

| Instrument | Modality | Spatial Coverage | Depth Sensitivity |
|-----------|----------|-----------------|-------------------|
| Pathfinder | Magnetic gradiometry | Wide area (surface) | 0-2m (dipole decay) |
| Pathfinder | EMI conductivity | Wide area (surface) | 0-1.5m (FDEM) |
| Pathfinder | IR thermal | Wide area (surface) | 0-0.3m (thermal diffusion) |
| Pathfinder | LiDAR topography | Wide area (surface) | Surface only |
| HIRT | MIT-3D (inductive) | Between probes (subsurface) | Full depth of probes |
| HIRT | ERT-Lite (resistivity) | Between probes (subsurface) | Full depth of probes |

### Complementary Strengths

1. **Pathfinder** covers large areas quickly but has decreasing sensitivity with depth (surface-based measurements).
2. **HIRT** provides high-resolution 3D imaging at depth but only between probe locations (limited lateral coverage).
3. **Combined**: Pathfinder identifies anomaly locations for targeted HIRT deployment. HIRT provides depth detail that Pathfinder cannot achieve.

## Surface Data as Boundary Condition

In the joint inversion framework, Pathfinder's surface measurements serve as **boundary conditions** for HIRT's 3D inversion:

### Magnetic Susceptibility Constraint

Pathfinder's magnetic gradient map provides a 2D surface estimate of magnetic susceptibility distribution. This constrains the top layer of HIRT's 3D model:

```
HIRT 3D Model (cross-section):

Surface ─────────────────────────────
  Layer 0: χ constrained by Pathfinder magnetics
  Layer 1: χ from HIRT MIT-3D inversion
  Layer 2: χ from HIRT MIT-3D inversion
  ...
  Layer N: χ from HIRT MIT-3D inversion
Bottom ──────────────────────────────
```

### Conductivity Constraint

Pathfinder's EMI channel provides surface apparent conductivity (σ_a). This constrains the upper layers of HIRT's ERT inversion:

```python
# Conceptual: Add Pathfinder conductivity as regularization constraint
sigma_surface = pathfinder_emi_data.interpolate_to_hirt_grid()
reg_surface = tikhonov.SmoothDeriv(sigma_ref=sigma_surface, alpha=0.5)
inversion.add_regularization(reg_surface, depth_range=[0, 0.5])  # Top 0.5m
```

### Topographic Correction

Pathfinder's LiDAR provides a micro-DEM (Digital Elevation Model) at the survey site. This corrects HIRT's inversion mesh for actual terrain:

- Standard HIRT assumes flat ground between probes
- LiDAR DEM provides actual surface topography at ~1 cm resolution
- HIRT's tetrahedral mesh is deformed to match the true surface
- This improves inversion accuracy on sloped or uneven terrain

## Implementation in SimPEG

SimPEG supports joint inversion through its modular framework:

```python
import SimPEG
from SimPEG import maps, regularization, optimization, inversion

# Define shared model mesh (covers full 3D volume)
mesh = TensorMesh([hx, hy, hz], x0='CCC')

# Pathfinder forward operators (surface measurements)
sim_mag = magnetics.Simulation3DIntegral(mesh, survey=pathfinder_mag_survey)
sim_emi = fdem.Simulation3DPrimarySecondary(mesh, survey=pathfinder_emi_survey)

# HIRT forward operators (crosshole measurements)
sim_mit = fdem.Simulation3DElectricField(mesh, survey=hirt_mit_survey)
sim_ert = resistivity.Simulation3DCellCentered(mesh, survey=hirt_ert_survey)

# Joint data misfit
dmis = (
    data_misfit.L2DataMisfit(data=pathfinder_mag_data, simulation=sim_mag) +
    data_misfit.L2DataMisfit(data=pathfinder_emi_data, simulation=sim_emi) +
    data_misfit.L2DataMisfit(data=hirt_mit_data, simulation=sim_mit) +
    data_misfit.L2DataMisfit(data=hirt_ert_data, simulation=sim_ert)
)

# Cross-gradient structural coupling (Gallardo & Meju, 2003)
# Encourages boundaries to align across physical properties
reg_cross = regularization.CrossGradient(mesh, model_mag, model_ert)

# Inversion
inv_prob = inverse_problem.BaseInvProblem(dmis, reg + reg_cross, opt)
inv = inversion.BaseInversion(inv_prob)
model_recovered = inv.run(m0)
```

### Cross-Gradient Coupling

The cross-gradient regularization term is key to joint inversion. It penalizes models where the gradients of different physical properties (e.g., resistivity and susceptibility) are not parallel or anti-parallel. This encodes the assumption that geological boundaries affect multiple physical properties simultaneously.

```
Cross-gradient: ∇m₁ × ∇m₂ = 0  (at structural boundaries)
```

This means: wherever resistivity changes, susceptibility should also change (and vice versa). This is geologically reasonable for most archaeological and forensic targets.

## Operational Workflow

```
1. Pathfinder Survey (30-60 min)
   ├── Walk survey area at 1 m/s
   ├── Collect: magnetics + EMI + IR + LiDAR + camera
   ├── Real-time anomaly detection on Jetson
   └── Flag anomaly locations for HIRT follow-up

2. HIRT Deployment (2-4 hours per anomaly)
   ├── Deploy probes around flagged anomaly
   ├── Record probe positions with sensor pod GPS
   ├── Run MIT-3D + ERT-Lite measurement sequences
   └── Progressive inversion on Jetson (real-time model update)

3. Joint Inversion (post-survey, 30-60 min on workstation)
   ├── Import Pathfinder surface data as boundary conditions
   ├── Import HIRT crosshole data
   ├── Import LiDAR DEM for mesh generation
   ├── Run joint inversion with cross-gradient coupling
   └── Output: 3D multi-physics model of the subsurface
```

## Data Handoff: GPS Registration

The shared sensor pod ensures that Pathfinder and HIRT data are in the same coordinate system:

1. Both instruments use the same ZED-F9P GPS receiver
2. Both use RTK corrections from the same NTRIP source
3. Probe positions are recorded with the same GPS that collected the Pathfinder survey
4. No manual coordinate transformation needed — all data is in WGS84

## Expected Benefits

| Metric | Pathfinder Only | HIRT Only | Joint Inversion |
|--------|----------------|-----------|-----------------|
| Lateral coverage | Excellent | Limited to probe array | Excellent |
| Depth resolution | Poor below 1m | Excellent between probes | Good everywhere |
| Multi-physics | 2 modalities (mag + EMI) | 2 modalities (MIT + ERT) | 4+ modalities |
| Survey time | 30 min for 1 ha | 2-4 hr per anomaly | 3-5 hr total |
| Model uniqueness | Low (surface ambiguity) | Moderate | High (multi-constraint) |

## References

- Gallardo, L.A., & Meju, M.A. (2003). Characterization of heterogeneous near-surface materials by joint 2D inversion of dc resistivity and seismic data. *Geophysical Research Letters*, 30(13).
- Haber, E., & Gazit, M.H. (2013). Model fusion and joint inversion. *Surveys in Geophysics*, 34(5), 675-695.
- SimPEG documentation: https://simpeg.xyz/
