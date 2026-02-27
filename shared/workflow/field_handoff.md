# Pathfinder → HIRT Field Handoff Procedure

Two-stage geophysical survey workflow: rapid reconnaissance with Pathfinder followed by targeted 3D imaging with HIRT.

## Overview

1. **Stage 1 — Pathfinder Surface Survey** (30-60 min per hectare)
   Walk the site at ~1 m/s with Pathfinder. Seven sensor modalities capture magnetic gradients, EMI conductivity, RTK GPS positions, orientation, IR thermal, LiDAR micro-topography, and camera imagery. The Jetson edge node runs real-time anomaly detection and flags targets.

2. **Stage 2 — HIRT Targeted Investigation** (2-4 hours per anomaly)
   Deploy HIRT crosshole probes at GPS-flagged anomaly locations. MIT-3D and ERT-Lite channels provide 3D tomographic imaging of the subsurface target.

3. **Stage 3 — Joint Inversion** (30-60 min on workstation)
   Combine Pathfinder surface data with HIRT subsurface data using cross-gradient regularization in SimPEG for a unified multi-physics 3D model.

## Pre-Survey Setup

### Establish Grid Origin
1. Drive a survey stake at the southwest corner of the area
2. Record RTK GPS position using the shared sensor pod (ZED-F9P)
3. This stake defines the local coordinate origin: X=East, Y=North, Z=Up
4. Both Pathfinder and HIRT reference this origin — no coordinate transformation needed

### Sensor Pod Transfer
The sensor pod (ZED-F9P + BNO055 + BMP390 + DS3231) is physically shared:
- Attach to Pathfinder crossbar during Stage 1
- Detach and use handheld during Stage 2 to record probe positions
- Same GPS receiver ensures both instruments operate in identical coordinate frame

## Stage 1: Pathfinder Survey

### Procedure
1. Attach sensor pod to Pathfinder crossbar via M8 8-pin connector
2. Power on Pathfinder and verify sensor pod detection (green LED on PCA9615 breakout)
3. Wait for RTK fix (fix_type = 5, typically 30-120 seconds with NTRIP corrections)
4. Walk parallel transects at ~1 m/s spacing, maintaining consistent height (~20 cm above ground)
5. Monitor Jetson tablet display for real-time anomaly flags

### Anomaly Flagging
The edge anomaly detector runs on the Jetson at <1 ms latency:
- **Detection threshold:** 3.0 sigma above rolling background (50-sample window)
- **Minimum duration:** 3 consecutive samples at 10 Hz (300 ms)
- **Multi-physics:** Magnetic gradient AND/OR EMI conductivity AND/OR IR thermal

Each flagged anomaly is published to `terrascry/pathfinder/anomaly/detected` with:
- GPS position (lat/lon)
- Anomaly strength (nT for magnetics, S/m for conductivity)
- Anomaly type classification
- Confidence score (0.0-1.0)

### Deliverables
- Pathfinder survey CSV (SD card + Jetson NVMe)
- Anomaly position list with classifications
- LiDAR DEM of survey area
- Georeferenced camera imagery

## Stage 2: HIRT Deployment

### Site Selection
1. Review Pathfinder anomaly map on Jetson tablet
2. Select highest-priority targets for HIRT investigation
3. Expand probe grid 2-3 m beyond anomaly boundary for inversion context

### Probe Position Recording
1. Detach sensor pod from Pathfinder
2. At each planned probe location:
   a. Hold sensor pod at ground level over insertion point
   b. Wait for RTK fix
   c. Record position (button press or voice command)
   d. Move to next probe location
3. Probe positions saved to CSV (see `protocols/csv_schemas.md` for format)

### HIRT Survey
1. Insert pilot rods at all recorded positions
2. Insert HIRT probes into pilot holes
3. Connect probes to base hub
4. Run MIT-3D measurement sequence (all TX-RX pairs)
5. Run ERT-Lite measurement sequence (all electrode combinations)
6. Data streams to Jetson via MQTT; progressive inversion updates in real time

## Stage 3: Joint Inversion

### Data Inputs
| Source | Data Type | Role in Inversion |
|--------|-----------|-------------------|
| Pathfinder | Magnetic gradient map | Constrains top susceptibility layer |
| Pathfinder | EMI conductivity map | Constrains top resistivity layers |
| Pathfinder | LiDAR DEM | Corrects inversion mesh for terrain |
| HIRT | MIT-3D measurements | 3D conductivity structure at depth |
| HIRT | ERT-Lite measurements | 3D resistivity structure at depth |

### Coordinate Registration
Both instruments used the same ZED-F9P GPS receiver referenced to the same grid origin stake:
- Pathfinder positions: continuous GPS track at 10 Hz
- HIRT probe positions: discrete GPS fixes per probe
- Transformation: WGS84 → local Cartesian (meters, origin at grid stake)

### Cross-Gradient Regularization
The joint inversion couples different physical properties using:

```
nabla(m1) x nabla(m2) = 0
```

This forces property boundaries to align across modalities — if a conductivity anomaly has a sharp boundary, the susceptibility model should also show a boundary at the same location.

Implementation uses SimPEG with shared tetrahedral mesh. See `geosim/docs/research/joint-inversion-concept.md`.

### Output
- 3D multi-physics subsurface model
- Confidence volumes showing resolution vs. depth
- Target characterization: material, size, depth, orientation
- Report with cross-sections and isosurface renderings
