# TERRASCRY Platform Context

TERRASCRY (*terra* + *scry* — "seeing what the earth hides") is a unified multi-sensor geophysical survey platform combining three instruments in a single monorepo.

## Sub-Projects

| Project | Path | Purpose |
|---------|------|---------|
| Pathfinder | `pathfinder/` | Handheld multi-sensor fluxgate gradiometer for rapid reconnaissance |
| HIRT | `hirt/` | Crosshole dual-channel subsurface tomography (MIT-3D + ERT-Lite) |
| GeoSim | `geosim/` | Physics simulation engine for both instruments |
| Shared | `shared/` | Cross-instrument firmware, protocols, hardware, and deployment |

Each sub-project has its own `CLAUDE.md` with project-specific conventions:
- `@pathfinder/CLAUDE.md` — Firmware commands, multi-sensor architecture, design principles
- `@hirt/CLAUDE.md` — Quarto docs, RIPER workflow, writing style (narrative prose for theory)
- `@geosim/CLAUDE.md` — Physics standards, testing conventions, code style

## Platform Conventions

### Performance Claims
All specifications across all sub-projects must use these qualifiers:
- **(Measured)** — Bench or field tested
- **(Modeled)** — Theoretical analysis or simulation
- **(Target)** — Design goal, not yet validated

### Units
SI units throughout. Right-handed coordinate system: X=East, Y=North, Z=Up. Positions in meters, magnetic fields in Tesla (T), gradients in T/m or nT, angles in degrees.

### Voice
Active voice, first person plural for design decisions ("We selected..." not "It was decided...").

## Shared Architecture

### Sensor Pod
Shared between Pathfinder and HIRT. Contains ZED-F9P RTK GPS (0x42), BNO055 IMU (0x29), BMP390 barometer (0x77), DS3231 RTC (0x68) in an IP67 enclosure. Connected via M8 8-pin connector and PCA9615 differential I2C over Cat5 STP cable (1-2m). Hardware specs and firmware in `shared/hardware/sensor_pod/` and `shared/firmware/sensor_pod/`.

### MQTT Streaming
All instruments stream data through MQTT to an NVIDIA Jetson edge compute node. Topic hierarchy:

```
terrascry/{instrument}/data/raw         — Raw sensor readings
terrascry/{instrument}/data/corrected   — After tilt/orientation correction
terrascry/{instrument}/anomaly/detected — Real-time anomaly flags
terrascry/{instrument}/model/update     — Latest model output
terrascry/status/{instrument}           — System health (1 Hz)
```

QoS: 0 for raw data (high rate, loss tolerable), 1 for corrected data (logged), 2 for anomaly alerts (critical), 1 for status. Messages are JSON with ISO 8601 UTC timestamps. See `shared/protocols/mqtt_topics.md`.

### Joint Inversion
Pathfinder surface data serves as boundary conditions for HIRT 3D inversion:
- Magnetic gradient map constrains top susceptibility layer
- EMI conductivity constrains top resistivity layer
- LiDAR DEM corrects inversion mesh for terrain
- Cross-gradient regularization (Gallardo & Meju 2003) couples multi-physics models in SimPEG

See `geosim/docs/research/joint-inversion-concept.md`.

### Edge Deployment
NVIDIA Jetson AGX Orin 64GB running Docker containers: Mosquitto MQTT broker, per-instrument processing pipelines, cloud bridge via Starlink. Config in `shared/edge/`. See `geosim/docs/streaming-architecture.md`.

## Consensus Validation
Critical design decisions are validated using multi-model consensus:
- **Standard tier:** GPT-5.2 + Gemini 3 Pro Preview via PAL consensus tool
- **Critical tier** (math/physics): GPT-5.2-Pro + Gemini 3 Pro Preview

## Key Research Archives
- Pathfinder multi-sensor architecture: `pathfinder/research/multi-sensor-architecture/` (26 files)
- Interference matrix: `pathfinder/research/multi-sensor-architecture/interference-matrix.md`
- HIRT sensor pod consensus: `hirt/research/electronics/sensor-pod-consensus.md`
- Joint inversion concept: `geosim/docs/research/joint-inversion-concept.md`
- Streaming architecture: `geosim/docs/streaming-architecture.md`

## Key Commands

```bash
# Pathfinder
cd pathfinder/firmware && pio run -e esp32dev          # Build
cd pathfinder/firmware && pio run -e esp32dev -t upload # Upload
cd pathfinder/firmware && pio test                      # Test

# HIRT
cd hirt/docs && quarto render                          # Build docs
cd hirt/docs && quarto preview                         # Preview docs

# GeoSim
cd geosim && pip install -e ".[dev]"                   # Install
cd geosim && pytest tests/                             # Test
cd geosim && ruff check geosim/                        # Lint
```
