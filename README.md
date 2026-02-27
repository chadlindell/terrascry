# TERRASCRY

**Multi-sensor geophysical survey platform** (*terra* + *scry* — "seeing what the earth hides")

TERRASCRY unifies three open-source geophysical instruments into a single platform for rapid site characterization: walk a field with Pathfinder, flag anomalies, deploy HIRT for 3D imaging, and validate with GeoSim physics simulations.

## Instruments

### Pathfinder — Handheld Multi-Sensor Gradiometer

Rapid reconnaissance instrument carried by a single operator. Seven sensor modalities in one pass: magnetic gradiometry (8x FG-3+ fluxgates), frequency-domain EMI conductivity, RTK GPS (ZED-F9P), 9-axis IMU (BNO055), IR thermal mapping, LiDAR micro-topography, and georeferenced camera. Covers >3,000 m²/hour at walking speed.

- **Path:** [`pathfinder/`](pathfinder/)
- **Firmware:** ESP32 + PlatformIO
- **BOM:** $1,052-1,262 **(Target)**

### HIRT — Crosshole Subsurface Tomography

Detailed 3D imaging of specific targets flagged by Pathfinder. Dual-channel system: MIT-3D (magnetic induction tomography, 2-50 kHz) for metal detection including aluminum, and ERT-Lite (electrical resistivity) for soil disturbance mapping. Probes inserted into boreholes for true crosshole geometry.

- **Path:** [`hirt/`](hirt/)
- **Documentation:** Quarto technical manual (30 sections)
- **BOM:** <$4,000 **(Target)**

### GeoSim — Physics Simulation Engine

Physics-realistic simulation engine for both instruments. Magnetic dipole forward modeling, FDEM analytical, ERT stubs, noise models, and a real-time MQTT streaming pipeline. Physics accuracy is non-negotiable. Includes Godot 4 interactive visualization frontend and ZeroMQ server for external clients.

- **Path:** [`geosim/`](geosim/)
- **Language:** Python 3.10+
- **Visualization:** PyVista + Godot 4

### Shared Components

Cross-instrument libraries, protocols, and hardware designs shared between Pathfinder and HIRT.

- **Path:** [`shared/`](shared/)
- **Contents:** Sensor pod firmware/hardware, MQTT protocol definitions, CSV schemas, edge deployment configs, field workflow documentation

## Architecture

```
Pathfinder (surface survey, 10 Hz)
     │ MQTT over WiFi
     ▼
NVIDIA Jetson (edge compute)  ◄── HIRT (crosshole, per-sequence)
     │ tilt correction, anomaly detection, data logging
     ▼
Workstation / HPC
     │ SimPEG joint inversion (magnetics + EMI + MIT + ERT)
     ▼
3D multi-physics subsurface model
```

**Shared sensor pod** (ZED-F9P + BNO055 + BMP390 + DS3231) moves between instruments, ensuring both operate in the same coordinate frame — no coordinate transformation errors.

**Joint inversion:** Pathfinder surface data provides boundary conditions for HIRT 3D inversion via cross-gradient regularization (Gallardo & Meju 2003).

## Quick Start

```bash
# Pathfinder firmware
cd pathfinder/firmware && pio run -e esp32dev

# HIRT documentation
cd hirt/docs && quarto render

# GeoSim simulation
cd geosim && pip install -e ".[dev]" && pytest tests/
```

## Licenses

- **Software/Firmware:** [MIT License](LICENSE-SOFTWARE.md)
- **Hardware Designs:** [CERN-OHL-S v2.0](LICENSE-HARDWARE.md)

## Repository Structure

```
terrascry/
├── pathfinder/          # Handheld multi-sensor gradiometer
├── hirt/                # Crosshole subsurface tomography
├── geosim/              # Physics simulation engine
├── shared/              # Cross-instrument shared components
│   ├── firmware/        # Sensor pod, MQTT, SD logger libraries
│   ├── protocols/       # MQTT topics, CSV schemas
│   ├── hardware/        # Sensor pod schematics, BOM, connectors
│   ├── edge/            # Jetson Docker deployment
│   └── workflow/        # Field handoff procedures
└── .github/workflows/   # Path-filtered CI/CD
```

## Links

- **Domain:** [terrascry.com](https://terrascry.com)
- **GitHub:** [chadlindell/terrascry](https://github.com/chadlindell/terrascry)
