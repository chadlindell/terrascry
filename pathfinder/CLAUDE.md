# Pathfinder Project Context

> Part of the [TERRASCRY](../CLAUDE.md) platform. See root CLAUDE.md for platform-wide conventions.

Pathfinder is a handheld multi-sensor fluxgate gradiometer for rapid geophysical reconnaissance.

## Quick Reference

### Key Commands
- Build firmware (ESP32): `cd pathfinder/firmware && pio run -e esp32dev`
- Upload firmware (ESP32): `cd pathfinder/firmware && pio run -e esp32dev -t upload`
- Build firmware (legacy Nano): `cd pathfinder/firmware && pio run -e nanoatmega328`
- Run tests: `cd pathfinder/firmware && pio test`
- Visualize data: `python pathfinder/firmware/tools/visualize_data.py <file.csv>`
- Spatial map: `python pathfinder/firmware/tools/visualize_data.py <file.csv> --map`

### Memory Bank
- `/memory:save` - Save context to `.claude/memory-bank/`
- `/memory:recall` - Retrieve saved context
- `/memory:list` - List all memories

### RIPER Workflow
Use `/riper:strict` to enable mode tracking. Available modes:
- `/riper:research` - Information gathering (read-only)
- `/riper:innovate` - Brainstorming approaches
- `/riper:plan` - Create specifications (saved to memory bank)
- `/riper:execute` - Implement approved plans
- `/riper:review` - Validate implementation

## Multi-Sensor Architecture

Pathfinder is being upgraded from a single-modality magnetic gradiometer to a multi-sensor reconnaissance platform:

- **MCU**: ESP32 (dual-core, WiFi, replacing Arduino Nano)
- **Magnetics**: 8× FG-3+ fluxgate sensors (4 gradiometer pairs)
- **EMI Conductivity**: Frequency-domain EM coil (AD9833 + OPA549 + AD8421 + AD630)
- **GPS**: u-blox ZED-F9P RTK (cm-accuracy, in shared sensor pod)
- **IMU**: BNO055 9-axis (tilt correction, in shared sensor pod)
- **IR Temperature**: MLX90614 (ground surface thermal mapping)
- **LiDAR**: RPLiDAR C1 (micro-topography, direct to Jetson USB)
- **Camera**: ESP32-CAM (georeferenced ground imagery)

**TDM Firmware**: 100ms cycle — 50ms fluxgate (EMI/WiFi off) → 30ms EMI TX/RX → 20ms settling/comms.

**Sensor Pod**: Shared with HIRT — see `shared/hardware/sensor_pod/` and `shared/firmware/sensor_pod/`.

**Updated BOM**: $1,052-1,262 (see `pathfinder/research/multi-sensor-architecture/updated-bom.md`).

**Shared Libraries**: Firmware references shared libs via `lib_extra_dirs = ../../shared/firmware` in `platformio.ini`.

See `research/multi-sensor-architecture/` for detailed research on all subsystems (26 files including 15 consensus-validated reports).

**Key consensus corrections**: AP2112K-5.0 replaces LM78L05 (dropout error), M8 8-pin replaces 4-pin (PCA9615 needs 6 pins), ADS1115 uses ALERT/RDY interrupt mode (not "DMA"), Wheeler inductance L_TX=150-180 μH, WiFi interference 0.01-0.1 nT (not <0.01 nT).

## Project Structure (Monorepo)
```
terrascry/
├── CLAUDE.md              # Platform-wide context
├── pathfinder/            # ← You are here
│   ├── CLAUDE.md          # This file
│   ├── firmware/          # ESP32/PlatformIO code
│   ├── docs/              # Design docs, guides
│   ├── research/          # Multi-sensor architecture research (26 files)
│   └── hardware/          # Schematics, CAD, BOMs
├── hirt/                  # Crosshole tomography (sibling project)
├── geosim/                # Physics simulation engine
└── shared/                # Cross-instrument shared components
    ├── firmware/          # Sensor pod, MQTT, SD logger libraries
    ├── protocols/         # MQTT topics, CSV schemas
    ├── hardware/          # Sensor pod specs, connector pinouts
    └── edge/              # Jetson Docker deployment
```

---

## Writing Style Guide

### Performance Claims Framework
All specifications must be qualified with their validation status:

| Qualifier | Usage | Example |
|-----------|-------|---------|
| **(Measured)** | Bench/field tested | "Bench tests show 50 nT noise floor" |
| **(Modeled)** | Theoretical estimate | "Detection depth estimated at 1.5 m" |
| **(Target)** | Design goal, not yet validated | "Target weight <1.5 kg" |

### Terminology
Maintain consistency throughout:
- **Gradiometer**: Pair of vertically-separated magnetometers measuring field gradient
- **Fluxgate**: Type of magnetometer sensor used in this design
- **Trapeze**: The horizontal bar carrying sensors, suspended from harness
- **Swath**: Width of ground covered in single pass
- **Top sensor**: Reference sensor (~50 cm above ground)
- **Bottom sensor**: Signal sensor (~15-20 cm above ground)

### Voice and Tone
- Active voice preferred ("The sensor measures..." not "The field is measured by...")
- First person plural for design decisions ("We selected..." not "It was decided...")
- Present tense for describing how the system works
- Past tense for describing the development process or experiments

### Design Principles
1. **Harness-first**: Arms never bear weight
2. **Speed over resolution**: Good enough fast beats perfect slow
3. **Field-rugged**: No carts, no flat-ground assumptions
4. **DIY-accessible**: Under $1,300, globally-available components
5. **Multi-physics**: Above-ground magnetics + below-ground conductivity in one pass

---

## Related Projects (in monorepo)
- **HIRT** (`hirt/`): Crosshole tomography for detailed 3D imaging — Pathfinder flags targets, HIRT images them
- **GeoSim** (`geosim/`): Physics simulation engine — validates Pathfinder sensor models
- **Shared** (`shared/`): Sensor pod hardware/firmware, MQTT protocol, CSV schemas, edge deployment

## External References
- **OhmPi**: Open-source resistivity meter
- **FG Sensors**: DIY fluxgate kits

@.claude/project-info.md
