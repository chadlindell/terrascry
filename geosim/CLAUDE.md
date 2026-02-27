# GeoSim Project Context

> Part of the [TERRASCRY](../CLAUDE.md) platform. See root CLAUDE.md for platform-wide conventions.

Physics-realistic simulation engine for HIRT (subsurface tomography) and Pathfinder (magnetic gradiometry) geophysical instruments. The physics engine IS the product; visualization layers are consumers. Physics accuracy is non-negotiable — this is not a game engine.

## Commands

```bash
cd geosim && pip install -e ".[dev]"              # Install for development
cd geosim && pip install -e ".[all]"              # Install everything (viz + hirt + dev)
cd geosim && pytest tests/                        # Run all tests
cd geosim && pytest tests/ -k dipole              # Run dipole tests only
cd geosim && pytest tests/test_zmq_server.py      # Run a single test file
cd geosim && ruff check geosim/                   # Lint
cd geosim && geosim-server --scenario scenarios/single-ferrous-target.json  # Start ZeroMQ server
```

## Architecture

**Data flow:** Scenario JSON → physics engine → sensor data → CSV/visualization/ZMQ

- **`geosim/magnetics/dipole.py`**: Core physics — dipole field B(r), superposition, gradiometer readings. Everything depends on this. NumPy only, no viz dependencies.
- **`geosim/scenarios/loader.py`**: Loads scenario JSON into `Scenario` objects. Auto-computes induced magnetic moments from susceptibility/radius when not explicitly set. Scenario files are the single source of truth for ground truth.
- **`geosim/sensors/pathfinder.py`**: Full survey simulation pipeline: walk path → field computation → noise → ADC conversion → CSV export. Output CSVs must be loadable by Pathfinder's existing `visualize_data.py`.
- **`geosim/noise/models.py`**: Three noise sources combined via `NoiseModel`: `SensorNoise` (white + 1/f), `DiurnalDrift` (geomagnetic variation), `HeadingError` (orientation systematic). Applied after clean physics.
- **`geosim/viz/`**: PyVista 3D visualization (terrain, buried objects, field volumes, survey scenes). Imports from core but never modifies physics. Optional dependency.
- **`geosim/server.py`**: ZeroMQ REQ-REP server wrapping physics for external clients (Godot). Commands: `ping`, `load_scenario`, `query_field`, `query_gradient`, `get_scenario_info`, `shutdown`.
- **`godot/`**: Godot 4 interactive frontend. Currently uses mock responses; real connection requires godot-zmq GDExtension.
- **`geosim/em/`, `geosim/resistivity/`**: Phase 2 stubs (SimPEG/pyGIMLi forward models, not yet implemented).
- **`geosim/streaming/`**: Real-time MQTT streaming module for instrument data. Messages are JSON-serialized dataclasses. MQTT topic hierarchy documented in `shared/protocols/mqtt_topics.md`. See `geosim/docs/streaming-architecture.md`.

## Coordinate Convention

Right-handed: X=East, Y=North, Z=Up. All positions in meters, magnetic moments in A·m², fields in Tesla (T), gradients in T/m. SI units throughout.

## Physics Standards

- Magnetic dipole: `B(r) = μ₀/4π [3(m·r̂)r̂ - m] / r³`
- Gradient: `ΔB = B(lower_sensor) - B(upper_sensor)` (firmware convention)
- Vacuum permeability: `μ₀ = 4π × 10⁻⁷ T·m/A`
- Magnetic sources represented as `[{'position': [x,y,z], 'moment': [mx,my,mz]}, ...]`
- Default Earth field: `[0, 20μT, 45μT]` (mid-latitude, ~65° inclination)

## Pathfinder Instrument Model

4 sensor pairs, 0.50m horizontal spacing, 0.35m vertical baseline, bottom sensors at 0.175m height, 10 Hz sample rate, ADC: 2.5e9 counts/Tesla with ±32000 saturation.

## Testing Conventions

- Physics tests validate against analytical textbook formulas (Jackson/Griffiths)
- Typical tolerances: `rel=1e-10` for field values, `abs=1e-20` for near-zero, `rtol=1e-14` for batch consistency
- ZMQ server tests enforce latency SLAs: ping <2ms, field query <5ms, 100-point batch <20ms
- Physics validation tests required for any new forward model

## Code Style

- Python 3.10+, type hints on public APIs
- NumPy-style docstrings on public functions
- ruff for linting (line-length=100, rules: E, F, W, I, N, UP)

## EMI/FDEM Forward Modeling (Phase 2)

The `geosim/em/` module will support frequency-domain electromagnetic forward modeling using SimPEG's FDEM API. This enables validation of Pathfinder's EMI conductivity channel against synthetic ground truth. Key capabilities:

- 1D layered earth forward model (EM38-equivalent)
- LIN approximation: σ_a = (4/(ωμ₀s²)) × Im(Hs/Hp)
- Multi-coil geometry (HCP, VCP)
- Integration with `geosim/scenarios/` for synthetic survey generation

## Joint Inversion Concept

Pathfinder surface data serves as boundary conditions for HIRT subsurface inversion:
- Magnetic gradient map → constrains top susceptibility layer
- EMI conductivity map → constrains top resistivity layer
- LiDAR DEM → corrects inversion mesh for terrain
- Cross-gradient regularization (Gallardo & Meju 2003) couples multi-physics

See `docs/research/joint-inversion-concept.md` for implementation approach using SimPEG.

## Fiber vs WiFi Research

Evaluated fiber optic data links as alternative to ESP32 WiFi. Conclusion: WiFi with TDM is sufficient (< 0.01 nT interference), POF fiber available as upgrade for high-precision work. See `docs/research/fiber-vs-wifi-analysis.md`.

## Related Projects (in monorepo)

- **Pathfinder** (`pathfinder/`): Handheld multi-sensor gradiometer — GeoSim simulates its sensor physics
- **HIRT** (`hirt/`): Crosshole tomography instrument — GeoSim provides forward models for inversion
- **Shared** (`shared/`): MQTT protocol specs, CSV schemas consumed by GeoSim's data loaders
