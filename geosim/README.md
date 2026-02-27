# GeoSim

Physics-realistic geophysical simulation engine for [HIRT](../HIRT/) (subsurface tomography) and [Pathfinder](../Pathfinder/) (magnetic gradiometry) instruments.

## Quick Start

```bash
# Install
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run ZeroMQ server
geosim-server --scenario scenarios/single-ferrous-target.json
```

## What It Does

GeoSim simulates the complete data acquisition pipeline:

1. **Define a scenario** (JSON): terrain, buried objects, soil properties
2. **Run the physics engine**: magnetic dipole fields, EM induction, resistivity
3. **Simulate an instrument**: walk path, sensor geometry, noise
4. **Export sensor data**: CSV files matching real firmware formats

The exported CSV files can be loaded directly by existing tools like Pathfinder's `visualize_data.py`.

## Project Structure

```
geosim/              # Python physics engine
├── magnetics/       # Dipole models, gradiometer simulation
├── em/              # EM induction forward modeling (Phase 2)
├── resistivity/     # ERT forward modeling (Phase 2)
├── sensors/         # Instrument models (Pathfinder, HIRT)
├── scenarios/       # Scenario loader
├── noise/           # Realistic noise models
└── server.py        # ZeroMQ server for external clients

scenarios/           # JSON scenario files
notebooks/           # Jupyter notebooks (education + validation)
tests/               # Physics validation tests
godot/               # Godot 4 interactive frontend (Phase 3)
```

## Dependencies

**Core (Phase 1):** numpy, scipy, pyzmq

**Visualization:** pyvista, jupyter, matplotlib

**HIRT Physics (Phase 2):** SimPEG, pyGIMLi

## License

MIT
