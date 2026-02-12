# GeoSim - Claude Code Guidelines

## What This Is
Physics-realistic simulation engine for HIRT (subsurface tomography) and Pathfinder (magnetic gradiometry) geophysical instruments. The physics engine IS the product; visualization layers are consumers.

## Architecture
- **Python physics engine** (`geosim/`): Core product. Must produce data matching real instruments.
- **Scenario JSON files** (`scenarios/`): Single source of truth for ground truth.
- **Jupyter notebooks** (`notebooks/`): Education, validation, research frontend.
- **Godot 4** (`godot/`): Interactive 3D frontend, communicates via ZeroMQ.
- **ZeroMQ server** (`geosim/server.py`): Bridge for any external client.

## Coordinate Convention
- Right-handed coordinate system: X=East, Y=North, Z=Up
- All positions in meters
- All magnetic moments in A·m²
- All magnetic fields in Tesla (T), gradients in T/m

## Physics Standards
- Magnetic dipole: B(r) = μ₀/4π [3(m·r̂)r̂ - m] / r³
- Gradient: ΔB = B(lower_sensor) - B(upper_sensor)
- Vacuum permeability: μ₀ = 4π × 10⁻⁷ T·m/A
- SI units throughout

## Key Commands
```bash
pip install -e ".[dev]"     # Install for development
pytest tests/               # Run all tests
pytest tests/ -k dipole     # Run dipole tests only
geosim-server               # Start ZeroMQ server
```

## Related Projects
- `../HIRT/`: Subsurface tomography instrument (EM induction + ERT)
- `../Pathfinder/`: Magnetic gradiometry instrument

## Code Style
- Python 3.10+, type hints on public APIs
- NumPy-style docstrings on public functions
- ruff for linting (line-length=100)
- Physics validation tests required for any new forward model
