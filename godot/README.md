# GeoSim Godot Frontend

Interactive 3D visualization frontend for GeoSim physics engine.

## Status: Phase 2 — Real Physics Connection

The Godot frontend connects to the Python physics engine via ZeroMQ REQ-REP.
When the `godot-zmq` addon is installed and the server is running, the HUD
shows **"GeoSim LIVE"** (green) and gradient readings come from the real
dipole/gradiometer model. Without the addon, it falls back to mock responses
(**"GeoSim OFFLINE"**, red) so you can still develop the scene.

## Quick Start

```bash
# Terminal 1 — start the physics server
geosim-server --scenario ../scenarios/scattered-debris.json

# Terminal 2 — run the integration test (no Godot required)
./tools/test_godot_integration.sh

# Or open the project in Godot 4.3+ and press Play
```

## Installing godot-zmq

1. Download the precompiled release for your platform from
   [godot_zeromq_bin](https://github.com/funatsufumiya/godot_zeromq_bin)
2. Extract to `godot/addons/godot_zeromq/`
3. Verify the `.gdextension` file is present
4. Godot will auto-detect the addon on next launch — `PhysicsClient` switches
   from mock mode to real ZMQ transport automatically

## Architecture

```
Godot 4 (this project)          Python Physics Engine
┌─────────────────────┐         ┌──────────────────┐
│ Main                │         │ geosim-server    │
│  ├ startup sequence │         │ (REQ-REP)        │
│  └ scenario config  │         │                  │
│                     │         │ Dipole model     │
│ PhysicsClient       │──ZMQ──→ │ Gradiometer sim  │
│ (autoload singleton)│←──REP── │ EM / ERT forward │
│  ├ ZMQ or mock      │         │ Scenario loader  │
│  └ is_busy() guard  │         │                  │
│                     │         │                  │
│ OperatorController  │         │                  │
│ TerrainGenerator    │         │                  │
│ HUD (LIVE/OFFLINE)  │         │                  │
└─────────────────────┘         └──────────────────┘
```

## Project Structure

```
godot/
├── project.godot              # Godot project config
├── addons/
│   └── godot_zeromq/          # ZMQ GDExtension (install manually)
├── scenes/
│   └── main.tscn              # Main scene (terrain + operator + HUD)
├── scripts/
│   ├── main.gd                # Startup orchestrator
│   ├── physics_client.gd      # ZMQ client (autoloaded singleton)
│   ├── terrain_generator.gd
│   ├── operator_controller.gd
│   └── hud.gd                 # Gradient display + connection status
└── shaders/
    └── dipole_field.glsl      # GPU compute shader (future)
```

## Controls

- **WASD**: Walk around the survey area
- **Mouse**: Look around
- **Escape**: Release mouse cursor

## Integration Testing

```bash
# Runs server, sends ZMQ test queries, verifies responses, shuts down
./tools/test_godot_integration.sh [scenario_path]
```

The test exercises: ping, scenario info, single-point field/gradient queries,
batch queries (11 points), and shutdown.

## Requirements

- Godot 4.3+
- godot-zmq GDExtension (for real physics connection; optional for mock mode)
- Python physics server running (`geosim-server`)
