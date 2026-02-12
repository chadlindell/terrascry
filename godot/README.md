# GeoSim Godot Frontend

Interactive 3D visualization frontend for GeoSim physics engine.

## Status: Phase 1d Proof-of-Concept

This is the initial scaffolding. The Godot scene currently uses **mock responses** for physics queries. To connect to the real Python physics engine:

1. Start the physics server: `geosim-server --scenario ../scenarios/scattered-debris.json`
2. Install the [godot-zmq](https://github.com/zeromq/godot-zmq) GDExtension
3. Update `physics_client.gd` to use real ZMQ sockets instead of mock responses

## Architecture

```
Godot 4 (this project)          Python Physics Engine
┌─────────────────────┐         ┌──────────────────┐
│ PhysicsClient       │──ZMQ──→ │ geosim-server    │
│ (autoload singleton)│←──REP── │ (REQ-REP)        │
│                     │         │                  │
│ OperatorController  │         │ Dipole model     │
│ TerrainGenerator    │         │ Gradiometer sim  │
│ HUD                 │         │ Scenario loader  │
└─────────────────────┘         └──────────────────┘
```

## Project Structure

```
godot/
├── project.godot           # Godot project config
├── scenes/
│   └── main.tscn           # Main scene (terrain + operator + HUD)
├── scripts/
│   ├── physics_client.gd   # ZMQ client (autoloaded)
│   ├── terrain_generator.gd
│   ├── operator_controller.gd
│   └── hud.gd
└── shaders/
    └── dipole_field.glsl   # GPU compute shader (Phase 3)
```

## Controls

- **WASD**: Walk around the survey area
- **Mouse**: Look around
- **Escape**: Release mouse cursor

## Requirements

- Godot 4.3+
- godot-zmq GDExtension (for real physics connection)
- Python physics server running (`geosim-server`)
