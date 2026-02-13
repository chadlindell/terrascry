# godot-zmq addon

This directory should contain the godot-zmq GDExtension for ZeroMQ communication.

## Installation

1. Download the precompiled release for your platform from:
   https://github.com/funatsufumiya/godot_zeromq_bin/releases

2. Extract the contents into this directory. You should have:
   - `godot_zeromq.gdextension`
   - `bin/` directory with platform-specific libraries

3. Restart Godot. The addon should load automatically.

## Verification

After installation, the `ZMQSender` class should be available. The `PhysicsClient`
autoload will detect this and switch from mock mode to live ZMQ communication.

## Without the addon

If the addon is not installed, `PhysicsClient` falls back to mock responses.
This allows scene development and testing without the Python server.
