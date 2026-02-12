## ZeroMQ client for GeoSim physics server.
##
## Autoloaded singleton that provides physics queries to all scenes.
## Communicates with the Python physics engine via ZeroMQ REQ-REP.
##
## Usage from any node:
##   var result = await PhysicsClient.query_field([[5.0, 5.0, 0.3]])
##   var gradient = await PhysicsClient.query_gradient([[5.0, 5.0, 0.175]])
extends Node

## Server address (configurable)
@export var server_address: String = "tcp://127.0.0.1:5555"

## Connection state
var _connected: bool = false
var _scenario_loaded: bool = false
var _scenario_name: String = ""

## HTTP-based fallback when ZeroMQ addon is not available.
## For the proof-of-concept, we use Godot's HTTPRequest + a thin
## HTTP wrapper around the ZMQ server. In production, use the
## godot-zmq GDExtension for direct ZMQ.
##
## NOTE: This POC uses StreamPeerTCP for direct TCP communication
## with a simple JSON-over-TCP protocol. For production, replace
## with godot-zmq GDExtension or WebSocket bridge.

var _tcp: StreamPeerTCP = null
var _request_queue: Array[Dictionary] = []
var _pending_callback: Callable = Callable()


func _ready() -> void:
	print("[PhysicsClient] GeoSim physics client initialized")
	print("[PhysicsClient] Server: ", server_address)
	print("[PhysicsClient] NOTE: ZMQ requires godot-zmq GDExtension.")
	print("[PhysicsClient] For POC testing, use the Python test_client.py instead.")


## Send a command to the physics server.
## Returns a Dictionary with "status" and "data" keys.
func send_command(command: String, params: Dictionary = {}) -> Dictionary:
	var request := {
		"command": command,
		"params": params,
	}
	# In production, this would use ZMQ REQ socket.
	# For now, return a mock response for scene development.
	print("[PhysicsClient] Would send: ", JSON.stringify(request))
	return _mock_response(command, params)


## Query the magnetic field at given positions.
## positions: Array of [x, y, z] arrays
## Returns: Dictionary with "B" key containing field vectors
func query_field(positions: Array) -> Dictionary:
	return await send_command("query_field", {"positions": positions})


## Query gradiometer readings at given positions.
## Returns: Dictionary with "gradient" key
func query_gradient(positions: Array, sensor_separation: float = 0.35) -> Dictionary:
	return await send_command("query_gradient", {
		"positions": positions,
		"sensor_separation": sensor_separation,
		"component": 2,
	})


## Load a scenario file on the server.
func load_scenario(path: String) -> Dictionary:
	var result = await send_command("load_scenario", {"path": path})
	if result.get("status") == "ok":
		_scenario_loaded = true
		_scenario_name = result.get("data", {}).get("name", "unknown")
		print("[PhysicsClient] Loaded scenario: ", _scenario_name)
	return result


## Ping the server.
func ping() -> Dictionary:
	return await send_command("ping")


## Mock responses for development without the Python server running.
func _mock_response(command: String, params: Dictionary) -> Dictionary:
	match command:
		"ping":
			return {"status": "ok", "data": {"message": "pong (mock)"}}
		"load_scenario":
			return {"status": "ok", "data": {"name": "mock_scenario", "n_sources": 3}}
		"query_field":
			var n_points := (params.get("positions", []) as Array).size()
			var B := []
			for i in range(n_points):
				B.append([0.0, 0.0, 1e-9])  # ~1 nT placeholder
			return {"status": "ok", "data": {"B": B}}
		"query_gradient":
			var n_points := (params.get("positions", []) as Array).size()
			var grad := []
			for i in range(n_points):
				grad.append(0.0)
			return {"status": "ok", "data": {"gradient": grad}}
		"get_scenario_info":
			return {"status": "ok", "data": {
				"name": "mock", "n_sources": 0,
				"terrain": {"x_extent": [0, 20], "y_extent": [0, 20]},
			}}
		_:
			return {"status": "error", "message": "Unknown command (mock)"}
