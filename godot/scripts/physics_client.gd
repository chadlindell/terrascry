## ZeroMQ client for GeoSim physics server.
##
## Autoloaded singleton that provides physics queries to all scenes.
## Communicates with the Python physics engine via ZeroMQ REQ-REP
## when the godot-zmq addon is installed. Falls back to mock
## responses for development without the server.
##
## Usage from any node:
##   var result = await PhysicsClient.query_field([[5.0, 5.0, 0.3]])
##   var gradient = await PhysicsClient.query_gradient([[5.0, 5.0, 0.175]])
extends Node

## Server address (configurable)
@export var server_address: String = "tcp://127.0.0.1:5555"

## Timeout for server responses (seconds)
@export var response_timeout: float = 2.0

## Connection state
var _connected: bool = false
var _scenario_loaded: bool = false
var _scenario_name: String = ""

## ZMQ availability and state
var _zmq_available: bool = false
var _zmq_socket = null  # ZMQSender when available
var _awaiting_reply: bool = false

## Signals
signal connection_state_changed(connected: bool)
signal server_error(message: String)
signal _response_received(data: Dictionary)


func _ready() -> void:
	print("[PhysicsClient] GeoSim physics client initialized")
	print("[PhysicsClient] Server: ", server_address)

	# Detect godot-zmq addon
	_zmq_available = ClassDB.class_exists("ZMQSender")
	if _zmq_available:
		print("[PhysicsClient] godot-zmq addon detected, using ZMQ transport")
		_setup_zmq()
	else:
		print("[PhysicsClient] godot-zmq addon not found, using mock responses")
		print("[PhysicsClient] Install godot-zmq from: https://github.com/funatsufumiya/godot_zeromq_bin")


func _setup_zmq() -> void:
	_zmq_socket = ClassDB.instantiate("ZMQSender")
	_zmq_socket.createSocket(1)  # ZMQ_REQ = 1
	var err = _zmq_socket.connectSocket(server_address)
	if err == OK:
		_connected = true
		_zmq_socket.onMessageString.connect(_on_zmq_message)
		print("[PhysicsClient] Connected to ", server_address)
		connection_state_changed.emit(true)
	else:
		_connected = false
		print("[PhysicsClient] Failed to connect to ", server_address)
		connection_state_changed.emit(false)
		server_error.emit("Failed to connect to %s" % server_address)


func _on_zmq_message(message: String) -> void:
	_awaiting_reply = false
	var json = JSON.new()
	var parse_result = json.parse(message)
	if parse_result == OK:
		_response_received.emit(json.data)
	else:
		var err_response := {"status": "error", "message": "Failed to parse server response"}
		_response_received.emit(err_response)
		server_error.emit("JSON parse error: %s" % json.get_error_message())


## Whether a request is currently in-flight (prevents double-send on REQ-REP).
func is_busy() -> bool:
	return _awaiting_reply


## Send a command to the physics server.
## Returns a Dictionary with "status" and "data" keys.
func send_command(command: String, params: Dictionary = {}) -> Dictionary:
	var request := {
		"command": command,
		"params": params,
	}

	if _zmq_available and _connected:
		if _awaiting_reply:
			return {"status": "error", "message": "Client busy (awaiting reply)"}

		_awaiting_reply = true
		_zmq_socket.sendString(JSON.stringify(request))

		# Wait for response with timeout
		var response: Dictionary = {}
		var timed_out := true

		# Use a timer for timeout
		var timer := get_tree().create_timer(response_timeout)
		var result = await _await_with_timeout(timer)

		if result.is_empty():
			_awaiting_reply = false
			server_error.emit("Request timed out: %s" % command)
			return {"status": "error", "message": "Request timed out"}

		return result
	else:
		# Fallback to mock responses
		return _mock_response(command, params)


## Helper to await response with timeout.
func _await_with_timeout(timer: SceneTreeTimer) -> Dictionary:
	var response := {}
	var waiting := true

	var on_response := func(data: Dictionary):
		response = data
		waiting = false

	var on_timeout := func():
		waiting = false

	_response_received.connect(on_response, CONNECT_ONE_SHOT)
	timer.timeout.connect(on_timeout, CONNECT_ONE_SHOT)

	while waiting:
		await get_tree().process_frame

	# Clean up if one fired before the other
	if _response_received.is_connected(on_response):
		_response_received.disconnect(on_response)

	return response


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


## Get scenario info from the server.
func get_scenario_info() -> Dictionary:
	return await send_command("get_scenario_info")


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
