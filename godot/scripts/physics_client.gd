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

## Reconnection interval (seconds)
@export var reconnect_interval: float = 5.0

## Connection state — only true after a successful server response
var _connected: bool = false
var _scenario_loaded: bool = false
var _scenario_name: String = ""

## ZMQ availability and state
var _zmq_available: bool = false
var _zmq_socket = null
var _awaiting_reply: bool = false
var _zmq_setup_done: bool = false
var _reconnect_timer: float = 0.0

## True when ZMQ addon is not installed (pure mock mode)
var is_mock_mode: bool:
	get:
		return not _zmq_available

## Signals
signal connection_state_changed(connected: bool)
signal server_error(message: String)
signal _response_received(data: Dictionary)


func _ready() -> void:
	print("[PhysicsClient] GeoSim physics client initialized")
	print("[PhysicsClient] Server: ", server_address)

	_zmq_available = ClassDB.class_exists("ZMQSender")
	if _zmq_available:
		print("[PhysicsClient] godot-zmq addon detected, using ZMQ transport")
		_setup_zmq()
	else:
		print("[PhysicsClient] godot-zmq addon not found, using mock responses")
		print("[PhysicsClient] Install from: https://github.com/funatsufumiya/godot_zeromq_bin")


func _process(delta: float) -> void:
	# Periodic reconnection when ZMQ is available but server is unreachable
	if _zmq_available and _zmq_setup_done and not _connected and not _awaiting_reply:
		_reconnect_timer += delta
		if _reconnect_timer >= reconnect_interval:
			_reconnect_timer = 0.0
			_try_reconnect()


func _setup_zmq() -> void:
	_zmq_socket = ClassDB.instantiate("ZMQSender")
	_zmq_socket.createSocket(1)  # ZMQ_REQ = 1
	var err = _zmq_socket.connectSocket(server_address)
	if err == OK:
		_zmq_socket.onMessageString.connect(_on_zmq_message)
		_zmq_setup_done = true
		# Don't set _connected — ZMQ connect is async; wait for first successful response
		print("[PhysicsClient] ZMQ socket bound to ", server_address)
	else:
		print("[PhysicsClient] Failed to bind ZMQ socket to ", server_address)
		server_error.emit("Failed to bind ZMQ socket to %s" % server_address)


func _try_reconnect() -> void:
	print("[PhysicsClient] Attempting reconnection...")
	var result = await ping()
	if result.get("status") == "ok":
		var msg: String = result.get("data", {}).get("message", "")
		if not msg.contains("mock"):
			_connected = true
			connection_state_changed.emit(true)
			print("[PhysicsClient] Reconnected to server")


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

	if _zmq_available and _zmq_setup_done:
		if _awaiting_reply:
			return {"status": "error", "message": "Client busy (awaiting reply)"}

		_awaiting_reply = true
		_zmq_socket.sendString(JSON.stringify(request))

		var timer := get_tree().create_timer(response_timeout)
		var result = await _await_with_timeout(timer)

		if result.is_empty():
			_awaiting_reply = false
			if _connected:
				_connected = false
				connection_state_changed.emit(false)
				print("[PhysicsClient] Server timeout — marking disconnected")
			server_error.emit("Request timed out: %s" % command)
			return {"status": "error", "message": "Request timed out"}

		# Successful response confirms connectivity
		if not _connected and result.get("status") == "ok":
			_connected = true
			connection_state_changed.emit(true)

		return result
	else:
		return _mock_response(command, params)


## Helper to await response with timeout.
## Cleans up both callbacks regardless of which fires first.
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

	# Clean up whichever callback didn't fire
	if _response_received.is_connected(on_response):
		_response_received.disconnect(on_response)
	if timer.timeout.is_connected(on_timeout):
		timer.timeout.disconnect(on_timeout)

	return response


# ---------- Convenience query methods ----------

## Query the magnetic field at given positions.
## positions: Array of [x, y, z] arrays
func query_field(positions: Array) -> Dictionary:
	return await send_command("query_field", {"positions": positions})


## Query gradiometer readings at given positions.
func query_gradient(positions: Array, sensor_separation: float = 0.35) -> Dictionary:
	return await send_command("query_gradient", {
		"positions": positions,
		"sensor_separation": sensor_separation,
		"component": 2,
	})


## Query EM response at given positions.
func query_em_response(positions: Array, frequency: float = 1000.0) -> Dictionary:
	return await send_command("query_em_response", {
		"positions": positions,
		"frequency": frequency,
	})


## Query apparent resistivity for electrode configuration.
func query_apparent_resistivity(electrode_positions: Array, measurements: Array) -> Dictionary:
	return await send_command("query_apparent_resistivity", {
		"electrode_positions": electrode_positions,
		"measurements": measurements,
	})


## Query skin depth for given frequency and conductivity.
func query_skin_depth(frequency: float, conductivity: float) -> Dictionary:
	return await send_command("query_skin_depth", {
		"frequency": frequency,
		"conductivity": conductivity,
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


# ---------- Mock responses ----------

## Mock responses for development without the Python server running.
## Gradient mock uses a synthetic dipole at terrain center for realistic
## position-dependent readings so the HUD can be developed offline.
func _mock_response(command: String, params: Dictionary) -> Dictionary:
	match command:
		"ping":
			return {"status": "ok", "data": {"message": "pong (mock)"}}
		"load_scenario":
			_scenario_loaded = true
			_scenario_name = "mock_scenario"
			return {"status": "ok", "data": {"name": "mock_scenario", "n_sources": 1}}
		"query_field":
			var positions: Array = params.get("positions", [])
			var b_arr := []
			for pos in positions:
				b_arr.append(_mock_dipole_field(pos))
			return {"status": "ok", "data": {"B": b_arr}}
		"query_gradient":
			var positions: Array = params.get("positions", [])
			var sep: float = params.get("sensor_separation", 0.35)
			var grad := []
			for pos in positions:
				grad.append(_mock_dipole_gradient(pos, sep))
			return {"status": "ok", "data": {"gradient": grad}}
		"get_scenario_info":
			return {"status": "ok", "data": {
				"name": "mock_scenario",
				"n_sources": 1,
				"n_objects": 1,
				"terrain": {
					"x_extent": [0, 20],
					"y_extent": [0, 20],
					"surface_elevation": 0.0,
				},
				"objects": [
					{
						"name": "mock_target",
						"position": [10.0, 10.0, -1.0],
						"radius": 0.05,
						"object_type": "ferrous_sphere",
					},
				],
			}}
		"query_em_response":
			var positions: Array = params.get("positions", [])
			var resp_real := []
			var resp_imag := []
			for pos in positions:
				var r := _mock_distance_to_target(pos)
				resp_real.append(1e-4 / max(r, 0.1))
				resp_imag.append(5e-5 / max(r, 0.1))
			return {"status": "ok", "data": {
				"response_real": resp_real,
				"response_imag": resp_imag,
				"frequency": params.get("frequency", 1000.0),
			}}
		"query_apparent_resistivity":
			var measurements: Array = params.get("measurements", [])
			var rho := []
			for i in range(measurements.size()):
				rho.append(100.0)
			return {"status": "ok", "data": {"apparent_resistivity": rho}}
		"query_skin_depth":
			var f: float = params.get("frequency", 1000.0)
			var sigma: float = params.get("conductivity", 0.01)
			var delta: float = 1.0 / sqrt(PI * f * 4e-7 * PI * sigma)
			return {"status": "ok", "data": {"skin_depth": delta, "unit": "meters"}}
		_:
			return {"status": "error", "message": "Unknown command: %s (mock)" % command}


## Mock dipole field: single source at (10, 10, -1) with moment [0, 0, 0.01] A·m².
func _mock_dipole_field(pos: Array) -> Array:
	var mu0_4pi := 1e-7
	var src := Vector3(10.0, 10.0, -1.0)
	var moment := Vector3(0.0, 0.0, 0.01)
	var r := Vector3(pos[0] - src.x, pos[1] - src.y, pos[2] - src.z)
	var r_mag := r.length()
	if r_mag < 0.01:
		return [0.0, 0.0, 0.0]
	var r_hat := r / r_mag
	var r3 := r_mag * r_mag * r_mag
	var m_dot_rhat := moment.dot(r_hat)
	var b := mu0_4pi * (3.0 * m_dot_rhat * r_hat - moment) / r3
	return [b.x, b.y, b.z]


## Mock gradient: finite difference of Bz component.
func _mock_dipole_gradient(pos: Array, separation: float) -> float:
	var b_lower := _mock_dipole_field(pos)
	var b_upper := _mock_dipole_field([pos[0], pos[1], pos[2] + separation])
	return (b_lower[2] - b_upper[2]) / separation


## Distance from position to mock target (for EM response falloff).
func _mock_distance_to_target(pos: Array) -> float:
	var dx: float = pos[0] - 10.0
	var dy: float = pos[1] - 10.0
	var dz: float = pos[2] + 1.0
	return sqrt(dx * dx + dy * dy + dz * dz)
