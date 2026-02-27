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
@export var response_timeout: float = 5.0

## Reconnection interval (seconds)
@export var reconnect_interval: float = 5.0

## Retry policy for simulated packet drops/timeouts
@export var max_retries: int = 2
@export var retry_backoff_seconds: float = 0.12

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

## Polling-based response handling (avoids signal/lambda timing issues)
var _response_ready: bool = false
var _last_response: Dictionary = {}

## True when ZMQ addon is not installed (pure mock mode)
var is_mock_mode: bool:
	get:
		return not _zmq_available

## Signals
signal connection_state_changed(connected: bool)
signal server_error(message: String)


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
	# Create ZMQSender dynamically to avoid parse-time dependency on the addon.
	# ZMQ.SocketType.REQ = 3, ZMQ.ConnectionMode.CONNECT = 2
	# auto_receive = false — we call beginReceiveRequest() explicitly after each send.
	var helper := GDScript.new()
	helper.source_code = "static func make(addr: String):\n\treturn ZMQSender.new_from(addr, 3, 2, \"\", false)\n"
	if helper.reload() != OK:
		push_error("[PhysicsClient] Failed to compile ZMQ helper")
		return

	_zmq_socket = helper.make(server_address)
	if _zmq_socket:
		if _zmq_socket is Node:
			add_child(_zmq_socket)
			print("[PhysicsClient] ZMQSender added as child node")
		_zmq_socket.onMessageString(_on_zmq_message)
		_zmq_setup_done = true
		print("[PhysicsClient] ZMQ REQ socket connected to ", server_address)
	else:
		push_error("[PhysicsClient] ZMQSender.new_from() returned null")
		server_error.emit("Failed to create ZMQ socket to %s" % server_address)


func _exit_tree() -> void:
	if _zmq_socket and _zmq_socket.has_method("stop"):
		_zmq_socket.stop()


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
	# ZMQ callback fires from a background thread — defer to main thread
	call_deferred("_handle_zmq_response", message)


func _handle_zmq_response(message: String) -> void:
	var json = JSON.new()
	var parse_result = json.parse(message)
	if parse_result == OK:
		if not _connected:
			_connected = true
			connection_state_changed.emit(true)
			print("[PhysicsClient] Connected to physics server")
		_last_response = json.data
	else:
		_last_response = {"status": "error", "message": "Failed to parse server response"}
		server_error.emit("JSON parse error: %s" % json.get_error_message())
	_awaiting_reply = false
	_response_ready = true


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
		var attempts := maxi(max_retries, 0) + 1
		for attempt in range(attempts):
			if _awaiting_reply:
				return {"status": "error", "message": "Client busy (awaiting reply)"}

			_awaiting_reply = true
			_response_ready = false
			_last_response = {}

			var json_str := JSON.stringify(request)
			_zmq_socket.sendString(json_str)
			_zmq_socket.beginReceiveRequest()

			# Poll for response with timeout — simple and reliable
			var elapsed := 0.0
			var poll_interval := 0.016  # ~1 frame at 60fps
			while not _response_ready and elapsed < response_timeout:
				await get_tree().create_timer(poll_interval).timeout
				elapsed += poll_interval

			if not _response_ready:
				_awaiting_reply = false
				var timed_out := (attempt + 1 >= attempts)
				if timed_out:
					print("[PhysicsClient] Timeout: ", command)
					if _connected:
						_connected = false
						connection_state_changed.emit(false)
					server_error.emit("Request timed out: %s" % command)
					return {"status": "error", "message": "Request timed out"}
				var delay := retry_backoff_seconds * pow(2.0, float(attempt))
				await get_tree().create_timer(delay).timeout
				continue

			# Successful response confirms connectivity
			if not _connected and _last_response.get("status") == "ok":
				_connected = true
				connection_state_changed.emit(true)

			if _last_response.get("status") == "error":
				var err_code := str(_last_response.get("error_code", ""))
				var msg := str(_last_response.get("message", "")).to_lower()
				var retryable := (
					err_code == "simulated_packet_drop"
					or err_code == "simulated_timeout"
					or msg.contains("simulated packet drop")
					or msg.contains("simulated timeout")
				)
				if retryable and attempt + 1 < attempts:
					var delay := retry_backoff_seconds * pow(2.0, float(attempt))
					await get_tree().create_timer(delay).timeout
					continue
				# Fall back to mock for commands the server doesn't support
				if msg.contains("unknown command"):
					print("[PhysicsClient] Server lacks '%s', using mock fallback" % command)
					return _mock_response(command, params)

			return _last_response
		return {
			"status": "error",
			"message": "No response after retries",
		}
	else:
		return _mock_response(command, params)


# ---------- Convenience query methods ----------

## Query the magnetic field at given positions.
## positions: Array of [x, y, z] arrays
func query_field(positions: Array) -> Dictionary:
	return await send_command("query_field", {"positions": positions})


## Query gradiometer readings at given positions.
func query_gradient(positions: Array, sensor_separation: float = 0.35) -> Dictionary:
	var result = await send_command("query_gradient", {
		"positions": positions,
		"sensor_separation": sensor_separation,
		"component": 2,
	})
	# Synthesize per_channel/adc_counts if server didn't provide them
	if result.get("status") == "ok":
		var data: Dictionary = result.get("data", {})
		if not data.has("per_channel"):
			var grad: Array = data.get("gradient", [])
			var per_channel := []
			var adc_counts := []
			for g in grad:
				var channels := [g * 0.8, g * 1.1, g * 0.95, g * 1.05]
				per_channel.append(channels)
				var counts := []
				for ch in channels:
					counts.append(clampi(int(ch * 2.5e9), -32000, 32000))
				adc_counts.append(counts)
			data["per_channel"] = per_channel
			data["adc_counts"] = adc_counts
	return result


## Query EM response at given positions.
func query_em_response(positions: Array, frequency: float = 1000.0) -> Dictionary:
	return await send_command("query_em_response", {
		"positions": positions,
		"frequency": frequency,
	})


## Query apparent resistivity for electrode configuration.
## Overlays client-side target conductivity effects when the server returns
## spatially uniform data (older scenarios lack per-object anomaly zones).
func query_apparent_resistivity(electrode_positions: Array, measurements: Array) -> Dictionary:
	var result = await send_command("query_apparent_resistivity", {
		"electrode_positions": electrode_positions,
		"measurements": measurements,
	})
	if result.get("status") == "ok":
		var data: Dictionary = result.get("data", {})
		var rho: Array = data.get("apparent_resistivity", [])
		_apply_ert_target_perturbation(rho, electrode_positions, measurements)
	return result


## Query metal detector ground-balanced anomaly (ΔT) at given positions.
func query_metal_detector(positions: Array) -> Dictionary:
	return await send_command("query_metal_detector", {"positions": positions})


## Query multi-frequency EM sweep at given positions.
func query_em_sweep(positions: Array, frequencies: Array = []) -> Dictionary:
	var params := {"positions": positions}
	if not frequencies.is_empty():
		params["frequencies"] = frequencies
	return await send_command("query_em_sweep", params)


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


## Configure server-side comms simulation profile.
func set_comms_profile(profile: Dictionary) -> Dictionary:
	return await send_command("set_comms_profile", profile)


## Retrieve server-side stats and recent request history.
func get_server_stats() -> Dictionary:
	return await send_command("get_server_stats")


## Set soil environment conditions on the server.
func set_environment(env: Dictionary) -> Dictionary:
	return await send_command("set_environment", env)


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
			var per_channel := []
			var adc_counts := []
			for pos in positions:
				var g: float = _mock_dipole_gradient(pos, sep)
				grad.append(g)
				# Per-channel: 4 sensor pairs with slight variation
				var channels := [g * 0.8, g * 1.1, g * 0.95, g * 1.05]
				per_channel.append(channels)
				var counts := []
				for ch in channels:
					counts.append(clampi(int(ch * 2.5e9), -32000, 32000))
				adc_counts.append(counts)
			return {"status": "ok", "data": {
				"gradient": grad,
				"per_channel": per_channel,
				"adc_counts": adc_counts,
			}}
		"get_scenario_info":
			return {"status": "ok", "data": {
				"name": "mock_scenario",
				"n_sources": 1,
				"n_objects": 1,
				"available_instruments": [
					"mag_gradiometer", "metal_detector", "em_fdem", "resistivity",
				],
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
						"susceptibility": 1000.0,
						"conductivity": 1e6,
					},
				],
			}}
		"query_metal_detector":
			var positions: Array = params.get("positions", [])
			var delta_t := []
			var earth_mag := sqrt(20e-6 * 20e-6 + 45e-6 * 45e-6)
			var eh_y := 20e-6 / earth_mag
			var eh_z := 45e-6 / earth_mag
			for pos in positions:
				# Magnetic component: project anomaly field onto earth direction
				var b: Array = _mock_dipole_field(pos)
				var dt: float = b[1] * eh_y + b[2] * eh_z
				# EM induction component: conductive targets (brass, copper, aluminium)
				for obj in _mock_targets():
					var sigma: float = obj.get("conductivity", 0.0)
					var radius: float = obj.get("radius", 0.0)
					if sigma <= 0 or radius <= 0:
						continue
					var sp: Array = obj.get("position", [0, 0, -1])
					var dx: float = pos[0] - sp[0]
					var dy: float = pos[1] - sp[1]
					var dz: float = pos[2] - sp[2]
					var dist := sqrt(dx * dx + dy * dy + dz * dz)
					if dist < radius:
						dist = radius * 1.01
					# Simplified EM: response ~ sigma * radius^5 / dist^3 (low induction)
					var em_resp: float = sigma * pow(radius, 5) * 1e-8 / pow(dist, 3)
					dt += em_resp * earth_mag
				delta_t.append(dt)
			# Compute enriched fields for instrument display
			var target_ids := []
			var depth_estimates := []
			var ground_minerals := []
			var ferrous_ratios := []
			for i in range(delta_t.size()):
				var abs_dt: float = abs(delta_t[i])
				# Simple mock target ID: higher delta_t = more ferrous
				var tid := clampi(int(abs_dt * 1e8 * 5.0), 0, 99)
				target_ids.append(tid)
				var depth := _mock_distance_to_target(positions[i])
				depth_estimates.append(depth)
				ground_minerals.append(5.0)  # Low mineral mock
				ferrous_ratios.append(0.8 if tid < 40 else 0.3)
			return {"status": "ok", "data": {
				"delta_t": delta_t, "unit": "T",
				"target_id": target_ids, "depth_estimate": depth_estimates,
				"ground_mineral_level": ground_minerals, "ferrous_ratio": ferrous_ratios,
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
		"query_em_sweep":
			var positions: Array = params.get("positions", [])
			var frequencies: Array = params.get("frequencies", [1000.0, 5000.0, 10000.0])
			var sweep := []
			for freq in frequencies:
				var resp_real := []
				var resp_imag := []
				for pos in positions:
					var r := _mock_distance_to_target(pos)
					resp_real.append(1e-4 / max(r, 0.1) * freq / 1000.0)
					resp_imag.append(5e-5 / max(r, 0.1) * freq / 1000.0)
				sweep.append({
					"frequency": freq,
					"response_real": resp_real,
					"response_imag": resp_imag,
				})
			return {"status": "ok", "data": {"sweep": sweep}}
		"query_apparent_resistivity":
			var electrodes: Array = params.get("electrode_positions", [])
			var measurements: Array = params.get("measurements", [])
			var rho := []
			for meas in measurements:
				# Compute midpoint of the 4-electrode array
				var mx := 0.0
				var my := 0.0
				for idx in meas:
					if idx < electrodes.size():
						mx += electrodes[idx][0]
						my += electrodes[idx][1]
				mx /= max(meas.size(), 1)
				my /= max(meas.size(), 1)
				var r := _mock_distance_to_target([mx, my, 0.0])
				# Background 100 ohm-m, drops near conductive targets
				rho.append(100.0 / (1.0 + 10.0 / max(r * r, 0.01)))
			return {"status": "ok", "data": {"apparent_resistivity": rho}}
		"query_skin_depth":
			var f: float = params.get("frequency", 1000.0)
			var sigma: float = params.get("conductivity", 0.01)
			var delta: float = 1.0 / sqrt(PI * f * 4e-7 * PI * sigma)
			return {"status": "ok", "data": {"skin_depth": delta, "unit": "meters"}}
		"set_comms_profile":
			return {"status": "ok", "data": {"comms_profile": params}}
		"set_environment":
			return {"status": "ok", "data": {"environment": params}}
		"get_server_stats":
			return {"status": "ok", "data": {
				"stats": {
					"request_count": 0,
					"error_count": 0,
					"dropped_count": 0,
					"timed_out_count": 0,
					"avg_latency_ms": 0.0,
				},
				"comms_profile": {},
				"recent_requests": [],
			}}
		_:
			return {"status": "error", "message": "Unknown command: %s (mock)" % command}


## Get mock target list from scenario info, or fallback to single dipole.
## Ensures each object has susceptibility/conductivity for mock physics,
## synthesizing defaults from object name/type when the server doesn't provide them.
func _mock_targets() -> Array:
	var objects: Array = SurveyManager.scenario_info.get("objects", [])
	if objects.is_empty():
		return [{"position": [10.0, 10.0, -1.0], "radius": 0.05,
				 "susceptibility": 1000.0, "conductivity": 1e6}]
	var enriched := []
	for obj in objects:
		var o: Dictionary = obj.duplicate()
		if not o.has("susceptibility"):
			_assign_mock_properties(o)
		if not o.has("conductivity"):
			o["conductivity"] = 0.0
		enriched.append(o)
	return enriched


## Assign realistic susceptibility and conductivity based on object name/type.
## Non-ferrous metals (Al, Cu, brass) have chi≈0 but high conductivity.
## Ferrous metals (steel, iron, engine parts) have high chi and conductivity.
func _assign_mock_properties(o: Dictionary) -> void:
	var name_lc: String = str(o.get("name", "")).to_lower()
	var type_lc: String = str(o.get("type", o.get("object_type", ""))).to_lower()
	var combined := name_lc + " " + type_lc

	# --- Non-ferrous metals: chi≈0, high conductivity (detected via EM only) ---
	if combined.contains("alumin") or combined.contains("aluminum"):
		o["susceptibility"] = 0.0
		o["conductivity"] = 3.7e7
		return
	if combined.contains("copper") or combined.contains("wire bundle"):
		o["susceptibility"] = 0.0
		o["conductivity"] = 5.8e7
		return
	if combined.contains("brass") or combined.contains("cartridge") or combined.contains("casing"):
		o["susceptibility"] = 0.0
		o["conductivity"] = 1.5e7
		return

	# --- Weakly magnetic / mixed metals ---
	if combined.contains("dog tag") or combined.contains("wristwatch") or combined.contains("watch"):
		# Stainless steel or aluminum — weakly or non-magnetic
		o["susceptibility"] = 5.0
		o["conductivity"] = 1.4e6
		return
	if combined.contains("zipper") or combined.contains("buckle") or combined.contains("button"):
		o["susceptibility"] = 30.0
		o["conductivity"] = 1e6
		return
	if combined.contains("boot") or combined.contains("heel"):
		o["susceptibility"] = 50.0
		o["conductivity"] = 1e6
		return

	# --- Strongly ferrous metals ---
	if combined.contains("engine") or combined.contains("iron") \
			or combined.contains("strut") or combined.contains("gear"):
		o["susceptibility"] = 2000.0
		o["conductivity"] = 1e6
		return
	if combined.contains("ferrous") or type_lc.contains("ferrous"):
		# Generic ferrous — scale chi by object size (larger = more detectable)
		var radius: float = o.get("radius", 0.05)
		# Small fasteners: chi~200, large parts: chi~2000
		o["susceptibility"] = clampf(radius * 10000.0, 100.0, 3000.0)
		o["conductivity"] = 1e6
		return

	# --- Natural materials ---
	if combined.contains("rock") or combined.contains("natural"):
		o["susceptibility"] = 3.0
		o["conductivity"] = 0.01
		return
	if combined.contains("burnt") or combined.contains("soil") or combined.contains("ash"):
		o["susceptibility"] = 8.0
		o["conductivity"] = 0.05
		return

	# --- Unknown: assume mild steel fragment ---
	var radius: float = o.get("radius", 0.05)
	o["susceptibility"] = clampf(radius * 5000.0, 50.0, 1000.0)
	o["conductivity"] = 1e5


## Compute induced dipole moment for a ferrous sphere in Earth field.
## Returns moment vector [mx, my, mz] aligned with Earth field.
func _mock_induced_moment(obj: Dictionary) -> Vector3:
	var chi: float = obj.get("susceptibility", 0.0)
	var radius: float = obj.get("radius", 0.05)
	if chi <= 0 or radius <= 0:
		return Vector3.ZERO
	var volume: float = (4.0 / 3.0) * PI * radius * radius * radius
	var eff_chi: float = 3.0 * chi / (chi + 3.0)
	var mu0 := 4.0 * PI * 1e-7
	var earth_mag := sqrt(20e-6 * 20e-6 + 45e-6 * 45e-6)
	var M: float = eff_chi * earth_mag / mu0
	var m_mag: float = volume * M
	# Moment aligned with earth field direction
	var eh_y := 20e-6 / earth_mag
	var eh_z := 45e-6 / earth_mag
	return Vector3(0.0, m_mag * eh_y, m_mag * eh_z)


## Compute dipole field from a single source at observation point.
func _mock_dipole_b(pos: Array, src_pos: Array, moment: Vector3) -> Vector3:
	var mu0_4pi := 1e-7
	var r := Vector3(pos[0] - src_pos[0], pos[1] - src_pos[1], pos[2] - src_pos[2])
	var r_mag := r.length()
	if r_mag < 0.01:
		return Vector3.ZERO
	var r_hat := r / r_mag
	var r3 := r_mag * r_mag * r_mag
	var m_dot_rhat := moment.dot(r_hat)
	return mu0_4pi * (3.0 * m_dot_rhat * r_hat - moment) / r3


## Mock dipole field: superposition of all scenario targets.
func _mock_dipole_field(pos: Array) -> Array:
	var b_total := Vector3.ZERO
	for obj in _mock_targets():
		var moment := _mock_induced_moment(obj)
		if moment.length_squared() < 1e-30:
			continue
		var src_pos: Array = obj.get("position", [10.0, 10.0, -1.0])
		b_total += _mock_dipole_b(pos, src_pos, moment)
	return [b_total.x, b_total.y, b_total.z]


## Mock gradient: finite difference of Bz component.
func _mock_dipole_gradient(pos: Array, separation: float) -> float:
	var b_lower := _mock_dipole_field(pos)
	var b_upper := _mock_dipole_field([pos[0], pos[1], pos[2] + separation])
	return (b_lower[2] - b_upper[2]) / separation


## Apply client-side target conductivity perturbation to ERT results.
## The server may return uniform background if the scenario lacks anomaly zones.
## This overlays the effect of conductive buried objects based on proximity.
func _apply_ert_target_perturbation(rho: Array, electrodes: Array, measurements: Array) -> void:
	var targets := _mock_targets()
	for i in range(rho.size()):
		var meas: Array = measurements[i] if i < measurements.size() else []
		# Compute midpoint of the electrode array
		var mx := 0.0
		var my := 0.0
		var count := 0
		for idx in meas:
			if idx < electrodes.size():
				mx += electrodes[idx][0]
				my += electrodes[idx][1]
				count += 1
		if count > 0:
			mx /= float(count)
			my /= float(count)
		var background: float = rho[i]
		# Accumulate conductivity effects from all nearby targets
		var total_effect := 0.0
		for obj in targets:
			var sigma: float = obj.get("conductivity", 0.0)
			var radius: float = obj.get("radius", 0.0)
			if sigma < 1.0 or radius <= 0:
				continue
			var sp: Array = obj.get("position", [0, 0, -1])
			var dx: float = mx - sp[0]
			var dy: float = my - sp[1]
			var depth: float = abs(sp[2])
			var dist := sqrt(dx * dx + dy * dy + depth * depth)
			# Anomaly scales with conductivity*volume / distance^3
			# Calibrated so 0.2m metallic sphere at 2.5m depth gives ~20% anomaly
			var volume := (4.0 / 3.0) * PI * pow(radius, 3)
			var effect := clampf(sigma * volume * 1e-4 / max(pow(dist, 3), 0.001), 0.0, 0.8)
			total_effect = clampf(total_effect + effect, 0.0, 0.95)
		# Conductive target lowers apparent resistivity
		rho[i] = background * (1.0 - total_effect)


## Distance from position to nearest mock target.
func _mock_distance_to_target(pos: Array) -> float:
	var min_dist := 1e10
	for obj in _mock_targets():
		var sp: Array = obj.get("position", [10.0, 10.0, -1.0])
		var dx: float = pos[0] - sp[0]
		var dy: float = pos[1] - sp[1]
		var dz: float = pos[2] - sp[2]
		var d := sqrt(dx * dx + dy * dy + dz * dz)
		if d < min_dist:
			min_dist = d
	return min_dist
