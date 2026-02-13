## Main scene orchestrator.
##
## Handles startup sequence: connect to physics server, load scenario,
## configure terrain, and position the operator.
extends Node3D

## Default scenario to load on startup
@export var default_scenario: String = ""


func _ready() -> void:
	print("[Main] GeoSim starting...")
	_startup_sequence()


func _startup_sequence() -> void:
	# Wait one frame for autoloads to initialize
	await get_tree().process_frame

	# Try to connect (ping the server)
	var ping_result = await PhysicsClient.ping()
	if ping_result.get("status") == "ok":
		var msg: String = ping_result.get("data", {}).get("message", "")
		var is_live := not msg.contains("mock")
		if is_live:
			print("[Main] Connected to GeoSim physics server")
		else:
			print("[Main] Running in mock mode (no server)")

	# Load scenario if specified
	if not default_scenario.is_empty():
		print("[Main] Loading scenario: ", default_scenario)
		var load_result = await PhysicsClient.load_scenario(default_scenario)
		if load_result.get("status") == "ok":
			print("[Main] Scenario loaded successfully")
			await _configure_from_scenario()
		else:
			print("[Main] Failed to load scenario: ", load_result.get("message", "unknown error"))
	else:
		print("[Main] No scenario specified, using defaults")


func _configure_from_scenario() -> void:
	var info_result = await PhysicsClient.get_scenario_info()
	if info_result.get("status") != "ok":
		return

	var data: Dictionary = info_result.get("data", {})

	# Update terrain
	var terrain := get_node_or_null("Terrain")
	if terrain and terrain.has_method("update_from_scenario"):
		terrain.update_from_scenario(data)

	# Center operator on terrain
	var terrain_info: Dictionary = data.get("terrain", {})
	if terrain_info.has("x_extent") and terrain_info.has("y_extent"):
		var x_ext: Array = terrain_info["x_extent"]
		var y_ext: Array = terrain_info["y_extent"]
		var center_x: float = (x_ext[0] + x_ext[1]) / 2.0
		var center_z: float = (y_ext[0] + y_ext[1]) / 2.0

		var operator := get_node_or_null("Operator")
		if operator:
			operator.position.x = center_x
			operator.position.z = center_z
			print("[Main] Operator centered at (%.1f, %.1f)" % [center_x, center_z])
