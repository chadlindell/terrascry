## Main scene orchestrator.
##
## Responds to SurveyManager state changes to configure the world,
## manage operators, and coordinate between subsystems.
extends Node3D

## Visual containers
var _object_markers: Node3D
var _survey_lines: Node3D

## References to operators (set up in _ready)
var _ground_operator: CharacterBody3D
var _drone_operator: CharacterBody3D

## World state
var _scenario_configured: bool = false


func _ready() -> void:
	print("[Main] GeoSim starting...")

	# Container for buried object markers
	_object_markers = Node3D.new()
	_object_markers.name = "ObjectMarkers"
	add_child(_object_markers)

	# Container for survey line visualizations
	_survey_lines = Node3D.new()
	_survey_lines.name = "SurveyLines"
	add_child(_survey_lines)

	# Cache operator references
	_ground_operator = get_node_or_null("Operator")
	_drone_operator = get_node_or_null("DroneOperator")

	# Wire operator signals
	var hud := get_node_or_null("UI/HUD")
	if _ground_operator and hud:
		_ground_operator.gradient_reading.connect(hud.update_reading)

	# Connect to SurveyManager
	SurveyManager.state_changed.connect(_on_state_changed)
	SurveyManager.scenario_loaded.connect(_on_scenario_loaded)
	SurveyManager.operator_switched.connect(_on_operator_switched)
	SurveyManager.survey_started.connect(_on_survey_started)
	SurveyManager.survey_finished.connect(_on_survey_finished)

	# Initial state — show main menu, disable gameplay
	_set_gameplay_active(false)

	# Initialize server connection
	await get_tree().process_frame
	_check_server()


func _process(_delta: float) -> void:
	if SurveyManager.current_state != SurveyManager.State.SURVEYING:
		return

	# Push live stats from operator to HUD each frame
	var hud := get_node_or_null("UI/HUD")
	if _ground_operator and hud and hud.has_method("update_stats"):
		hud.update_stats(_ground_operator.distance_traveled,
			DataRecorder.samples.size())


func _check_server() -> void:
	var ping_result = await PhysicsClient.ping()
	if ping_result.get("status") == "ok":
		var msg: String = ping_result.get("data", {}).get("message", "")
		if not msg.contains("mock"):
			print("[Main] Connected to GeoSim physics server")
		else:
			print("[Main] Running in mock mode (no server)")


func _on_state_changed(new_state: SurveyManager.State) -> void:
	match new_state:
		SurveyManager.State.MAIN_MENU:
			_set_gameplay_active(false)
		SurveyManager.State.SCENARIO_SELECT:
			_set_gameplay_active(false)
		SurveyManager.State.SURVEY_PLANNING:
			_set_gameplay_active(false)
			# Show terrain preview if scenario loaded
			if _scenario_configured:
				_show_terrain_preview()
		SurveyManager.State.SURVEYING:
			_set_gameplay_active(true)
		SurveyManager.State.PAUSED:
			_set_gameplay_active(false)
		SurveyManager.State.POST_SURVEY:
			_set_gameplay_active(false)
		SurveyManager.State.HIRT_SURVEY:
			_set_gameplay_active(false)
		SurveyManager.State.TRAINING:
			pass  # Handled by training manager


func _on_scenario_loaded(info: Dictionary) -> void:
	_configure_from_scenario(info)


func _on_operator_switched(mode: SurveyManager.OperatorMode) -> void:
	match mode:
		SurveyManager.OperatorMode.GROUND:
			_activate_operator(_ground_operator)
			if _drone_operator:
				_deactivate_operator(_drone_operator)
		SurveyManager.OperatorMode.DRONE:
			if _drone_operator:
				_activate_operator(_drone_operator)
			_deactivate_operator(_ground_operator)


func _on_survey_started() -> void:
	# Start recording
	var inst_name: String = SurveyManager.Instrument.keys()[SurveyManager.current_instrument].to_lower()
	var mode_name: String = SurveyManager.OperatorMode.keys()[SurveyManager.current_operator_mode].to_lower()
	DataRecorder.start_recording(
		SurveyManager.scenario_info.get("name", "unknown"),
		inst_name,
		mode_name,
		SurveyManager.survey_plan,
	)


func _on_survey_finished() -> void:
	DataRecorder.stop_recording()


func _set_gameplay_active(active: bool) -> void:
	if _ground_operator:
		_ground_operator.set_process(active)
		_ground_operator.set_physics_process(active)
		_ground_operator.set_process_unhandled_input(active)
	if _drone_operator:
		_drone_operator.set_process(active and
			SurveyManager.current_operator_mode == SurveyManager.OperatorMode.DRONE)
		_drone_operator.set_physics_process(active and
			SurveyManager.current_operator_mode == SurveyManager.OperatorMode.DRONE)


func _activate_operator(op: Node3D) -> void:
	if not op:
		return
	op.set_process(true)
	op.set_physics_process(true)
	op.set_process_unhandled_input(true)
	var cam := op.get_node_or_null("Camera3D") as Camera3D
	if cam:
		cam.current = true
	SurveyManager.active_operator = op


func _deactivate_operator(op: Node3D) -> void:
	if not op:
		return
	op.set_process(false)
	op.set_physics_process(false)
	op.set_process_unhandled_input(false)


func _show_terrain_preview() -> void:
	# Position camera for an overview if possible
	pass


func _configure_from_scenario(data: Dictionary) -> void:
	# Update terrain
	var terrain := get_node_or_null("Terrain")
	if terrain and terrain.has_method("update_from_scenario"):
		terrain.update_from_scenario(data)

	# Extract terrain geometry
	var terrain_info: Dictionary = data.get("terrain", {})
	var surface_elev: float = terrain_info.get("surface_elevation", 0.0)
	var x_ext: Array = terrain_info.get("x_extent", [0, 20])
	var y_ext: Array = terrain_info.get("y_extent", [0, 20])
	var center_x: float = (x_ext[0] + x_ext[1]) / 2.0
	var center_z: float = (y_ext[0] + y_ext[1]) / 2.0

	# Configure ground operator with terrain bounds and starting position
	if _ground_operator:
		_ground_operator.terrain_x_extent = Vector2(x_ext[0], x_ext[1])
		_ground_operator.terrain_y_extent = Vector2(y_ext[0], y_ext[1])
		_ground_operator.surface_elevation = surface_elev
		_ground_operator.position = Vector3(center_x, surface_elev + 2.0, center_z)
		print("[Main] Operator placed at (%.1f, %.1f), elevation %.1f" % [
			center_x, center_z, surface_elev])

	# Configure drone operator similarly
	if _drone_operator:
		_drone_operator.position = Vector3(center_x, surface_elev + 5.0, center_z)

	# Update HUD with scenario name
	var hud := get_node_or_null("UI/HUD")
	if hud and hud.has_method("set_scenario_name"):
		hud.set_scenario_name(data.get("name", ""))

	# Create visual markers for buried objects
	var objects: Array = data.get("objects", [])
	_create_object_markers(objects, surface_elev)

	_scenario_configured = true


func _create_object_markers(objects: Array, surface_elev: float) -> void:
	# Clear existing markers
	for child in _object_markers.get_children():
		child.queue_free()

	for obj in objects:
		var obj_pos: Array = obj.get("position", [0, 0, 0])
		var obj_name: String = obj.get("name", "unknown")
		var obj_radius: float = obj.get("radius", 0.05)

		# GeoSim [X_east, Y_north, Z_up] -> Godot [X, Y_up, Z]
		var godot_pos := CoordUtil.to_godot(Vector3(
			obj_pos[0],
			obj_pos[1] if obj_pos.size() > 1 else 0.0,
			obj_pos[2] if obj_pos.size() > 2 else -1.0,
		))

		# Red disk on the surface marking the object's XZ position
		var surface_marker := _create_surface_marker(obj_name)
		surface_marker.position = Vector3(godot_pos.x, surface_elev + 0.02, godot_pos.z)
		_object_markers.add_child(surface_marker)

		# Translucent sphere at actual burial depth
		var buried_sphere := _create_buried_sphere(obj_radius, obj_name)
		buried_sphere.position = godot_pos
		_object_markers.add_child(buried_sphere)


func _create_surface_marker(obj_name: String) -> MeshInstance3D:
	var mesh_inst := MeshInstance3D.new()
	mesh_inst.name = "Marker_" + obj_name

	var cyl := CylinderMesh.new()
	cyl.top_radius = 0.15
	cyl.bottom_radius = 0.15
	cyl.height = 0.04
	mesh_inst.mesh = cyl

	var mat := StandardMaterial3D.new()
	mat.albedo_color = Color(1.0, 0.2, 0.2, 0.6)
	mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	mesh_inst.material_override = mat

	return mesh_inst


func _create_buried_sphere(radius: float, obj_name: String) -> MeshInstance3D:
	var mesh_inst := MeshInstance3D.new()
	mesh_inst.name = "Object_" + obj_name

	var sphere := SphereMesh.new()
	sphere.radius = max(radius, 0.05) * 3.0  # Scale up for visibility
	sphere.height = sphere.radius * 2.0
	mesh_inst.mesh = sphere

	var mat := StandardMaterial3D.new()
	mat.albedo_color = Color(0.8, 0.3, 0.1, 0.4)
	mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	mesh_inst.material_override = mat

	return mesh_inst


## Draw planned survey lines on the terrain.
func draw_survey_lines(lines: Array, surface_elev: float) -> void:
	# Clear existing
	for child in _survey_lines.get_children():
		child.queue_free()

	if lines.is_empty():
		return

	var im := ImmediateMesh.new()
	var mesh_inst := MeshInstance3D.new()
	mesh_inst.name = "PlannedLines"

	im.surface_begin(Mesh.PRIMITIVE_LINES)
	for line_idx in range(lines.size()):
		var line: PackedVector3Array = lines[line_idx]
		for i in range(line.size() - 1):
			var p0 := line[i]
			var p1 := line[i + 1]
			im.surface_add_vertex(Vector3(p0.x, surface_elev + 0.03, p0.z))
			im.surface_add_vertex(Vector3(p1.x, surface_elev + 0.03, p1.z))
	im.surface_end()

	mesh_inst.mesh = im
	var mat := StandardMaterial3D.new()
	mat.albedo_color = Color(1.0, 1.0, 1.0, 0.6)
	mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	mesh_inst.material_override = mat

	_survey_lines.add_child(mesh_inst)
