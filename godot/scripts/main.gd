## Main scene orchestrator.
##
## Responds to SurveyManager state changes to configure the world,
## manage operators, and coordinate between subsystems.
extends Node3D

## Visual containers
var _object_markers: Node3D
var _survey_lines: Node3D
var _event_markers: Node3D
var _site_dressing: Node3D = null

## References to operators (set up in _ready)
var _ground_operator: CharacterBody3D
var _drone_operator: CharacterBody3D

## World state
var _scenario_configured: bool = false
var _objects_visible: bool = false  # Hidden by default (realistic mode)

## Cached survey line data for color re-drawing
var _cached_lines: Array = []
var _cached_surface_elev := 0.0
var _current_active_line := 0


func _ready() -> void:
	print("[Main] GeoSim starting...")
	get_tree().set_auto_accept_quit(false)

	# Container for buried object markers
	_object_markers = Node3D.new()
	_object_markers.name = "ObjectMarkers"
	add_child(_object_markers)

	# Container for survey line visualizations
	_survey_lines = Node3D.new()
	_survey_lines.name = "SurveyLines"
	add_child(_survey_lines)

	# Container for event marker stakes
	_event_markers = Node3D.new()
	_event_markers.name = "EventMarkers"
	add_child(_event_markers)

	# Site dressing (stakes, tape, equipment)
	var dressing_script := load("res://scripts/site_dressing.gd")
	if dressing_script:
		_site_dressing = Node3D.new()
		_site_dressing.set_script(dressing_script)
		add_child(_site_dressing)

	# Cache operator references
	_ground_operator = get_node_or_null("Operator")
	_drone_operator = get_node_or_null("DroneOperator")

	# Wire operator signals
	var hud := get_node_or_null("UI/HUD")
	if _ground_operator and hud:
		_ground_operator.gradient_reading.connect(hud.update_reading)
	if _ground_operator:
		_ground_operator.event_marked.connect(_on_event_marked)
		_ground_operator.line_changed.connect(_on_line_changed)
		if hud and _ground_operator.has_signal("line_completed"):
			_ground_operator.line_completed.connect(hud.on_line_completed)

	# Wire drone progress to HUD
	if _drone_operator and hud and _drone_operator.has_signal("drone_progress"):
		_drone_operator.drone_progress.connect(hud.update_drone_progress)

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


func _unhandled_input(event: InputEvent) -> void:
	if event.is_action_pressed("toggle_targets"):
		_toggle_object_visibility()
	# Fallback raw keycode check in case input action doesn't register
	elif event is InputEventKey and event.pressed and not event.echo:
		if event.keycode == KEY_H or event.physical_keycode == KEY_H:
			_toggle_object_visibility()


func _notification(what: int) -> void:
	if what == NOTIFICATION_WM_CLOSE_REQUEST:
		if DataRecorder.is_recording:
			DataRecorder.export_session()
			DataRecorder.stop_recording()
		get_tree().quit()


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
	op.visible = true
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
	op.visible = false
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

	# Set up site dressing (stakes, boundary tape, equipment)
	if _site_dressing and _site_dressing.has_method("setup"):
		_site_dressing.setup(terrain, Vector2(x_ext[0], x_ext[1]),
			Vector2(y_ext[0], y_ext[1]), surface_elev)

	_scenario_configured = true


func _toggle_object_visibility() -> void:
	_objects_visible = not _objects_visible
	for child in _object_markers.get_children():
		child.visible = _objects_visible

	var hud := get_node_or_null("UI/HUD")
	if hud and hud.has_method("set_targets_visible"):
		hud.set_targets_visible(_objects_visible)


func _create_object_markers(objects: Array, surface_elev: float) -> void:
	# Clear existing markers
	for child in _object_markers.get_children():
		child.queue_free()

	var terrain := get_node_or_null("Terrain")

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

		# Get actual terrain height at this XZ position (follows noise displacement)
		var terrain_h := surface_elev
		if terrain and terrain.has_method("get_height_at"):
			terrain_h = terrain.get_height_at(godot_pos.x, godot_pos.z)

		# Red disk on the surface marking the object's XZ position
		var surface_marker := _create_surface_marker(obj_name)
		surface_marker.position = Vector3(godot_pos.x, terrain_h + 0.05, godot_pos.z)
		surface_marker.visible = _objects_visible
		_object_markers.add_child(surface_marker)

		# Translucent sphere at actual burial depth
		var buried_sphere := _create_buried_sphere(obj_radius, obj_name)
		buried_sphere.position = godot_pos
		buried_sphere.visible = _objects_visible
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


## Draw planned survey lines on the terrain with per-line color coding.
## Lines are draped onto terrain height and subdivided for contour following.
func draw_survey_lines(lines: Array, surface_elev: float) -> void:
	_cached_lines = lines
	_cached_surface_elev = surface_elev
	_current_active_line = 0
	_redraw_survey_lines()

	# Add pin flags along survey lines
	if _site_dressing and _site_dressing.has_method("add_survey_line_flags"):
		_site_dressing.add_survey_line_flags(lines)


func _redraw_survey_lines() -> void:
	# Clear existing
	for child in _survey_lines.get_children():
		child.queue_free()

	if _cached_lines.is_empty():
		return

	var terrain := get_node_or_null("Terrain")

	for line_idx in range(_cached_lines.size()):
		var line: PackedVector3Array = _cached_lines[line_idx]
		if line.size() < 2:
			continue

		# Determine line color based on state
		var line_color: Color
		if line_idx < _current_active_line:
			line_color = Color(0.2, 1.0, 0.3, 0.6)  # Green = completed
		elif line_idx == _current_active_line:
			line_color = Color(0.3, 0.9, 1.0, 0.8)  # Cyan = active
		else:
			line_color = Color(0.9, 0.9, 0.9, 0.4)  # White = planned

		var im := ImmediateMesh.new()
		var mesh_inst := MeshInstance3D.new()
		mesh_inst.name = "Line_%d" % line_idx

		im.surface_begin(Mesh.PRIMITIVE_LINES)
		for i in range(line.size() - 1):
			var p0 := line[i]
			var p1 := line[i + 1]
			# Subdivide long segments for terrain contour following
			var seg_len := p0.distance_to(p1)
			var num_subdivs := int(ceil(seg_len / 0.5))
			num_subdivs = maxi(num_subdivs, 1)

			for s in range(num_subdivs):
				var t0 := float(s) / float(num_subdivs)
				var t1 := float(s + 1) / float(num_subdivs)
				var v0 := p0.lerp(p1, t0)
				var v1 := p0.lerp(p1, t1)
				var h0 := _get_line_height(terrain, v0.x, v0.z) + 0.03
				var h1 := _get_line_height(terrain, v1.x, v1.z) + 0.03
				im.surface_add_vertex(Vector3(v0.x, h0, v0.z))
				im.surface_add_vertex(Vector3(v1.x, h1, v1.z))
		im.surface_end()

		mesh_inst.mesh = im
		var mat := StandardMaterial3D.new()
		mat.albedo_color = line_color
		mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
		mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
		mesh_inst.material_override = mat

		_survey_lines.add_child(mesh_inst)


func _get_line_height(terrain: Node, world_x: float, world_z: float) -> float:
	if terrain and terrain.has_method("get_height_at"):
		return terrain.get_height_at(world_x, world_z)
	return _cached_surface_elev


func _on_line_changed(new_index: int) -> void:
	_current_active_line = new_index
	_redraw_survey_lines()


func _on_event_marked(world_pos: Vector3) -> void:
	_create_event_stake(world_pos)


func _create_event_stake(pos: Vector3) -> void:
	## Create a visible 3D stake at the event marker position.
	var stake := MeshInstance3D.new()
	stake.name = "EventStake_%d" % _event_markers.get_child_count()

	# Thin vertical cylinder (survey stake)
	var cyl := CylinderMesh.new()
	cyl.top_radius = 0.02
	cyl.bottom_radius = 0.02
	cyl.height = 0.5
	stake.mesh = cyl

	var mat := StandardMaterial3D.new()
	mat.albedo_color = Color(1.0, 0.8, 0.0)
	mat.emission_enabled = true
	mat.emission = Color(1.0, 0.8, 0.0)
	mat.emission_energy_multiplier = 0.3
	stake.material_override = mat

	# Position stake sticking up from ground
	stake.position = Vector3(pos.x, pos.y + 0.25, pos.z)
	_event_markers.add_child(stake)

	# Small flag at top
	var flag := MeshInstance3D.new()
	var flag_mesh := BoxMesh.new()
	flag_mesh.size = Vector3(0.12, 0.06, 0.005)
	flag.mesh = flag_mesh

	var flag_mat := StandardMaterial3D.new()
	flag_mat.albedo_color = Color(1.0, 0.3, 0.0)
	flag_mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	flag.material_override = flag_mat

	flag.position = Vector3(0.06, 0.25, 0)  # At top of stake, offset to side
	stake.add_child(flag)
