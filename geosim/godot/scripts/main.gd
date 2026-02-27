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
var _world_env: WorldEnvironment
var _sun_light: DirectionalLight3D
var _sky_material: PhysicalSkyMaterial = null

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

	# Atmospheric particles (dust, pollen, insects, mist)
	var atmo_script := load("res://scripts/atmosphere_particles.gd")
	if atmo_script:
		var atmo := Node3D.new()
		atmo.name = "AtmosphereParticles"
		atmo.set_script(atmo_script)
		add_child(atmo)

	# Debug capture mode (auto-removes if --debug-capture not passed)
	var debug_script := load("res://scripts/debug_capture.gd")
	if debug_script:
		var debug_node := Node.new()
		debug_node.name = "DebugCapture"
		debug_node.set_script(debug_script)
		add_child(debug_node)

	# Cache operator references
	_ground_operator = get_node_or_null("Operator")
	_drone_operator = get_node_or_null("DroneOperator")
	_world_env = get_node_or_null("WorldEnvironment")
	_sun_light = get_node_or_null("DirectionalLight")

	if _world_env and _world_env.environment:
		var sky := _world_env.environment.sky
		if sky and sky.sky_material is PhysicalSkyMaterial:
			_sky_material = sky.sky_material as PhysicalSkyMaterial

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

	# Hide drone by default (ground mode is the initial operator mode)
	if _drone_operator:
		_drone_operator.visible = false

	# Initialize server connection
	await get_tree().process_frame
	_check_server()


func _input(event: InputEvent) -> void:
	# Use _input (not _unhandled_input) so H works even when UI Controls have focus
	if event.is_action_pressed("toggle_targets"):
		_toggle_object_visibility()
		get_viewport().set_input_as_handled()
	elif event is InputEventKey and event.pressed and not event.echo:
		if event.keycode == KEY_H or event.physical_keycode == KEY_H:
			_toggle_object_visibility()
			get_viewport().set_input_as_handled()


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
		var drone_active: bool = (active and
			SurveyManager.current_operator_mode == SurveyManager.OperatorMode.DRONE)
		_drone_operator.set_process(drone_active)
		_drone_operator.set_physics_process(drone_active)
		_drone_operator.set_process_unhandled_input(drone_active)


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
	_apply_environment_styling(data)

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

	# Configure drone operator with terrain bounds and starting position
	if _drone_operator:
		_drone_operator.terrain_x_extent = Vector2(x_ext[0], x_ext[1])
		_drone_operator.terrain_y_extent = Vector2(y_ext[0], y_ext[1])
		_drone_operator.surface_elevation = surface_elev
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
			Vector2(y_ext[0], y_ext[1]), surface_elev, data)

	# Create distant ground plane extending to horizon (fills gap beyond terrain edge)
	_create_background_ground(terrain_info, data)

	# Water table — standing water plane + shader parameter for wet ground
	_setup_water_features(terrain_info, data, terrain)

	# Dynamic ReflectionProbe for sharp specular reflections on wet terrain
	_create_reflection_probe(terrain_info)

	_scenario_configured = true


func _create_background_ground(terrain_info: Dictionary, _data: Dictionary) -> void:
	## Ring-shaped ground extending from the terrain edge outward to the fog
	## distance, so it never pokes through terrain features (craters, etc).
	var old_bg := get_node_or_null("BackgroundGround")
	if old_bg:
		old_bg.queue_free()

	var x_ext: Array = terrain_info.get("x_extent", [0, 20])
	var y_ext: Array = terrain_info.get("y_extent", [0, 20])
	var surface_elev: float = terrain_info.get("surface_elevation", 0.0)

	# Inner boundary = terrain edge (with small overlap to avoid gap)
	var inner_margin := 0.5
	var x_min: float = x_ext[0] - inner_margin
	var x_max: float = x_ext[1] + inner_margin
	var z_min: float = y_ext[0] - inner_margin
	var z_max: float = y_ext[1] + inner_margin

	# Outer boundary = 250m beyond terrain edge (fog hides the far edge)
	var outer_dist := 250.0
	var ox_min: float = x_ext[0] - outer_dist
	var ox_max: float = x_ext[1] + outer_dist
	var oz_min: float = y_ext[0] - outer_dist
	var oz_max: float = y_ext[1] + outer_dist

	# Build ring mesh as 4 rectangular strips (N, S, E, W) + 4 corners
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)

	var y: float = surface_elev - 0.02

	# North strip (far Z)
	_add_bg_quad(st, ox_min, oz_min, ox_max, z_min, y)
	# South strip (near Z)
	_add_bg_quad(st, ox_min, z_max, ox_max, oz_max, y)
	# West strip (left X, between inner Z bounds)
	_add_bg_quad(st, ox_min, z_min, x_min, z_max, y)
	# East strip (right X, between inner Z bounds)
	_add_bg_quad(st, x_max, z_min, ox_max, z_max, y)

	st.generate_normals()
	var ring_mesh := st.commit()

	var mesh_inst := MeshInstance3D.new()
	mesh_inst.name = "BackgroundGround"
	mesh_inst.mesh = ring_mesh

	# Color: muted version of top terrain layer
	var layers: Array = terrain_info.get("layers", [])
	var ground_color := Color(0.30, 0.28, 0.22)
	if not layers.is_empty():
		var hex := str(layers[0].get("color", ""))
		if not hex.is_empty():
			ground_color = Color.from_string(hex, ground_color)
	ground_color = ground_color.darkened(0.15)

	var mat := StandardMaterial3D.new()
	mat.albedo_color = ground_color
	mat.roughness = 1.0
	mat.metallic = 0.0
	mesh_inst.material_override = mat
	mesh_inst.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF

	add_child(mesh_inst)


func _add_bg_quad(st: SurfaceTool, x0: float, z0: float, x1: float, z1: float, y: float) -> void:
	## Add a flat quad to the SurfaceTool at height y.
	st.set_uv(Vector2(0, 0))
	st.add_vertex(Vector3(x0, y, z0))
	st.set_uv(Vector2(1, 0))
	st.add_vertex(Vector3(x1, y, z0))
	st.set_uv(Vector2(1, 1))
	st.add_vertex(Vector3(x1, y, z1))

	st.set_uv(Vector2(0, 0))
	st.add_vertex(Vector3(x0, y, z0))
	st.set_uv(Vector2(1, 1))
	st.add_vertex(Vector3(x1, y, z1))
	st.set_uv(Vector2(0, 1))
	st.add_vertex(Vector3(x0, y, z1))


func _setup_water_features(terrain_info: Dictionary, data: Dictionary, terrain: Node) -> void:
	# Remove previous water plane
	var old_water := get_node_or_null("WaterPlane")
	if old_water:
		old_water.queue_free()

	var metadata: Dictionary = data.get("metadata", {})
	if not metadata.has("water_table_depth"):
		return

	var surface_elev: float = terrain_info.get("surface_elevation", 0.0)
	var wt_depth := float(metadata.get("water_table_depth", 2.0))
	var water_height := surface_elev - wt_depth

	# Set shader parameter on terrain material
	if terrain and terrain is MeshInstance3D:
		var mat: Material = (terrain as MeshInstance3D).material_override
		if not mat:
			mat = (terrain as MeshInstance3D).get_surface_override_material(0)
		if mat is ShaderMaterial:
			(mat as ShaderMaterial).set_shader_parameter("water_table_height", water_height)

	# Only create visible water plane if water table is near surface
	if wt_depth > 1.5:
		return

	var x_ext: Array = terrain_info.get("x_extent", [0, 20])
	var y_ext: Array = terrain_info.get("y_extent", [0, 20])
	var margin := 2.0

	var water_mesh := PlaneMesh.new()
	water_mesh.size = Vector2(
		x_ext[1] - x_ext[0] + margin * 2,
		y_ext[1] - y_ext[0] + margin * 2)
	water_mesh.subdivide_width = 32
	water_mesh.subdivide_depth = 32

	var water_inst := MeshInstance3D.new()
	water_inst.name = "WaterPlane"
	water_inst.mesh = water_mesh

	var center_x := (float(x_ext[0]) + float(x_ext[1])) / 2.0
	var center_z := (float(y_ext[0]) + float(y_ext[1])) / 2.0
	water_inst.position = Vector3(center_x, water_height, center_z)

	# Try water surface shader for advanced rendering, fallback to StandardMaterial3D
	var water_shader := load("res://shaders/water_surface.gdshader") as Shader
	if water_shader:
		var shader_mat := ShaderMaterial.new()
		shader_mat.shader = water_shader

		# Scenario-driven water color
		var name_lower := str(data.get("name", "")).to_lower()
		if name_lower.contains("swamp") or name_lower.contains("marsh"):
			shader_mat.set_shader_parameter("shallow_color", Vector3(0.12, 0.16, 0.10))
			shader_mat.set_shader_parameter("deep_color", Vector3(0.04, 0.06, 0.04))
			shader_mat.set_shader_parameter("wave_amplitude_1", 0.01)
			shader_mat.set_shader_parameter("wave_amplitude_2", 0.005)
			shader_mat.set_shader_parameter("wave_amplitude_3", 0.003)
			shader_mat.set_shader_parameter("foam_intensity", 0.2)
			shader_mat.set_shader_parameter("base_alpha", 0.8)
		else:
			shader_mat.set_shader_parameter("shallow_color", Vector3(0.15, 0.22, 0.18))
			shader_mat.set_shader_parameter("deep_color", Vector3(0.03, 0.06, 0.08))

		# Use procedural normal maps if no real textures available
		var water_normal := _create_water_normal_texture(256, 42)
		var water_normal2 := _create_water_normal_texture(256, 99)
		shader_mat.set_shader_parameter("normal_map_1", water_normal)
		shader_mat.set_shader_parameter("normal_map_2", water_normal2)

		water_inst.material_override = shader_mat
	else:
		# Fallback: semi-transparent dark water material with reflectivity
		var mat := StandardMaterial3D.new()
		mat.albedo_color = Color(0.05, 0.08, 0.1, 0.6)
		mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
		mat.metallic = 0.1
		mat.roughness = 0.15
		mat.clearcoat_enabled = true
		mat.clearcoat = 0.8
		mat.clearcoat_roughness = 0.1
		mat.cull_mode = BaseMaterial3D.CULL_DISABLED
		water_inst.material_override = mat

	water_inst.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF

	add_child(water_inst)


func _create_reflection_probe(terrain_info: Dictionary) -> void:
	# Remove previous probe
	var old_probe := get_node_or_null("ReflectionProbe")
	if old_probe:
		old_probe.queue_free()

	var x_ext: Array = terrain_info.get("x_extent", [0, 20])
	var y_ext: Array = terrain_info.get("y_extent", [0, 20])
	var surface_elev: float = terrain_info.get("surface_elevation", 0.0)

	var center_x := (float(x_ext[0]) + float(x_ext[1])) / 2.0
	var center_z := (float(y_ext[0]) + float(y_ext[1])) / 2.0
	var extent_x := float(x_ext[1]) - float(x_ext[0])
	var extent_z := float(y_ext[1]) - float(y_ext[0])

	var probe := ReflectionProbe.new()
	probe.name = "ReflectionProbe"
	probe.transform.origin = Vector3(center_x, surface_elev + 2.0, center_z)
	probe.box_projection = true
	probe.size = Vector3(extent_x + 10.0, 10.0, extent_z + 10.0)
	probe.interior = false
	probe.update_mode = ReflectionProbe.UPDATE_ONCE
	add_child(probe)


func _apply_environment_styling(data: Dictionary) -> void:
	if not _world_env or not _world_env.environment:
		return

	var env := _world_env.environment
	var profile := _environment_profile(data)
	var wetness: float = profile.get("wetness", 0.3)
	var ruggedness: float = profile.get("ruggedness", 0.3)
	var fog_density_override: float = profile.get("fog_density", -1.0)
	var fog_color: Color = profile.get("fog_color", Color(0.7, 0.75, 0.8))
	var ambient_color: Color = profile.get("ambient_color", Color(0.65, 0.72, 0.82))
	var sun_color: Color = profile.get("sun_color", Color(1.0, 0.95, 0.84))
	var rayleigh_color: Color = profile.get("rayleigh_color", Color(0.3, 0.5, 0.75))

	env.ambient_light_color = ambient_color
	env.ambient_light_energy = lerpf(0.62, 0.44, wetness)
	env.fog_enabled = true
	env.fog_light_color = fog_color
	env.fog_density = fog_density_override if fog_density_override >= 0.0 else lerpf(0.0035, 0.013, wetness)
	env.fog_aerial_perspective = lerpf(0.22, 0.46, wetness)
	env.fog_height = lerpf(2.8, 1.2, wetness)
	env.fog_height_density = lerpf(0.1, 0.22, wetness)
	env.volumetric_fog_enabled = true
	env.volumetric_fog_density = lerpf(0.005, 0.016, wetness)
	env.volumetric_fog_albedo = fog_color.lightened(0.1)
	env.volumetric_fog_length = lerpf(120.0, 70.0, wetness)

	# Glow: lower intensity, HDR threshold so only sun/specular bloom
	env.glow_intensity = lerpf(0.18, 0.30, 1.0 - wetness * 0.5)
	env.glow_bloom = lerpf(0.06, 0.14, ruggedness)
	_set_env_property_if_present(env, "glow_hdr_threshold", lerpf(0.9, 1.2, wetness))

	# SSAO tuned per ruggedness
	env.ssao_intensity = lerpf(0.45, 0.8, ruggedness)
	_set_env_property_if_present(env, "ssao_radius", lerpf(1.5, 3.0, ruggedness))
	_set_env_property_if_present(env, "ssao_sharpness", 0.98)

	# SDFGI: bounce lighting energy varies by wetness
	_set_env_property_if_present(env, "sdfgi_energy", lerpf(0.7, 1.0, wetness))

	# SSR: only on wet scenarios for specular reflections on terrain
	_set_env_property_if_present(env, "ssr_enabled", wetness > 0.15)

	# Color grading per wetness
	_set_env_property_if_present(env, "adjustment_enabled", true)
	_set_env_property_if_present(env, "adjustment_brightness", lerpf(1.0, 0.96, wetness))
	_set_env_property_if_present(env, "adjustment_contrast", lerpf(1.02, 1.06, wetness))
	_set_env_property_if_present(env, "adjustment_saturation", lerpf(1.05, 0.88, wetness))

	_set_env_property_if_present(env, "dof_blur_far_distance", lerpf(52.0, 34.0, wetness))
	_set_env_property_if_present(env, "dof_blur_far_transition", lerpf(24.0, 14.0, wetness))

	# HDRI sky integration — load based on lighting preset / time of day
	var hdri_loaded := _try_load_hdri_sky(profile, env)

	# PhysicalSkyMaterial parameters (used as fallback if no HDRI loaded)
	if not hdri_loaded and _sky_material:
		_sky_material.turbidity = lerpf(6.0, 16.0, wetness)
		_sky_material.rayleigh_coefficient = 2.0
		_sky_material.rayleigh_color = rayleigh_color
		_sky_material.mie_coefficient = lerpf(0.003, 0.012, wetness)
		_sky_material.energy_multiplier = lerpf(1.0, 0.85, wetness)

	# SSIL tuning
	_set_env_property_if_present(env, "ssil_enabled", true)
	_set_env_property_if_present(env, "ssil_radius", lerpf(4.0, 6.0, ruggedness))
	_set_env_property_if_present(env, "ssil_intensity", lerpf(0.5, 0.7, ruggedness))

	if _sun_light:
		# Apply lighting preset if specified, otherwise derive from wetness
		var lighting_preset: String = profile.get("lighting_preset", "")
		var preset := _get_lighting_preset(lighting_preset, wetness)
		_sun_light.light_color = preset.get("sun_color", sun_color)
		_sun_light.light_energy = preset.get("sun_energy", lerpf(1.35, 1.02, wetness))
		_set_object_property_if_present(_sun_light, ["shadow_contact"], true)
		_set_object_property_if_present(_sun_light, ["light_angular_distance", "light_angular_size"], lerpf(0.35, 0.75, wetness))
		var sun_altitude: float = preset.get("sun_altitude", lerpf(38.0, 24.0, wetness))
		var rot := _sun_light.rotation_degrees
		rot.x = -sun_altitude
		_sun_light.rotation_degrees = rot

		# Apply preset overrides to environment
		if preset.has("ambient_energy"):
			env.ambient_light_energy = preset["ambient_energy"]
		if preset.has("fog_density"):
			env.fog_density = preset["fog_density"]
		if _sky_material and preset.has("turbidity"):
			_sky_material.turbidity = preset["turbidity"]
		if _sky_material and preset.has("rayleigh"):
			_sky_material.rayleigh_coefficient = preset["rayleigh"]


func _set_env_property_if_present(env: Environment, property_name: String, value: Variant) -> void:
	for prop in env.get_property_list():
		if str(prop.get("name", "")) == property_name:
			env.set(property_name, value)
			return


func _set_object_property_if_present(target: Object, property_names: Array[String], value: Variant) -> void:
	for property_name in property_names:
		for prop in target.get_property_list():
			if str(prop.get("name", "")) == property_name:
				target.set(property_name, value)
				return


func _environment_profile(data: Dictionary) -> Dictionary:
	var terrain_data: Dictionary = data.get("terrain", {})
	var metadata: Dictionary = data.get("metadata", {})
	var layers: Array = terrain_data.get("layers", [])
	var anomalies: Array = data.get("anomaly_zones", [])
	var env_profile: Dictionary = data.get("environment_profile", {})

	var name := str(data.get("name", "")).to_lower()
	var desc := str(data.get("description", "")).to_lower()
	var category := str(metadata.get("category", "")).to_lower()

	var mean_cond := 0.05
	if not layers.is_empty():
		var s := 0.0
		for layer in layers:
			s += float(layer.get("conductivity", 0.05))
		mean_cond = s / float(layers.size())

	var wetness := clampf(mean_cond / 0.25, 0.0, 1.0)
	if metadata.has("water_table_depth"):
		var wt_depth := float(metadata.get("water_table_depth", 2.0))
		wetness = clampf(wetness + clampf((1.0 - wt_depth) / 1.0, 0.0, 0.55), 0.0, 1.0)
	if name.contains("swamp") or name.contains("marsh") or desc.contains("waterlogged"):
		wetness = clampf(wetness + 0.35, 0.0, 1.0)
	# Incorporate soil_environment saturation from scenario defaults
	var soil_env: Dictionary = data.get("soil_environment", {})
	if not soil_env.is_empty():
		var sat := float(soil_env.get("saturation", 0.5))
		wetness = clampf(wetness * 0.6 + sat * 0.4, 0.0, 1.0)
	# Incorporate survey plan conditions if available
	var plan: Dictionary = SurveyManager.survey_plan
	if plan.has("saturation"):
		var plan_sat := float(plan.get("saturation", 0.6))
		wetness = clampf(wetness * 0.5 + plan_sat * 0.5, 0.0, 1.0)
	if env_profile.has("wetness"):
		wetness = clampf(float(env_profile.get("wetness", wetness)), 0.0, 1.0)

	var ruggedness := clampf(float(anomalies.size()) * 0.08, 0.0, 0.35)
	if name.contains("crater") or desc.contains("crater") or metadata.has("crater"):
		ruggedness = clampf(ruggedness + 0.4, 0.0, 1.0)
	if category.contains("uxo"):
		ruggedness = clampf(ruggedness + 0.2, 0.0, 1.0)
	if category.contains("forensic"):
		ruggedness = clampf(ruggedness - 0.15, 0.05, 1.0)
	if env_profile.has("ruggedness"):
		ruggedness = clampf(float(env_profile.get("ruggedness", ruggedness)), 0.0, 1.0)

	var top_layer_color := Color(0.45, 0.36, 0.28)
	if not layers.is_empty():
		top_layer_color = _color_from_hex(str(layers[0].get("color", "")), top_layer_color)

	var fog_color := Color(0.74, 0.79, 0.84).lerp(Color(0.56, 0.63, 0.67), wetness)
	var ambient := top_layer_color.lightened(0.7).lerp(Color(0.6, 0.68, 0.76), 0.6)
	var sun_color := Color(1.0, 0.95, 0.84).lerp(Color(0.95, 0.93, 0.88), wetness)
	var rayleigh_color := Color(0.30, 0.50, 0.75).lerp(Color(0.35, 0.42, 0.55), wetness)

	return {
		"wetness": wetness,
		"ruggedness": ruggedness,
		"fog_density": float(env_profile.get("fog_density", -1.0)),
		"fog_color": fog_color,
		"ambient_color": ambient,
		"sun_color": sun_color,
		"rayleigh_color": rayleigh_color,
		"lighting_preset": str(env_profile.get("lighting_preset", "")),
	}


func _get_lighting_preset(preset_name: String, wetness: float) -> Dictionary:
	## Return lighting parameters for named presets.
	## Scenarios can specify "lighting_preset" in environment_profile.
	match preset_name.to_lower():
		"midday":
			return {
				"sun_energy": 1.4,
				"sun_color": Color(1.0, 0.95, 0.83),
				"sun_altitude": 45.0,
				"ambient_energy": 0.5,
				"fog_density": 0.004,
				"turbidity": 8.0,
				"rayleigh": 2.0,
			}
		"overcast":
			return {
				"sun_energy": 0.8,
				"sun_color": Color(0.91, 0.89, 0.86),
				"sun_altitude": 50.0,
				"ambient_energy": 0.7,
				"fog_density": 0.012,
				"turbidity": 16.0,
				"rayleigh": 2.0,
			}
		"golden_hour":
			return {
				"sun_energy": 1.0,
				"sun_color": Color(1.0, 0.83, 0.63),
				"sun_altitude": 12.0,
				"ambient_energy": 0.4,
				"fog_density": 0.006,
				"turbidity": 6.0,
				"rayleigh": 3.0,
			}
		_:
			# No named preset — return defaults derived from wetness
			return {
				"sun_energy": lerpf(1.35, 1.02, wetness),
				"sun_altitude": lerpf(38.0, 24.0, wetness),
			}


func _try_load_hdri_sky(profile: Dictionary, env: Environment) -> bool:
	## Try to load an HDRI skybox based on lighting preset.
	## Returns true if an HDRI was loaded, false to fall back to PhysicalSkyMaterial.
	var preset_name: String = profile.get("lighting_preset", "").to_lower()

	# Map presets to HDRI filenames (downloaded CC0 from Poly Haven)
	var hdri_map := {
		"midday": [
			"res://assets/hdri/kloofendal_48d_partly_cloudy_puresky.hdr",
			"res://assets/hdri/kloofendal_48d_partly_cloudy_puresky.exr",
		],
		"overcast": [
			"res://assets/hdri/meadow_2.hdr",
			"res://assets/hdri/meadow_2.exr",
		],
		"golden_hour": [
			"res://assets/hdri/golden_bay.hdr",
			"res://assets/hdri/golden_bay.exr",
		],
	}

	if not hdri_map.has(preset_name):
		return false

	var candidates: Array = hdri_map[preset_name]
	var hdri_texture: Texture2D = null

	for path in candidates:
		if ResourceLoader.exists(path):
			hdri_texture = load(path) as Texture2D
			if hdri_texture:
				break

	if not hdri_texture:
		return false

	var panorama_mat := PanoramaSkyMaterial.new()
	panorama_mat.panorama = hdri_texture

	# Adjust exposure based on preset
	match preset_name:
		"midday":
			panorama_mat.energy_multiplier = 1.0
		"overcast":
			panorama_mat.energy_multiplier = 0.8
		"golden_hour":
			panorama_mat.energy_multiplier = 0.9

	if not env.sky:
		env.sky = Sky.new()
	env.sky.sky_material = panorama_mat

	# Rotate HDRI to match sun direction if we have a DirectionalLight3D
	if _sun_light:
		env.sky.process_mode = Sky.PROCESS_MODE_QUALITY

	print("[Main] Loaded HDRI sky for preset: %s" % preset_name)
	return true


func _create_water_normal_texture(size: int, seed_val: int) -> ImageTexture:
	## Generate procedural water normal map (scrolling ripple pattern).
	var img := Image.create(size, size, false, Image.FORMAT_RGBA8)
	var n := FastNoiseLite.new()
	n.noise_type = FastNoiseLite.TYPE_SIMPLEX
	n.frequency = 0.03
	n.fractal_octaves = 3
	n.seed = seed_val
	for y in range(size):
		for x in range(size):
			var dx := (n.get_noise_2d(x + 1, y) - n.get_noise_2d(x - 1, y)) * 2.0
			var dy := (n.get_noise_2d(x, y + 1) - n.get_noise_2d(x, y - 1)) * 2.0
			img.set_pixel(x, y, Color(
				clampf(0.5 + dx * 0.4, 0, 1),
				clampf(0.5 + dy * 0.4, 0, 1), 1.0))
	return ImageTexture.create_from_image(img)


func _color_from_hex(hex: String, fallback: Color) -> Color:
	if hex.is_empty():
		return fallback
	return Color.from_string(hex, fallback)


func _toggle_object_visibility() -> void:
	_objects_visible = not _objects_visible
	for child in _object_markers.get_children():
		child.visible = _objects_visible

	var hud := get_node_or_null("UI/HUD")
	if hud and hud.has_method("set_targets_visible"):
		hud.set_targets_visible(_objects_visible)

	if _objects_visible:
		print("[Main] Artifact overlay ON — showing %d buried objects" % _object_markers.get_child_count())
	else:
		print("[Main] Artifact overlay OFF")


func _create_object_markers(objects: Array, surface_elev: float) -> void:
	# Clear existing markers
	for child in _object_markers.get_children():
		child.queue_free()

	var terrain := get_node_or_null("Terrain")

	for obj in objects:
		var obj_pos: Array = obj.get("position", [0, 0, 0])
		var obj_name: String = obj.get("name", "unknown")
		var obj_radius: float = obj.get("radius", 0.05)
		var obj_metadata: Dictionary = obj.get("metadata", {})
		var obj_description: String = obj_metadata.get("description", "")

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

		# Container node for the whole artifact marker group
		var artifact_group := Node3D.new()
		artifact_group.name = "Artifact_" + obj_name
		artifact_group.visible = _objects_visible

		# Red pulsing disk on the surface marking the object's XZ position
		var surface_marker := _create_surface_marker(obj_name)
		surface_marker.position = Vector3(godot_pos.x, terrain_h + 0.05, godot_pos.z)
		artifact_group.add_child(surface_marker)

		# Dashed vertical line from surface down to buried object
		var depth_line := _create_depth_line(
			Vector3(godot_pos.x, terrain_h, godot_pos.z), godot_pos)
		artifact_group.add_child(depth_line)

		# Try loading a realistic 3D model for this artifact type
		var artifact_model := _load_artifact_model(obj_name, obj_description)
		if artifact_model:
			var model_scale: float = max(obj_radius, 0.05) * 4.0
			artifact_model.scale = Vector3.ONE * model_scale
			artifact_model.position = godot_pos
			_apply_artifact_material(artifact_model)
			artifact_group.add_child(artifact_model)
		else:
			# Fallback: shape-appropriate translucent mesh at burial depth
			var buried_mesh := _create_artifact_fallback(obj_radius, obj_name)
			buried_mesh.position = godot_pos
			artifact_group.add_child(buried_mesh)

		# Floating label above the surface showing object name and depth
		var depth_m := absf(obj_pos[2]) if obj_pos.size() > 2 else 0.0
		var label_text := "%s\n%.1fm deep" % [obj_name, depth_m]
		if obj_metadata.has("hazard"):
			label_text += "\n⚠ " + obj_metadata["hazard"]
		var label_3d := _create_artifact_label(label_text)
		label_3d.position = Vector3(godot_pos.x, terrain_h + 0.6, godot_pos.z)
		artifact_group.add_child(label_3d)

		_object_markers.add_child(artifact_group)


func _create_xray_material(color: Color, emission: Color, emission_strength: float = 0.5) -> ShaderMaterial:
	## Create X-ray overlay material using custom shader (avoids GPU crash from
	## StandardMaterial3D no_depth_test + alpha in Vulkan Forward+).
	var shader := load("res://shaders/xray_overlay.gdshader") as Shader
	if not shader:
		push_warning("[Main] xray_overlay.gdshader not found, using fallback material")
		return null
	var mat := ShaderMaterial.new()
	mat.shader = shader
	mat.set_shader_parameter("overlay_color", color)
	mat.set_shader_parameter("emission_color", Color(emission.r, emission.g, emission.b, 1.0))
	mat.set_shader_parameter("emission_strength", emission_strength)
	mat.render_priority = 127
	return mat


func _create_surface_marker(obj_name: String) -> MeshInstance3D:
	var mesh_inst := MeshInstance3D.new()
	mesh_inst.name = "Marker_" + obj_name

	var cyl := CylinderMesh.new()
	cyl.top_radius = 0.2
	cyl.bottom_radius = 0.2
	cyl.height = 0.02
	cyl.radial_segments = 24
	mesh_inst.mesh = cyl

	var xray_mat := _create_xray_material(
		Color(1.0, 0.15, 0.1, 0.7),
		Color(1.0, 0.2, 0.1),
		0.6)
	if xray_mat:
		mesh_inst.material_override = xray_mat
	else:
		var mat := StandardMaterial3D.new()
		mat.albedo_color = Color(1.0, 0.15, 0.1, 0.7)
		mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
		mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
		mesh_inst.material_override = mat

	return mesh_inst


func _create_depth_line(surface_pos: Vector3, buried_pos: Vector3) -> MeshInstance3D:
	## Thin vertical line from surface down to buried object.
	var mesh_inst := MeshInstance3D.new()
	var depth := surface_pos.y - buried_pos.y
	if depth < 0.01:
		depth = 0.5

	var cyl := CylinderMesh.new()
	cyl.top_radius = 0.01
	cyl.bottom_radius = 0.01
	cyl.height = depth
	cyl.radial_segments = 4
	mesh_inst.mesh = cyl

	var xray_mat := _create_xray_material(
		Color(1.0, 0.3, 0.1, 0.5),
		Color(1.0, 0.2, 0.1),
		0.4)
	if xray_mat:
		mesh_inst.material_override = xray_mat
	else:
		var mat := StandardMaterial3D.new()
		mat.albedo_color = Color(1.0, 0.3, 0.1, 0.5)
		mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
		mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
		mesh_inst.material_override = mat

	# Center the cylinder between surface and buried position
	mesh_inst.position = Vector3(
		surface_pos.x,
		buried_pos.y + depth / 2.0,
		surface_pos.z)
	return mesh_inst


func _create_artifact_label(text: String) -> Node3D:
	## Billboard label showing artifact name and depth.
	var label := Label3D.new()
	label.text = text
	label.font_size = 48
	label.pixel_size = 0.003
	label.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	label.shaded = false
	label.modulate = Color(1.0, 0.9, 0.7, 0.9)
	label.outline_modulate = Color(0, 0, 0, 0.8)
	label.outline_size = 6
	label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	label.vertical_alignment = VERTICAL_ALIGNMENT_BOTTOM
	return label


## Map object name keywords to artifact model directory paths.
## Tries .glb first (Kenney CC0 models), then .fbx fallback.
func _get_artifact_model_path(obj_name: String, obj_description: String) -> String:
	var name_lower := obj_name.to_lower()
	var desc_lower := obj_description.to_lower()
	var combined := name_lower + " " + desc_lower

	# UXO / ordnance — no model yet, will use procedural fallback
	if "bomb" in combined or "uxb" in combined or "uxo" in combined:
		return "res://assets/models/artifacts/uxo/bomb"
	if "ordnance" in combined or "submunition" in combined or "shell" in combined:
		return "res://assets/models/artifacts/uxo/shell"
	if "ammunition" in combined or "ammo" in combined:
		return "res://assets/models/artifacts/weapons/ammo_box"
	# Aircraft parts
	if "engine" in combined:
		return "res://assets/models/artifacts/aircraft/engine"
	if "propeller" in combined or "prop" in combined:
		return "res://assets/models/artifacts/aircraft/propeller"
	if "wing" in combined or "spar" in combined:
		return "res://assets/models/artifacts/aircraft/wing_section"
	if "landing gear" in combined or "strut" in combined:
		return "res://assets/models/artifacts/aircraft/landing_gear"
	if "instrument panel" in combined or "panel" in combined:
		return "res://assets/models/artifacts/aircraft/panel"
	# Weapons
	if "machine gun" in combined or "caliber" in combined or "browning" in combined:
		return "res://assets/models/artifacts/weapons/machine_gun"
	if "weapon" in combined or "gun" in combined:
		return "res://assets/models/artifacts/weapons/weapon"
	# Forensic
	if "skeleton" in combined or "remains" in combined or "body" in combined:
		return "res://assets/models/artifacts/forensic/skeleton"
	if "coffin" in combined or "casket" in combined or "burial" in combined:
		return "res://assets/models/artifacts/forensic/coffin"
	if "belt" in combined or "buckle" in combined or "clothing" in combined:
		return "res://assets/models/artifacts/forensic/skeleton"
	if "shoe" in combined or "boot" in combined or "zipper" in combined:
		return "res://assets/models/artifacts/forensic/skeleton"
	if "suit" in combined or "flight" in combined or "copper" in combined:
		return "res://assets/models/artifacts/forensic/skeleton"
	# Debris
	if "pipe" in combined or "rebar" in combined:
		return "res://assets/models/artifacts/debris/barrel"
	if "fragment" in combined or "debris" in combined:
		return "res://assets/models/artifacts/debris/fragments"
	if "cable" in combined or "wire" in combined:
		return "res://assets/models/artifacts/debris/fragments"
	if "shovel" in combined or "tool" in combined:
		return "res://assets/models/artifacts/debris/shovel"
	# Fallback
	return "res://assets/models/artifacts/general/generic"


func _load_artifact_model(obj_name: String, obj_description: String) -> Node3D:
	## Try to load a realistic 3D model for this artifact type.
	var model_path := _get_artifact_model_path(obj_name, obj_description)
	for ext in [".glb", ".fbx"]:
		var path: String = model_path.get_basename() + ext
		if ResourceLoader.exists(path):
			var scene := load(path) as PackedScene
			if scene:
				return scene.instantiate()
	return null


func _apply_artifact_material(node: Node3D) -> void:
	## Apply X-ray style material — renders through terrain so buried objects
	## are visible as glowing translucent shapes when highlight is toggled.
	var xray_mat := _create_xray_material(
		Color(0.9, 0.6, 0.2, 0.45),
		Color(0.8, 0.4, 0.1),
		0.5)
	var fallback_mat: StandardMaterial3D = null
	if not xray_mat:
		fallback_mat = StandardMaterial3D.new()
		fallback_mat.albedo_color = Color(0.9, 0.6, 0.2, 0.45)
		fallback_mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
		fallback_mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
		fallback_mat.render_priority = 127
	var mat: Material = xray_mat if xray_mat else fallback_mat
	var nodes := [node]
	while not nodes.is_empty():
		var n: Node = nodes.pop_back()
		if n is MeshInstance3D:
			(n as MeshInstance3D).material_override = mat
		for child in n.get_children():
			nodes.append(child)


func _create_artifact_fallback(radius: float, obj_name: String) -> MeshInstance3D:
	## Create a shape-appropriate fallback mesh based on the object name.
	var mesh_inst := MeshInstance3D.new()
	mesh_inst.name = "Object_" + obj_name
	var name_lower := obj_name.to_lower()

	# Choose mesh shape based on object type
	if "bomb" in name_lower or "uxb" in name_lower or "ordnance" in name_lower:
		# Elongated capsule for bombs/ordnance
		var capsule := CapsuleMesh.new()
		capsule.radius = max(radius, 0.05) * 2.0
		capsule.height = max(radius, 0.05) * 8.0
		mesh_inst.mesh = capsule
		mesh_inst.rotate_z(deg_to_rad(15.0))
	elif "engine" in name_lower:
		# Large box for engine blocks
		var box := BoxMesh.new()
		var s: float = max(radius, 0.1) * 3.0
		box.size = Vector3(s * 1.5, s, s * 1.2)
		mesh_inst.mesh = box
	elif "pipe" in name_lower or "cable" in name_lower or "rebar" in name_lower:
		# Long thin cylinder for pipes/cables
		var cyl := CylinderMesh.new()
		cyl.top_radius = max(radius, 0.02) * 1.5
		cyl.bottom_radius = max(radius, 0.02) * 1.5
		cyl.height = max(radius, 0.02) * 12.0
		mesh_inst.mesh = cyl
		mesh_inst.rotate_z(deg_to_rad(80.0))
	elif "coffin" in name_lower or "burial" in name_lower:
		# Elongated box for coffin
		var box := BoxMesh.new()
		box.size = Vector3(0.6, 0.3, 1.8)
		mesh_inst.mesh = box
	elif "skeleton" in name_lower or "remains" in name_lower:
		# Elongated capsule for remains
		var capsule := CapsuleMesh.new()
		capsule.radius = 0.2
		capsule.height = 1.7
		mesh_inst.mesh = capsule
		mesh_inst.rotate_z(deg_to_rad(90.0))
	elif "propeller" in name_lower:
		# Flat cylinder for propeller
		var cyl := CylinderMesh.new()
		cyl.top_radius = max(radius, 0.05) * 4.0
		cyl.bottom_radius = max(radius, 0.05) * 4.0
		cyl.height = max(radius, 0.05) * 0.5
		mesh_inst.mesh = cyl
	elif "wing" in name_lower or "panel" in name_lower:
		# Flat wide box for wing/panel sections
		var box := BoxMesh.new()
		var s: float = max(radius, 0.05) * 3.0
		box.size = Vector3(s * 4.0, s * 0.2, s * 2.0)
		mesh_inst.mesh = box
	else:
		# Default sphere for unrecognized objects
		var sphere := SphereMesh.new()
		sphere.radius = max(radius, 0.05) * 3.0
		sphere.height = sphere.radius * 2.0
		mesh_inst.mesh = sphere

	var xray_mat := _create_xray_material(
		Color(0.9, 0.5, 0.15, 0.4),
		Color(0.8, 0.4, 0.1),
		0.5)
	if xray_mat:
		mesh_inst.material_override = xray_mat
	else:
		var mat := StandardMaterial3D.new()
		mat.albedo_color = Color(0.9, 0.5, 0.15, 0.4)
		mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
		mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
		mat.render_priority = 127
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
