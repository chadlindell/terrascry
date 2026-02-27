## First-person operator controller for survey simulation.
##
## WASD movement on the terrain surface with gravity. The operator carries
## the geophysical sensor rig and queries the physics server at each
## position for readings. Tracks heading, records walk path, computes
## cross-track error, and integrates with DataRecorder and SurveyManager.
extends CharacterBody3D

## Movement parameters
@export var walk_speed := 1.5  # m/s (typical survey speed)
@export var mouse_sensitivity := 0.002

## Sensor query rate (Hz) — matches Pathfinder's 10 Hz sample rate
@export var sample_rate := 10.0

## Sensor rig offset from body (Pathfinder boom: 0.5m forward, bottom at 0.175m)
@export var sensor_forward_offset := 0.5
@export var sensor_height := 0.175

## Eye height above ground (camera position)
@export var eye_height := 1.7

## Gravity acceleration
@export var gravity := 9.8

## Acceleration ramp (simulates carrying heavy survey equipment)
@export var accel_time := 0.3  # seconds to reach walk_speed

## Head bob parameters (gait noise)
@export var head_bob_enabled := true
@export var head_bob_frequency := 2.0  # Hz (step frequency)
@export var head_bob_amplitude := 0.02  # meters
@export var head_bob_lateral := 0.008  # meters (lateral sway amplitude)

## Current readings
var current_gradient := 0.0
var peak_gradient := 0.0

## Heading in GeoSim convention: 0 = North (+Y), PI/2 = East (+X), radians
var heading := 0.0

## Speed tracking
var current_speed := 0.0

## Survey recording
var recorded_positions: PackedVector3Array = []
var recorded_gradients: PackedFloat64Array = []
var recorded_timestamps: PackedFloat64Array = []
var recorded_headings: PackedFloat64Array = []
var distance_traveled := 0.0
var survey_start_time := 0.0

## Terrain bounds (set by Main after scenario load)
var terrain_x_extent := Vector2(0.0, 20.0)
var terrain_y_extent := Vector2(0.0, 20.0)
var surface_elevation := 0.0

## Line guidance
var _current_line_index := 0
var _survey_lines: Array = []
var _target_speed := 1.5
var _dropout_count := 0

## Sensor latency buffer (simulates fluxgate time constant)
var _reading_buffer: Array = []
var _buffer_delay := 2  # samples of delay

## Dust particle emitter
var _dust_particles: GPUParticles3D = null

## Instrument rig (for sway animation)
var _instrument_rig: Node3D = null
var _instrument_sway_timer := 0.0
var _instrument_visible := true

## Display shader and LED references
var _display_node: MeshInstance3D = null
var _display_shader_mat: ShaderMaterial = null
var _led_recording: MeshInstance3D = null
var _display_readings := [0.0, 0.0, 0.0, 0.0]

## Internal state
var _query_timer := 0.0
var _query_interval := 0.1
var _last_position := Vector3.ZERO
var _mouse_captured := true
var _head_bob_timer := 0.0
var _head_bob_prev_sign := 1.0  # For footstep zero-crossing detection

## Emitted when a new gradient reading arrives. HUD listens to this.
signal gradient_reading(value: float, world_pos: Vector3)

## Emitted when the operator places an event marker.
signal event_marked(world_pos: Vector3)

## Emitted when the current survey line changes (for line color updates).
signal line_changed(new_index: int)

## Emitted when a survey line is completed with QC evaluation.
signal line_completed(line_index: int, qc_passed: bool, stats: Dictionary)


func _ready() -> void:
	_query_interval = 1.0 / sample_rate
	survey_start_time = Time.get_ticks_msec() / 1000.0
	Input.mouse_mode = Input.MOUSE_MODE_CAPTURED
	_last_position = global_position

	# Setup dust puff particles
	_dust_particles = _create_dust_particles()
	add_child(_dust_particles)

	# Build visible survey instrument
	_build_pathfinder_model()

	# Load survey lines from plan
	if not SurveyManager.survey_plan.is_empty():
		_survey_lines = SurveyManager.survey_plan.get("lines", [])
		_target_speed = SurveyManager.survey_plan.get("speed_target", 1.5)
		walk_speed = _target_speed


func _unhandled_input(event: InputEvent) -> void:
	# Sync local flag with actual mouse mode (SurveyManager controls capture on pause/resume)
	_mouse_captured = (Input.mouse_mode == Input.MOUSE_MODE_CAPTURED)

	# Mouse look (only when captured)
	if event is InputEventMouseMotion and _mouse_captured:
		rotate_y(-event.relative.x * mouse_sensitivity)
		$Camera3D.rotate_x(-event.relative.y * mouse_sensitivity)
		$Camera3D.rotation.x = clamp($Camera3D.rotation.x, -PI / 4, PI / 4)

	# Click recaptures mouse (after resume from pause)
	if event is InputEventMouseButton and event.pressed and not _mouse_captured:
		_mouse_captured = true
		Input.mouse_mode = Input.MOUSE_MODE_CAPTURED

	# Event marker
	if event.is_action_pressed("mark_event"):
		if DataRecorder.is_recording:
			DataRecorder.mark_event("mark", global_position)
			event_marked.emit(global_position)
			AudioManager.play_beep()

	# VCO toggle (Tab key or toggle_audio action)
	if event.is_action_pressed("toggle_audio"):
		AudioManager.set_vco_enabled(not AudioManager.vco_enabled)

	# Instrument visibility toggle
	if event.is_action_pressed("toggle_instrument"):
		_instrument_visible = not _instrument_visible
		if _instrument_rig:
			_instrument_rig.visible = _instrument_visible


func _physics_process(delta: float) -> void:
	# Gravity — fall until on ground
	if not is_on_floor():
		velocity.y -= gravity * delta
	else:
		velocity.y = 0.0

	# WASD movement
	var input_dir := Vector2.ZERO
	if Input.is_action_pressed("move_forward"):
		input_dir.y -= 1
	if Input.is_action_pressed("move_back"):
		input_dir.y += 1
	if Input.is_action_pressed("move_left"):
		input_dir.x -= 1
	if Input.is_action_pressed("move_right"):
		input_dir.x += 1

	input_dir = input_dir.normalized()
	var direction := (transform.basis * Vector3(input_dir.x, 0, input_dir.y)).normalized()

	# Acceleration ramp — simulates carrying heavy survey equipment
	var accel_rate := walk_speed / accel_time if accel_time > 0 else walk_speed * 100.0
	if direction:
		velocity.x = move_toward(velocity.x, direction.x * walk_speed, accel_rate * delta)
		velocity.z = move_toward(velocity.z, direction.z * walk_speed, accel_rate * delta)
	else:
		# Frame-rate-independent deceleration (faster than accel)
		velocity.x = move_toward(velocity.x, 0, accel_rate * 2.0 * delta)
		velocity.z = move_toward(velocity.z, 0, accel_rate * 2.0 * delta)

	move_and_slide()

	# Clamp to terrain boundaries
	position.x = clamp(position.x, terrain_x_extent.x, terrain_x_extent.y)
	position.z = clamp(position.z, terrain_y_extent.x, terrain_y_extent.y)

	# Track distance and speed
	var moved := global_position.distance_to(_last_position)
	current_speed = moved / delta if delta > 0 else 0.0
	if moved > 0.001:
		distance_traveled += moved
	_last_position = global_position

	# Head bob (gait noise simulation)
	if head_bob_enabled and current_speed > 0.3 and is_on_floor():
		_head_bob_timer += delta * head_bob_frequency * TAU
		var speed_factor := clampf(current_speed / walk_speed, 0.0, 1.0)
		var bob_value := sin(_head_bob_timer)
		var bob_offset := bob_value * head_bob_amplitude * speed_factor
		# Lateral sway in antiphase to vertical bob (half frequency)
		var lateral_sway := sin(_head_bob_timer * 0.5) * head_bob_lateral * speed_factor
		$Camera3D.position.y = eye_height + bob_offset
		$Camera3D.position.x = lateral_sway

		# Footstep on downward zero-crossing (foot strike)
		var current_sign := signf(bob_value)
		if current_sign < 0.0 and _head_bob_prev_sign >= 0.0:
			AudioManager.play_footstep()
			# Emit dust puff on footstep
			if _dust_particles:
				_dust_particles.restart()
				_dust_particles.emitting = true
		_head_bob_prev_sign = current_sign
	else:
		$Camera3D.position.y = eye_height
		$Camera3D.position.x = 0.0
		_head_bob_timer = 0.0

	# Instrument sway (loose grip simulation)
	if _instrument_rig:
		_instrument_sway_timer += delta
		var sway_speed_factor := clampf(current_speed / maxf(walk_speed, 0.1), 0.0, 1.0)
		# Low-frequency sway: ±2° at rest, increases with speed
		var sway_deg := lerpf(0.5, 2.0, sway_speed_factor)
		var sway_x := sin(_instrument_sway_timer * 1.7) * deg_to_rad(sway_deg)
		var sway_z := sin(_instrument_sway_timer * 1.3 + 0.8) * deg_to_rad(sway_deg * 0.7)
		_instrument_rig.rotation.x = sway_x
		_instrument_rig.rotation.z = sway_z

	# FOV shift — subtle "rushing" feel at walk speed
	var cam := $Camera3D as Camera3D
	if cam:
		var speed_ratio := clampf(current_speed / maxf(walk_speed, 0.1), 0.0, 1.0)
		cam.fov = lerpf(cam.fov, lerpf(70.0, 74.0, speed_ratio), delta * 4.0)

	# Compute heading: Godot rotation.y -> GeoSim compass bearing
	heading = fposmod(rotation.y - PI, TAU)

	# Query physics at sample rate
	_query_timer += delta
	if _query_timer >= _query_interval:
		_query_timer -= _query_interval
		_query_physics()

	# Update guidance if we have survey lines
	if not _survey_lines.is_empty():
		_update_guidance()


func _get_sensor_position() -> Vector3:
	## Get the world-space sensor position (offset from body).
	var forward := -transform.basis.z.normalized()
	var sensor_world := global_position + forward * sensor_forward_offset
	return sensor_world


func _query_physics() -> void:
	if PhysicsClient.is_busy():
		_dropout_count += 1
		return

	var sensor_world := _get_sensor_position()

	# Convert to GeoSim coordinates
	var gs := CoordUtil.to_geosim(sensor_world)
	var sensor_pos := [gs.x, gs.y, surface_elevation + sensor_height]

	var start_ms := Time.get_ticks_msec()
	var result: Dictionary

	# Query based on current instrument
	match SurveyManager.current_instrument:
		SurveyManager.Instrument.MAG_GRADIOMETER:
			result = await PhysicsClient.query_gradient([sensor_pos])
		SurveyManager.Instrument.EM_FDEM:
			result = await PhysicsClient.query_em_response([sensor_pos])
		SurveyManager.Instrument.RESISTIVITY:
			result = await PhysicsClient.query_apparent_resistivity([sensor_pos], [[0, 1, 2, 3]])
		_:
			result = await PhysicsClient.query_gradient([sensor_pos])

	var latency_ms := Time.get_ticks_msec() - start_ms

	if result.get("status") == "ok":
		var reading := 0.0
		var raw_data: Dictionary = result.get("data", {})

		match SurveyManager.current_instrument:
			SurveyManager.Instrument.MAG_GRADIOMETER:
				var grad_data: Array = raw_data.get("gradient", [0.0])
				if grad_data.size() > 0:
					reading = grad_data[0]
			SurveyManager.Instrument.EM_FDEM:
				var resp_real: Array = raw_data.get("response_real", [0.0])
				if resp_real.size() > 0:
					reading = resp_real[0]
			SurveyManager.Instrument.RESISTIVITY:
				var rho: Array = raw_data.get("apparent_resistivity", [0.0])
				if rho.size() > 0:
					reading = rho[0]

		# Sensor latency buffer
		_reading_buffer.append(reading)
		if _reading_buffer.size() > _buffer_delay:
			current_gradient = _reading_buffer[0]
			_reading_buffer.remove_at(0)
		else:
			current_gradient = reading

		if abs(current_gradient) > abs(peak_gradient):
			peak_gradient = current_gradient

		# Record in legacy arrays
		var now := Time.get_ticks_msec() / 1000.0
		recorded_positions.append(global_position)
		recorded_gradients.append(current_gradient)
		recorded_timestamps.append(now - survey_start_time)
		recorded_headings.append(heading)

		# Record in DataRecorder
		if DataRecorder.is_recording:
			var xte := _compute_xte()
			var quality := _compute_quality(xte)
			var inst_name: String = SurveyManager.Instrument.keys()[SurveyManager.current_instrument].to_lower()

			DataRecorder.record_sample({
				"x_e": gs.x,
				"y_n": gs.y,
				"z_up": surface_elevation + sensor_height,
				"heading": heading,
				"speed": current_speed,
				"line_id": _current_line_index if not _survey_lines.is_empty() else -1,
				"instrument": inst_name,
				"reading": current_gradient,
				"reading_raw": raw_data,
				"mock": PhysicsClient.is_mock_mode,
				"xte": xte,
				"quality": quality,
				"latency_ms": latency_ms,
			})

		# Update QC dashboard
		var hud := _get_hud()
		if hud and hud.has_method("update_qc"):
			hud.update_qc(latency_ms, _dropout_count)
	else:
		_dropout_count += 1

	# Update instrument display shader with per-sensor readings
	if _display_shader_mat:
		# Distribute reading across 4 sensor pairs with slight variation
		var base_nT := current_gradient * 1e9
		_display_readings[0] = clampf(abs(base_nT * 0.8) / 100.0, 0.0, 1.0)
		_display_readings[1] = clampf(abs(base_nT * 1.1) / 100.0, 0.0, 1.0)
		_display_readings[2] = clampf(abs(base_nT * 0.95) / 100.0, 0.0, 1.0)
		_display_readings[3] = clampf(abs(base_nT * 1.05) / 100.0, 0.0, 1.0)
		_display_shader_mat.set_shader_parameter("reading0", _display_readings[0])
		_display_shader_mat.set_shader_parameter("reading1", _display_readings[1])
		_display_shader_mat.set_shader_parameter("reading2", _display_readings[2])
		_display_shader_mat.set_shader_parameter("reading3", _display_readings[3])

	# Blink recording LED
	if _led_recording:
		_led_recording.visible = DataRecorder.is_recording and (int(Time.get_ticks_msec() / 500) % 2 == 0)

	# Feed reading to VCO sonification
	AudioManager.update_vco_reading(current_gradient)

	# Notify HUD and other listeners
	gradient_reading.emit(current_gradient, global_position)


func _compute_xte() -> float:
	## Compute cross-track error (perpendicular distance to current survey line).
	if _survey_lines.is_empty() or _current_line_index >= _survey_lines.size():
		return -1.0

	var line: PackedVector3Array = _survey_lines[_current_line_index]
	if line.size() < 2:
		return -1.0

	# Find closest segment
	var pos_2d := Vector2(global_position.x, global_position.z)
	var min_dist := INF

	for i in range(line.size() - 1):
		# Lines are in GeoSim coords, convert to Godot for comparison
		var p0 := CoordUtil.to_godot(line[i])
		var p1 := CoordUtil.to_godot(line[i + 1])
		var a := Vector2(p0.x, p0.z)
		var b := Vector2(p1.x, p1.z)

		var dist := _point_to_segment_distance(pos_2d, a, b)
		min_dist = min(min_dist, dist)

	return min_dist


func _point_to_segment_distance(p: Vector2, a: Vector2, b: Vector2) -> float:
	var ab := b - a
	var len_sq := ab.length_squared()
	if len_sq < 1e-10:
		return p.distance_to(a)
	var ap := p - a
	var t := clampf(ap.dot(ab) / len_sq, 0.0, 1.0)
	var closest := a + t * ab
	return p.distance_to(closest)


func _compute_quality(xte: float) -> int:
	## Quality codes: 0=dropout, 1=good, 2=too fast, 3=off-line
	if current_speed > _target_speed * 1.5:
		return 2  # Too fast
	if xte > 0.5:
		return 3  # Off-line
	return 1  # Good


func _update_guidance() -> void:
	## Update line guidance info and send to HUD.
	if _survey_lines.is_empty():
		return

	var xte := _compute_xte()
	var heading_error := _compute_heading_error()

	var hud := _get_hud()
	if hud and hud.has_method("update_guidance"):
		hud.update_guidance(
			_current_line_index,
			_survey_lines.size(),
			xte,
			current_speed,
			_target_speed,
			heading_error,
		)

	# Auto-advance to next line when near end of current line
	if _current_line_index < _survey_lines.size():
		var line: PackedVector3Array = _survey_lines[_current_line_index]
		if line.size() >= 2:
			var end_pt := CoordUtil.to_godot(line[-1])
			var dist_to_end := Vector2(global_position.x, global_position.z).distance_to(
				Vector2(end_pt.x, end_pt.z))
			if dist_to_end < 1.0 and _current_line_index < _survey_lines.size() - 1:
				# Evaluate QC for the completed line
				var completed_index := _current_line_index
				var qc_result := DataRecorder.evaluate_line_qc(completed_index)
				var qc_stats: Dictionary = qc_result.get("stats", {})
				qc_stats["reason"] = qc_result.get("reason", "")
				line_completed.emit(
					completed_index,
					qc_result.get("passed", false),
					qc_stats,
				)

				_current_line_index += 1
				line_changed.emit(_current_line_index)


func _compute_heading_error() -> float:
	## Compute angle between current heading and survey line direction.
	if _survey_lines.is_empty() or _current_line_index >= _survey_lines.size():
		return 0.0

	var line: PackedVector3Array = _survey_lines[_current_line_index]
	if line.size() < 2:
		return 0.0

	var p0 := line[0]
	var p1 := line[-1]
	var line_heading := atan2(p1.x - p0.x, p1.y - p0.y)  # GeoSim: atan2(east, north)
	var diff := fposmod(heading - line_heading + PI, TAU) - PI
	return diff


func _get_hud() -> Control:
	var main := get_parent()
	if main:
		return main.get_node_or_null("UI/HUD")
	return null


func _create_dust_particles() -> GPUParticles3D:
	## Create dust puff particle emitter for footstep ground contact.
	var particles := GPUParticles3D.new()
	particles.name = "DustPuff"
	particles.emitting = false
	particles.one_shot = true
	particles.amount = 4
	particles.lifetime = 0.5
	particles.explosiveness = 1.0

	var mat := ParticleProcessMaterial.new()
	mat.direction = Vector3(0, 1, 0)
	mat.spread = 15.0
	mat.initial_velocity_min = 0.3
	mat.initial_velocity_max = 0.6
	mat.gravity = Vector3(0, -2.0, 0)
	mat.scale_min = 0.02
	mat.scale_max = 0.04
	mat.color = Color(0.5, 0.45, 0.35, 0.6)

	particles.process_material = mat

	# Simple quad mesh for particle
	var mesh := QuadMesh.new()
	mesh.size = Vector2(0.03, 0.03)
	particles.draw_pass_1 = mesh

	# Position at feet
	particles.position = Vector3(0, 0.05, 0)

	return particles


func _build_pathfinder_model() -> void:
	## Procedurally build an accurate Pathfinder gradiometer model.
	## Real instrument: 2m vertical boom, 1.5m horizontal cross-arm with
	## 4 sensor pairs (8 pods), 0.50m horizontal spacing, 0.35m vertical
	## baseline per pair. Bottom sensors at 0.175m from ground.
	_instrument_rig = Node3D.new()
	_instrument_rig.name = "InstrumentRig"
	# Position: slightly right of center (right-handed carry), forward
	_instrument_rig.position = Vector3(0.15, 0, -0.3)
	add_child(_instrument_rig)

	# --- Realistic PBR materials ---
	var carbon_mat := StandardMaterial3D.new()
	carbon_mat.albedo_color = Color(0.08, 0.08, 0.10)
	carbon_mat.roughness = 0.25
	carbon_mat.metallic = 0.05

	var grip_mat := StandardMaterial3D.new()
	grip_mat.albedo_color = Color(0.15, 0.14, 0.13)
	grip_mat.roughness = 0.85
	grip_mat.metallic = 0.0

	var sensor_mat := StandardMaterial3D.new()
	sensor_mat.albedo_color = Color(0.9, 0.55, 0.1)
	sensor_mat.roughness = 0.3
	sensor_mat.metallic = 0.15

	var sensor_dark_mat := StandardMaterial3D.new()
	sensor_dark_mat.albedo_color = Color(0.1, 0.1, 0.12)
	sensor_dark_mat.roughness = 0.4
	sensor_dark_mat.metallic = 0.05

	var cable_mat := StandardMaterial3D.new()
	cable_mat.albedo_color = Color(0.06, 0.06, 0.08)
	cable_mat.roughness = 0.6
	cable_mat.metallic = 0.0

	var bracket_mat := StandardMaterial3D.new()
	bracket_mat.albedo_color = Color(0.12, 0.12, 0.14)
	bracket_mat.roughness = 0.35
	bracket_mat.metallic = 0.3

	# 1. Main vertical boom (2.0m tall, 25mm diameter)
	var boom := MeshInstance3D.new()
	var boom_mesh := CylinderMesh.new()
	boom_mesh.top_radius = 0.0125
	boom_mesh.bottom_radius = 0.0125
	boom_mesh.height = 2.0
	boom.mesh = boom_mesh
	boom.position = Vector3(0, 1.0, 0)
	boom.material_override = carbon_mat
	_instrument_rig.add_child(boom)

	# 2. Grip section: thicker at ~1.0m height (waist)
	var grip := MeshInstance3D.new()
	var grip_mesh := CylinderMesh.new()
	grip_mesh.top_radius = 0.018
	grip_mesh.bottom_radius = 0.018
	grip_mesh.height = 0.20
	grip.mesh = grip_mesh
	grip.position = Vector3(0, 1.0, 0)
	grip.material_override = grip_mat
	_instrument_rig.add_child(grip)

	# Carry strap ring (torus at grip)
	var strap_ring := MeshInstance3D.new()
	var torus_mesh := TorusMesh.new()
	torus_mesh.inner_radius = 0.012
	torus_mesh.outer_radius = 0.022
	strap_ring.mesh = torus_mesh
	strap_ring.position = Vector3(0, 1.12, 0)
	strap_ring.material_override = bracket_mat
	_instrument_rig.add_child(strap_ring)

	# 3. Cross-arm at top of boom (1.5m wide, 20mm diameter)
	var cross_arm := MeshInstance3D.new()
	var cross_mesh := CylinderMesh.new()
	cross_mesh.top_radius = 0.010
	cross_mesh.bottom_radius = 0.010
	cross_mesh.height = 1.5
	cross_arm.mesh = cross_mesh
	cross_arm.rotation.z = deg_to_rad(90)
	cross_arm.position = Vector3(0, 1.9, 0)
	cross_arm.material_override = carbon_mat
	_instrument_rig.add_child(cross_arm)

	# 4. Sensor pods: 4 pairs at 0.50m horizontal spacing
	var sensor_x_positions := [-0.75, -0.25, 0.25, 0.75]
	var bottom_height := sensor_height  # 0.175m
	var top_height := bottom_height + 0.35  # 0.525m

	for x_pos in sensor_x_positions:
		# Bottom sensor housing (dark polymer)
		var bot_housing := MeshInstance3D.new()
		var bot_h_mesh := CylinderMesh.new()
		bot_h_mesh.top_radius = 0.025
		bot_h_mesh.bottom_radius = 0.025
		bot_h_mesh.height = 0.02
		bot_housing.mesh = bot_h_mesh
		bot_housing.position = Vector3(x_pos, bottom_height - 0.04, 0)
		bot_housing.material_override = sensor_dark_mat
		_instrument_rig.add_child(bot_housing)

		# Bottom sensor pod (orange)
		var bot_pod := MeshInstance3D.new()
		var bot_mesh := CylinderMesh.new()
		bot_mesh.top_radius = 0.022
		bot_mesh.bottom_radius = 0.022
		bot_mesh.height = 0.08
		bot_pod.mesh = bot_mesh
		bot_pod.position = Vector3(x_pos, bottom_height, 0)
		bot_pod.material_override = sensor_mat
		_instrument_rig.add_child(bot_pod)

		# Top sensor housing
		var top_housing := MeshInstance3D.new()
		var top_h_mesh := CylinderMesh.new()
		top_h_mesh.top_radius = 0.025
		top_h_mesh.bottom_radius = 0.025
		top_h_mesh.height = 0.02
		top_housing.mesh = top_h_mesh
		top_housing.position = Vector3(x_pos, top_height + 0.04, 0)
		top_housing.material_override = sensor_dark_mat
		_instrument_rig.add_child(top_housing)

		# Top sensor pod (orange)
		var top_pod := MeshInstance3D.new()
		var top_mesh := CylinderMesh.new()
		top_mesh.top_radius = 0.022
		top_mesh.bottom_radius = 0.022
		top_mesh.height = 0.08
		top_pod.mesh = top_mesh
		top_pod.position = Vector3(x_pos, top_height, 0)
		top_pod.material_override = sensor_mat
		_instrument_rig.add_child(top_pod)

		# Vertical stub connecting sensors to cross-arm
		var stub := MeshInstance3D.new()
		var stub_mesh := CylinderMesh.new()
		stub_mesh.top_radius = 0.008
		stub_mesh.bottom_radius = 0.008
		var stub_height := 1.9 - top_height
		stub_mesh.height = stub_height
		stub.mesh = stub_mesh
		stub.position = Vector3(x_pos, top_height + stub_height / 2.0, 0)
		stub.material_override = sensor_dark_mat
		_instrument_rig.add_child(stub)

		# Mounting brackets at cross-arm junction (top and bottom of connection)
		for bracket_y in [1.9 - 0.01, 1.9 + 0.01]:
			var bracket := MeshInstance3D.new()
			var bracket_mesh := BoxMesh.new()
			bracket_mesh.size = Vector3(0.015, 0.010, 0.005)
			bracket.mesh = bracket_mesh
			bracket.position = Vector3(x_pos, bracket_y, 0)
			bracket.material_override = bracket_mat
			_instrument_rig.add_child(bracket)

		# Cable running from display down the boom to sensor pod
		var cable := MeshInstance3D.new()
		var cable_mesh := CylinderMesh.new()
		cable_mesh.top_radius = 0.0025
		cable_mesh.bottom_radius = 0.0025
		var cable_h := 1.9 - bottom_height
		cable_mesh.height = cable_h
		cable.mesh = cable_mesh
		cable.position = Vector3(x_pos + 0.015, bottom_height + cable_h / 2.0, 0.01)
		cable.material_override = cable_mat
		_instrument_rig.add_child(cable)

	# Main cable bundle from display box down the boom
	var main_cable := MeshInstance3D.new()
	var main_cable_mesh := CylinderMesh.new()
	main_cable_mesh.top_radius = 0.004
	main_cable_mesh.bottom_radius = 0.004
	main_cable_mesh.height = 0.6
	main_cable.mesh = main_cable_mesh
	main_cable.position = Vector3(0.015, 1.6, 0.02)
	main_cable.material_override = cable_mat
	_instrument_rig.add_child(main_cable)

	# 5. Display box at chest height (facing operator)
	_display_node = MeshInstance3D.new()
	_display_node.name = "Display"
	var display_mesh := BoxMesh.new()
	display_mesh.size = Vector3(0.15, 0.10, 0.03)
	_display_node.mesh = display_mesh
	_display_node.position = Vector3(0, 1.3, 0.05)
	_display_node.rotation.x = deg_to_rad(-25)

	_display_shader_mat = ShaderMaterial.new()
	var display_shader := Shader.new()
	display_shader.code = """
shader_type spatial;
render_mode unshaded;
uniform float reading0 = 0.0;
uniform float reading1 = 0.0;
uniform float reading2 = 0.0;
uniform float reading3 = 0.0;
void fragment() {
	vec2 uv = UV;
	vec3 bg = vec3(0.08, 0.18, 0.08);
	vec3 bar_color = vec3(0.7, 1.0, 0.4);
	vec3 col = bg;
	float margin = 0.08;
	float bar_w = (1.0 - margin * 5.0) / 4.0;
	float readings[4] = float[](reading0, reading1, reading2, reading3);
	for (int i = 0; i < 4; i++) {
		float x0 = margin + float(i) * (bar_w + margin);
		float x1 = x0 + bar_w;
		float h = clamp(abs(readings[i]), 0.02, 0.9);
		if (uv.x > x0 && uv.x < x1 && (1.0 - uv.y) < h + 0.05 && (1.0 - uv.y) > 0.05) {
			col = bar_color * (0.6 + 0.4 * h);
		}
	}
	ALBEDO = col;
	EMISSION = col * 1.5;
}
"""
	_display_shader_mat.shader = display_shader
	_display_node.material_override = _display_shader_mat
	_instrument_rig.add_child(_display_node)

	# Connector port on side of display box
	var connector := MeshInstance3D.new()
	var connector_mesh := BoxMesh.new()
	connector_mesh.size = Vector3(0.010, 0.008, 0.005)
	connector.mesh = connector_mesh
	connector.position = Vector3(0.08, 1.3, 0.05)
	connector.material_override = sensor_dark_mat
	_instrument_rig.add_child(connector)

	# LED indicators on display box
	var led_green := MeshInstance3D.new()
	var led_mesh := SphereMesh.new()
	led_mesh.radius = 0.003
	led_mesh.height = 0.006
	led_green.mesh = led_mesh
	led_green.position = Vector3(-0.06, 1.36, 0.065)
	var led_green_mat := StandardMaterial3D.new()
	led_green_mat.albedo_color = Color(0.1, 0.9, 0.1)
	led_green_mat.emission_enabled = true
	led_green_mat.emission = Color(0.2, 1.0, 0.2)
	led_green_mat.emission_energy_multiplier = 2.0
	led_green.material_override = led_green_mat
	_instrument_rig.add_child(led_green)

	var led_amber := MeshInstance3D.new()
	var led_amber_mesh := SphereMesh.new()
	led_amber_mesh.radius = 0.003
	led_amber_mesh.height = 0.006
	led_amber.mesh = led_amber_mesh
	led_amber.position = Vector3(-0.05, 1.36, 0.065)
	var led_amber_mat := StandardMaterial3D.new()
	led_amber_mat.albedo_color = Color(0.9, 0.6, 0.05)
	led_amber_mat.emission_enabled = true
	led_amber_mat.emission = Color(1.0, 0.7, 0.1)
	led_amber_mat.emission_energy_multiplier = 1.5
	_led_recording = led_amber
	led_amber.material_override = led_amber_mat
	_instrument_rig.add_child(led_amber)


## Reset all survey recording state.
func reset_survey() -> void:
	recorded_positions.clear()
	recorded_gradients.clear()
	recorded_timestamps.clear()
	recorded_headings.clear()
	distance_traveled = 0.0
	peak_gradient = 0.0
	survey_start_time = Time.get_ticks_msec() / 1000.0
	_current_line_index = 0
	_dropout_count = 0
	_reading_buffer.clear()
