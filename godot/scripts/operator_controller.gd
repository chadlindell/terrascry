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

## Head bob parameters (gait noise)
@export var head_bob_enabled := true
@export var head_bob_frequency := 2.0  # Hz (step frequency)
@export var head_bob_amplitude := 0.02  # meters

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

## Internal state
var _query_timer := 0.0
var _query_interval := 0.1
var _last_position := Vector3.ZERO
var _mouse_captured := true
var _head_bob_timer := 0.0

## Emitted when a new gradient reading arrives. HUD listens to this.
signal gradient_reading(value: float, world_pos: Vector3)


func _ready() -> void:
	_query_interval = 1.0 / sample_rate
	survey_start_time = Time.get_ticks_msec() / 1000.0
	Input.mouse_mode = Input.MOUSE_MODE_CAPTURED
	_last_position = global_position

	# Load survey lines from plan
	if not SurveyManager.survey_plan.is_empty():
		_survey_lines = SurveyManager.survey_plan.get("lines", [])
		_target_speed = SurveyManager.survey_plan.get("speed_target", 1.5)
		walk_speed = _target_speed


func _unhandled_input(event: InputEvent) -> void:
	# Mouse look (only when captured)
	if event is InputEventMouseMotion and _mouse_captured:
		rotate_y(-event.relative.x * mouse_sensitivity)
		$Camera3D.rotate_x(-event.relative.y * mouse_sensitivity)
		$Camera3D.rotation.x = clamp($Camera3D.rotation.x, -PI / 4, PI / 4)

	# Escape releases mouse
	if event.is_action_pressed("ui_cancel"):
		_mouse_captured = false
		Input.mouse_mode = Input.MOUSE_MODE_VISIBLE

	# Click recaptures mouse
	if event is InputEventMouseButton and event.pressed and not _mouse_captured:
		_mouse_captured = true
		Input.mouse_mode = Input.MOUSE_MODE_CAPTURED

	# Event marker
	if event.is_action_pressed("mark_event"):
		if DataRecorder.is_recording:
			DataRecorder.mark_event("mark", global_position)


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

	if direction:
		velocity.x = direction.x * walk_speed
		velocity.z = direction.z * walk_speed
	else:
		# Frame-rate-independent deceleration
		velocity.x = move_toward(velocity.x, 0, walk_speed * delta * 10.0)
		velocity.z = move_toward(velocity.z, 0, walk_speed * delta * 10.0)

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
		var bob_offset := sin(_head_bob_timer) * head_bob_amplitude * (current_speed / walk_speed)
		$Camera3D.position.y = eye_height + bob_offset
	else:
		$Camera3D.position.y = eye_height
		_head_bob_timer = 0.0

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
				_current_line_index += 1


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
