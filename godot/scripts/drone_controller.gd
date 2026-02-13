## Drone/UAV controller for aerial survey simulation.
##
## 6DOF movement without gravity, altitude hold via RayCast3D, autopilot mode
## following planned waypoints with PID controller, battery simulation,
## and first/third-person camera toggle.
extends CharacterBody3D

## Movement parameters
@export var max_speed := 5.0  # m/s
@export var acceleration := 3.0
@export var deceleration := 5.0
@export var vertical_speed := 2.0  # m/s
@export var yaw_speed := 2.0  # rad/s
@export var mouse_sensitivity := 0.002

## Flight parameters
@export var target_altitude_agl := 2.0  # meters above ground level
@export var min_altitude_agl := 0.5
@export var max_altitude_agl := 20.0

## Sensor parameters
@export var sensor_height_offset := 0.0  # sensor on belly
@export var sample_rate := 10.0

## Battery simulation
@export var battery_capacity_s := 1200.0  # 20 minutes
@export var rth_threshold := 0.2  # Return-to-home at 20%
var battery_remaining := 1.0  # 0.0 to 1.0
var _battery_warning := false

## Autopilot state
var autopilot_enabled := false
var _waypoint_index := 0
var _waypoints: PackedVector3Array = []  # Godot coordinates
var _home_position := Vector3.ZERO

## PID controller gains (lateral cross-track correction)
var _pid_kp := 2.0
var _pid_ki := 0.1
var _pid_kd := 0.5
var _pid_integral := 0.0
var _pid_prev_error := 0.0

## Camera mode
var _third_person := false
var _chase_distance := 5.0
var _chase_height := 3.0

## Survey state
var current_speed := 0.0
var heading := 0.0
var current_altitude_agl := 0.0
var distance_traveled := 0.0
var _last_position := Vector3.ZERO

## Sensor query
var _query_timer := 0.0
var _query_interval := 0.1
var _dropout_count := 0
var current_reading := 0.0

## Terrain bounds
var terrain_x_extent := Vector2(0, 20)
var terrain_y_extent := Vector2(0, 20)
var surface_elevation := 0.0

## Mouse state
var _mouse_captured := true

## Signals
signal position_updated(world_pos: Vector3)
signal battery_warning(remaining: float)
signal altitude_warning(agl: float)


func _ready() -> void:
	_query_interval = 1.0 / sample_rate
	_home_position = global_position
	_last_position = global_position

	# Load waypoints from survey plan
	var plan: Dictionary = SurveyManager.survey_plan
	if not plan.is_empty():
		var lines: Array = plan.get("lines", [])
		for line in lines:
			for pt in line:
				_waypoints.append(CoordUtil.to_godot(pt))

		max_speed = plan.get("speed_target", 5.0)


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseMotion and _mouse_captured:
		if not autopilot_enabled:
			rotate_y(-event.relative.x * mouse_sensitivity)

	if event.is_action_pressed("ui_cancel"):
		_mouse_captured = false
		Input.mouse_mode = Input.MOUSE_MODE_VISIBLE

	if event is InputEventMouseButton and event.pressed and not _mouse_captured:
		_mouse_captured = true
		Input.mouse_mode = Input.MOUSE_MODE_CAPTURED

	# Toggle camera mode
	if event.is_action_pressed("toggle_camera"):
		_third_person = not _third_person
		_update_camera_mode()

	# Toggle autopilot (double-tap move_forward or specific key)
	if event.is_action_pressed("interact"):
		autopilot_enabled = not autopilot_enabled
		if autopilot_enabled:
			_pid_integral = 0.0
			_pid_prev_error = 0.0
			print("[Drone] Autopilot ENGAGED")
		else:
			print("[Drone] Autopilot DISENGAGED")

	# Event marker
	if event.is_action_pressed("mark_event"):
		if DataRecorder.is_recording:
			DataRecorder.mark_event("drone_mark", global_position)


func _physics_process(delta: float) -> void:
	# Battery drain
	battery_remaining -= delta / battery_capacity_s
	battery_remaining = maxf(battery_remaining, 0.0)

	if battery_remaining < rth_threshold and not _battery_warning:
		_battery_warning = true
		battery_warning.emit(battery_remaining)
		print("[Drone] LOW BATTERY — RTH recommended")

	if autopilot_enabled and not _waypoints.is_empty():
		_autopilot_update(delta)
	else:
		_manual_control(delta)

	# Altitude measurement via raycast
	var space_state := get_world_3d().direct_space_state
	var ray_params := PhysicsRayQueryParameters3D.create(
		global_position,
		global_position + Vector3.DOWN * 50.0
	)
	var ray_result := space_state.intersect_ray(ray_params)
	if ray_result:
		current_altitude_agl = global_position.y - ray_result.position.y
	else:
		current_altitude_agl = global_position.y - surface_elevation

	# Altitude warning
	if current_altitude_agl < min_altitude_agl:
		altitude_warning.emit(current_altitude_agl)

	# Clamp to terrain boundaries
	position.x = clamp(position.x, terrain_x_extent.x, terrain_x_extent.y)
	position.z = clamp(position.z, terrain_y_extent.x, terrain_y_extent.y)

	# Track distance and speed
	var moved := global_position.distance_to(_last_position)
	current_speed = moved / delta if delta > 0 else 0.0
	if moved > 0.001:
		distance_traveled += moved
	_last_position = global_position

	# Heading
	heading = fposmod(rotation.y - PI, TAU)

	# Position signal
	position_updated.emit(global_position)

	# Sensor query
	_query_timer += delta
	if _query_timer >= _query_interval:
		_query_timer -= _query_interval
		_query_physics()


func _manual_control(delta: float) -> void:
	## Manual WASD + Q/E flight control.
	var input_dir := Vector3.ZERO

	if Input.is_action_pressed("move_forward"):
		input_dir.z -= 1
	if Input.is_action_pressed("move_back"):
		input_dir.z += 1
	if Input.is_action_pressed("move_left"):
		input_dir.x -= 1
	if Input.is_action_pressed("move_right"):
		input_dir.x += 1
	if Input.is_action_pressed("altitude_up"):
		input_dir.y += 1
	if Input.is_action_pressed("altitude_down"):
		input_dir.y -= 1

	input_dir = input_dir.normalized()
	var world_dir := transform.basis * input_dir

	if input_dir.length() > 0.01:
		velocity = velocity.move_toward(world_dir * max_speed, acceleration * delta)
	else:
		velocity = velocity.move_toward(Vector3.ZERO, deceleration * delta)

	# Altitude hold: if no vertical input, maintain target AGL
	if abs(input_dir.y) < 0.01:
		var alt_error := target_altitude_agl - current_altitude_agl
		velocity.y += alt_error * 2.0 * delta  # Gentle correction

	move_and_slide()


func _autopilot_update(delta: float) -> void:
	## PID-based waypoint following with altitude hold.
	if _waypoint_index >= _waypoints.size():
		autopilot_enabled = false
		print("[Drone] Autopilot: all waypoints reached")
		return

	var target := _waypoints[_waypoint_index]
	target.y = surface_elevation + target_altitude_agl

	var to_target := target - global_position
	var horiz_dist := Vector2(to_target.x, to_target.z).length()

	# Waypoint reached?
	if horiz_dist < 0.5:
		_waypoint_index += 1
		_pid_integral = 0.0
		if _waypoint_index >= _waypoints.size():
			velocity = Vector3.ZERO
			autopilot_enabled = false
			return
		return

	# PID for lateral correction
	var error := horiz_dist
	_pid_integral += error * delta
	_pid_integral = clampf(_pid_integral, -10.0, 10.0)
	var derivative := (error - _pid_prev_error) / delta if delta > 0 else 0.0
	_pid_prev_error = error

	var correction := _pid_kp * error + _pid_ki * _pid_integral + _pid_kd * derivative
	var speed := minf(correction, max_speed)

	# Direction to target
	var dir := to_target.normalized()
	velocity = dir * speed

	# Altitude hold
	var alt_error := target_altitude_agl - current_altitude_agl
	velocity.y = alt_error * 3.0

	# Face direction of travel
	if horiz_dist > 0.1:
		var look_target := global_position + Vector3(to_target.x, 0, to_target.z)
		look_at(look_target, Vector3.UP)

	move_and_slide()


func _query_physics() -> void:
	if PhysicsClient.is_busy():
		_dropout_count += 1
		return

	var gs := CoordUtil.to_geosim(global_position)
	var sensor_pos := [gs.x, gs.y, gs.z + sensor_height_offset]

	var start_ms := Time.get_ticks_msec()
	var result: Dictionary

	match SurveyManager.current_instrument:
		SurveyManager.Instrument.MAG_GRADIOMETER:
			result = await PhysicsClient.query_gradient([sensor_pos])
		SurveyManager.Instrument.EM_FDEM:
			result = await PhysicsClient.query_em_response([sensor_pos])
		_:
			result = await PhysicsClient.query_gradient([sensor_pos])

	var latency_ms := Time.get_ticks_msec() - start_ms

	if result.get("status") == "ok":
		var raw_data: Dictionary = result.get("data", {})
		var reading := 0.0

		match SurveyManager.current_instrument:
			SurveyManager.Instrument.MAG_GRADIOMETER:
				var grad: Array = raw_data.get("gradient", [0.0])
				if grad.size() > 0:
					reading = grad[0]
			SurveyManager.Instrument.EM_FDEM:
				var resp: Array = raw_data.get("response_real", [0.0])
				if resp.size() > 0:
					reading = resp[0]

		current_reading = reading

		# Record in DataRecorder
		if DataRecorder.is_recording:
			var inst_name: String = SurveyManager.Instrument.keys()[SurveyManager.current_instrument].to_lower()
			DataRecorder.record_sample({
				"x_e": gs.x,
				"y_n": gs.y,
				"z_up": gs.z,
				"heading": heading,
				"speed": current_speed,
				"line_id": -1,
				"instrument": inst_name,
				"reading": reading,
				"reading_raw": raw_data,
				"mock": PhysicsClient.is_mock_mode,
				"xte": -1.0,
				"quality": 1 if current_altitude_agl >= min_altitude_agl else 0,
				"latency_ms": latency_ms,
			})


func _update_camera_mode() -> void:
	var cam := $Camera3D as Camera3D
	if not cam:
		return

	if _third_person:
		cam.position = Vector3(0, _chase_height, _chase_distance)
		cam.look_at(global_position, Vector3.UP)
	else:
		cam.position = Vector3.ZERO
		cam.rotation = Vector3.ZERO
