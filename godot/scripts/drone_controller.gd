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
var _auto_engaged := false  # True if autopilot was auto-started

## Line tracking: maps waypoint index ranges to line IDs
var _waypoint_line_map: PackedInt32Array = []  # line_id per waypoint
var _current_line_id := -1
var _total_survey_lines := 0

## PID controller gains (lateral cross-track correction)
var _pid_kp := 2.0
var _pid_ki := 0.1
var _pid_kd := 0.5
var _pid_integral := 0.0
var _pid_prev_error := 0.0

## Camera mode
var _third_person := true
var _chase_distance := 8.0
var _chase_height := 5.0

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

## Drone model
var _drone_model: Node3D
var _rotors: Array[MeshInstance3D] = []

## Mouse state
var _mouse_captured := true

## Signals
signal position_updated(world_pos: Vector3)
signal battery_warning(remaining: float)
signal altitude_warning(agl: float)
signal drone_progress(info: Dictionary)


func _ready() -> void:
	_query_interval = 1.0 / sample_rate
	_home_position = global_position
	_last_position = global_position

	_build_drone_model()
	_update_camera_mode()

	# Load waypoints when the survey actually starts (plan isn't ready at _ready time)
	SurveyManager.survey_started.connect(_on_survey_started)


func _on_survey_started() -> void:
	## Load waypoints from survey plan and auto-engage autopilot.
	_waypoints.clear()
	_waypoint_line_map.clear()
	_waypoint_index = 0
	_current_line_id = -1
	_pid_integral = 0.0
	_pid_prev_error = 0.0
	autopilot_enabled = false

	var plan: Dictionary = SurveyManager.survey_plan
	if plan.is_empty():
		return

	var lines: Array = plan.get("lines", [])
	_total_survey_lines = lines.size()
	for line_idx in range(lines.size()):
		var line: Array = lines[line_idx]
		for pt in line:
			_waypoints.append(CoordUtil.to_godot(pt))
			_waypoint_line_map.append(line_idx)

	max_speed = plan.get("speed_target", 5.0)
	_home_position = global_position

	# Auto-engage autopilot for drone surveys
	if not _waypoints.is_empty():
		autopilot_enabled = true
		_auto_engaged = true
		print("[Drone] Autopilot AUTO-ENGAGED — %d waypoints" % _waypoints.size())


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

	# Toggle autopilot — E key disengages (manual override) or re-engages
	if event.is_action_pressed("interact"):
		if autopilot_enabled:
			autopilot_enabled = false
			_auto_engaged = false
			print("[Drone] Autopilot DISENGAGED — manual control")
		elif not _waypoints.is_empty() and _waypoint_index < _waypoints.size():
			autopilot_enabled = true
			_pid_integral = 0.0
			_pid_prev_error = 0.0
			print("[Drone] Autopilot RE-ENGAGED")

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

	# Spin rotors
	for rotor in _rotors:
		if is_instance_valid(rotor):
			rotor.rotate_y(25.0 * delta)

	# Position signal
	position_updated.emit(global_position)

	# Emit progress info for HUD
	if autopilot_enabled and not _waypoints.is_empty():
		var remaining_dist := _compute_remaining_distance()
		var eta := remaining_dist / maxf(current_speed, 0.5)
		var lines_done := _current_line_id if _current_line_id >= 0 else 0
		drone_progress.emit({
			"waypoint": _waypoint_index,
			"total_waypoints": _waypoints.size(),
			"eta_seconds": eta,
			"lines_completed": lines_done,
			"total_lines": _total_survey_lines,
			"battery": battery_remaining,
			"autopilot": true,
		})
	else:
		drone_progress.emit({
			"waypoint": _waypoint_index,
			"total_waypoints": _waypoints.size(),
			"eta_seconds": 0.0,
			"lines_completed": _current_line_id if _current_line_id >= 0 else 0,
			"total_lines": _total_survey_lines,
			"battery": battery_remaining,
			"autopilot": false,
		})

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
		return

	var target := _waypoints[_waypoint_index]
	target.y = surface_elevation + target_altitude_agl

	var to_target := target - global_position
	var horiz_dist := Vector2(to_target.x, to_target.z).length()

	# Track current line ID from waypoint map
	if _waypoint_index < _waypoint_line_map.size():
		_current_line_id = _waypoint_line_map[_waypoint_index]

	# Waypoint reached?
	if horiz_dist < 0.5:
		_waypoint_index += 1
		_pid_integral = 0.0
		if _waypoint_index >= _waypoints.size():
			velocity = Vector3.ZERO
			autopilot_enabled = false
			print("[Drone] Autopilot: all waypoints reached")
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
				"line_id": _current_line_id,
				"instrument": inst_name,
				"reading": reading,
				"reading_raw": raw_data,
				"mock": PhysicsClient.is_mock_mode,
				"xte": -1.0,
				"quality": 1 if current_altitude_agl >= min_altitude_agl else 0,
				"latency_ms": latency_ms,
			})


func _compute_remaining_distance() -> float:
	## Sum remaining waypoint-to-waypoint distances from current position.
	if _waypoints.is_empty() or _waypoint_index >= _waypoints.size():
		return 0.0
	var total := global_position.distance_to(_waypoints[_waypoint_index])
	for i in range(_waypoint_index, _waypoints.size() - 1):
		total += _waypoints[i].distance_to(_waypoints[i + 1])
	return total


func _update_camera_mode() -> void:
	var cam := $Camera3D as Camera3D
	if not cam:
		return

	if _third_person:
		cam.position = Vector3(0, _chase_height, _chase_distance)
		cam.rotation = Vector3(-deg_to_rad(20.0), 0, 0)
	else:
		cam.position = Vector3(0, 0.1, 0.3)
		cam.rotation = Vector3.ZERO


func _build_drone_model() -> void:
	## Build a procedural quadcopter mesh.
	_drone_model = Node3D.new()
	_drone_model.name = "DroneModel"
	add_child(_drone_model)

	# Central body — flat dark-gray box
	var body := MeshInstance3D.new()
	body.name = "Body"
	var body_mesh := BoxMesh.new()
	body_mesh.size = Vector3(0.25, 0.06, 0.25)
	body.mesh = body_mesh
	var body_mat := StandardMaterial3D.new()
	body_mat.albedo_color = Color(0.2, 0.22, 0.25)
	body.material_override = body_mat
	_drone_model.add_child(body)

	# 4 arms extending diagonally + rotor discs at tips
	var arm_positions: Array[Vector3] = [
		Vector3(0.2, 0, 0.2),
		Vector3(-0.2, 0, 0.2),
		Vector3(0.2, 0, -0.2),
		Vector3(-0.2, 0, -0.2),
	]

	var arm_mat := StandardMaterial3D.new()
	arm_mat.albedo_color = Color(0.15, 0.15, 0.18)

	var rotor_mat := StandardMaterial3D.new()
	rotor_mat.albedo_color = Color(0.5, 0.5, 0.55, 0.5)
	rotor_mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA

	for i in range(4):
		var arm_pos := arm_positions[i]

		# Arm
		var arm := MeshInstance3D.new()
		arm.name = "Arm_%d" % i
		var arm_mesh := CylinderMesh.new()
		arm_mesh.top_radius = 0.012
		arm_mesh.bottom_radius = 0.012
		arm_mesh.height = arm_pos.length() * 2.0
		arm.mesh = arm_mesh
		arm.material_override = arm_mat
		arm.position = arm_pos * 0.5
		# Rotate cylinder to lie along the diagonal
		var angle := atan2(arm_pos.x, arm_pos.z)
		arm.rotation = Vector3(0, 0, deg_to_rad(90.0))
		arm.rotate(Vector3.UP, angle)
		_drone_model.add_child(arm)

		# Motor housing (small cylinder at arm tip)
		var motor := MeshInstance3D.new()
		motor.name = "Motor_%d" % i
		var motor_mesh := CylinderMesh.new()
		motor_mesh.top_radius = 0.025
		motor_mesh.bottom_radius = 0.025
		motor_mesh.height = 0.03
		motor.mesh = motor_mesh
		motor.material_override = arm_mat
		motor.position = arm_pos
		motor.position.y = 0.015
		_drone_model.add_child(motor)

		# Rotor disc
		var rotor := MeshInstance3D.new()
		rotor.name = "Rotor_%d" % i
		var rotor_mesh := CylinderMesh.new()
		rotor_mesh.top_radius = 0.12
		rotor_mesh.bottom_radius = 0.12
		rotor_mesh.height = 0.005
		rotor.mesh = rotor_mesh
		rotor.material_override = rotor_mat
		rotor.position = arm_pos
		rotor.position.y = 0.035
		_drone_model.add_child(rotor)
		_rotors.append(rotor)

	# Sensor pod — small orange cylinder underneath center (magnetometer)
	var sensor := MeshInstance3D.new()
	sensor.name = "SensorPod"
	var sensor_mesh := CylinderMesh.new()
	sensor_mesh.top_radius = 0.03
	sensor_mesh.bottom_radius = 0.02
	sensor_mesh.height = 0.06
	sensor.mesh = sensor_mesh
	var sensor_mat := StandardMaterial3D.new()
	sensor_mat.albedo_color = Color(0.9, 0.5, 0.1)
	sensor.material_override = sensor_mat
	sensor.position = Vector3(0, -0.06, 0)
	_drone_model.add_child(sensor)

	# Landing skids — 2 thin bars beneath body
	var skid_mat := StandardMaterial3D.new()
	skid_mat.albedo_color = Color(0.3, 0.3, 0.32)
	for side in [-1.0, 1.0]:
		var skid := MeshInstance3D.new()
		skid.name = "Skid_%s" % ("L" if side < 0 else "R")
		var skid_mesh := BoxMesh.new()
		skid_mesh.size = Vector3(0.01, 0.01, 0.3)
		skid.mesh = skid_mesh
		skid.material_override = skid_mat
		skid.position = Vector3(side * 0.1, -0.05, 0)
		_drone_model.add_child(skid)
