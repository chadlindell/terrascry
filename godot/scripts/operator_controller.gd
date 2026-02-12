## First-person operator controller for survey simulation.
##
## WASD movement on the terrain surface. The operator carries
## the Pathfinder gradiometer and queries the physics server
## at each position for sensor readings.
extends CharacterBody3D

## Movement parameters
@export var walk_speed := 1.5  # m/s (typical survey speed)
@export var mouse_sensitivity := 0.002

## Sensor query rate (Hz) — matches Pathfinder's 10 Hz sample rate
@export var sample_rate := 10.0

## Sensor height above ground (bottom sensor)
@export var sensor_height := 0.175

## Current gradient reading (ADC counts equivalent)
var current_gradient := 0.0

## Query timer
var _query_timer := 0.0
var _query_interval := 0.1  # 1/sample_rate


func _ready() -> void:
	_query_interval = 1.0 / sample_rate
	# Capture mouse for first-person camera
	Input.mouse_mode = Input.MOUSE_MODE_CAPTURED


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseMotion:
		rotate_y(-event.relative.x * mouse_sensitivity)
		$Camera3D.rotate_x(-event.relative.y * mouse_sensitivity)
		$Camera3D.rotation.x = clamp($Camera3D.rotation.x, -PI/4, PI/4)

	if event.is_action_pressed("ui_cancel"):
		Input.mouse_mode = Input.MOUSE_MODE_VISIBLE


func _physics_process(delta: float) -> void:
	# Movement
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
		velocity.x = move_toward(velocity.x, 0, walk_speed)
		velocity.z = move_toward(velocity.z, 0, walk_speed)

	# Keep on ground
	position.y = 1.7  # Eye height ~1.7m

	move_and_slide()

	# Query physics at sample rate
	_query_timer += delta
	if _query_timer >= _query_interval:
		_query_timer = 0.0
		_query_physics()


func _query_physics() -> void:
	## Query gradient at current position
	var pos := global_position
	var sensor_pos := [pos.x, pos.z, sensor_height]  # Godot Y-up → GeoSim Z-up

	var result = await PhysicsClient.query_gradient([sensor_pos])
	if result.get("status") == "ok":
		var grad_data: Array = result.get("data", {}).get("gradient", [0.0])
		if grad_data.size() > 0:
			current_gradient = grad_data[0]

	# Update HUD (if connected)
	var hud := get_node_or_null("/root/Main/HUD")
	if hud and hud.has_method("update_reading"):
		hud.update_reading(current_gradient, global_position)
