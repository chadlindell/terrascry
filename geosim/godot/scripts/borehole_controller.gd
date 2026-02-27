## HIRT borehole probe controller.
##
## Controls a virtual probe being lowered into a borehole for subsurface
## EM/resistivity measurements. Supports stop-and-go and continuous
## descent modes with winch mechanics and cable tension simulation.
extends Node3D

## Borehole parameters
@export var max_depth := 5.0  # meters
@export var depth_interval := 0.1  # meters (stop-and-go mode)
@export var descent_speed := 0.1  # m/s (adjustable via W/S)
@export var min_descent_speed := 0.05
@export var max_descent_speed := 0.5

## Probe state
var current_depth := 0.0
var target_depth := 0.0
var is_descending := false
var is_ascending := false
var cable_tension := 1.0  # 0=slack, 1=taut

## Measurement mode
enum Mode { STOP_AND_GO, CONTINUOUS }
var current_mode: Mode = Mode.STOP_AND_GO

## Borehole info
var borehole_position := Vector3.ZERO  # GeoSim coords
var surface_elevation := 0.0

## Recorded depth log
var depth_log: Array[Dictionary] = []

## Query state
var _query_timer := 0.0
var _stabilization_timer := 0.0
var _stabilized := false
var _continuous_sample_interval := 0.1  # seconds between samples in continuous mode

## Signals
signal depth_changed(depth: float, reading: float)
signal log_complete(log_data: Array)


func _ready() -> void:
	set_process(false)  # Activated by SurveyManager


func _unhandled_input(event: InputEvent) -> void:
	# Q/E adjusts descent speed (discrete events)
	if event.is_action_pressed("altitude_up"):
		descent_speed = min(descent_speed + 0.05, max_descent_speed)
	if event.is_action_pressed("altitude_down"):
		descent_speed = max(descent_speed - 0.05, min_descent_speed)

	# Space marks a reading at current depth
	if event.is_action_pressed("mark_event"):
		_take_measurement()


func _process(delta: float) -> void:
	# Poll movement inputs each frame (not in _unhandled_input to avoid reset on unrelated events)
	is_descending = Input.is_action_pressed("move_forward")
	is_ascending = Input.is_action_pressed("move_back") and not is_descending

	# Update probe depth
	if is_descending and current_depth < max_depth:
		current_depth += descent_speed * delta
		current_depth = min(current_depth, max_depth)
		_stabilized = false
		_stabilization_timer = 0.0
	elif is_ascending and current_depth > 0.0:
		current_depth -= descent_speed * delta * 1.5  # Ascend slightly faster
		current_depth = max(current_depth, 0.0)
		_stabilized = false
		_stabilization_timer = 0.0

	# Cable tension simulation
	if is_descending:
		cable_tension = clampf(1.0 - descent_speed * 0.5, 0.3, 1.0)
	elif is_ascending:
		cable_tension = 1.0
	else:
		cable_tension = lerpf(cable_tension, 1.0, delta * 2.0)

	# Stop-and-go: stabilize at each depth interval
	if current_mode == Mode.STOP_AND_GO and not is_descending and not is_ascending:
		_stabilization_timer += delta
		if _stabilization_timer > 1.0 and not _stabilized:
			_stabilized = true
			_take_measurement()

	# Continuous mode: sample at regular intervals while moving
	if current_mode == Mode.CONTINUOUS and (is_descending or is_ascending):
		_query_timer += delta
		if _query_timer >= _continuous_sample_interval:
			_query_timer = 0.0
			_take_measurement()


func _take_measurement() -> void:
	## Query physics and record a depth reading.
	if PhysicsClient.is_busy():
		return

	# Probe position: borehole surface position minus current depth
	var probe_pos := [
		borehole_position.x,
		borehole_position.y,
		surface_elevation - current_depth,
	]

	var result: Dictionary
	match SurveyManager.current_instrument:
		SurveyManager.Instrument.EM_FDEM:
			result = await PhysicsClient.query_em_response([probe_pos])
		SurveyManager.Instrument.RESISTIVITY:
			result = await PhysicsClient.query_apparent_resistivity(
				[probe_pos], [[0, 1, 2, 3]])
		_:
			result = await PhysicsClient.query_em_response([probe_pos])

	if result.get("status") != "ok":
		return

	var raw_data: Dictionary = result.get("data", {})
	var em_real := 0.0
	var em_imag := 0.0
	var rho := 0.0

	var resp_real: Array = raw_data.get("response_real", [0.0])
	if resp_real.size() > 0:
		em_real = resp_real[0]
	var resp_imag: Array = raw_data.get("response_imag", [0.0])
	if resp_imag.size() > 0:
		em_imag = resp_imag[0]
	var rho_arr: Array = raw_data.get("apparent_resistivity", [0.0])
	if rho_arr.size() > 0:
		rho = rho_arr[0]

	var entry := {
		"depth_m": current_depth,
		"em_real": em_real,
		"em_imag": em_imag,
		"rho_apparent": rho,
		"cable_tension": cable_tension,
		"descent_rate": descent_speed if is_descending else 0.0,
		"stabilized": _stabilized,
	}

	depth_log.append(entry)
	depth_changed.emit(current_depth, em_real)

	# Also record in DataRecorder
	if DataRecorder.is_recording:
		DataRecorder.record_sample({
			"x_e": borehole_position.x,
			"y_n": borehole_position.y,
			"z_up": surface_elevation - current_depth,
			"heading": 0.0,
			"speed": descent_speed if is_descending else 0.0,
			"line_id": -1,
			"instrument": "em_borehole",
			"reading": em_real,
			"reading_raw": raw_data,
			"mock": PhysicsClient.is_mock_mode,
			"xte": -1.0,
			"quality": 1 if _stabilized else 2,
			"latency_ms": 0.0,
		})


## Export depth log as CSV.
func export_depth_log(file_path: String) -> void:
	var file := FileAccess.open(file_path, FileAccess.WRITE)
	if not file:
		return

	file.store_line("depth_m,em_real,em_imag,rho_apparent,cable_tension,descent_rate,stabilized")
	for entry in depth_log:
		file.store_line("%.3f,%.6e,%.6e,%.3f,%.2f,%.3f,%d" % [
			entry.get("depth_m", 0.0),
			entry.get("em_real", 0.0),
			entry.get("em_imag", 0.0),
			entry.get("rho_apparent", 0.0),
			entry.get("cable_tension", 0.0),
			entry.get("descent_rate", 0.0),
			1 if entry.get("stabilized", false) else 0,
		])
	file.close()


## Initialize borehole from scenario data.
func setup_borehole(pos: Vector3, depth: float, elev: float) -> void:
	borehole_position = pos
	max_depth = depth
	surface_elevation = elev
	current_depth = 0.0
	depth_log.clear()
	_stabilized = false
	_stabilization_timer = 0.0
