## Central state machine and orchestration (autoload singleton).
##
## Manages application state transitions, active scenario/instrument tracking,
## and operator switching. All major UI and gameplay systems respond to signals
## emitted by this manager.
extends Node

## Application states
enum State {
	MAIN_MENU,
	SCENARIO_SELECT,
	SURVEY_PLANNING,
	SURVEYING,
	PAUSED,
	POST_SURVEY,
	HIRT_SURVEY,
	TRAINING,
}

## Available instrument types
enum Instrument {
	MAG_GRADIOMETER,
	EM_FDEM,
	RESISTIVITY,
}

## Operator modes
enum OperatorMode {
	GROUND,
	DRONE,
	BOREHOLE,
}

## Current state
var current_state: State = State.MAIN_MENU
var previous_state: State = State.MAIN_MENU

## Active instrument and operator
var current_instrument: Instrument = Instrument.MAG_GRADIOMETER
var current_operator_mode: OperatorMode = OperatorMode.GROUND
var active_operator: Node3D = null

## Scenario info (populated after load)
var scenario_info: Dictionary = {}
var scenario_path: String = ""

## Survey plan (populated in planning phase)
var survey_plan: Dictionary = {}

## Instruments available for current scenario
var available_instruments: Array[Instrument] = [Instrument.MAG_GRADIOMETER]

## Whether the physics server is live (vs mock)
var is_live: bool = false

## Signals
signal state_changed(new_state: State)
signal scenario_loaded(info: Dictionary)
signal instrument_changed(instrument_id: Instrument)
signal operator_switched(mode: OperatorMode)
signal survey_started()
signal survey_finished()


func _ready() -> void:
	print("[SurveyManager] Initialized")


func _unhandled_input(event: InputEvent) -> void:
	if event.is_action_pressed("pause"):
		if current_state == State.SURVEYING or current_state == State.HIRT_SURVEY:
			transition(State.PAUSED)
		elif current_state == State.PAUSED:
			transition(previous_state)

	if event.is_action_pressed("switch_instrument"):
		if current_state == State.SURVEYING:
			cycle_instrument()


## Transition to a new state. Validates the transition is legal.
func transition(new_state: State) -> void:
	if new_state == current_state:
		return

	var old_state := current_state
	previous_state = current_state
	current_state = new_state

	print("[SurveyManager] State: %s -> %s" % [State.keys()[old_state], State.keys()[new_state]])

	match new_state:
		State.SURVEYING:
			Input.mouse_mode = Input.MOUSE_MODE_CAPTURED
			survey_started.emit()
		State.PAUSED, State.MAIN_MENU, State.SCENARIO_SELECT, State.SURVEY_PLANNING, \
				State.POST_SURVEY, State.TRAINING:
			Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
		State.HIRT_SURVEY:
			Input.mouse_mode = Input.MOUSE_MODE_VISIBLE

	state_changed.emit(new_state)


## Load a scenario by path and store info.
func load_scenario(path: String) -> bool:
	scenario_path = path
	var result = await PhysicsClient.load_scenario(path)
	if result.get("status") != "ok":
		push_warning("[SurveyManager] Failed to load scenario: %s" % result.get("message", ""))
		return false

	var info_result = await PhysicsClient.get_scenario_info()
	if info_result.get("status") != "ok":
		return false

	scenario_info = info_result.get("data", {})
	_determine_available_instruments()

	# Check server mode
	var ping_result = await PhysicsClient.ping()
	var msg: String = ping_result.get("data", {}).get("message", "")
	is_live = not msg.contains("mock")

	scenario_loaded.emit(scenario_info)
	return true


## Determine which instruments are applicable for current scenario.
func _determine_available_instruments() -> void:
	available_instruments.clear()
	available_instruments.append(Instrument.MAG_GRADIOMETER)

	# EM and ERT available based on scenario metadata or hirt_config presence
	var meta: Dictionary = scenario_info.get("metadata", {})
	var difficulty: String = meta.get("difficulty", "")
	var has_hirt: bool = scenario_info.has("hirt_config")

	# Bomb crater and swamp scenarios support EM
	var name: String = scenario_info.get("name", "").to_lower()
	if name.contains("crater") or name.contains("swamp") or name.contains("crash") \
			or has_hirt:
		available_instruments.append(Instrument.EM_FDEM)

	# Swamp / crash scenarios support ERT
	if name.contains("swamp") or name.contains("crash") or has_hirt:
		available_instruments.append(Instrument.RESISTIVITY)


## Cycle to next available instrument.
func cycle_instrument() -> void:
	if available_instruments.size() <= 1:
		return

	var idx := available_instruments.find(current_instrument)
	idx = (idx + 1) % available_instruments.size()
	current_instrument = available_instruments[idx]
	instrument_changed.emit(current_instrument)
	print("[SurveyManager] Instrument: %s" % Instrument.keys()[current_instrument])


## Switch active operator mode.
func switch_operator(mode: OperatorMode) -> void:
	if mode == current_operator_mode:
		return
	current_operator_mode = mode
	operator_switched.emit(mode)
	print("[SurveyManager] Operator: %s" % OperatorMode.keys()[mode])


## Get human-readable instrument name.
static func instrument_name(inst: Instrument) -> String:
	match inst:
		Instrument.MAG_GRADIOMETER:
			return "Mag Gradiometer"
		Instrument.EM_FDEM:
			return "EM (FDEM)"
		Instrument.RESISTIVITY:
			return "Resistivity (ERT)"
		_:
			return "Unknown"


## Get instrument units string.
static func instrument_units(inst: Instrument) -> String:
	match inst:
		Instrument.MAG_GRADIOMETER:
			return "nT"
		Instrument.EM_FDEM:
			return "ppm"
		Instrument.RESISTIVITY:
			return "Ohm-m"
		_:
			return ""


## Get query command for instrument.
static func instrument_query_command(inst: Instrument) -> String:
	match inst:
		Instrument.MAG_GRADIOMETER:
			return "query_gradient"
		Instrument.EM_FDEM:
			return "query_em_response"
		Instrument.RESISTIVITY:
			return "query_apparent_resistivity"
		_:
			return "query_gradient"
