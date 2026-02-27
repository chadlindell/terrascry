## Training system manager — guided exercises with grading.
##
## Defines training modules that teach survey techniques, track performance
## metrics, and grade the operator. Provides real-time feedback and
## end-of-module report cards.
class_name TrainingManager
extends Node

## Training module definitions
enum Module {
	WALK_STRAIGHT_LINE,
	COMPLETE_GRID,
	FIND_TARGET,
	MULTI_TARGET,
	DRONE_SURVEY,
	BOREHOLE_LOGGING,
}

## Module descriptions
const MODULE_INFO := {
	Module.WALK_STRAIGHT_LINE: {
		"name": "Walk a Straight Line",
		"description": "Maintain heading for 20m. Graded on cross-track error.",
		"scenario": "scenarios/single-ferrous-target.json",
		"duration_limit": 60,  # seconds
		"target_distance": 20.0,
	},
	Module.COMPLETE_GRID: {
		"name": "Complete a Grid",
		"description": "Cover a 10x10m area with 1m line spacing.",
		"scenario": "scenarios/single-ferrous-target.json",
		"duration_limit": 300,
		"target_coverage": 85.0,
	},
	Module.FIND_TARGET: {
		"name": "Find the Target",
		"description": "Locate a single buried object. Graded on detection and marking accuracy.",
		"scenario": "scenarios/single-ferrous-target.json",
		"duration_limit": 180,
	},
	Module.MULTI_TARGET: {
		"name": "Multi-Target Discrimination",
		"description": "Scattered debris — identify and mark all anomalies.",
		"scenario": "scenarios/scattered-debris.json",
		"duration_limit": 600,
	},
	Module.DRONE_SURVEY: {
		"name": "Drone Survey",
		"description": "Fly planned lines at constant altitude. Graded on speed/altitude stability.",
		"scenario": "scenarios/scattered-debris.json",
		"duration_limit": 300,
		"operator_mode": "drone",
	},
	Module.BOREHOLE_LOGGING: {
		"name": "Borehole Logging",
		"description": "HIRT borehole exercise — log a complete depth profile.",
		"scenario": "scenarios/swamp-crash-site.json",
		"duration_limit": 300,
		"operator_mode": "borehole",
	},
}

## Current training state
var active_module: Module = Module.WALK_STRAIGHT_LINE
var module_active := false
var module_start_time := 0.0
var module_elapsed := 0.0

## Grading metrics (accumulated during module)
var total_xte := 0.0
var xte_count := 0
var max_xte := 0.0
var xte_samples: PackedFloat64Array = []
var speed_samples: PackedFloat64Array = []
var marked_positions: Array[Vector3] = []

## Signals
signal module_started(module: Module)
signal module_completed(module: Module, grade: Dictionary)
signal objective_updated(text: String)


func _ready() -> void:
	SurveyManager.state_changed.connect(_on_state_changed)
	DataRecorder.sample_recorded.connect(_on_sample_recorded)


func _on_state_changed(new_state: SurveyManager.State) -> void:
	if new_state != SurveyManager.State.TRAINING and module_active:
		end_module()


func _process(delta: float) -> void:
	if not module_active:
		return

	module_elapsed = (Time.get_ticks_msec() / 1000.0) - module_start_time

	# Check time limit
	var info: Dictionary = MODULE_INFO[active_module]
	var limit: float = info.get("duration_limit", 999)
	if module_elapsed > limit:
		end_module()


func _on_sample_recorded(sample_index: int) -> void:
	if not module_active or DataRecorder.samples.is_empty():
		return

	var sample: Dictionary = DataRecorder.samples[sample_index - 1]
	var xte: float = sample.get("xte", -1.0)
	var speed: float = sample.get("speed", 0.0)

	if xte >= 0:
		total_xte += xte
		xte_count += 1
		max_xte = max(max_xte, xte)
		xte_samples.append(xte)

	speed_samples.append(speed)


## Start a training module.
func start_module(module: Module) -> void:
	active_module = module
	module_active = true
	module_start_time = Time.get_ticks_msec() / 1000.0

	# Reset metrics
	total_xte = 0.0
	xte_count = 0
	max_xte = 0.0
	xte_samples.clear()
	speed_samples.clear()
	marked_positions.clear()

	module_started.emit(module)

	var info: Dictionary = MODULE_INFO[module]
	objective_updated.emit(info.get("description", ""))
	print("[Training] Module started: %s" % info.get("name", ""))


## End current module and compute grade.
func end_module() -> void:
	if not module_active:
		return

	module_active = false
	var grade := _compute_grade()

	module_completed.emit(active_module, grade)
	print("[Training] Module complete — Grade: %s (%.0f%%)" % [
		grade.get("letter", "?"), grade.get("score", 0.0)])


## Compute grade for the completed module.
func _compute_grade() -> Dictionary:
	var score := 0.0
	var breakdown := {}

	match active_module:
		Module.WALK_STRAIGHT_LINE:
			# Grade on XTE
			var rms_xte := _compute_rms(xte_samples)
			breakdown["rms_xte_m"] = rms_xte
			breakdown["max_xte_m"] = max_xte
			# Score: 100 if RMS < 0.1m, 0 if RMS > 1.0m
			score = clampf((1.0 - rms_xte) / 0.9, 0.0, 1.0) * 100.0

		Module.COMPLETE_GRID:
			# Grade on coverage completeness
			var coverage := 0.0
			# Approximate coverage from heatmap
			var stats := DataRecorder.get_stats()
			breakdown["distance_m"] = stats.get("distance_m", 0.0)
			breakdown["samples"] = stats.get("sample_count", 0)
			# Spacing quality
			var rms_xte := _compute_rms(xte_samples)
			breakdown["rms_xte_m"] = rms_xte
			score = clampf(1.0 - rms_xte * 2.0, 0.0, 1.0) * 100.0

		Module.FIND_TARGET:
			# Grade on detection accuracy
			breakdown["marks_placed"] = marked_positions.size()
			var peak: float = abs(DataRecorder.peak_reading)
			breakdown["peak_reading"] = peak
			if peak > 0 and marked_positions.size() > 0:
				score = 80.0  # Basic pass for finding something
			elif peak > 0:
				score = 40.0  # Detected but didn't mark

		Module.MULTI_TARGET:
			breakdown["marks_placed"] = marked_positions.size()
			var stats := DataRecorder.get_stats()
			breakdown["samples"] = stats.get("sample_count", 0)
			score = clampf(float(marked_positions.size()) / 6.0, 0.0, 1.0) * 100.0

		Module.DRONE_SURVEY:
			# Grade on speed and altitude stability
			var speed_var := _compute_variance(speed_samples)
			breakdown["speed_variance"] = speed_var
			score = clampf(1.0 - speed_var * 10.0, 0.0, 1.0) * 100.0

		Module.BOREHOLE_LOGGING:
			var stats := DataRecorder.get_stats()
			breakdown["samples"] = stats.get("sample_count", 0)
			score = clampf(float(stats.get("sample_count", 0)) / 50.0, 0.0, 1.0) * 100.0

	var letter := _score_to_letter(score)

	return {
		"module": Module.keys()[active_module],
		"score": score,
		"letter": letter,
		"duration_s": module_elapsed,
		"breakdown": breakdown,
	}


func _score_to_letter(score: float) -> String:
	if score >= 90:
		return "A"
	elif score >= 80:
		return "B"
	elif score >= 70:
		return "C"
	elif score >= 60:
		return "D"
	else:
		return "F"


func _compute_rms(values: PackedFloat64Array) -> float:
	if values.is_empty():
		return 0.0
	var sum_sq := 0.0
	for v in values:
		sum_sq += v * v
	return sqrt(sum_sq / values.size())


func _compute_variance(values: PackedFloat64Array) -> float:
	if values.size() < 2:
		return 0.0
	var mean := 0.0
	for v in values:
		mean += v
	mean /= values.size()
	var variance := 0.0
	for v in values:
		variance += (v - mean) * (v - mean)
	return variance / values.size()


## Get list of available modules with descriptions.
func get_module_list() -> Array[Dictionary]:
	var list: Array[Dictionary] = []
	for module_key in MODULE_INFO:
		var info: Dictionary = MODULE_INFO[module_key]
		list.append({
			"module": module_key,
			"name": info.get("name", ""),
			"description": info.get("description", ""),
		})
	return list
