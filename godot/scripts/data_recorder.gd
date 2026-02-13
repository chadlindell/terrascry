## Survey data recording and export (autoload singleton).
##
## Buffers all measurement samples during a survey, manages session metadata,
## and exports to CSV (Pathfinder-compatible) and XYZ (Geosoft/Surfer/QGIS)
## formats. Each survey session gets a timestamped directory with all data files.
extends Node

## Sample buffer — each entry is a Dictionary (see schema in plan)
var samples: Array[Dictionary] = []

## Event markers (user-placed via Space key)
var events: Array[Dictionary] = []

## Session metadata
var session_id: String = ""
var session_start_time: float = 0.0
var session_scenario: String = ""
var session_instrument: String = ""
var session_operator_mode: String = ""
var session_survey_plan: Dictionary = {}

## Recording state
var is_recording: bool = false
var _sample_index: int = 0

## Statistics (updated live)
var total_distance: float = 0.0
var peak_reading: float = 0.0
var min_reading: float = INF
var max_reading: float = -INF
var _last_sample_pos: Vector3 = Vector3.ZERO

## Signals
signal sample_recorded(sample_index: int)
signal export_completed(file_path: String)
signal recording_started()
signal recording_stopped()


func _ready() -> void:
	print("[DataRecorder] Initialized")


## Begin a new recording session.
func start_recording(scenario_name: String, instrument: String, operator_mode: String,
		plan: Dictionary = {}) -> void:
	clear()
	session_id = _generate_session_id()
	session_start_time = Time.get_ticks_msec() / 1000.0
	session_scenario = scenario_name
	session_instrument = instrument
	session_operator_mode = operator_mode
	session_survey_plan = plan
	is_recording = true
	recording_started.emit()
	print("[DataRecorder] Recording started — session %s" % session_id)


## Stop recording.
func stop_recording() -> void:
	is_recording = false
	recording_stopped.emit()
	print("[DataRecorder] Recording stopped — %d samples" % samples.size())


## Record a single measurement sample.
func record_sample(data: Dictionary) -> void:
	if not is_recording:
		return

	data["t"] = (Time.get_ticks_msec() / 1000.0) - session_start_time
	_sample_index += 1

	# Track distance
	var pos := Vector3(data.get("x_e", 0.0), data.get("y_n", 0.0), data.get("z_up", 0.0))
	if _sample_index > 1:
		total_distance += pos.distance_to(_last_sample_pos)
	_last_sample_pos = pos

	# Track reading extremes
	var reading: float = data.get("reading", 0.0)
	if abs(reading) > abs(peak_reading):
		peak_reading = reading
	min_reading = min(min_reading, reading)
	max_reading = max(max_reading, reading)

	samples.append(data)
	sample_recorded.emit(_sample_index)


## Add an event marker at current time.
func mark_event(label: String = "mark", position: Vector3 = Vector3.ZERO) -> void:
	var evt := {
		"t": (Time.get_ticks_msec() / 1000.0) - session_start_time,
		"label": label,
		"x_e": position.x,
		"y_n": position.z,
		"z_up": position.y,
		"sample_index": samples.size(),
	}
	events.append(evt)
	print("[DataRecorder] Event marked: %s at sample %d" % [label, samples.size()])


## Clear all recorded data.
func clear() -> void:
	samples.clear()
	events.clear()
	_sample_index = 0
	total_distance = 0.0
	peak_reading = 0.0
	min_reading = INF
	max_reading = -INF
	_last_sample_pos = Vector3.ZERO
	session_id = ""


## Export session to directory. Returns the session directory path.
func export_session(base_path: String = "") -> String:
	if samples.is_empty():
		push_warning("[DataRecorder] No samples to export")
		return ""

	if base_path.is_empty():
		base_path = "user://sessions"

	var date_str := Time.get_datetime_string_from_system().replace(":", "-").replace("T", "_")
	var safe_scenario := session_scenario.replace(" ", "-").to_lower()
	var dir_name := "%s_%s" % [date_str, safe_scenario]
	var session_dir := base_path.path_join(dir_name)

	# Create directory
	DirAccess.make_dir_recursive_absolute(session_dir)

	# Export all files
	_export_session_json(session_dir)
	_export_samples_csv(session_dir)
	_export_samples_xyz(session_dir)
	_export_events_csv(session_dir)
	_export_survey_plan(session_dir)

	var full_path := ProjectSettings.globalize_path(session_dir)
	print("[DataRecorder] Session exported to: %s" % full_path)
	export_completed.emit(full_path)
	return full_path


## Export session metadata as JSON.
func _export_session_json(dir_path: String) -> void:
	var duration := 0.0
	if samples.size() > 0:
		duration = samples[-1].get("t", 0.0)

	var coverage_pct := 0.0  # Calculated by heatmap overlay if available

	var meta := {
		"session_id": session_id,
		"timestamp": Time.get_datetime_string_from_system(),
		"scenario": session_scenario,
		"instrument": session_instrument,
		"operator_mode": session_operator_mode,
		"survey_plan": session_survey_plan,
		"server_mode": "live" if SurveyManager.is_live else "mock",
		"stats": {
			"duration_s": duration,
			"distance_m": total_distance,
			"sample_count": samples.size(),
			"event_count": events.size(),
			"peak_reading": peak_reading,
			"min_reading": min_reading,
			"max_reading": max_reading,
		},
	}

	var file := FileAccess.open(dir_path.path_join("session.json"), FileAccess.WRITE)
	if file:
		file.store_string(JSON.stringify(meta, "  "))
		file.close()


## Export samples as Pathfinder-compatible CSV.
func _export_samples_csv(dir_path: String) -> void:
	var file := FileAccess.open(dir_path.path_join("samples.csv"), FileAccess.WRITE)
	if not file:
		return

	file.store_line("timestamp,x_east,y_north,z_up,heading_rad,speed_ms,line_id,instrument,reading,reading_nT,distance_m,xte_m,quality,latency_ms")

	var cumulative_dist := 0.0
	var prev_pos := Vector3.ZERO
	for i in range(samples.size()):
		var s: Dictionary = samples[i]
		var pos := Vector3(s.get("x_e", 0.0), s.get("y_n", 0.0), s.get("z_up", 0.0))
		if i > 0:
			cumulative_dist += pos.distance_to(prev_pos)
		prev_pos = pos

		var reading: float = s.get("reading", 0.0)
		var reading_nT: float = reading * 1e9 if session_instrument == "mag_gradiometer" else reading

		file.store_line("%.4f,%.3f,%.3f,%.3f,%.4f,%.3f,%d,%s,%.6e,%.3f,%.3f,%.3f,%d,%.1f" % [
			s.get("t", 0.0),
			s.get("x_e", 0.0),
			s.get("y_n", 0.0),
			s.get("z_up", 0.0),
			s.get("heading", 0.0),
			s.get("speed", 0.0),
			s.get("line_id", -1),
			s.get("instrument", "mag"),
			reading,
			reading_nT,
			cumulative_dist,
			s.get("xte", -1.0),
			s.get("quality", 1),
			s.get("latency_ms", 0.0),
		])
	file.close()


## Export in XYZ format (Geosoft/Surfer/QGIS compatible).
func _export_samples_xyz(dir_path: String) -> void:
	var file := FileAccess.open(dir_path.path_join("samples.xyz"), FileAccess.WRITE)
	if not file:
		return

	file.store_line("/ GeoSim Export — scenario: %s — %s" % [
		session_scenario, Time.get_datetime_string_from_system()])
	file.store_line("/ East, North, Elevation, Reading, Quality")

	for s in samples:
		var reading: float = s.get("reading", 0.0)
		var reading_display: float = reading * 1e9 if session_instrument == "mag_gradiometer" else reading

		file.store_line("%.1f  %.1f  %.3f  %.4f  %d" % [
			s.get("x_e", 0.0) + CoordUtil.UTM_EASTING_OFFSET,
			s.get("y_n", 0.0) + CoordUtil.UTM_NORTHING_OFFSET,
			s.get("z_up", 0.0),
			reading_display,
			s.get("quality", 1),
		])
	file.close()


## Export events CSV.
func _export_events_csv(dir_path: String) -> void:
	if events.is_empty():
		return

	var file := FileAccess.open(dir_path.path_join("events.csv"), FileAccess.WRITE)
	if not file:
		return

	file.store_line("timestamp,label,x_east,y_north,z_up,sample_index")
	for e in events:
		file.store_line("%.4f,%s,%.3f,%.3f,%.3f,%d" % [
			e.get("t", 0.0),
			e.get("label", ""),
			e.get("x_e", 0.0),
			e.get("y_n", 0.0),
			e.get("z_up", 0.0),
			e.get("sample_index", 0),
		])
	file.close()


## Export survey plan as JSON.
func _export_survey_plan(dir_path: String) -> void:
	if session_survey_plan.is_empty():
		return

	var file := FileAccess.open(dir_path.path_join("survey_plan.json"), FileAccess.WRITE)
	if file:
		file.store_string(JSON.stringify(session_survey_plan, "  "))
		file.close()


## Generate a short unique session ID.
func _generate_session_id() -> String:
	var t := Time.get_ticks_msec()
	return "%x" % t


## Get session statistics as a Dictionary.
func get_stats() -> Dictionary:
	var duration := 0.0
	if samples.size() > 0:
		duration = samples[-1].get("t", 0.0)

	return {
		"duration_s": duration,
		"distance_m": total_distance,
		"sample_count": samples.size(),
		"event_count": events.size(),
		"peak_reading": peak_reading,
		"min_reading": min_reading if min_reading != INF else 0.0,
		"max_reading": max_reading if max_reading != -INF else 0.0,
	}
