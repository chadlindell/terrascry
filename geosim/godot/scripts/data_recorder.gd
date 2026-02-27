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

## Per-line QC statistics — keyed by line_id (int -> Dictionary)
var line_stats: Dictionary = {}

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

	# Accumulate per-line QC statistics
	var line_id: int = data.get("line_id", -1)
	if line_id >= 0:
		if not line_stats.has(line_id):
			line_stats[line_id] = {
				"sample_count": 0,
				"dropout_count": 0,
				"xte_sum": 0.0,
				"xte_max": 0.0,
				"speed_sum": 0.0,
				"speed_sq_sum": 0.0,
				"in_tolerance_count": 0,
			}
		var ls: Dictionary = line_stats[line_id]
		ls["sample_count"] += 1

		var xte: float = data.get("xte", -1.0)
		if xte >= 0:
			ls["xte_sum"] += xte
			ls["xte_max"] = maxf(ls["xte_max"], xte)
			if xte < 0.5:
				ls["in_tolerance_count"] += 1

		var speed: float = data.get("speed", 0.0)
		ls["speed_sum"] += speed
		ls["speed_sq_sum"] += speed * speed

		var quality: int = data.get("quality", 1)
		if quality == 0:
			ls["dropout_count"] += 1

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
	line_stats.clear()


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
	_export_multichannel_csv(session_dir)
	_export_samples_xyz(session_dir)
	_export_events_csv(session_dir)
	_export_survey_plan(session_dir)
	_export_geojson(session_dir)
	_export_qc_report(session_dir)

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


## Export multi-channel CSV matching Pathfinder firmware format.
## If raw data contains B_bottom/B_top/gradient arrays, writes one column per
## sensor pair channel. Converts Tesla to ADC counts (2.5e9 counts/T, ±32000).
func _export_multichannel_csv(dir_path: String) -> void:
	if samples.is_empty():
		return

	# Check if first sample has multi-channel raw data
	var first_raw: Dictionary = samples[0].get("reading_raw", {})
	var has_gradient: bool = first_raw.has("gradient")
	var has_B_bottom: bool = first_raw.has("B_bottom")

	if not has_gradient and not has_B_bottom:
		return  # No multi-channel data, skip

	var file := FileAccess.open(dir_path.path_join("samples_multichannel.csv"), FileAccess.WRITE)
	if not file:
		return

	# Determine number of channels from first sample
	var n_pairs := 0
	if has_gradient:
		var grad_arr: Array = first_raw.get("gradient", [])
		if grad_arr is Array:
			n_pairs = grad_arr.size()

	if n_pairs == 0:
		n_pairs = 4  # Default Pathfinder

	# Build header
	var header := "timestamp,lat,lon"
	for i in range(1, n_pairs + 1):
		header += ",g%d_top,g%d_bot,g%d_grad" % [i, i, i]
	file.store_line(header)

	# ADC conversion constants
	var adc_scale := 2.5e9  # counts per Tesla
	var adc_sat := 32000

	for s in samples:
		var raw: Dictionary = s.get("reading_raw", {})
		var t_ms: int = int(float(s.get("t", 0.0)) * 1000)
		var lat: float = float(s.get("y_n", 0.0))
		var lon: float = float(s.get("x_e", 0.0))
		var line := "%d,%.6f,%.6f" % [t_ms, lat, lon]

		var b_bot_arr: Array = raw.get("B_bottom", [])
		var b_top_arr: Array = raw.get("B_top", [])
		var grad_arr: Array = raw.get("gradient", [])

		for i in range(n_pairs):
			var b_top := 0.0
			var b_bot := 0.0
			var grad := 0.0

			if i < b_top_arr.size():
				b_top = b_top_arr[i]
			if i < b_bot_arr.size():
				b_bot = b_bot_arr[i]
			if i < grad_arr.size():
				grad = grad_arr[i]

			# Convert to ADC counts with saturation
			var top_adc := clampi(int(round(b_top * adc_scale)), -adc_sat, adc_sat)
			var bot_adc := clampi(int(round(b_bot * adc_scale)), -adc_sat, adc_sat)
			var grad_adc := bot_adc - top_adc

			line += ",%d,%d,%d" % [top_adc, bot_adc, grad_adc]

		file.store_line(line)
	file.close()


## Export track as GeoJSON for GIS visualization (QGIS/Google Earth).
func _export_geojson(dir_path: String) -> void:
	if samples.is_empty():
		return

	var file := FileAccess.open(dir_path.path_join("track.geojson"), FileAccess.WRITE)
	if not file:
		return

	# Build coordinates array for LineString (UTM-offset positions)
	var coords: Array = []
	for s in samples:
		var east: float = float(s.get("x_e", 0.0)) + CoordUtil.UTM_EASTING_OFFSET
		var north: float = float(s.get("y_n", 0.0)) + CoordUtil.UTM_NORTHING_OFFSET
		var elev: float = float(s.get("z_up", 0.0))
		coords.append([east, north, elev])

	# Build event marker points
	var point_features: Array = []
	for e in events:
		var east: float = float(e.get("x_e", 0.0)) + CoordUtil.UTM_EASTING_OFFSET
		var north: float = float(e.get("y_n", 0.0)) + CoordUtil.UTM_NORTHING_OFFSET
		var elev: float = float(e.get("z_up", 0.0))
		point_features.append({
			"type": "Feature",
			"geometry": {
				"type": "Point",
				"coordinates": [east, north, elev],
			},
			"properties": {
				"label": e.get("label", "mark"),
				"timestamp": e.get("t", 0.0),
				"sample_index": e.get("sample_index", 0),
			},
		})

	var features: Array = [{
		"type": "Feature",
		"geometry": {
			"type": "LineString",
			"coordinates": coords,
		},
		"properties": {
			"session_id": session_id,
			"scenario": session_scenario,
			"instrument": session_instrument,
			"sample_count": samples.size(),
		},
	}]
	features.append_array(point_features)

	var geojson := {
		"type": "FeatureCollection",
		"features": features,
	}

	file.store_string(JSON.stringify(geojson, "  "))
	file.close()


## Export per-line QC summary report as JSON.
func _export_qc_report(dir_path: String) -> void:
	if samples.is_empty():
		return

	var file := FileAccess.open(dir_path.path_join("qc_report.json"), FileAccess.WRITE)
	if not file:
		return

	# Compute per-line statistics
	var line_data: Dictionary = {}
	var quality_counts := {0: 0, 1: 0, 2: 0, 3: 0}

	for s in samples:
		var lid: int = s.get("line_id", -1)
		var lid_str := str(lid)
		if not line_data.has(lid_str):
			line_data[lid_str] = {
				"sample_count": 0,
				"dropout_count": 0,
				"xte_sum": 0.0,
				"xte_max": 0.0,
				"speed_sum": 0.0,
				"speed_sq_sum": 0.0,
			}

		var ld: Dictionary = line_data[lid_str]
		ld["sample_count"] += 1

		var xte: float = s.get("xte", -1.0)
		if xte >= 0:
			ld["xte_sum"] += xte
			ld["xte_max"] = maxf(ld["xte_max"], xte)

		var speed: float = s.get("speed", 0.0)
		ld["speed_sum"] += speed
		ld["speed_sq_sum"] += speed * speed

		var quality: int = s.get("quality", 1)
		if quality == 0:
			ld["dropout_count"] += 1
		if quality_counts.has(quality):
			quality_counts[quality] += 1

	# Compute derived stats per line
	var per_line: Array = []
	for lid_str in line_data:
		var ld: Dictionary = line_data[lid_str]
		var n: int = ld["sample_count"]
		var mean_xte: float = float(ld["xte_sum"]) / maxf(n, 1)
		var speed_mean: float = float(ld["speed_sum"]) / maxf(n, 1)
		var speed_var: float = (float(ld["speed_sq_sum"]) / maxf(n, 1)) - speed_mean * speed_mean
		var speed_std: float = sqrt(maxf(speed_var, 0.0))
		var dropout_rate: float = float(ld["dropout_count"]) / maxf(n, 1)

		# Coverage: fraction of samples with XTE < 0.5m
		var in_tolerance := 0
		for s in samples:
			if str(s.get("line_id", -1)) == lid_str:
				if s.get("xte", -1.0) >= 0 and s.get("xte", 999.0) < 0.5:
					in_tolerance += 1
		var coverage_pct: float = (float(in_tolerance) / maxf(n, 1)) * 100.0

		per_line.append({
			"line_id": int(lid_str),
			"sample_count": n,
			"dropout_count": ld["dropout_count"],
			"mean_xte": mean_xte,
			"max_xte": ld["xte_max"],
			"speed_mean": speed_mean,
			"speed_std": speed_std,
			"dropout_rate": dropout_rate,
			"coverage_pct": coverage_pct,
		})

	# Overall stats
	var duration := 0.0
	if samples.size() > 0:
		duration = samples[-1].get("t", 0.0)

	var total_samples: int = samples.size()
	var quality_breakdown := {}
	for q in quality_counts:
		var pct := (float(quality_counts[q]) / maxf(total_samples, 1)) * 100.0
		quality_breakdown[str(q)] = {"count": quality_counts[q], "percent": pct}

	var report := {
		"session_id": session_id,
		"scenario": session_scenario,
		"instrument": session_instrument,
		"overall": {
			"duration_s": duration,
			"total_distance_m": total_distance,
			"sample_count": total_samples,
			"peak_reading": peak_reading,
			"mean_reading": _compute_mean_reading(),
		},
		"quality_breakdown": quality_breakdown,
		"per_line": per_line,
	}

	file.store_string(JSON.stringify(report, "  "))
	file.close()


## Evaluate QC for a completed line. Returns {passed: bool, stats: Dictionary, reason: String}.
func evaluate_line_qc(line_id: int) -> Dictionary:
	if not line_stats.has(line_id):
		return {"passed": false, "stats": {}, "reason": "no data"}

	var ls: Dictionary = line_stats[line_id]
	var n: int = ls["sample_count"]
	if n == 0:
		return {"passed": false, "stats": {}, "reason": "no samples"}

	var mean_xte: float = float(ls["xte_sum"]) / n
	var speed_mean: float = float(ls["speed_sum"]) / n
	var speed_var: float = (float(ls["speed_sq_sum"]) / n) - speed_mean * speed_mean
	var speed_std: float = sqrt(maxf(speed_var, 0.0))
	var coverage_pct: float = (float(ls["in_tolerance_count"]) / n) * 100.0
	var dropout_rate: float = (float(ls["dropout_count"]) / n) * 100.0

	var stats := {
		"sample_count": n,
		"mean_xte": mean_xte,
		"max_xte": ls["xte_max"],
		"speed_mean": speed_mean,
		"speed_std": speed_std,
		"coverage_pct": coverage_pct,
		"dropout_rate": dropout_rate,
	}

	# QC criteria
	var passed := true
	var reason := ""

	if mean_xte > 0.3:
		passed = false
		reason = "XTE %.2fm" % mean_xte
	elif speed_std > 0.3:
		passed = false
		reason = "Speed var %.2f" % speed_std
	elif coverage_pct < 90.0:
		passed = false
		reason = "Coverage %.0f%%" % coverage_pct
	elif dropout_rate > 5.0:
		passed = false
		reason = "Dropouts %.0f%%" % dropout_rate

	return {"passed": passed, "stats": stats, "reason": reason}


## Compute mean reading across all samples.
func _compute_mean_reading() -> float:
	if samples.is_empty():
		return 0.0
	var total := 0.0
	for s in samples:
		total += s.get("reading", 0.0)
	return total / samples.size()


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
