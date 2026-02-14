## Heads-up display for survey readings.
##
## Shows gradient reading with color coding, peak value, scrolling strip
## chart, position (local + mock UTM), survey statistics, connection status,
## instrument panel, guidance overlay, QC dashboard, and coverage indicator.
## All UI is created programmatically for consistent layout.
extends Control

## UI elements — left panel (readings)
var _gradient_label: Label
var _peak_label: Label
var _position_label: Label
var _utm_label: Label
var _stats_label: Label
var _status_label: Label
var _scenario_label: Label
var _instrument_label: Label
var _strip_chart  # StripChart Control

## UI elements — right panel (guidance)
var _guidance_panel: PanelContainer
var _line_progress_label: Label
var _xte_bar: ProgressBar
var _xte_label: Label
var _speed_label: Label
var _heading_label: Label
var _coverage_label: Label

## UI elements — bottom-right panel (QC dashboard)
var _qc_panel: PanelContainer
var _latency_label: Label
var _dropout_label: Label
var _noise_label: Label
var _speed_stability_label: Label

## UI elements — recording indicator
var _record_indicator: Label

## UI elements — line QC toast and progress
var _line_qc_toast: Label
var _line_qc_toast_timer := 0.0
var _line_progress_bar: HBoxContainer
var _line_qc_results: Array = []  ## Array of {passed: bool} per completed line

## UI elements — drone progress panel
var _drone_panel: PanelContainer
var _drone_wp_label: Label
var _drone_eta_label: Label
var _drone_lines_label: Label
var _drone_battery_label: Label
var _drone_status_label: Label

## UI elements — toggle legend
var _toggle_legend: VBoxContainer
var _targets_visible := false

## UI elements — compass and crosshair
var _compass_label: Label
var _crosshair: Control
var _controls_label: Label

## Tracking
var _reading_count := 0
var _peak_gradient_nT := 0.0
var _recent_readings: PackedFloat64Array = []
var _recent_speeds: PackedFloat64Array = []


func _ready() -> void:
	mouse_filter = Control.MOUSE_FILTER_IGNORE
	_create_hud()

	PhysicsClient.connection_state_changed.connect(_on_connection_changed)
	SurveyManager.state_changed.connect(_on_state_changed)
	SurveyManager.instrument_changed.connect(_on_instrument_changed)
	_update_connection_status(PhysicsClient._connected)
	_update_visibility(SurveyManager.current_state)


func _process(delta: float) -> void:
	if not visible:
		return
	# Update compass bearing from operator heading
	if _compass_label and SurveyManager.active_operator:
		var heading: float = SurveyManager.active_operator.get("heading") if SurveyManager.active_operator.get("heading") != null else 0.0
		var deg := rad_to_deg(heading)
		var cardinal := _heading_to_cardinal(deg)
		_compass_label.text = "%s %.0f" % [cardinal, fposmod(deg, 360.0)]

	# Fade out line QC toast
	if _line_qc_toast and _line_qc_toast.visible:
		_line_qc_toast_timer -= delta
		if _line_qc_toast_timer <= 0:
			_line_qc_toast.visible = false
		elif _line_qc_toast_timer < 0.5:
			_line_qc_toast.modulate.a = _line_qc_toast_timer / 0.5

	# Update toggle legend
	if _toggle_legend:
		_update_toggle_legend()


func _on_state_changed(new_state: SurveyManager.State) -> void:
	_update_visibility(new_state)


func _update_visibility(state: SurveyManager.State) -> void:
	visible = (state == SurveyManager.State.SURVEYING or
		state == SurveyManager.State.HIRT_SURVEY)

	var is_surveying: bool = (state == SurveyManager.State.SURVEYING)
	var is_drone: bool = (SurveyManager.current_operator_mode == SurveyManager.OperatorMode.DRONE)
	var has_plan: bool = (not SurveyManager.survey_plan.is_empty() and
		SurveyManager.survey_plan.get("pattern", "freeform") != "freeform")

	if _guidance_panel:
		_guidance_panel.visible = is_surveying and has_plan and not is_drone

	if _drone_panel:
		_drone_panel.visible = is_surveying and is_drone

	if _controls_label:
		if is_drone:
			_controls_label.text = "WASD=Fly | Q/Z=Alt | E=Autopilot | C=Camera | Esc=Pause | H=Targets"
		else:
			_controls_label.text = "WASD=Move | Esc=Pause | Tab=Instrument | Space=Mark | V=VCO | H=Targets | T=Tool"


func _on_connection_changed(connected: bool) -> void:
	_update_connection_status(connected)


func _on_instrument_changed(instrument: SurveyManager.Instrument) -> void:
	if _instrument_label:
		_instrument_label.text = SurveyManager.instrument_name(instrument)
		_instrument_label.add_theme_color_override("font_color", Color.CYAN)

	# Reset peak for new instrument
	_peak_gradient_nT = 0.0
	if _peak_label:
		_peak_label.text = "Peak: -- "


func _update_connection_status(connected: bool) -> void:
	if not _status_label:
		return
	if connected:
		_status_label.text = "LIVE"
		_status_label.add_theme_color_override("font_color", Color.GREEN)
	elif PhysicsClient.is_mock_mode:
		_status_label.text = "MOCK"
		_status_label.add_theme_color_override("font_color", Color.ORANGE_RED)
	else:
		_status_label.text = "OFFLINE"
		_status_label.add_theme_color_override("font_color", Color.RED)


## Called by OperatorController via gradient_reading signal.
func update_reading(gradient: float, world_pos: Vector3) -> void:
	_reading_count += 1

	# Track recent readings for noise estimate
	_recent_readings.append(gradient)
	if _recent_readings.size() > 20:
		_recent_readings = _recent_readings.slice(_recent_readings.size() - 20)

	var gradient_nT := gradient * 1e9
	var units := SurveyManager.instrument_units(SurveyManager.current_instrument)

	if abs(gradient_nT) > abs(_peak_gradient_nT):
		_peak_gradient_nT = gradient_nT

	# Main reading display
	if _gradient_label:
		if SurveyManager.current_instrument == SurveyManager.Instrument.MAG_GRADIOMETER:
			_gradient_label.text = "%.1f %s" % [gradient_nT, units]
		else:
			_gradient_label.text = "%.4f %s" % [gradient, units]
		_gradient_label.add_theme_color_override("font_color", _gradient_color(gradient_nT))

	# Peak
	if _peak_label:
		_peak_label.text = "Peak: %.1f %s" % [_peak_gradient_nT, units]

	# Position (local)
	if _position_label:
		_position_label.text = CoordUtil.format_local(world_pos)

	# Position (UTM)
	if _utm_label:
		_utm_label.text = CoordUtil.format_utm(world_pos)

	# Strip chart
	if _strip_chart:
		_strip_chart.add_value(gradient_nT)

	# Recording indicator
	if _record_indicator:
		_record_indicator.visible = DataRecorder.is_recording
		if DataRecorder.is_recording:
			# Blink effect
			var t := Time.get_ticks_msec() / 500
			_record_indicator.modulate.a = 1.0 if t % 2 == 0 else 0.5


## Called by Main each frame with operator stats.
func update_stats(distance: float, sample_count: int) -> void:
	if _stats_label:
		_stats_label.text = "Dist: %.1f m | N=%d" % [distance, sample_count]


## Update guidance display for line-following surveys.
func update_guidance(line_index: int, total_lines: int, xte: float, speed: float,
		target_speed: float, heading_error: float) -> void:
	if _line_progress_label:
		_line_progress_label.text = "Line %d / %d" % [line_index + 1, total_lines]

	if _xte_bar:
		# XTE bar: center is 0, range ±1.0m mapped to 0-100
		var normalized_xte := clampf((xte + 1.0) / 2.0, 0.0, 1.0) * 100.0
		_xte_bar.value = normalized_xte
		# Color code
		var abs_xte := absf(xte)
		if abs_xte < 0.25:
			_xte_bar.modulate = Color.GREEN
		elif abs_xte < 0.5:
			_xte_bar.modulate = Color.YELLOW
		else:
			_xte_bar.modulate = Color.RED

	if _xte_label:
		_xte_label.text = "XTE: %.2f m" % xte

	if _speed_label:
		var speed_ratio := speed / target_speed if target_speed > 0 else 1.0
		_speed_label.text = "Speed: %.1f / %.1f m/s" % [speed, target_speed]
		if speed_ratio > 1.3 or speed_ratio < 0.7:
			_speed_label.add_theme_color_override("font_color", Color.RED)
		elif speed_ratio > 1.1 or speed_ratio < 0.9:
			_speed_label.add_theme_color_override("font_color", Color.YELLOW)
		else:
			_speed_label.add_theme_color_override("font_color", Color.GREEN)

	# Track recent speeds for stability
	_recent_speeds.append(speed)
	if _recent_speeds.size() > 100:
		_recent_speeds = _recent_speeds.slice(_recent_speeds.size() - 100)

	if _heading_label:
		if abs(heading_error) < 0.1:
			_heading_label.text = "Heading: ON"
			_heading_label.add_theme_color_override("font_color", Color.GREEN)
		else:
			_heading_label.text = "Heading: %.0f deg off" % rad_to_deg(heading_error)
			_heading_label.add_theme_color_override("font_color", Color.YELLOW)


## Update coverage percentage display.
func update_coverage(pct: float) -> void:
	if _coverage_label:
		_coverage_label.text = "Coverage: %.0f%%" % pct


## Update drone autopilot progress display.
func update_drone_progress(info: Dictionary) -> void:
	if not _drone_panel:
		return
	_drone_panel.visible = true

	if _drone_wp_label:
		var wp: int = info.get("waypoint", 0)
		var total: int = info.get("total_waypoints", 0)
		_drone_wp_label.text = "WP %d / %d" % [wp + 1, total]

	if _drone_eta_label:
		var eta: float = info.get("eta_seconds", 0.0)
		if eta > 0:
			_drone_eta_label.text = "ETA: %d:%02d" % [int(eta) / 60, int(eta) % 60]
		else:
			_drone_eta_label.text = "ETA: --"

	if _drone_lines_label:
		var done: int = info.get("lines_completed", 0)
		var total: int = info.get("total_lines", 0)
		if total > 0:
			_drone_lines_label.text = "Lines: %d / %d (%.0f%%)" % [done, total, float(done) / float(total) * 100.0]
		else:
			_drone_lines_label.text = "Lines: --"

	if _drone_battery_label:
		var batt: float = info.get("battery", 1.0)
		_drone_battery_label.text = "Battery: %.0f%%" % (batt * 100.0)
		if batt < 0.2:
			_drone_battery_label.add_theme_color_override("font_color", Color.RED)
		elif batt < 0.5:
			_drone_battery_label.add_theme_color_override("font_color", Color.YELLOW)
		else:
			_drone_battery_label.add_theme_color_override("font_color", Color.GREEN)

	if _drone_status_label:
		var ap: bool = info.get("autopilot", false)
		if ap:
			_drone_status_label.text = "AUTOPILOT"
			_drone_status_label.add_theme_color_override("font_color", Color.GREEN)
		else:
			_drone_status_label.text = "MANUAL"
			_drone_status_label.add_theme_color_override("font_color", Color.YELLOW)


## Update QC dashboard values.
func update_qc(latency_ms: float, dropout_count: int) -> void:
	if _latency_label:
		_latency_label.text = "Latency: %.1f ms" % latency_ms
		if latency_ms < 5.0:
			_latency_label.add_theme_color_override("font_color", Color.GREEN)
		elif latency_ms < 20.0:
			_latency_label.add_theme_color_override("font_color", Color.YELLOW)
		else:
			_latency_label.add_theme_color_override("font_color", Color.RED)

	if _dropout_label:
		_dropout_label.text = "Dropouts: %d" % dropout_count

	# Noise estimate (4th difference)
	if _noise_label and _recent_readings.size() >= 5:
		var n := _recent_readings.size()
		var fourth_diffs: PackedFloat64Array = []
		for i in range(4, n):
			var d4 := _recent_readings[i] - 4 * _recent_readings[i-1] + \
				6 * _recent_readings[i-2] - 4 * _recent_readings[i-3] + _recent_readings[i-4]
			fourth_diffs.append(abs(d4))

		if fourth_diffs.size() > 0:
			var noise_est := 0.0
			for d in fourth_diffs:
				noise_est += d
			noise_est = (noise_est / fourth_diffs.size()) * 1e9
			_noise_label.text = "Noise: %.2f nT" % noise_est

	# Speed stability
	if _speed_stability_label and _recent_speeds.size() >= 10:
		var mean_speed := 0.0
		for s in _recent_speeds:
			mean_speed += s
		mean_speed /= _recent_speeds.size()
		var variance := 0.0
		for s in _recent_speeds:
			variance += (s - mean_speed) * (s - mean_speed)
		variance /= _recent_speeds.size()
		_speed_stability_label.text = "Speed Var: %.3f" % variance


## Handle line completion QC result — show toast and update progress bar.
func on_line_completed(line_index: int, qc_passed: bool, stats: Dictionary) -> void:
	var total_lines := 0
	if SurveyManager.active_operator:
		var lines: Array = SurveyManager.active_operator.get("_survey_lines")
		if lines:
			total_lines = lines.size()

	# Show toast
	if _line_qc_toast:
		if qc_passed:
			_line_qc_toast.text = "Line %d/%d — PASS" % [line_index + 1, total_lines]
			_line_qc_toast.add_theme_color_override("font_color", Color.GREEN)
		else:
			var reason: String = stats.get("reason", "FAIL")
			if reason.is_empty():
				reason = "FAIL"
			_line_qc_toast.text = "Line %d/%d — FAIL: %s" % [line_index + 1, total_lines, reason]
			_line_qc_toast.add_theme_color_override("font_color", Color.RED)
		_line_qc_toast.visible = true
		_line_qc_toast.modulate.a = 1.0
		_line_qc_toast_timer = 3.0  # Show for 3 seconds

	# Update progress bar segments
	while _line_qc_results.size() <= line_index:
		_line_qc_results.append({"passed": false})
	_line_qc_results[line_index] = {"passed": qc_passed}
	_update_line_progress_bar(total_lines)


## Update the line progress bar with per-line pass/fail color segments.
func _update_line_progress_bar(total_lines: int) -> void:
	if not _line_progress_bar:
		return

	# Clear existing segments
	for child in _line_progress_bar.get_children():
		child.queue_free()

	if total_lines <= 0:
		return

	for i in range(total_lines):
		var segment := ColorRect.new()
		segment.custom_minimum_size = Vector2(maxf(180.0 / total_lines, 4), 8)
		segment.mouse_filter = Control.MOUSE_FILTER_IGNORE

		if i < _line_qc_results.size():
			if _line_qc_results[i].get("passed", false):
				segment.color = Color(0.2, 0.8, 0.3, 0.8)  # Green
			else:
				segment.color = Color(0.9, 0.2, 0.2, 0.8)  # Red
		else:
			segment.color = Color(0.3, 0.35, 0.4, 0.5)  # Gray (pending)

		_line_progress_bar.add_child(segment)


## Display the loaded scenario name.
func set_scenario_name(scenario_name: String) -> void:
	if _scenario_label:
		_scenario_label.text = scenario_name


## Update targets visibility indicator on HUD.
func set_targets_visible(visible_flag: bool) -> void:
	_targets_visible = visible_flag


func _update_toggle_legend() -> void:
	## Rebuild toggle legend labels to reflect current state.
	## Runs each frame but only updates text — no allocations after first build.
	var is_drone: bool = (SurveyManager.current_operator_mode == SurveyManager.OperatorMode.DRONE)

	# Gather toggle entries: [key, name, on/off]
	var entries: Array = []
	if is_drone:
		var ap := false
		var cam_mode := "3rd"
		if SurveyManager.active_operator:
			ap = SurveyManager.active_operator.get("autopilot_enabled") == true
			var tp = SurveyManager.active_operator.get("_third_person")
			cam_mode = "3rd" if tp else "1st"
		entries.append(["E", "Autopilot", ap])
		entries.append(["C", "Camera", cam_mode])
	else:
		entries.append(["V", "VCO", AudioManager.vco_enabled])
		var tool_vis := true
		if SurveyManager.active_operator:
			var iv = SurveyManager.active_operator.get("_instrument_visible")
			if iv != null:
				tool_vis = iv
		entries.append(["T", "Tool", tool_vis])
	entries.append(["H", "Targets", _targets_visible])

	# Ensure correct number of child labels
	while _toggle_legend.get_child_count() < entries.size():
		var lbl := Label.new()
		lbl.add_theme_font_size_override("font_size", 11)
		lbl.mouse_filter = Control.MOUSE_FILTER_IGNORE
		_toggle_legend.add_child(lbl)
	while _toggle_legend.get_child_count() > entries.size():
		_toggle_legend.get_child(_toggle_legend.get_child_count() - 1).queue_free()

	for i in range(entries.size()):
		var entry: Array = entries[i]
		var lbl: Label = _toggle_legend.get_child(i) as Label
		if not lbl:
			continue
		var key: String = entry[0]
		var label_name: String = entry[1]
		var value = entry[2]

		if value is bool:
			var state_str := "ON" if value else "OFF"
			lbl.text = "[%s] %s: %s" % [key, label_name, state_str]
			if value:
				lbl.add_theme_color_override("font_color", Color.CYAN)
			else:
				lbl.add_theme_color_override("font_color", Color(0.35, 0.4, 0.45))
		else:
			lbl.text = "[%s] %s: %s" % [key, label_name, str(value)]
			lbl.add_theme_color_override("font_color", Color(0.6, 0.7, 0.8))


func _gradient_color(value_nT: float) -> Color:
	var abs_val: float = absf(value_nT)
	if abs_val < 1.0:
		return Color.WHITE
	elif abs_val < 10.0:
		return Color.YELLOW
	elif abs_val < 100.0:
		return Color.ORANGE_RED
	else:
		return Color.RED


func _create_hud() -> void:
	# ---- LEFT PANEL: Readings ----
	var panel := PanelContainer.new()
	panel.name = "ReadingsPanel"
	panel.set_anchors_preset(PRESET_TOP_LEFT)
	panel.offset_left = 12
	panel.offset_top = 12
	panel.offset_right = 330
	panel.offset_bottom = 340

	var panel_style := StyleBoxFlat.new()
	panel_style.bg_color = Color(0.05, 0.08, 0.12, 0.65)
	panel_style.corner_radius_top_left = 6
	panel_style.corner_radius_top_right = 6
	panel_style.corner_radius_bottom_left = 6
	panel_style.corner_radius_bottom_right = 6
	panel_style.border_color = Color(0.3, 0.35, 0.45, 0.4)
	panel_style.border_width_left = 1
	panel_style.border_width_right = 1
	panel_style.border_width_top = 1
	panel_style.border_width_bottom = 1
	panel_style.content_margin_left = 10
	panel_style.content_margin_right = 10
	panel_style.content_margin_top = 8
	panel_style.content_margin_bottom = 8
	panel.add_theme_stylebox_override("panel", panel_style)
	panel.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(panel)

	var vbox := VBoxContainer.new()
	vbox.add_theme_constant_override("separation", 4)
	panel.add_child(vbox)

	# Row 1: Connection status + scenario + instrument
	var status_row := HBoxContainer.new()
	status_row.add_theme_constant_override("separation", 8)
	vbox.add_child(status_row)

	_status_label = Label.new()
	_status_label.text = "OFFLINE"
	_status_label.add_theme_font_size_override("font_size", 13)
	_status_label.add_theme_color_override("font_color", Color.RED)
	status_row.add_child(_status_label)

	_scenario_label = Label.new()
	_scenario_label.text = ""
	_scenario_label.add_theme_font_size_override("font_size", 13)
	_scenario_label.add_theme_color_override("font_color", Color.DIM_GRAY)
	status_row.add_child(_scenario_label)

	_instrument_label = Label.new()
	_instrument_label.text = SurveyManager.instrument_name(SurveyManager.current_instrument)
	_instrument_label.add_theme_font_size_override("font_size", 13)
	_instrument_label.add_theme_color_override("font_color", Color.CYAN)
	status_row.add_child(_instrument_label)

	# Row 2: Recording indicator
	_record_indicator = Label.new()
	_record_indicator.text = "REC"
	_record_indicator.add_theme_font_size_override("font_size", 14)
	_record_indicator.add_theme_color_override("font_color", Color.RED)
	_record_indicator.visible = false
	vbox.add_child(_record_indicator)

	# Row 3: Gradient reading (large) + peak
	var grad_row := HBoxContainer.new()
	grad_row.add_theme_constant_override("separation", 12)
	vbox.add_child(grad_row)

	_gradient_label = Label.new()
	_gradient_label.text = "-- nT"
	_gradient_label.add_theme_font_size_override("font_size", 24)
	_gradient_label.add_theme_color_override("font_color", Color.WHITE)
	grad_row.add_child(_gradient_label)

	_peak_label = Label.new()
	_peak_label.text = "Peak: -- nT"
	_peak_label.add_theme_font_size_override("font_size", 13)
	_peak_label.add_theme_color_override("font_color", Color.LIGHT_GRAY)
	_peak_label.size_flags_vertical = Control.SIZE_SHRINK_END
	grad_row.add_child(_peak_label)

	# Row 4: Strip chart
	var chart_script: Resource = load("res://scripts/strip_chart.gd")
	if chart_script:
		_strip_chart = Control.new()
		_strip_chart.set_script(chart_script)
		_strip_chart.custom_minimum_size = Vector2(290, 80)
		vbox.add_child(_strip_chart)

	# Row 5: Position (local)
	_position_label = Label.new()
	_position_label.text = "(--, --) m"
	_position_label.add_theme_font_size_override("font_size", 13)
	_position_label.add_theme_color_override("font_color", Color.LIGHT_GRAY)
	vbox.add_child(_position_label)

	# Row 6: Position (UTM)
	_utm_label = Label.new()
	_utm_label.text = ""
	_utm_label.add_theme_font_size_override("font_size", 11)
	_utm_label.add_theme_color_override("font_color", Color(0.4, 0.45, 0.5))
	vbox.add_child(_utm_label)

	# Row 7: Survey stats
	_stats_label = Label.new()
	_stats_label.text = "Dist: 0.0 m | N=0"
	_stats_label.add_theme_font_size_override("font_size", 13)
	_stats_label.add_theme_color_override("font_color", Color.LIGHT_GRAY)
	vbox.add_child(_stats_label)

	# Row 8: Coverage
	_coverage_label = Label.new()
	_coverage_label.text = ""
	_coverage_label.add_theme_font_size_override("font_size", 13)
	_coverage_label.add_theme_color_override("font_color", Color.LIGHT_GRAY)
	vbox.add_child(_coverage_label)

	# ---- RIGHT PANEL: Guidance ----
	_guidance_panel = PanelContainer.new()
	_guidance_panel.name = "GuidancePanel"
	_guidance_panel.set_anchors_preset(PRESET_TOP_RIGHT)
	_guidance_panel.offset_left = -250
	_guidance_panel.offset_top = 12
	_guidance_panel.offset_right = -12
	_guidance_panel.offset_bottom = 200
	_guidance_panel.visible = false

	var guidance_style := panel_style.duplicate()
	_guidance_panel.add_theme_stylebox_override("panel", guidance_style)
	_guidance_panel.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(_guidance_panel)

	var g_vbox := VBoxContainer.new()
	g_vbox.add_theme_constant_override("separation", 6)
	_guidance_panel.add_child(g_vbox)

	var g_title := Label.new()
	g_title.text = "Survey Guidance"
	g_title.add_theme_font_size_override("font_size", 14)
	g_title.add_theme_color_override("font_color", Color(0.6, 0.65, 0.7))
	g_vbox.add_child(g_title)

	_line_progress_label = Label.new()
	_line_progress_label.text = "Line -- / --"
	_line_progress_label.add_theme_font_size_override("font_size", 16)
	g_vbox.add_child(_line_progress_label)

	_xte_label = Label.new()
	_xte_label.text = "XTE: -- m"
	_xte_label.add_theme_font_size_override("font_size", 13)
	g_vbox.add_child(_xte_label)

	_xte_bar = ProgressBar.new()
	_xte_bar.min_value = 0
	_xte_bar.max_value = 100
	_xte_bar.value = 50
	_xte_bar.custom_minimum_size = Vector2(0, 12)
	_xte_bar.show_percentage = false
	g_vbox.add_child(_xte_bar)

	_speed_label = Label.new()
	_speed_label.text = "Speed: -- m/s"
	_speed_label.add_theme_font_size_override("font_size", 13)
	g_vbox.add_child(_speed_label)

	_heading_label = Label.new()
	_heading_label.text = "Heading: --"
	_heading_label.add_theme_font_size_override("font_size", 13)
	g_vbox.add_child(_heading_label)

	# Line QC progress bar (colored segments per completed line)
	_line_progress_bar = HBoxContainer.new()
	_line_progress_bar.add_theme_constant_override("separation", 2)
	g_vbox.add_child(_line_progress_bar)

	# ---- TOP-RIGHT: Line QC Toast ----
	_line_qc_toast = Label.new()
	_line_qc_toast.set_anchors_preset(PRESET_TOP_RIGHT)
	_line_qc_toast.offset_right = -12
	_line_qc_toast.offset_top = 220
	_line_qc_toast.offset_left = -280
	_line_qc_toast.text = ""
	_line_qc_toast.add_theme_font_size_override("font_size", 18)
	_line_qc_toast.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	_line_qc_toast.visible = false
	_line_qc_toast.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(_line_qc_toast)

	# (Toggle legend replaces individual VCO/Targets indicators)

	# ---- BOTTOM-RIGHT: QC Dashboard ----
	_qc_panel = PanelContainer.new()
	_qc_panel.name = "QCPanel"
	_qc_panel.set_anchors_preset(PRESET_BOTTOM_RIGHT)
	_qc_panel.offset_left = -220
	_qc_panel.offset_top = -130
	_qc_panel.offset_right = -12
	_qc_panel.offset_bottom = -12

	var qc_style := panel_style.duplicate()
	_qc_panel.add_theme_stylebox_override("panel", qc_style)
	_qc_panel.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(_qc_panel)

	var qc_vbox := VBoxContainer.new()
	qc_vbox.add_theme_constant_override("separation", 3)
	_qc_panel.add_child(qc_vbox)

	var qc_title := Label.new()
	qc_title.text = "QC Dashboard"
	qc_title.add_theme_font_size_override("font_size", 12)
	qc_title.add_theme_color_override("font_color", Color(0.5, 0.55, 0.6))
	qc_vbox.add_child(qc_title)

	_latency_label = Label.new()
	_latency_label.text = "Latency: -- ms"
	_latency_label.add_theme_font_size_override("font_size", 11)
	_latency_label.add_theme_color_override("font_color", Color.LIGHT_GRAY)
	qc_vbox.add_child(_latency_label)

	_dropout_label = Label.new()
	_dropout_label.text = "Dropouts: 0"
	_dropout_label.add_theme_font_size_override("font_size", 11)
	_dropout_label.add_theme_color_override("font_color", Color.LIGHT_GRAY)
	qc_vbox.add_child(_dropout_label)

	_noise_label = Label.new()
	_noise_label.text = "Noise: -- nT"
	_noise_label.add_theme_font_size_override("font_size", 11)
	_noise_label.add_theme_color_override("font_color", Color.LIGHT_GRAY)
	qc_vbox.add_child(_noise_label)

	_speed_stability_label = Label.new()
	_speed_stability_label.text = "Speed Var: --"
	_speed_stability_label.add_theme_font_size_override("font_size", 11)
	_speed_stability_label.add_theme_color_override("font_color", Color.LIGHT_GRAY)
	qc_vbox.add_child(_speed_stability_label)

	# ---- TOP-RIGHT: Drone Progress Panel ----
	_drone_panel = PanelContainer.new()
	_drone_panel.name = "DronePanel"
	_drone_panel.set_anchors_preset(PRESET_TOP_RIGHT)
	_drone_panel.offset_left = -250
	_drone_panel.offset_top = 12
	_drone_panel.offset_right = -12
	_drone_panel.offset_bottom = 180
	_drone_panel.visible = false

	var drone_style := panel_style.duplicate()
	_drone_panel.add_theme_stylebox_override("panel", drone_style)
	_drone_panel.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(_drone_panel)

	var d_vbox := VBoxContainer.new()
	d_vbox.add_theme_constant_override("separation", 4)
	_drone_panel.add_child(d_vbox)

	_drone_status_label = Label.new()
	_drone_status_label.text = "AUTOPILOT"
	_drone_status_label.add_theme_font_size_override("font_size", 16)
	_drone_status_label.add_theme_color_override("font_color", Color.GREEN)
	d_vbox.add_child(_drone_status_label)

	_drone_wp_label = Label.new()
	_drone_wp_label.text = "WP -- / --"
	_drone_wp_label.add_theme_font_size_override("font_size", 13)
	_drone_wp_label.add_theme_color_override("font_color", Color.LIGHT_GRAY)
	d_vbox.add_child(_drone_wp_label)

	_drone_lines_label = Label.new()
	_drone_lines_label.text = "Lines: --"
	_drone_lines_label.add_theme_font_size_override("font_size", 13)
	_drone_lines_label.add_theme_color_override("font_color", Color.LIGHT_GRAY)
	d_vbox.add_child(_drone_lines_label)

	_drone_eta_label = Label.new()
	_drone_eta_label.text = "ETA: --"
	_drone_eta_label.add_theme_font_size_override("font_size", 13)
	_drone_eta_label.add_theme_color_override("font_color", Color.LIGHT_GRAY)
	d_vbox.add_child(_drone_eta_label)

	_drone_battery_label = Label.new()
	_drone_battery_label.text = "Battery: 100%"
	_drone_battery_label.add_theme_font_size_override("font_size", 13)
	_drone_battery_label.add_theme_color_override("font_color", Color.GREEN)
	d_vbox.add_child(_drone_battery_label)


	# ---- BOTTOM-LEFT: Toggle legend ----
	_toggle_legend = VBoxContainer.new()
	_toggle_legend.name = "ToggleLegend"
	_toggle_legend.set_anchors_preset(PRESET_BOTTOM_LEFT)
	_toggle_legend.offset_left = 12
	_toggle_legend.offset_bottom = -30
	_toggle_legend.offset_top = -110
	_toggle_legend.add_theme_constant_override("separation", 1)
	_toggle_legend.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(_toggle_legend)

	# ---- BOTTOM-LEFT: Controls hint ----
	_controls_label = Label.new()
	_controls_label.set_anchors_preset(PRESET_BOTTOM_LEFT)
	_controls_label.offset_left = 12
	_controls_label.offset_bottom = -12
	_controls_label.offset_top = -28
	_controls_label.text = "WASD=Move | Esc=Pause | Tab=Instrument | Space=Mark | V=VCO | H=Targets | T=Tool"
	_controls_label.add_theme_font_size_override("font_size", 11)
	_controls_label.add_theme_color_override("font_color", Color(0.35, 0.4, 0.45))
	_controls_label.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(_controls_label)

	# ---- TOP-CENTER: Compass bearing ----
	_compass_label = Label.new()
	_compass_label.set_anchors_preset(PRESET_CENTER_TOP)
	_compass_label.offset_top = 12
	_compass_label.offset_left = -40
	_compass_label.offset_right = 40
	_compass_label.text = "N 0"
	_compass_label.add_theme_font_size_override("font_size", 16)
	_compass_label.add_theme_color_override("font_color", Color(0.8, 0.85, 0.9))
	_compass_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	_compass_label.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(_compass_label)

	# ---- CENTER: Crosshair ----
	_crosshair = Control.new()
	_crosshair.name = "Crosshair"
	_crosshair.set_anchors_preset(PRESET_CENTER)
	_crosshair.offset_left = -8
	_crosshair.offset_top = -8
	_crosshair.offset_right = 8
	_crosshair.offset_bottom = 8
	_crosshair.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_crosshair.draw.connect(_draw_crosshair)
	add_child(_crosshair)

	# ---- VIGNETTE: Subtle corner darkening ----
	var vignette := ColorRect.new()
	vignette.name = "Vignette"
	vignette.set_anchors_preset(PRESET_FULL_RECT)
	vignette.mouse_filter = Control.MOUSE_FILTER_IGNORE
	var vignette_shader := Shader.new()
	vignette_shader.code = """
shader_type canvas_item;
void fragment() {
	float d = length(UV - 0.5) * 1.6;
	float v = mix(0.0, 0.25, pow(d, 2.5));
	COLOR = vec4(0.0, 0.0, 0.0, v);
}
"""
	var vignette_mat := ShaderMaterial.new()
	vignette_mat.shader = vignette_shader
	vignette.material = vignette_mat
	add_child(vignette)


func _draw_crosshair() -> void:
	var center := _crosshair.size / 2.0
	var gap := 3.0
	var arm := 6.0
	var color := Color(1.0, 1.0, 1.0, 0.4)
	# Top
	_crosshair.draw_line(Vector2(center.x, center.y - gap - arm), Vector2(center.x, center.y - gap), color, 1.0)
	# Bottom
	_crosshair.draw_line(Vector2(center.x, center.y + gap), Vector2(center.x, center.y + gap + arm), color, 1.0)
	# Left
	_crosshair.draw_line(Vector2(center.x - gap - arm, center.y), Vector2(center.x - gap, center.y), color, 1.0)
	# Right
	_crosshair.draw_line(Vector2(center.x + gap, center.y), Vector2(center.x + gap + arm, center.y), color, 1.0)
	# Center dot
	_crosshair.draw_circle(center, 1.0, color)


func _heading_to_cardinal(deg: float) -> String:
	deg = fposmod(deg, 360.0)
	if deg < 22.5 or deg >= 337.5:
		return "N"
	elif deg < 67.5:
		return "NE"
	elif deg < 112.5:
		return "E"
	elif deg < 157.5:
		return "SE"
	elif deg < 202.5:
		return "S"
	elif deg < 247.5:
		return "SW"
	elif deg < 292.5:
		return "W"
	else:
		return "NW"
