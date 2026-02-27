## Borehole survey HUD — depth log display with reading vs depth plot.
##
## Shows a well-log-style display: reading on X axis, depth on Y axis
## (increasing downward), cable tension, winch speed, and probe status.
extends Control

var _log_panel: PanelContainer
var _depth_label: Label
var _speed_label: Label
var _tension_label: Label
var _mode_label: Label
var _reading_label: Label
var _log_chart: Control  # Custom draw for depth log

## Chart data
var _log_entries: Array[Dictionary] = []
var _max_reading := 1e-4
var _chart_rect := Rect2()

## Dual-track data (MIT + ERT)
var _mit_inphase: Array[Dictionary] = []
var _mit_quad: Array[Dictionary] = []
var _ert_log: Array[Dictionary] = []
var _max_mit := 1e-4
var _max_ert := 100.0
var _min_ert := 1.0

## Probe status labels
var _temp_label: Label
var _tilt_label: Label
var _cycle_label: Label
var _qc_label: Label


func _ready() -> void:
	mouse_filter = Control.MOUSE_FILTER_IGNORE
	visible = false
	_create_ui()
	SurveyManager.state_changed.connect(_on_state_changed)


func _on_state_changed(new_state: SurveyManager.State) -> void:
	visible = (new_state == SurveyManager.State.HIRT_SURVEY)


func _create_ui() -> void:
	# Left panel: depth log chart
	_log_panel = PanelContainer.new()
	_log_panel.set_anchors_preset(PRESET_LEFT_WIDE)
	_log_panel.offset_left = 12
	_log_panel.offset_right = 320
	_log_panel.offset_top = 12
	_log_panel.offset_bottom = -12

	var style := StyleBoxFlat.new()
	style.bg_color = Color(0.0, 0.0, 0.0, 0.7)
	style.corner_radius_top_left = 6
	style.corner_radius_top_right = 6
	style.corner_radius_bottom_left = 6
	style.corner_radius_bottom_right = 6
	style.content_margin_left = 10
	style.content_margin_right = 10
	style.content_margin_top = 8
	style.content_margin_bottom = 8
	_log_panel.add_theme_stylebox_override("panel", style)
	add_child(_log_panel)

	var vbox := VBoxContainer.new()
	vbox.add_theme_constant_override("separation", 6)
	_log_panel.add_child(vbox)

	# Title
	var title := Label.new()
	title.text = "Borehole Log  [MIT | ERT]"
	title.add_theme_font_size_override("font_size", 16)
	title.add_theme_color_override("font_color", Color(0.8, 0.85, 0.9))
	vbox.add_child(title)

	# Status row
	var status_row := HBoxContainer.new()
	status_row.add_theme_constant_override("separation", 10)
	vbox.add_child(status_row)

	_depth_label = Label.new()
	_depth_label.text = "Depth: 0.00 m"
	_depth_label.add_theme_font_size_override("font_size", 18)
	_depth_label.add_theme_color_override("font_color", Color.WHITE)
	status_row.add_child(_depth_label)

	_reading_label = Label.new()
	_reading_label.text = "Reading: --"
	_reading_label.add_theme_font_size_override("font_size", 14)
	_reading_label.add_theme_color_override("font_color", Color.LIGHT_GRAY)
	status_row.add_child(_reading_label)

	# Info row
	var info_row := HBoxContainer.new()
	info_row.add_theme_constant_override("separation", 10)
	vbox.add_child(info_row)

	_speed_label = Label.new()
	_speed_label.text = "Speed: 0.10 m/s"
	_speed_label.add_theme_font_size_override("font_size", 12)
	_speed_label.add_theme_color_override("font_color", Color.LIGHT_GRAY)
	info_row.add_child(_speed_label)

	_tension_label = Label.new()
	_tension_label.text = "Tension: 1.00"
	_tension_label.add_theme_font_size_override("font_size", 12)
	_tension_label.add_theme_color_override("font_color", Color.GREEN)
	info_row.add_child(_tension_label)

	_mode_label = Label.new()
	_mode_label.text = "Mode: Stop-and-Go"
	_mode_label.add_theme_font_size_override("font_size", 12)
	_mode_label.add_theme_color_override("font_color", Color(0.5, 0.55, 0.6))
	info_row.add_child(_mode_label)

	# Probe status row
	var probe_row := HBoxContainer.new()
	probe_row.add_theme_constant_override("separation", 10)
	vbox.add_child(probe_row)

	_temp_label = Label.new()
	_temp_label.text = "Temp: --°C"
	_temp_label.add_theme_font_size_override("font_size", 11)
	_temp_label.add_theme_color_override("font_color", Color(0.6, 0.65, 0.7))
	probe_row.add_child(_temp_label)

	_tilt_label = Label.new()
	_tilt_label.text = "Tilt: --°"
	_tilt_label.add_theme_font_size_override("font_size", 11)
	_tilt_label.add_theme_color_override("font_color", Color(0.6, 0.65, 0.7))
	probe_row.add_child(_tilt_label)

	_cycle_label = Label.new()
	_cycle_label.text = "Cycle: --"
	_cycle_label.add_theme_font_size_override("font_size", 11)
	_cycle_label.add_theme_color_override("font_color", Color(0.5, 0.55, 0.6))
	probe_row.add_child(_cycle_label)

	_qc_label = Label.new()
	_qc_label.text = "QC: --"
	_qc_label.add_theme_font_size_override("font_size", 11)
	_qc_label.add_theme_color_override("font_color", Color.GREEN)
	probe_row.add_child(_qc_label)

	# Log chart area (custom draw)
	_log_chart = Control.new()
	_log_chart.custom_minimum_size = Vector2(280, 400)
	_log_chart.size_flags_vertical = Control.SIZE_EXPAND_FILL
	_log_chart.draw.connect(_draw_log_chart)
	vbox.add_child(_log_chart)

	# Controls hint
	var hint := Label.new()
	hint.text = "W=Descend  S=Ascend  Q/E=Speed  Space=Mark"
	hint.add_theme_font_size_override("font_size", 11)
	hint.add_theme_color_override("font_color", Color(0.35, 0.4, 0.45))
	vbox.add_child(hint)


## Update display with current probe state.
func update_borehole(depth: float, speed: float, tension: float,
		mode: String, reading: float) -> void:
	if _depth_label:
		_depth_label.text = "Depth: %.2f m" % depth
	if _speed_label:
		_speed_label.text = "Speed: %.2f m/s" % speed
	if _tension_label:
		_tension_label.text = "Tension: %.2f" % tension
		if tension > 0.8:
			_tension_label.add_theme_color_override("font_color", Color.GREEN)
		elif tension > 0.5:
			_tension_label.add_theme_color_override("font_color", Color.YELLOW)
		else:
			_tension_label.add_theme_color_override("font_color", Color.RED)
	if _mode_label:
		_mode_label.text = "Mode: %s" % mode
	if _reading_label:
		_reading_label.text = "Reading: %.2e" % reading


## Add a measurement point to the log chart.
func add_log_point(depth: float, reading: float) -> void:
	_log_entries.append({"depth": depth, "reading": reading})
	_max_reading = max(_max_reading, abs(reading) * 1.2)
	if _log_chart:
		_log_chart.queue_redraw()


## Add MIT (EM induction) measurement to dual-track display.
func add_mit_point(depth: float, inphase: float, quadrature: float) -> void:
	_mit_inphase.append({"depth": depth, "value": inphase})
	_mit_quad.append({"depth": depth, "value": quadrature})
	_max_mit = max(_max_mit, abs(inphase) * 1.2, abs(quadrature) * 1.2)
	if _log_chart:
		_log_chart.queue_redraw()


## Add ERT (resistivity) measurement to dual-track display.
func add_ert_point(depth: float, resistivity: float) -> void:
	_ert_log.append({"depth": depth, "value": resistivity})
	_max_ert = max(_max_ert, resistivity * 1.2)
	_min_ert = min(_min_ert, max(resistivity * 0.8, 0.1))
	if _log_chart:
		_log_chart.queue_redraw()


## Update probe status display.
func update_probe_status(temperature: float, tilt: float, cycle: String, qc_pass: bool) -> void:
	if _temp_label:
		_temp_label.text = "Temp: %.1f°C" % temperature
	if _tilt_label:
		_tilt_label.text = "Tilt: %.1f°" % tilt
		if tilt > 5.0:
			_tilt_label.add_theme_color_override("font_color", Color.RED)
		else:
			_tilt_label.add_theme_color_override("font_color", Color(0.6, 0.65, 0.7))
	if _cycle_label:
		_cycle_label.text = "Cycle: %s" % cycle
	if _qc_label:
		_qc_label.text = "QC: PASS" if qc_pass else "QC: FAIL"
		_qc_label.add_theme_color_override("font_color", Color.GREEN if qc_pass else Color.RED)


func _draw_log_chart() -> void:
	if not _log_chart:
		return

	var w := _log_chart.size.x
	var h := _log_chart.size.y

	# Background
	_log_chart.draw_rect(Rect2(Vector2.ZERO, _log_chart.size), Color(0.08, 0.08, 0.1, 0.9))

	var margin := 30.0
	var chart_h := h - margin * 2
	var mid_x := w / 2.0

	# Dual-track separator
	_log_chart.draw_line(Vector2(mid_x, margin - 10), Vector2(mid_x, h - margin),
		Color(0.25, 0.3, 0.35), 1.0)

	# Track labels
	var font := ThemeDB.fallback_font
	var font_size := 9
	var label_color := Color(0.5, 0.55, 0.6)
	_log_chart.draw_string(font, Vector2(margin, margin - 2), "MIT",
		HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, Color(0.4, 0.8, 1.0))
	_log_chart.draw_string(font, Vector2(mid_x + 4, margin - 2), "ERT",
		HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, Color(1.0, 0.9, 0.4))

	# Vertical axes (depth, increasing downward)
	_log_chart.draw_line(Vector2(margin, margin), Vector2(margin, h - margin),
		Color(0.3, 0.35, 0.4), 1.0)
	_log_chart.draw_line(Vector2(mid_x + 4, margin), Vector2(mid_x + 4, h - margin),
		Color(0.3, 0.35, 0.4), 1.0)

	# Find max depth across all data
	var max_depth := 1.0
	for entry in _log_entries:
		max_depth = max(max_depth, entry["depth"])
	for entry in _mit_inphase:
		max_depth = max(max_depth, entry["depth"])
	for entry in _ert_log:
		max_depth = max(max_depth, entry["depth"])
	max_depth *= 1.1

	# --- Left track: MIT (In-phase + Quadrature) or legacy single-track ---
	var left_w := mid_x - margin - 4

	# Draw MIT in-phase (cyan)
	if not _mit_inphase.is_empty():
		var prev_pt := Vector2.ZERO
		for i in range(_mit_inphase.size()):
			var entry: Dictionary = _mit_inphase[i]
			var x: float = margin + (abs(entry["value"]) / _max_mit) * left_w
			var y: float = margin + (entry["depth"] / max_depth) * chart_h
			var pt := Vector2(x, y)
			_log_chart.draw_circle(pt, 2.0, Color(0.3, 0.8, 1.0))
			if i > 0:
				_log_chart.draw_line(prev_pt, pt, Color(0.2, 0.6, 0.8, 0.7), 1.5)
			prev_pt = pt

	# Draw MIT quadrature (orange)
	if not _mit_quad.is_empty():
		var prev_pt := Vector2.ZERO
		for i in range(_mit_quad.size()):
			var entry: Dictionary = _mit_quad[i]
			var x: float = margin + (abs(entry["value"]) / _max_mit) * left_w
			var y: float = margin + (entry["depth"] / max_depth) * chart_h
			var pt := Vector2(x, y)
			_log_chart.draw_circle(pt, 2.0, Color(1.0, 0.6, 0.2))
			if i > 0:
				_log_chart.draw_line(prev_pt, pt, Color(0.8, 0.5, 0.15, 0.7), 1.5)
			prev_pt = pt

	# Legacy single-track fallback (cyan)
	if _mit_inphase.is_empty() and not _log_entries.is_empty():
		var prev_pt := Vector2.ZERO
		for i in range(_log_entries.size()):
			var entry: Dictionary = _log_entries[i]
			var x: float = margin + (abs(entry["reading"]) / _max_reading) * left_w
			var y: float = margin + (entry["depth"] / max_depth) * chart_h
			var pt := Vector2(x, y)
			_log_chart.draw_circle(pt, 3.0, Color(0.2, 0.7, 0.9))
			if i > 0:
				_log_chart.draw_line(prev_pt, pt, Color(0.15, 0.5, 0.7, 0.7), 1.5)
			prev_pt = pt

	# --- Right track: ERT (log-scale resistivity, yellow) ---
	var right_w := w - mid_x - margin - 4
	if not _ert_log.is_empty():
		var log_min := log(_min_ert) / log(10.0)
		var log_max := log(_max_ert) / log(10.0)
		var log_range: float = max(log_max - log_min, 0.1)
		var prev_pt := Vector2.ZERO
		for i in range(_ert_log.size()):
			var entry: Dictionary = _ert_log[i]
			var log_val: float = (log(max(entry["value"], 0.1)) / log(10.0) - log_min) / log_range
			var x: float = mid_x + 4 + clampf(log_val, 0.0, 1.0) * right_w
			var y: float = margin + (entry["depth"] / max_depth) * chart_h
			var pt := Vector2(x, y)
			_log_chart.draw_circle(pt, 2.5, Color(1.0, 0.85, 0.3))
			if i > 0:
				_log_chart.draw_line(prev_pt, pt, Color(0.8, 0.7, 0.2, 0.7), 1.5)
			prev_pt = pt

	# Depth labels
	_log_chart.draw_string(font, Vector2(2, margin + 12), "0.0m",
		HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, label_color)
	_log_chart.draw_string(font, Vector2(2, h - margin + 12), "%.1fm" % max_depth,
		HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, label_color)
