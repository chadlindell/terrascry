## Post-survey review UI — replay, stats, export.
##
## Shows survey results after completion: statistics, heatmap overview,
## anomaly list, timeline scrubber for replay, and export buttons.
extends Control

var _stats_label: Label
var _anomaly_list: VBoxContainer
var _scrubber: HSlider
var _scrubber_label: Label
var _replay_playing: bool = false
var _replay_index: int = 0
var _replay_speed: float = 10.0  # samples per second
var _replay_accumulator: float = 0.0


func _ready() -> void:
	mouse_filter = Control.MOUSE_FILTER_STOP
	visible = false
	_create_ui()
	SurveyManager.state_changed.connect(_on_state_changed)


func _on_state_changed(new_state: SurveyManager.State) -> void:
	visible = (new_state == SurveyManager.State.POST_SURVEY)
	if visible:
		_populate_results()


func _process(delta: float) -> void:
	if not visible or not _replay_playing:
		return

	# Advance replay at consistent speed regardless of frame rate
	_replay_accumulator += delta * _replay_speed
	var steps := int(_replay_accumulator)
	_replay_accumulator -= steps
	_replay_index += steps

	if _replay_index >= DataRecorder.samples.size():
		_replay_playing = false
		_replay_index = DataRecorder.samples.size() - 1

	if _scrubber:
		_scrubber.set_value_no_signal(float(_replay_index))
	_update_scrubber_label()


func _create_ui() -> void:
	# Background
	var bg := ColorRect.new()
	bg.color = Color(0.05, 0.07, 0.1, 0.95)
	bg.set_anchors_preset(PRESET_FULL_RECT)
	bg.mouse_filter = Control.MOUSE_FILTER_STOP
	add_child(bg)

	var margin := MarginContainer.new()
	margin.set_anchors_preset(PRESET_FULL_RECT)
	margin.add_theme_constant_override("margin_left", 60)
	margin.add_theme_constant_override("margin_right", 60)
	margin.add_theme_constant_override("margin_top", 30)
	margin.add_theme_constant_override("margin_bottom", 30)
	add_child(margin)

	var outer_vbox := VBoxContainer.new()
	outer_vbox.add_theme_constant_override("separation", 14)
	margin.add_child(outer_vbox)

	# Header
	var title := Label.new()
	title.text = "Survey Complete"
	title.add_theme_font_size_override("font_size", 28)
	title.add_theme_color_override("font_color", Color(0.8, 0.9, 1.0))
	outer_vbox.add_child(title)

	# Two-column layout
	var columns := HBoxContainer.new()
	columns.add_theme_constant_override("separation", 30)
	columns.size_flags_vertical = Control.SIZE_EXPAND_FILL
	outer_vbox.add_child(columns)

	# Left: Stats
	var left_panel := PanelContainer.new()
	left_panel.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	var left_style := StyleBoxFlat.new()
	left_style.bg_color = Color(0.08, 0.1, 0.13, 0.9)
	left_style.corner_radius_top_left = 6
	left_style.corner_radius_top_right = 6
	left_style.corner_radius_bottom_left = 6
	left_style.corner_radius_bottom_right = 6
	left_style.content_margin_left = 16
	left_style.content_margin_right = 16
	left_style.content_margin_top = 14
	left_style.content_margin_bottom = 14
	left_panel.add_theme_stylebox_override("panel", left_style)
	columns.add_child(left_panel)

	_stats_label = Label.new()
	_stats_label.text = "Loading results..."
	_stats_label.add_theme_font_size_override("font_size", 14)
	_stats_label.add_theme_color_override("font_color", Color(0.7, 0.75, 0.8))
	_stats_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	left_panel.add_child(_stats_label)

	# Right: Anomaly list
	var right_panel := PanelContainer.new()
	right_panel.custom_minimum_size.x = 300
	right_panel.add_theme_stylebox_override("panel", left_style.duplicate())
	columns.add_child(right_panel)

	var right_vbox := VBoxContainer.new()
	right_vbox.add_theme_constant_override("separation", 8)
	right_panel.add_child(right_vbox)

	var anomaly_title := Label.new()
	anomaly_title.text = "Detected Anomalies"
	anomaly_title.add_theme_font_size_override("font_size", 16)
	anomaly_title.add_theme_color_override("font_color", Color(0.8, 0.85, 0.9))
	right_vbox.add_child(anomaly_title)

	var anomaly_scroll := ScrollContainer.new()
	anomaly_scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	right_vbox.add_child(anomaly_scroll)

	_anomaly_list = VBoxContainer.new()
	_anomaly_list.add_theme_constant_override("separation", 4)
	anomaly_scroll.add_child(_anomaly_list)

	# Timeline scrubber
	var scrubber_row := HBoxContainer.new()
	scrubber_row.add_theme_constant_override("separation", 10)
	outer_vbox.add_child(scrubber_row)

	var play_btn := Button.new()
	play_btn.text = "Play"
	play_btn.custom_minimum_size.x = 60
	play_btn.pressed.connect(func(): _replay_playing = not _replay_playing; play_btn.text = "Pause" if _replay_playing else "Play")
	scrubber_row.add_child(play_btn)

	_scrubber = HSlider.new()
	_scrubber.min_value = 0
	_scrubber.max_value = 1
	_scrubber.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_scrubber.value_changed.connect(func(v): _replay_index = int(v); _update_scrubber_label())
	scrubber_row.add_child(_scrubber)

	_scrubber_label = Label.new()
	_scrubber_label.text = "0 / 0"
	_scrubber_label.custom_minimum_size.x = 100
	_scrubber_label.add_theme_font_size_override("font_size", 13)
	scrubber_row.add_child(_scrubber_label)

	# Bottom buttons
	var btn_row := HBoxContainer.new()
	btn_row.add_theme_constant_override("separation", 12)
	btn_row.alignment = BoxContainer.ALIGNMENT_END
	outer_vbox.add_child(btn_row)

	var export_btn := Button.new()
	export_btn.text = "Export Data"
	export_btn.custom_minimum_size = Vector2(150, 40)
	export_btn.add_theme_font_size_override("font_size", 16)
	export_btn.pressed.connect(_on_export)
	btn_row.add_child(export_btn)

	var menu_btn := Button.new()
	menu_btn.text = "Main Menu"
	menu_btn.custom_minimum_size = Vector2(150, 40)
	menu_btn.add_theme_font_size_override("font_size", 16)
	menu_btn.pressed.connect(func(): SurveyManager.transition(SurveyManager.State.MAIN_MENU))
	btn_row.add_child(menu_btn)


func _populate_results() -> void:
	# Reset replay state from any previous session
	_replay_playing = false
	_replay_index = 0
	_replay_accumulator = 0.0

	var stats := DataRecorder.get_stats()
	var duration: float = stats.get("duration_s", 0.0)

	var text := "Survey Statistics\n\n"
	text += "Scenario: %s\n" % DataRecorder.session_scenario
	text += "Instrument: %s\n" % DataRecorder.session_instrument
	text += "Operator: %s\n\n" % DataRecorder.session_operator_mode
	text += "Duration: %d:%02d\n" % [int(duration) / 60, int(duration) % 60]
	text += "Distance: %.1f m\n" % stats.get("distance_m", 0.0)
	text += "Samples: %d\n" % stats.get("sample_count", 0)
	text += "Events: %d\n\n" % stats.get("event_count", 0)

	var peak: float = stats.get("peak_reading", 0.0)
	var min_r: float = stats.get("min_reading", 0.0)
	var max_r: float = stats.get("max_reading", 0.0)

	if DataRecorder.session_instrument == "mag_gradiometer":
		text += "Peak gradient: %.1f nT\n" % (peak * 1e9)
		text += "Range: %.1f to %.1f nT\n" % [min_r * 1e9, max_r * 1e9]
	else:
		text += "Peak reading: %.4f\n" % peak
		text += "Range: %.4f to %.4f\n" % [min_r, max_r]

	text += "\nServer: %s\n" % ("Live" if SurveyManager.is_live else "Mock")

	_stats_label.text = text

	# Setup scrubber
	if DataRecorder.samples.size() > 0:
		_scrubber.max_value = DataRecorder.samples.size() - 1
		_scrubber.value = 0
		_replay_index = 0

	# Find anomalies (peaks in readings)
	_populate_anomalies()


func _populate_anomalies() -> void:
	for child in _anomaly_list.get_children():
		child.queue_free()

	if DataRecorder.samples.is_empty():
		var label := Label.new()
		label.text = "No data recorded"
		label.add_theme_font_size_override("font_size", 13)
		_anomaly_list.add_child(label)
		return

	# Find local peaks (simple peak detection)
	var peaks: Array[Dictionary] = []
	var window := 10

	for i in range(window, DataRecorder.samples.size() - window):
		var reading: float = abs(DataRecorder.samples[i].get("reading", 0.0))
		var is_peak := true
		for j in range(i - window, i + window + 1):
			if j != i and abs(DataRecorder.samples[j].get("reading", 0.0)) > reading:
				is_peak = false
				break

		if is_peak and reading > abs(DataRecorder.peak_reading) * 0.1:
			peaks.append({
				"index": i,
				"reading": DataRecorder.samples[i].get("reading", 0.0),
				"x": DataRecorder.samples[i].get("x_e", 0.0),
				"y": DataRecorder.samples[i].get("y_n", 0.0),
			})

	# Sort by magnitude (descending)
	peaks.sort_custom(func(a, b): return abs(a["reading"]) > abs(b["reading"]))

	# Display top 10
	var count := mini(peaks.size(), 10)
	for i in range(count):
		var p: Dictionary = peaks[i]
		var reading_display: float = p["reading"]
		if DataRecorder.session_instrument == "mag_gradiometer":
			reading_display *= 1e9

		var label := Label.new()
		label.text = "#%d: %.1f at (%.1f, %.1f)" % [i + 1, reading_display, p["x"], p["y"]]
		label.add_theme_font_size_override("font_size", 12)
		label.add_theme_color_override("font_color", Color(0.7, 0.75, 0.8))
		_anomaly_list.add_child(label)

	if peaks.is_empty():
		var label := Label.new()
		label.text = "No significant anomalies detected"
		label.add_theme_font_size_override("font_size", 13)
		_anomaly_list.add_child(label)


func _update_scrubber_label() -> void:
	if _scrubber_label:
		_scrubber_label.text = "%d / %d" % [_replay_index, DataRecorder.samples.size()]


func _on_export() -> void:
	var path := DataRecorder.export_session()
	if not path.is_empty():
		_stats_label.text += "\n\nExported to:\n%s" % path
