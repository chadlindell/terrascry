## Survey planning UI — configure survey parameters before starting.
##
## Allows user to select pattern type, line spacing, direction, operator mode,
## instrument, and speed target. Shows planned lines on terrain preview.
extends Control

## References
var _pattern_option: OptionButton
var _spacing_slider: HSlider
var _spacing_label: Label
var _direction_option: OptionButton
var _mode_option: OptionButton
var _instrument_option: OptionButton
var _speed_slider: HSlider
var _speed_label: Label
var _tie_check: CheckButton
var _tie_spacing_slider: HSlider
var _tie_spacing_label: Label
var _info_label: Label
var _scenario_label: Label

## Environment controls
var _temp_slider: HSlider
var _temp_label: Label
var _moisture_slider: HSlider
var _moisture_label: Label
var _season_option: OptionButton
var _weather_option: OptionButton
var _ground_option: OptionButton

## Planned survey lines (Godot coordinates)
var _planned_lines: Array = []


func _ready() -> void:
	mouse_filter = Control.MOUSE_FILTER_STOP
	visible = false
	_create_ui()
	SurveyManager.state_changed.connect(_on_state_changed)


func _on_state_changed(new_state: SurveyManager.State) -> void:
	visible = (new_state == SurveyManager.State.SURVEY_PLANNING)
	if visible:
		_update_instrument_options()
		_update_plan_preview()
		if _season_option and _season_option.selected == 0:
			_load_scenario_defaults()


func _create_ui() -> void:
	# Background
	var bg := ColorRect.new()
	bg.color = Color(0.05, 0.07, 0.1, 0.93)
	bg.set_anchors_preset(PRESET_FULL_RECT)
	bg.mouse_filter = Control.MOUSE_FILTER_STOP
	add_child(bg)

	var margin := MarginContainer.new()
	margin.set_anchors_preset(PRESET_FULL_RECT)
	margin.add_theme_constant_override("margin_left", 80)
	margin.add_theme_constant_override("margin_right", 80)
	margin.add_theme_constant_override("margin_top", 40)
	margin.add_theme_constant_override("margin_bottom", 40)
	add_child(margin)

	var outer_vbox := VBoxContainer.new()
	outer_vbox.add_theme_constant_override("separation", 16)
	margin.add_child(outer_vbox)

	# Header
	var header_row := HBoxContainer.new()
	header_row.add_theme_constant_override("separation", 20)
	outer_vbox.add_child(header_row)

	var back_btn := Button.new()
	back_btn.text = "< Back"
	back_btn.add_theme_font_size_override("font_size", 16)
	back_btn.pressed.connect(func(): SurveyManager.transition(SurveyManager.State.SCENARIO_SELECT))
	header_row.add_child(back_btn)

	var title := Label.new()
	title.text = "Survey Planning"
	title.add_theme_font_size_override("font_size", 28)
	title.add_theme_color_override("font_color", Color(0.8, 0.9, 1.0))
	header_row.add_child(title)

	# Scenario name
	_scenario_label = Label.new()
	_scenario_label.name = "ScenarioLabel"
	_scenario_label.add_theme_font_size_override("font_size", 14)
	_scenario_label.add_theme_color_override("font_color", Color(0.5, 0.6, 0.7))
	outer_vbox.add_child(_scenario_label)

	# Two-column layout
	var columns := HBoxContainer.new()
	columns.add_theme_constant_override("separation", 40)
	columns.size_flags_vertical = Control.SIZE_EXPAND_FILL
	outer_vbox.add_child(columns)

	# Left column: Settings
	var settings_scroll := ScrollContainer.new()
	settings_scroll.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	columns.add_child(settings_scroll)

	var settings_vbox := VBoxContainer.new()
	settings_vbox.add_theme_constant_override("separation", 14)
	settings_vbox.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	settings_scroll.add_child(settings_vbox)

	# Operator mode
	_add_section_label(settings_vbox, "Operator Mode")
	_mode_option = OptionButton.new()
	_mode_option.add_item("Ground Walker", 0)
	_mode_option.add_item("Drone/UAV", 1)
	_mode_option.add_theme_font_size_override("font_size", 14)
	_mode_option.item_selected.connect(func(_idx): _update_plan_preview())
	settings_vbox.add_child(_mode_option)

	# Instrument
	_add_section_label(settings_vbox, "Instrument")
	_instrument_option = OptionButton.new()
	_instrument_option.add_theme_font_size_override("font_size", 14)
	settings_vbox.add_child(_instrument_option)

	# Pattern type
	_add_section_label(settings_vbox, "Survey Pattern")
	_pattern_option = OptionButton.new()
	_pattern_option.add_item("Freeform", 0)
	_pattern_option.add_item("Parallel Lines", 1)
	_pattern_option.add_item("Zigzag", 2)
	_pattern_option.select(1)
	_pattern_option.add_theme_font_size_override("font_size", 14)
	_pattern_option.item_selected.connect(func(_idx): _update_plan_preview())
	settings_vbox.add_child(_pattern_option)

	# Line spacing
	_add_section_label(settings_vbox, "Line Spacing")
	var spacing_row := HBoxContainer.new()
	spacing_row.add_theme_constant_override("separation", 10)
	settings_vbox.add_child(spacing_row)

	_spacing_slider = HSlider.new()
	_spacing_slider.min_value = 0.5
	_spacing_slider.max_value = 5.0
	_spacing_slider.step = 0.25
	_spacing_slider.value = 1.0
	_spacing_slider.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_spacing_slider.value_changed.connect(func(v): _spacing_label.text = "%.2f m" % v; _update_plan_preview())
	spacing_row.add_child(_spacing_slider)

	_spacing_label = Label.new()
	_spacing_label.text = "1.00 m"
	_spacing_label.custom_minimum_size.x = 60
	_spacing_label.add_theme_font_size_override("font_size", 14)
	spacing_row.add_child(_spacing_label)

	# Line direction
	_add_section_label(settings_vbox, "Line Direction")
	_direction_option = OptionButton.new()
	_direction_option.add_item("N-S (0 deg)", 0)
	_direction_option.add_item("E-W (90 deg)", 1)
	_direction_option.add_item("NE-SW (45 deg)", 2)
	_direction_option.add_item("NW-SE (135 deg)", 3)
	_direction_option.add_theme_font_size_override("font_size", 14)
	_direction_option.item_selected.connect(func(_idx): _update_plan_preview())
	settings_vbox.add_child(_direction_option)

	# Tie lines
	_tie_check = CheckButton.new()
	_tie_check.text = "Include Tie Lines"
	_tie_check.add_theme_font_size_override("font_size", 14)
	_tie_check.toggled.connect(func(_on): _tie_spacing_slider.visible = _on; _tie_spacing_label.visible = _on; _update_plan_preview())
	settings_vbox.add_child(_tie_check)

	var tie_row := HBoxContainer.new()
	tie_row.add_theme_constant_override("separation", 10)
	settings_vbox.add_child(tie_row)

	_tie_spacing_slider = HSlider.new()
	_tie_spacing_slider.min_value = 5.0
	_tie_spacing_slider.max_value = 20.0
	_tie_spacing_slider.step = 1.0
	_tie_spacing_slider.value = 10.0
	_tie_spacing_slider.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_tie_spacing_slider.visible = false
	_tie_spacing_slider.value_changed.connect(func(v): _tie_spacing_label.text = "%.0f m" % v; _update_plan_preview())
	tie_row.add_child(_tie_spacing_slider)

	_tie_spacing_label = Label.new()
	_tie_spacing_label.text = "10 m"
	_tie_spacing_label.custom_minimum_size.x = 60
	_tie_spacing_label.add_theme_font_size_override("font_size", 14)
	_tie_spacing_label.visible = false
	tie_row.add_child(_tie_spacing_label)

	# Walk speed target
	_add_section_label(settings_vbox, "Speed Target")
	var speed_row := HBoxContainer.new()
	speed_row.add_theme_constant_override("separation", 10)
	settings_vbox.add_child(speed_row)

	_speed_slider = HSlider.new()
	_speed_slider.min_value = 0.5
	_speed_slider.max_value = 3.0
	_speed_slider.step = 0.1
	_speed_slider.value = 1.5
	_speed_slider.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_speed_slider.value_changed.connect(func(v): _speed_label.text = "%.1f m/s" % v)
	speed_row.add_child(_speed_slider)

	_speed_label = Label.new()
	_speed_label.text = "1.5 m/s"
	_speed_label.custom_minimum_size.x = 70
	_speed_label.add_theme_font_size_override("font_size", 14)
	speed_row.add_child(_speed_label)

	# ---- Conditions Section ----
	var cond_sep := HSeparator.new()
	settings_vbox.add_child(cond_sep)

	var cond_title := Label.new()
	cond_title.text = "Conditions"
	cond_title.add_theme_font_size_override("font_size", 16)
	cond_title.add_theme_color_override("font_color", Color(0.7, 0.8, 0.9))
	settings_vbox.add_child(cond_title)

	# Season preset
	_add_section_label(settings_vbox, "Season Preset")
	_season_option = OptionButton.new()
	_season_option.add_item("Scenario Default", 0)
	_season_option.add_item("Spring (12°C / 70%)", 1)
	_season_option.add_item("Summer (22°C / 40%)", 2)
	_season_option.add_item("Autumn (10°C / 75%)", 3)
	_season_option.add_item("Winter (3°C / 85%)", 4)
	_season_option.add_theme_font_size_override("font_size", 14)
	_season_option.item_selected.connect(_on_season_changed)
	settings_vbox.add_child(_season_option)

	# Temperature
	_add_section_label(settings_vbox, "Temperature (°C)")
	var temp_row := HBoxContainer.new()
	temp_row.add_theme_constant_override("separation", 10)
	settings_vbox.add_child(temp_row)

	_temp_slider = HSlider.new()
	_temp_slider.min_value = -10.0
	_temp_slider.max_value = 40.0
	_temp_slider.step = 0.5
	_temp_slider.value = 15.0
	_temp_slider.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_temp_slider.value_changed.connect(func(v): _temp_label.text = "%.1f°C" % v)
	temp_row.add_child(_temp_slider)

	_temp_label = Label.new()
	_temp_label.text = "15.0°C"
	_temp_label.custom_minimum_size.x = 70
	_temp_label.add_theme_font_size_override("font_size", 14)
	temp_row.add_child(_temp_label)

	# Soil Moisture
	_add_section_label(settings_vbox, "Soil Moisture (%)")
	var moist_row := HBoxContainer.new()
	moist_row.add_theme_constant_override("separation", 10)
	settings_vbox.add_child(moist_row)

	_moisture_slider = HSlider.new()
	_moisture_slider.min_value = 0.0
	_moisture_slider.max_value = 100.0
	_moisture_slider.step = 1.0
	_moisture_slider.value = 60.0
	_moisture_slider.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_moisture_slider.value_changed.connect(func(v): _moisture_label.text = "%.0f%%" % v)
	moist_row.add_child(_moisture_slider)

	_moisture_label = Label.new()
	_moisture_label.text = "60%"
	_moisture_label.custom_minimum_size.x = 60
	_moisture_label.add_theme_font_size_override("font_size", 14)
	moist_row.add_child(_moisture_label)

	# Weather preset
	_add_section_label(settings_vbox, "Weather")
	_weather_option = OptionButton.new()
	_weather_option.add_item("Dry", 0)
	_weather_option.add_item("Light Rain (+20% moisture)", 1)
	_weather_option.add_item("Heavy Rain (+40% moisture)", 2)
	_weather_option.add_item("Frost (frozen ground)", 3)
	_weather_option.add_theme_font_size_override("font_size", 14)
	_weather_option.item_selected.connect(_on_weather_changed)
	settings_vbox.add_child(_weather_option)

	# Ground type override
	_add_section_label(settings_vbox, "Ground Type")
	_ground_option = OptionButton.new()
	_ground_option.add_item("Scenario Default", 0)
	_ground_option.add_item("Sandy", 1)
	_ground_option.add_item("Loam", 2)
	_ground_option.add_item("Clay", 3)
	_ground_option.add_item("Peat", 4)
	_ground_option.add_item("Volcanic", 5)
	_ground_option.add_theme_font_size_override("font_size", 14)
	settings_vbox.add_child(_ground_option)

	# Right column: Plan info / preview
	var info_panel := PanelContainer.new()
	info_panel.custom_minimum_size.x = 300
	var info_style := StyleBoxFlat.new()
	info_style.bg_color = Color(0.08, 0.1, 0.13, 0.9)
	info_style.corner_radius_top_left = 6
	info_style.corner_radius_top_right = 6
	info_style.corner_radius_bottom_left = 6
	info_style.corner_radius_bottom_right = 6
	info_style.content_margin_left = 16
	info_style.content_margin_right = 16
	info_style.content_margin_top = 14
	info_style.content_margin_bottom = 14
	info_panel.add_theme_stylebox_override("panel", info_style)
	columns.add_child(info_panel)

	_info_label = Label.new()
	_info_label.text = "Plan Summary\n\nConfigure settings and click Begin Survey."
	_info_label.add_theme_font_size_override("font_size", 14)
	_info_label.add_theme_color_override("font_color", Color(0.7, 0.75, 0.8))
	_info_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	info_panel.add_child(_info_label)

	# Bottom: Begin button
	var bottom_row := HBoxContainer.new()
	bottom_row.alignment = BoxContainer.ALIGNMENT_END
	outer_vbox.add_child(bottom_row)

	var begin_btn := Button.new()
	begin_btn.text = "Begin Survey"
	begin_btn.custom_minimum_size = Vector2(200, 50)
	begin_btn.add_theme_font_size_override("font_size", 20)
	begin_btn.pressed.connect(_on_begin_survey)
	bottom_row.add_child(begin_btn)


func _add_section_label(parent: Control, text: String) -> void:
	var label := Label.new()
	label.text = text
	label.add_theme_font_size_override("font_size", 13)
	label.add_theme_color_override("font_color", Color(0.5, 0.55, 0.6))
	parent.add_child(label)


func _update_instrument_options() -> void:
	_instrument_option.clear()
	for inst in SurveyManager.available_instruments:
		_instrument_option.add_item(SurveyManager.instrument_name(inst), inst)

	# Update scenario label
	if _scenario_label:
		_scenario_label.text = "Scenario: %s" % SurveyManager.scenario_info.get("name", "")


func _get_direction_angle() -> float:
	match _direction_option.selected:
		0: return 0.0          # N-S
		1: return PI / 2.0     # E-W
		2: return PI / 4.0     # NE-SW
		3: return 3.0 * PI / 4.0  # NW-SE
		_: return 0.0


func _get_pattern_name() -> String:
	match _pattern_option.selected:
		0: return "freeform"
		1: return "parallel"
		2: return "zigzag"
		_: return "freeform"


func _update_plan_preview() -> void:
	var terrain_info: Dictionary = SurveyManager.scenario_info.get("terrain", {})
	var x_ext: Array = terrain_info.get("x_extent", [0, 20])
	var y_ext: Array = terrain_info.get("y_extent", [0, 20])
	var area := Rect2(x_ext[0], y_ext[0], x_ext[1] - x_ext[0], y_ext[1] - y_ext[0])

	var spacing := _spacing_slider.value
	var angle := _get_direction_angle()
	var pattern := _pattern_option.selected

	# Generate lines
	_planned_lines.clear()
	if pattern == 1:  # Parallel
		_planned_lines = PatternGenerator.parallel_lines(area, spacing, angle)
	elif pattern == 2:  # Zigzag
		_planned_lines = PatternGenerator.zigzag(area, spacing, angle)

	# Add tie lines if enabled
	if _tie_check.button_pressed and not _planned_lines.is_empty():
		var tie_lines := PatternGenerator.add_tie_lines(
			_planned_lines, area, _tie_spacing_slider.value, angle)
		_planned_lines.append_array(tie_lines)

	# Update info label
	var total_length := 0.0
	var line_count := _planned_lines.size()
	for line in _planned_lines:
		for i in range(line.size() - 1):
			total_length += line[i].distance_to(line[i + 1])

	var speed := _speed_slider.value
	var est_time := total_length / speed if speed > 0 else 0.0
	var sample_rate := 10.0
	var est_samples := int(est_time * sample_rate)

	var text := "Plan Summary\n\n"
	text += "Pattern: %s\n" % _get_pattern_name().capitalize()
	text += "Lines: %d\n" % line_count
	text += "Spacing: %.2f m\n" % spacing
	text += "Total length: %.0f m\n" % total_length
	text += "Est. time: %d:%02d\n" % [int(est_time) / 60, int(est_time) % 60]
	text += "Est. samples: %d\n" % est_samples
	text += "Speed target: %.1f m/s\n" % speed
	if _tie_check.button_pressed:
		text += "Tie lines: every %.0f m\n" % _tie_spacing_slider.value

	if _mode_option.selected == 0:
		text += "\nOperator: Ground"
		text += "\nSensor height: 0.175 m"
	else:
		text += "\nOperator: Drone"
		text += "\nFlight altitude: 2.0 m AGL"

	# Environment conditions
	if _temp_slider:
		text += "\n\nConditions:"
		text += "\nTemp: %.1f°C" % _temp_slider.value
		text += "\nMoisture: %.0f%%" % _moisture_slider.value
		if _weather_option.selected == 3:
			text += "\nFrost: Yes"

	_info_label.text = text

	# Draw lines on terrain in main scene
	var main := get_tree().root.get_node_or_null("Main")
	if main and main.has_method("draw_survey_lines"):
		var surface_elev: float = terrain_info.get("surface_elevation", 0.0)
		# Convert planned lines from GeoSim to Godot coordinates
		var godot_lines: Array = []
		for line in _planned_lines:
			var godot_line := PackedVector3Array()
			for pt in line:
				godot_line.append(CoordUtil.to_godot(pt))
			godot_lines.append(godot_line)
		main.draw_survey_lines(godot_lines, surface_elev)


func _on_season_changed(idx: int) -> void:
	match idx:
		1:  # Spring
			_temp_slider.value = 12.0
			_moisture_slider.value = 70.0
		2:  # Summer
			_temp_slider.value = 22.0
			_moisture_slider.value = 40.0
		3:  # Autumn
			_temp_slider.value = 10.0
			_moisture_slider.value = 75.0
		4:  # Winter
			_temp_slider.value = 3.0
			_moisture_slider.value = 85.0
		_:  # Scenario Default
			_load_scenario_defaults()


func _on_weather_changed(idx: int) -> void:
	var base_moisture := _moisture_slider.value
	match idx:
		1:  # Light rain
			_moisture_slider.value = minf(base_moisture + 20.0, 100.0)
		2:  # Heavy rain
			_moisture_slider.value = minf(base_moisture + 40.0, 100.0)


func _load_scenario_defaults() -> void:
	var env: Dictionary = SurveyManager.scenario_info.get("soil_environment", {})
	if not env.is_empty():
		_temp_slider.value = env.get("temperature_c", 15.0)
		_moisture_slider.value = env.get("saturation", 0.6) * 100.0
	else:
		_temp_slider.value = 15.0
		_moisture_slider.value = 60.0


func _send_environment() -> void:
	var env := {
		"temperature_c": _temp_slider.value,
		"saturation": _moisture_slider.value / 100.0,
		"frozen": _weather_option.selected == 3,
	}
	# Ground type override
	var ground_names := ["", "sandy", "loam", "clay", "peat", "volcanic"]
	if _ground_option.selected > 0 and _ground_option.selected < ground_names.size():
		env["ground_type"] = ground_names[_ground_option.selected]
	PhysicsClient.set_environment(env)


func _on_begin_survey() -> void:
	# Store survey plan in SurveyManager
	var inst_idx := _instrument_option.get_selected_id()
	if inst_idx >= 0:
		SurveyManager.current_instrument = inst_idx as SurveyManager.Instrument

	var op_mode := SurveyManager.OperatorMode.GROUND if _mode_option.selected == 0 \
		else SurveyManager.OperatorMode.DRONE
	SurveyManager.switch_operator(op_mode)

	# Send environment conditions to physics server
	_send_environment()

	SurveyManager.survey_plan = {
		"pattern": _get_pattern_name(),
		"spacing": _spacing_slider.value,
		"direction_deg": rad_to_deg(_get_direction_angle()),
		"speed_target": _speed_slider.value,
		"tie_lines": _tie_check.button_pressed,
		"tie_spacing": _tie_spacing_slider.value if _tie_check.button_pressed else 0.0,
		"line_count": _planned_lines.size(),
		"lines": _planned_lines,
		"temperature_c": _temp_slider.value,
		"saturation": _moisture_slider.value / 100.0,
		"frozen": _weather_option.selected == 3,
	}

	# Configure operator speed
	var main := get_tree().root.get_node_or_null("Main")
	if main:
		var operator := main.get_node_or_null("Operator")
		if operator:
			operator.walk_speed = _speed_slider.value

	SurveyManager.transition(SurveyManager.State.SURVEYING)
