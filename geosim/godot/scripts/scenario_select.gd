## Scenario selection UI — browse and load scenarios.
##
## Displays cards for each available scenario with metadata (size, object count,
## difficulty, applicable instruments). Selecting a card loads the scenario
## via PhysicsClient and transitions to survey planning.
extends Control

## Scenario metadata for card display
const SCENARIO_DATA := [
	{
		"path": "scenarios/single-ferrous-target.json",
		"name": "Single Ferrous Target",
		"size": "20x20m",
		"objects": 1,
		"depth_range": "1.0m",
		"instruments": ["Mag"],
		"difficulty": "Beginner",
		"difficulty_color": Color(0.3, 0.8, 0.3),
		"description": "Single steel sphere at 1m depth. Basic detection training.",
	},
	{
		"path": "scenarios/scattered-debris.json",
		"name": "Scattered Debris",
		"size": "30x30m",
		"objects": 6,
		"depth_range": "0.3-2.0m",
		"instruments": ["Mag"],
		"difficulty": "Intermediate",
		"difficulty_color": Color(0.9, 0.8, 0.2),
		"description": "Multiple targets at varying depths and sizes.",
	},
	{
		"path": "scenarios/clandestine-burial.json",
		"name": "Clandestine Burial",
		"size": "15x15m",
		"objects": 3,
		"depth_range": "0.5-1.5m",
		"instruments": ["Mag"],
		"difficulty": "Advanced",
		"difficulty_color": Color(0.9, 0.5, 0.2),
		"description": "Forensic scenario with subtle magnetic anomalies.",
	},
	{
		"path": "scenarios/bomb-crater-heterogeneous.json",
		"name": "Bomb Crater",
		"size": "25x25m",
		"objects": 3,
		"depth_range": "0.5-3.0m",
		"instruments": ["Mag", "EM"],
		"difficulty": "Advanced",
		"difficulty_color": Color(0.9, 0.5, 0.2),
		"description": "Heterogeneous fill with mixed debris. Requires multi-instrument approach.",
	},
	{
		"path": "scenarios/swamp-crash-site.json",
		"name": "Swamp Crash Site",
		"size": "25x25m",
		"objects": 5,
		"depth_range": "0.6-2.5m",
		"instruments": ["Mag", "EM", "ERT"],
		"difficulty": "Expert",
		"difficulty_color": Color(0.9, 0.2, 0.2),
		"description": "Waterlogged marsh with high conductivity. Full instrument suite needed.",
	},
]

var _loading_label: Label
var _card_container: VBoxContainer
var _is_loading := false


func _ready() -> void:
	mouse_filter = Control.MOUSE_FILTER_STOP
	visible = false
	_create_ui()
	SurveyManager.state_changed.connect(_on_state_changed)


func _on_state_changed(new_state: SurveyManager.State) -> void:
	visible = (new_state == SurveyManager.State.SCENARIO_SELECT)


func _create_ui() -> void:
	# Background
	var bg := ColorRect.new()
	bg.color = Color(0.05, 0.07, 0.1, 0.95)
	bg.set_anchors_preset(PRESET_FULL_RECT)
	bg.mouse_filter = Control.MOUSE_FILTER_STOP
	add_child(bg)

	# Scrollable container
	var margin := MarginContainer.new()
	margin.set_anchors_preset(PRESET_FULL_RECT)
	margin.add_theme_constant_override("margin_left", 80)
	margin.add_theme_constant_override("margin_right", 80)
	margin.add_theme_constant_override("margin_top", 40)
	margin.add_theme_constant_override("margin_bottom", 40)
	add_child(margin)

	var outer_vbox := VBoxContainer.new()
	outer_vbox.add_theme_constant_override("separation", 20)
	margin.add_child(outer_vbox)

	# Header row
	var header_row := HBoxContainer.new()
	header_row.add_theme_constant_override("separation", 20)
	outer_vbox.add_child(header_row)

	var back_btn := Button.new()
	back_btn.text = "< Back"
	back_btn.add_theme_font_size_override("font_size", 16)
	back_btn.pressed.connect(func(): SurveyManager.transition(SurveyManager.State.MAIN_MENU))
	header_row.add_child(back_btn)

	var title := Label.new()
	title.text = "Select Scenario"
	title.add_theme_font_size_override("font_size", 32)
	title.add_theme_color_override("font_color", Color(0.8, 0.9, 1.0))
	header_row.add_child(title)

	# Loading indicator
	_loading_label = Label.new()
	_loading_label.text = ""
	_loading_label.add_theme_font_size_override("font_size", 14)
	_loading_label.add_theme_color_override("font_color", Color.YELLOW)
	outer_vbox.add_child(_loading_label)

	# Scroll container for cards
	var scroll := ScrollContainer.new()
	scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	outer_vbox.add_child(scroll)

	_card_container = VBoxContainer.new()
	_card_container.add_theme_constant_override("separation", 12)
	_card_container.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	scroll.add_child(_card_container)

	# Create scenario cards
	for scenario in SCENARIO_DATA:
		_create_scenario_card(scenario)


func _create_scenario_card(data: Dictionary) -> void:
	var card := PanelContainer.new()
	card.custom_minimum_size = Vector2(0, 100)
	card.size_flags_horizontal = Control.SIZE_EXPAND_FILL

	var style := StyleBoxFlat.new()
	style.bg_color = Color(0.12, 0.14, 0.18, 0.9)
	style.corner_radius_top_left = 6
	style.corner_radius_top_right = 6
	style.corner_radius_bottom_left = 6
	style.corner_radius_bottom_right = 6
	style.content_margin_left = 16
	style.content_margin_right = 16
	style.content_margin_top = 12
	style.content_margin_bottom = 12
	style.border_width_left = 3
	style.border_color = data.get("difficulty_color", Color.WHITE)
	card.add_theme_stylebox_override("panel", style)
	_card_container.add_child(card)

	var hbox := HBoxContainer.new()
	hbox.add_theme_constant_override("separation", 20)
	card.add_child(hbox)

	# Left: Name and description
	var left_vbox := VBoxContainer.new()
	left_vbox.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	left_vbox.add_theme_constant_override("separation", 4)
	hbox.add_child(left_vbox)

	var name_label := Label.new()
	name_label.text = data.get("name", "")
	name_label.add_theme_font_size_override("font_size", 20)
	name_label.add_theme_color_override("font_color", Color(0.9, 0.92, 0.95))
	left_vbox.add_child(name_label)

	var desc_label := Label.new()
	desc_label.text = data.get("description", "")
	desc_label.add_theme_font_size_override("font_size", 13)
	desc_label.add_theme_color_override("font_color", Color(0.5, 0.55, 0.6))
	desc_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	left_vbox.add_child(desc_label)

	# Middle: Metadata
	var mid_vbox := VBoxContainer.new()
	mid_vbox.custom_minimum_size.x = 180
	mid_vbox.add_theme_constant_override("separation", 2)
	hbox.add_child(mid_vbox)

	_add_meta_label(mid_vbox, "Size: %s" % data.get("size", ""))
	_add_meta_label(mid_vbox, "Objects: %d" % data.get("objects", 0))
	_add_meta_label(mid_vbox, "Depth: %s" % data.get("depth_range", ""))

	var instruments: Array = data.get("instruments", [])
	_add_meta_label(mid_vbox, "Instruments: %s" % ", ".join(instruments))

	# Right: Difficulty badge + Select button
	var right_vbox := VBoxContainer.new()
	right_vbox.custom_minimum_size.x = 120
	right_vbox.add_theme_constant_override("separation", 8)
	hbox.add_child(right_vbox)

	var diff_label := Label.new()
	diff_label.text = data.get("difficulty", "")
	diff_label.add_theme_font_size_override("font_size", 14)
	diff_label.add_theme_color_override("font_color", data.get("difficulty_color", Color.WHITE))
	diff_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	right_vbox.add_child(diff_label)

	var select_btn := Button.new()
	select_btn.text = "Select"
	select_btn.custom_minimum_size = Vector2(100, 36)
	select_btn.add_theme_font_size_override("font_size", 15)
	select_btn.pressed.connect(_on_scenario_selected.bind(data.get("path", "")))
	right_vbox.add_child(select_btn)


func _add_meta_label(parent: Control, text: String) -> void:
	var label := Label.new()
	label.text = text
	label.add_theme_font_size_override("font_size", 12)
	label.add_theme_color_override("font_color", Color(0.6, 0.65, 0.7))
	parent.add_child(label)


func _on_scenario_selected(path: String) -> void:
	if _is_loading:
		return
	_is_loading = true
	_loading_label.text = "Loading scenario..."

	var success = await SurveyManager.load_scenario(path)
	_is_loading = false
	if not visible:
		return  # User navigated away during load
	if success:
		_loading_label.text = ""
		SurveyManager.transition(SurveyManager.State.SURVEY_PLANNING)
	else:
		_loading_label.text = "Failed to load scenario"
