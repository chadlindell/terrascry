## Training HUD overlay — objectives, real-time feedback, score.
##
## Displays current training module objective, live score updates,
## and end-of-module report card with letter grade.
extends Control

var _objective_label: Label
var _score_label: Label
var _timer_label: Label
var _feedback_label: Label
var _report_panel: PanelContainer
var _report_content: Label

## Module select panel
var _select_panel: Control
var _module_buttons: VBoxContainer

## Track signal connection for cleanup
var _objective_callable: Callable


func _ready() -> void:
	mouse_filter = Control.MOUSE_FILTER_IGNORE
	visible = false
	_create_ui()
	SurveyManager.state_changed.connect(_on_state_changed)


func _on_state_changed(new_state: SurveyManager.State) -> void:
	if new_state == SurveyManager.State.TRAINING:
		visible = true
		_show_module_select()
	else:
		visible = false
		if _report_panel:
			_report_panel.visible = false
		if _select_panel:
			_select_panel.visible = false


func _process(_delta: float) -> void:
	if not visible:
		return

	# Update timer during active training
	var training := _get_training_manager()
	if training and training.module_active:
		if _timer_label:
			var elapsed: float = training.module_elapsed
			_timer_label.text = "Time: %d:%02d" % [int(elapsed) / 60, int(elapsed) % 60]

		# Live XTE feedback
		if _feedback_label and training.xte_count > 0:
			var avg_xte: float = training.total_xte / training.xte_count
			if avg_xte < 0.25:
				_feedback_label.text = "Excellent line discipline!"
				_feedback_label.add_theme_color_override("font_color", Color.GREEN)
			elif avg_xte < 0.5:
				_feedback_label.text = "Good — stay on line"
				_feedback_label.add_theme_color_override("font_color", Color.YELLOW)
			else:
				_feedback_label.text = "Off-line — correct heading"
				_feedback_label.add_theme_color_override("font_color", Color.RED)


func _create_ui() -> void:
	# ---- Module select panel ----
	_select_panel = Control.new()
	_select_panel.set_anchors_preset(PRESET_FULL_RECT)
	_select_panel.visible = false
	add_child(_select_panel)

	var select_bg := ColorRect.new()
	select_bg.color = Color(0.05, 0.07, 0.1, 0.95)
	select_bg.set_anchors_preset(PRESET_FULL_RECT)
	select_bg.mouse_filter = Control.MOUSE_FILTER_STOP
	_select_panel.add_child(select_bg)

	var select_center := CenterContainer.new()
	select_center.set_anchors_preset(PRESET_FULL_RECT)
	_select_panel.add_child(select_center)

	var select_vbox := VBoxContainer.new()
	select_vbox.add_theme_constant_override("separation", 16)
	select_center.add_child(select_vbox)

	var select_title := Label.new()
	select_title.text = "Training Modules"
	select_title.add_theme_font_size_override("font_size", 28)
	select_title.add_theme_color_override("font_color", Color(0.8, 0.9, 1.0))
	select_title.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	select_vbox.add_child(select_title)

	_module_buttons = VBoxContainer.new()
	_module_buttons.add_theme_constant_override("separation", 8)
	_module_buttons.custom_minimum_size.x = 400
	select_vbox.add_child(_module_buttons)

	var back_btn := Button.new()
	back_btn.text = "< Back to Menu"
	back_btn.add_theme_font_size_override("font_size", 14)
	back_btn.pressed.connect(func(): SurveyManager.transition(SurveyManager.State.MAIN_MENU))
	select_vbox.add_child(back_btn)

	# ---- In-training HUD overlay ----
	# Top-center: Objective
	_objective_label = Label.new()
	_objective_label.set_anchors_preset(PRESET_CENTER_TOP)
	_objective_label.offset_top = 12
	_objective_label.offset_left = -200
	_objective_label.offset_right = 200
	_objective_label.text = ""
	_objective_label.add_theme_font_size_override("font_size", 16)
	_objective_label.add_theme_color_override("font_color", Color(0.9, 0.92, 0.95))
	_objective_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	_objective_label.autowrap_mode = TextServer.AUTOWRAP_WORD
	add_child(_objective_label)

	# Top-right: Timer + Score
	var top_right := VBoxContainer.new()
	top_right.set_anchors_preset(PRESET_TOP_RIGHT)
	top_right.offset_left = -180
	top_right.offset_top = 12
	top_right.offset_right = -12
	top_right.add_theme_constant_override("separation", 4)
	add_child(top_right)

	_timer_label = Label.new()
	_timer_label.text = "Time: 0:00"
	_timer_label.add_theme_font_size_override("font_size", 16)
	_timer_label.add_theme_color_override("font_color", Color.WHITE)
	_timer_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	top_right.add_child(_timer_label)

	_score_label = Label.new()
	_score_label.text = ""
	_score_label.add_theme_font_size_override("font_size", 14)
	_score_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	top_right.add_child(_score_label)

	# Bottom-center: Feedback
	_feedback_label = Label.new()
	_feedback_label.set_anchors_preset(PRESET_CENTER_BOTTOM)
	_feedback_label.offset_bottom = -60
	_feedback_label.offset_left = -200
	_feedback_label.offset_right = 200
	_feedback_label.text = ""
	_feedback_label.add_theme_font_size_override("font_size", 18)
	_feedback_label.add_theme_color_override("font_color", Color.WHITE)
	_feedback_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	add_child(_feedback_label)

	# ---- Report card panel (shown at end) ----
	_report_panel = PanelContainer.new()
	_report_panel.set_anchors_preset(PRESET_CENTER)
	_report_panel.offset_left = -200
	_report_panel.offset_right = 200
	_report_panel.offset_top = -150
	_report_panel.offset_bottom = 150
	_report_panel.visible = false

	var report_style := StyleBoxFlat.new()
	report_style.bg_color = Color(0.08, 0.1, 0.13, 0.95)
	report_style.corner_radius_top_left = 8
	report_style.corner_radius_top_right = 8
	report_style.corner_radius_bottom_left = 8
	report_style.corner_radius_bottom_right = 8
	report_style.content_margin_left = 20
	report_style.content_margin_right = 20
	report_style.content_margin_top = 16
	report_style.content_margin_bottom = 16
	_report_panel.add_theme_stylebox_override("panel", report_style)
	_report_panel.mouse_filter = Control.MOUSE_FILTER_STOP
	add_child(_report_panel)

	var report_vbox := VBoxContainer.new()
	report_vbox.add_theme_constant_override("separation", 10)
	_report_panel.add_child(report_vbox)

	var report_title := Label.new()
	report_title.text = "Report Card"
	report_title.add_theme_font_size_override("font_size", 22)
	report_title.add_theme_color_override("font_color", Color(0.8, 0.85, 0.9))
	report_title.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	report_vbox.add_child(report_title)

	_report_content = Label.new()
	_report_content.text = ""
	_report_content.add_theme_font_size_override("font_size", 14)
	_report_content.add_theme_color_override("font_color", Color(0.7, 0.75, 0.8))
	_report_content.autowrap_mode = TextServer.AUTOWRAP_WORD
	report_vbox.add_child(_report_content)

	var report_btn := Button.new()
	report_btn.text = "Continue"
	report_btn.custom_minimum_size = Vector2(120, 36)
	report_btn.pressed.connect(func():
		_report_panel.visible = false
		_show_module_select()
	)
	report_vbox.add_child(report_btn)


func _show_module_select() -> void:
	_select_panel.visible = true
	_select_panel.mouse_filter = Control.MOUSE_FILTER_STOP
	_objective_label.text = ""
	_feedback_label.text = ""

	# Clear old buttons
	for child in _module_buttons.get_children():
		child.queue_free()

	# Create module buttons
	var training := _get_training_manager()
	if not training:
		return

	for module_data in training.get_module_list():
		var btn := Button.new()
		btn.text = "%s — %s" % [module_data["name"], module_data["description"]]
		btn.custom_minimum_size = Vector2(400, 40)
		btn.add_theme_font_size_override("font_size", 14)
		btn.text_overrun_behavior = TextServer.OVERRUN_TRIM_ELLIPSIS
		var mod: int = module_data["module"]
		btn.pressed.connect(_on_module_selected.bind(mod))
		_module_buttons.add_child(btn)


func _on_module_selected(module: int) -> void:
	_select_panel.visible = false

	var training := _get_training_manager()
	if not training:
		return

	var info: Dictionary = training.MODULE_INFO[module]

	# Load scenario
	var scenario_path: String = info.get("scenario", "")
	if not scenario_path.is_empty():
		await SurveyManager.load_scenario(scenario_path)

	# Set up simple survey plan for training
	SurveyManager.survey_plan = {
		"pattern": "parallel",
		"spacing": 1.0,
		"direction_deg": 0.0,
		"speed_target": 1.5,
		"lines": [],
	}

	# Disconnect previous objective signal if connected
	if _objective_callable.is_valid() and training.objective_updated.is_connected(_objective_callable):
		training.objective_updated.disconnect(_objective_callable)

	# Connect signals before starting module to avoid missing immediate emissions
	training.module_completed.connect(_on_module_completed, CONNECT_ONE_SHOT)
	_objective_callable = func(text): _objective_label.text = text
	training.objective_updated.connect(_objective_callable)

	# Start module
	training.start_module(module as TrainingManager.Module)

	# Transition to surveying
	SurveyManager.transition(SurveyManager.State.SURVEYING)


func _on_module_completed(_module: TrainingManager.Module, grade: Dictionary) -> void:
	# Show report card
	var text := ""
	text += "Module: %s\n\n" % grade.get("module", "")
	text += "Grade: %s\n" % grade.get("letter", "?")
	text += "Score: %.0f%%\n\n" % grade.get("score", 0.0)

	var duration: float = grade.get("duration_s", 0.0)
	text += "Duration: %d:%02d\n\n" % [int(duration) / 60, int(duration) % 60]

	var breakdown: Dictionary = grade.get("breakdown", {})
	for key in breakdown:
		text += "%s: %s\n" % [key, str(breakdown[key])]

	_report_content.text = text
	_report_panel.visible = true

	# Return to training state for module selection
	SurveyManager.transition(SurveyManager.State.TRAINING)


func _get_training_manager() -> Node:
	return get_tree().root.get_node_or_null("Main/TrainingManager")
