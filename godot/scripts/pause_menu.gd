## Pause menu overlay during surveys.
##
## Provides Resume, Export Data, Quit to Menu options.
## Shown/hidden via SurveyManager.State.PAUSED.
extends Control


func _ready() -> void:
	mouse_filter = Control.MOUSE_FILTER_STOP
	visible = false
	_create_ui()
	SurveyManager.state_changed.connect(_on_state_changed)


func _on_state_changed(new_state: SurveyManager.State) -> void:
	visible = (new_state == SurveyManager.State.PAUSED)


func _create_ui() -> void:
	# Semi-transparent overlay
	var bg := ColorRect.new()
	bg.color = Color(0.0, 0.0, 0.0, 0.7)
	bg.set_anchors_preset(PRESET_FULL_RECT)
	bg.mouse_filter = Control.MOUSE_FILTER_STOP
	add_child(bg)

	var center := CenterContainer.new()
	center.set_anchors_preset(PRESET_FULL_RECT)
	add_child(center)

	var panel := PanelContainer.new()
	var panel_style := StyleBoxFlat.new()
	panel_style.bg_color = Color(0.1, 0.12, 0.15, 0.95)
	panel_style.corner_radius_top_left = 8
	panel_style.corner_radius_top_right = 8
	panel_style.corner_radius_bottom_left = 8
	panel_style.corner_radius_bottom_right = 8
	panel_style.content_margin_left = 30
	panel_style.content_margin_right = 30
	panel_style.content_margin_top = 20
	panel_style.content_margin_bottom = 20
	panel.add_theme_stylebox_override("panel", panel_style)
	center.add_child(panel)

	var vbox := VBoxContainer.new()
	vbox.add_theme_constant_override("separation", 14)
	panel.add_child(vbox)

	# Title
	var title := Label.new()
	title.text = "PAUSED"
	title.add_theme_font_size_override("font_size", 28)
	title.add_theme_color_override("font_color", Color(0.8, 0.85, 0.9))
	title.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	vbox.add_child(title)

	# Stats summary
	var stats := Label.new()
	stats.name = "StatsLabel"
	stats.add_theme_font_size_override("font_size", 13)
	stats.add_theme_color_override("font_color", Color(0.5, 0.55, 0.6))
	stats.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	vbox.add_child(stats)

	var sep := HSeparator.new()
	vbox.add_child(sep)

	# Buttons
	var btn_box := VBoxContainer.new()
	btn_box.add_theme_constant_override("separation", 10)
	btn_box.custom_minimum_size.x = 240
	vbox.add_child(btn_box)

	_add_button(btn_box, "Resume", _on_resume)
	_add_button(btn_box, "Export Data", _on_export)
	_add_button(btn_box, "Finish Survey", _on_finish)
	_add_button(btn_box, "Quit to Menu", _on_quit_menu)


func _add_button(parent: Control, text: String, callback: Callable) -> void:
	var btn := Button.new()
	btn.text = text
	btn.custom_minimum_size = Vector2(240, 40)
	btn.add_theme_font_size_override("font_size", 16)
	btn.pressed.connect(callback)
	parent.add_child(btn)


func _on_resume() -> void:
	SurveyManager.transition(SurveyManager.previous_state)


func _on_export() -> void:
	if DataRecorder.samples.size() > 0:
		DataRecorder.export_session()


func _on_finish() -> void:
	SurveyManager.survey_finished.emit()
	SurveyManager.transition(SurveyManager.State.POST_SURVEY)


func _on_quit_menu() -> void:
	DataRecorder.stop_recording()
	SurveyManager.transition(SurveyManager.State.MAIN_MENU)
