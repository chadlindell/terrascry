## Main menu UI — entry point for the application.
##
## Provides buttons: Start Survey, Training, Quit.
## Responds to SurveyManager state changes to show/hide.
extends Control

var _panel: PanelContainer
var _title_label: Label


func _ready() -> void:
	mouse_filter = Control.MOUSE_FILTER_STOP
	_create_ui()
	SurveyManager.state_changed.connect(_on_state_changed)
	_update_visibility(SurveyManager.current_state)


func _on_state_changed(new_state: SurveyManager.State) -> void:
	_update_visibility(new_state)


func _update_visibility(state: SurveyManager.State) -> void:
	var should_show := (state == SurveyManager.State.MAIN_MENU)
	if should_show and not visible:
		visible = true
		modulate.a = 0.0
		var tween := create_tween()
		tween.tween_property(self, "modulate:a", 1.0, 0.3)
	elif not should_show and visible:
		var tween := create_tween()
		tween.tween_property(self, "modulate:a", 0.0, 0.2)
		tween.tween_callback(func(): visible = false)


func _create_ui() -> void:
	# Semi-transparent background showing 3D scene behind
	var bg := ColorRect.new()
	bg.color = Color(0.03, 0.05, 0.08, 0.75)
	bg.set_anchors_preset(PRESET_FULL_RECT)
	bg.mouse_filter = Control.MOUSE_FILTER_STOP
	add_child(bg)

	# Center container
	var center := CenterContainer.new()
	center.set_anchors_preset(PRESET_FULL_RECT)
	add_child(center)

	var vbox := VBoxContainer.new()
	vbox.add_theme_constant_override("separation", 20)
	center.add_child(vbox)

	# Title
	_title_label = Label.new()
	_title_label.text = "GeoSim"
	_title_label.add_theme_font_size_override("font_size", 48)
	_title_label.add_theme_color_override("font_color", Color(0.8, 0.9, 1.0))
	_title_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	vbox.add_child(_title_label)

	# Subtitle
	var subtitle := Label.new()
	subtitle.text = "Geophysical Survey Simulation Platform"
	subtitle.add_theme_font_size_override("font_size", 16)
	subtitle.add_theme_color_override("font_color", Color(0.5, 0.6, 0.7))
	subtitle.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	vbox.add_child(subtitle)

	# Spacer
	var spacer := Control.new()
	spacer.custom_minimum_size.y = 30
	vbox.add_child(spacer)

	# Button container
	var btn_box := VBoxContainer.new()
	btn_box.add_theme_constant_override("separation", 12)
	btn_box.custom_minimum_size.x = 280
	vbox.add_child(btn_box)

	_add_menu_button(btn_box, "Start Survey", _on_start_survey)
	_add_menu_button(btn_box, "Training", _on_training)
	_add_menu_button(btn_box, "Quit", _on_quit)

	# Version label
	var version := Label.new()
	version.text = "v0.1.0"
	version.add_theme_font_size_override("font_size", 12)
	version.add_theme_color_override("font_color", Color(0.3, 0.35, 0.4))
	version.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	vbox.add_child(version)


func _add_menu_button(parent: Control, text: String, callback: Callable) -> Button:
	var btn := Button.new()
	btn.text = text
	btn.custom_minimum_size = Vector2(280, 48)
	btn.add_theme_font_size_override("font_size", 18)
	btn.pressed.connect(callback)
	parent.add_child(btn)
	return btn


func _on_start_survey() -> void:
	AudioManager.play_ui_click()
	SurveyManager.transition(SurveyManager.State.SCENARIO_SELECT)


func _on_training() -> void:
	AudioManager.play_ui_click()
	SurveyManager.transition(SurveyManager.State.TRAINING)


func _on_quit() -> void:
	AudioManager.play_ui_click()
	get_tree().quit()
