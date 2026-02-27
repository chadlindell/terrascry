## Post-survey review UI — top-down replay, stats, export.
##
## Shows survey results after completion: statistics, anomaly list, and a
## top-down camera replay with progressive heatmap, position marker, trail,
## and adjustable playback speed.
extends Control

var _stats_label: Label
var _anomaly_list: VBoxContainer
var _scrubber: HSlider
var _scrubber_label: Label
var _speed_label: Label
var _play_btn: Button

## Replay state
var _replay_playing: bool = false
var _replay_index: int = 0
var _replay_speed: float = 10.0  # samples per second (base rate)
var _replay_speed_multiplier: float = 1.0
var _replay_accumulator: float = 0.0

## 3D replay objects
var _replay_marker: MeshInstance3D
var _replay_trail: ImmediateMesh
var _replay_trail_mesh: MeshInstance3D
var _topdown_camera: Camera3D
var _previous_camera: Camera3D

## Speed multiplier options
const SPEED_OPTIONS := [0.5, 1.0, 2.0, 5.0, 10.0]
var _speed_index := 1  # Start at 1x


func _ready() -> void:
	mouse_filter = Control.MOUSE_FILTER_STOP
	visible = false
	_create_ui()
	SurveyManager.state_changed.connect(_on_state_changed)


func _on_state_changed(new_state: SurveyManager.State) -> void:
	visible = (new_state == SurveyManager.State.POST_SURVEY)

	if visible:
		_enter_post_survey()
	else:
		_exit_post_survey()


func _enter_post_survey() -> void:
	# Set up 3D objects FIRST (before scrubber can trigger callbacks)
	_setup_topdown_camera()
	_setup_replay_objects()
	_clear_heatmap()
	# Now populate UI (which sets scrubber range and may trigger callbacks)
	_populate_results()
	# Disable operator input
	var main := _get_main()
	if main:
		var op := main.get_node_or_null("Operator")
		if op:
			op.set_process(false)
			op.set_physics_process(false)
			op.set_process_unhandled_input(false)
	# Ensure mouse is visible
	Input.mouse_mode = Input.MOUSE_MODE_VISIBLE


func _exit_post_survey() -> void:
	_cleanup_replay_objects()
	_restore_camera()


func _unhandled_input(event: InputEvent) -> void:
	if not visible:
		return

	# Speed controls: [ slower, ] faster
	if event is InputEventKey and event.pressed and not event.echo:
		if event.keycode == KEY_BRACKETLEFT:
			_change_speed(-1)
		elif event.keycode == KEY_BRACKETRIGHT:
			_change_speed(1)


func _process(delta: float) -> void:
	if not visible or not _replay_playing:
		return

	# Advance replay
	_replay_accumulator += delta * _replay_speed * _replay_speed_multiplier
	var steps := int(_replay_accumulator)
	_replay_accumulator -= steps

	if steps > 0:
		var new_index := mini(_replay_index + steps, DataRecorder.samples.size() - 1)
		# Feed samples progressively to heatmap
		_feed_heatmap_range(_replay_index, new_index)
		_replay_index = new_index

	if _replay_index >= DataRecorder.samples.size() - 1:
		_replay_playing = false
		_replay_index = DataRecorder.samples.size() - 1
		if _play_btn:
			_play_btn.text = "Play"

	_update_replay_visuals()

	if _scrubber:
		_scrubber.set_value_no_signal(float(_replay_index))
	_update_scrubber_label()


func _setup_topdown_camera() -> void:
	## Create a top-down camera looking straight down, framing the survey area.
	var main := _get_main()
	if not main:
		return

	# Save reference to current camera
	_previous_camera = get_viewport().get_camera_3d()

	# Calculate survey bounds from terrain extents
	var terrain := main.get_node_or_null("Terrain")
	var x_min := 0.0
	var x_max := 20.0
	var z_min := 0.0
	var z_max := 20.0

	if terrain:
		x_min = terrain.x_extent.x
		x_max = terrain.x_extent.y
		z_min = terrain.y_extent.x
		z_max = terrain.y_extent.y

	var center_x := (x_min + x_max) / 2.0
	var center_z := (z_min + z_max) / 2.0
	var width := x_max - x_min
	var depth := z_max - z_min
	var extent := maxf(width, depth) + 4.0  # Padding

	# Calculate camera height for orthographic-like framing
	# Using perspective camera, height = extent / (2 * tan(fov/2))
	var fov_rad := deg_to_rad(50.0)
	var camera_height := (extent / 2.0) / tan(fov_rad / 2.0)

	_topdown_camera = Camera3D.new()
	_topdown_camera.name = "TopDownCamera"
	_topdown_camera.fov = 50.0
	_topdown_camera.near = 0.5
	_topdown_camera.far = camera_height + 50.0
	# Position above center, looking straight down
	_topdown_camera.position = Vector3(center_x, camera_height, center_z)
	# Rotate to look down: -90 degrees around X
	_topdown_camera.rotation = Vector3(deg_to_rad(-90), 0, 0)

	main.add_child(_topdown_camera)
	_topdown_camera.current = true


func _setup_replay_objects() -> void:
	## Create the replay marker and trail in the 3D scene.
	var main := _get_main()
	if not main:
		return

	# Replay marker: glowing cyan sphere
	_replay_marker = MeshInstance3D.new()
	_replay_marker.name = "ReplayMarker"
	var sphere := SphereMesh.new()
	sphere.radius = 0.4
	sphere.height = 0.8
	_replay_marker.mesh = sphere

	var mat := StandardMaterial3D.new()
	mat.albedo_color = Color(0.2, 1.0, 1.0, 0.9)
	mat.emission_enabled = true
	mat.emission = Color(0.2, 1.0, 1.0)
	mat.emission_energy_multiplier = 2.0
	mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	_replay_marker.material_override = mat
	_replay_marker.visible = true

	main.add_child(_replay_marker)

	# Trail mesh for breadcrumb path
	_replay_trail = ImmediateMesh.new()
	_replay_trail_mesh = MeshInstance3D.new()
	_replay_trail_mesh.name = "ReplayTrail"
	_replay_trail_mesh.mesh = _replay_trail

	var trail_mat := StandardMaterial3D.new()
	trail_mat.albedo_color = Color(0.0, 0.8, 1.0, 0.7)
	trail_mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	trail_mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	_replay_trail_mesh.material_override = trail_mat

	main.add_child(_replay_trail_mesh)


func _cleanup_replay_objects() -> void:
	if _replay_marker and is_instance_valid(_replay_marker):
		_replay_marker.queue_free()
		_replay_marker = null
	if _replay_trail_mesh and is_instance_valid(_replay_trail_mesh):
		_replay_trail_mesh.queue_free()
		_replay_trail_mesh = null
	_replay_trail = null
	if _topdown_camera and is_instance_valid(_topdown_camera):
		_topdown_camera.queue_free()
		_topdown_camera = null


func _restore_camera() -> void:
	if _previous_camera and is_instance_valid(_previous_camera):
		_previous_camera.current = true
	_previous_camera = null


func _update_replay_visuals() -> void:
	if DataRecorder.samples.is_empty() or _replay_index >= DataRecorder.samples.size():
		return

	var sample: Dictionary = DataRecorder.samples[_replay_index]

	# Get position using CoordUtil for correct coordinate mapping
	var x_e: float = sample.get("x_e", 0.0)
	var y_n: float = sample.get("y_n", 0.0)
	var z_up: float = sample.get("z_up", 0.0)
	var godot_pos := CoordUtil.to_godot(Vector3(x_e, y_n, z_up))

	# Update marker position
	if _replay_marker and is_instance_valid(_replay_marker):
		_replay_marker.global_position = godot_pos + Vector3(0, 0.5, 0)  # Raise above ground

		# Pulse animation: scale oscillates
		var pulse := 1.0 + 0.2 * sin(Time.get_ticks_msec() / 200.0)
		_replay_marker.scale = Vector3(pulse, pulse, pulse)

	# Update trail (redraw from start to current position)
	if _replay_trail:
		_replay_trail.clear_surfaces()
		if _replay_index > 0:
			_replay_trail.surface_begin(Mesh.PRIMITIVE_LINES)
			for i in range(mini(_replay_index, DataRecorder.samples.size() - 1)):
				var s0: Dictionary = DataRecorder.samples[i]
				var s1: Dictionary = DataRecorder.samples[i + 1]
				var p0 := CoordUtil.to_godot(Vector3(
					s0.get("x_e", 0.0), s0.get("y_n", 0.0), s0.get("z_up", 0.0)
				))
				var p1 := CoordUtil.to_godot(Vector3(
					s1.get("x_e", 0.0), s1.get("y_n", 0.0), s1.get("z_up", 0.0)
				))
				_replay_trail.surface_add_vertex(p0 + Vector3(0, 0.15, 0))
				_replay_trail.surface_add_vertex(p1 + Vector3(0, 0.15, 0))
			_replay_trail.surface_end()

	# Update reading display
	_update_reading_display(sample)


func _update_reading_display(sample: Dictionary) -> void:
	if not _stats_label:
		return

	var base_text := _stats_label.text.split("\n\n--- Replay ---")[0]
	var reading: float = sample.get("reading", 0.0)
	var qual: int = sample.get("quality", 1)
	var q_str := "Good"
	if qual == 0: q_str = "Dropout"
	elif qual == 2: q_str = "Fast"
	elif qual == 3: q_str = "Offline"

	var r_display := reading
	var unit := ""
	if DataRecorder.session_instrument == "mag_gradiometer":
		r_display *= 1e9
		unit = "nT/m"

	var replay_text := "\n\n--- Replay ---\n"
	replay_text += "Sample: %d / %d\n" % [_replay_index + 1, DataRecorder.samples.size()]
	replay_text += "Time: %.1fs\n" % sample.get("t", 0.0)
	replay_text += "Pos: (%.1f, %.1f) m\n" % [sample.get("x_e", 0.0), sample.get("y_n", 0.0)]
	replay_text += "Reading: %.2f %s\n" % [r_display, unit]
	replay_text += "Quality: %s" % q_str

	_stats_label.text = base_text + replay_text


func _feed_heatmap_range(from_index: int, to_index: int) -> void:
	## Feed samples from from_index to to_index into the heatmap overlay.
	var main := _get_main()
	if not main:
		return

	var heatmap := main.get_node_or_null("HeatmapOverlay")
	if not heatmap or not heatmap.has_method("add_replay_sample"):
		return

	for i in range(from_index, to_index + 1):
		if i >= 0 and i < DataRecorder.samples.size():
			heatmap.add_replay_sample(DataRecorder.samples[i])


func _clear_heatmap() -> void:
	var main := _get_main()
	if not main:
		return
	var heatmap := main.get_node_or_null("HeatmapOverlay")
	if heatmap and heatmap.has_method("clear_for_replay"):
		heatmap.clear_for_replay()


func _change_speed(direction: int) -> void:
	_speed_index = clampi(_speed_index + direction, 0, SPEED_OPTIONS.size() - 1)
	_replay_speed_multiplier = SPEED_OPTIONS[_speed_index]
	if _speed_label:
		_speed_label.text = "%.1fx" % _replay_speed_multiplier


func _create_ui() -> void:
	# Background (semi-transparent to see 3D world through top-down camera)
	var bg := ColorRect.new()
	bg.color = Color(0.02, 0.03, 0.06, 0.45)
	bg.set_anchors_preset(PRESET_FULL_RECT)
	bg.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(bg)

	# Top bar with title
	var top_bar := PanelContainer.new()
	top_bar.set_anchors_preset(PRESET_TOP_WIDE)
	top_bar.offset_bottom = 45
	var top_style := StyleBoxFlat.new()
	top_style.bg_color = Color(0.05, 0.07, 0.1, 0.85)
	top_style.content_margin_left = 20
	top_style.content_margin_right = 20
	top_style.content_margin_top = 8
	top_style.content_margin_bottom = 8
	top_bar.add_theme_stylebox_override("panel", top_style)
	top_bar.mouse_filter = Control.MOUSE_FILTER_STOP
	add_child(top_bar)

	var top_hbox := HBoxContainer.new()
	top_hbox.add_theme_constant_override("separation", 20)
	top_bar.add_child(top_hbox)

	var title := Label.new()
	title.text = "Survey Complete — Replay"
	title.add_theme_font_size_override("font_size", 20)
	title.add_theme_color_override("font_color", Color(0.8, 0.9, 1.0))
	top_hbox.add_child(title)

	# Spacer
	var spacer := Control.new()
	spacer.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	top_hbox.add_child(spacer)

	# Export and Menu buttons in top bar
	var export_btn := Button.new()
	export_btn.text = "Export Data"
	export_btn.custom_minimum_size = Vector2(120, 30)
	export_btn.pressed.connect(_on_export)
	top_hbox.add_child(export_btn)

	var menu_btn := Button.new()
	menu_btn.text = "Main Menu"
	menu_btn.custom_minimum_size = Vector2(120, 30)
	menu_btn.pressed.connect(func(): SurveyManager.transition(SurveyManager.State.MAIN_MENU))
	top_hbox.add_child(menu_btn)

	# Left panel: Stats (semi-transparent overlay)
	var left_panel := PanelContainer.new()
	left_panel.set_anchors_preset(PRESET_TOP_LEFT)
	left_panel.offset_left = 12
	left_panel.offset_top = 55
	left_panel.offset_right = 330
	left_panel.offset_bottom = 450
	var left_style := StyleBoxFlat.new()
	left_style.bg_color = Color(0.05, 0.08, 0.12, 0.8)
	left_style.corner_radius_top_left = 6
	left_style.corner_radius_top_right = 6
	left_style.corner_radius_bottom_left = 6
	left_style.corner_radius_bottom_right = 6
	left_style.content_margin_left = 14
	left_style.content_margin_right = 14
	left_style.content_margin_top = 12
	left_style.content_margin_bottom = 12
	left_panel.add_theme_stylebox_override("panel", left_style)
	left_panel.mouse_filter = Control.MOUSE_FILTER_STOP
	add_child(left_panel)

	_stats_label = Label.new()
	_stats_label.text = "Loading results..."
	_stats_label.add_theme_font_size_override("font_size", 13)
	_stats_label.add_theme_color_override("font_color", Color(0.7, 0.75, 0.8))
	_stats_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	left_panel.add_child(_stats_label)

	# Right panel: Anomaly list
	var right_panel := PanelContainer.new()
	right_panel.set_anchors_preset(PRESET_TOP_RIGHT)
	right_panel.offset_left = -280
	right_panel.offset_top = 55
	right_panel.offset_right = -12
	right_panel.offset_bottom = 350
	right_panel.add_theme_stylebox_override("panel", left_style.duplicate())
	right_panel.mouse_filter = Control.MOUSE_FILTER_STOP
	add_child(right_panel)

	var right_vbox := VBoxContainer.new()
	right_vbox.add_theme_constant_override("separation", 6)
	right_panel.add_child(right_vbox)

	var anomaly_title := Label.new()
	anomaly_title.text = "Detected Anomalies"
	anomaly_title.add_theme_font_size_override("font_size", 15)
	anomaly_title.add_theme_color_override("font_color", Color(0.8, 0.85, 0.9))
	right_vbox.add_child(anomaly_title)

	var anomaly_scroll := ScrollContainer.new()
	anomaly_scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	right_vbox.add_child(anomaly_scroll)

	_anomaly_list = VBoxContainer.new()
	_anomaly_list.add_theme_constant_override("separation", 4)
	anomaly_scroll.add_child(_anomaly_list)

	# Bottom bar: Scrubber + speed controls
	var bottom_bar := PanelContainer.new()
	bottom_bar.set_anchors_preset(PRESET_BOTTOM_WIDE)
	bottom_bar.offset_top = -70
	var bottom_style := StyleBoxFlat.new()
	bottom_style.bg_color = Color(0.05, 0.07, 0.1, 0.85)
	bottom_style.content_margin_left = 20
	bottom_style.content_margin_right = 20
	bottom_style.content_margin_top = 8
	bottom_style.content_margin_bottom = 8
	bottom_bar.add_theme_stylebox_override("panel", bottom_style)
	bottom_bar.mouse_filter = Control.MOUSE_FILTER_STOP
	add_child(bottom_bar)

	var bottom_vbox := VBoxContainer.new()
	bottom_vbox.add_theme_constant_override("separation", 6)
	bottom_bar.add_child(bottom_vbox)

	# Scrubber row
	var scrubber_row := HBoxContainer.new()
	scrubber_row.add_theme_constant_override("separation", 10)
	bottom_vbox.add_child(scrubber_row)

	_play_btn = Button.new()
	_play_btn.text = "Play"
	_play_btn.custom_minimum_size.x = 65
	_play_btn.pressed.connect(_toggle_play)
	scrubber_row.add_child(_play_btn)

	_scrubber = HSlider.new()
	_scrubber.min_value = 0
	_scrubber.max_value = 1
	_scrubber.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_scrubber.value_changed.connect(_on_scrubber_changed)
	scrubber_row.add_child(_scrubber)

	_scrubber_label = Label.new()
	_scrubber_label.text = "0 / 0"
	_scrubber_label.custom_minimum_size.x = 110
	_scrubber_label.add_theme_font_size_override("font_size", 13)
	_scrubber_label.add_theme_color_override("font_color", Color(0.7, 0.75, 0.8))
	scrubber_row.add_child(_scrubber_label)

	# Speed controls row
	var speed_row := HBoxContainer.new()
	speed_row.add_theme_constant_override("separation", 6)
	speed_row.alignment = BoxContainer.ALIGNMENT_CENTER
	bottom_vbox.add_child(speed_row)

	var speed_title := Label.new()
	speed_title.text = "Speed:"
	speed_title.add_theme_font_size_override("font_size", 12)
	speed_title.add_theme_color_override("font_color", Color(0.5, 0.55, 0.6))
	speed_row.add_child(speed_title)

	for i in range(SPEED_OPTIONS.size()):
		var spd: float = SPEED_OPTIONS[i]
		var btn := Button.new()
		btn.text = "%.1fx" % spd if spd != int(spd) else "%dx" % int(spd)
		btn.custom_minimum_size = Vector2(50, 24)
		btn.add_theme_font_size_override("font_size", 12)
		var idx := i
		btn.pressed.connect(func():
			_speed_index = idx
			_replay_speed_multiplier = SPEED_OPTIONS[idx]
			if _speed_label:
				_speed_label.text = "%.1fx" % _replay_speed_multiplier
		)
		speed_row.add_child(btn)

	_speed_label = Label.new()
	_speed_label.text = "1.0x"
	_speed_label.add_theme_font_size_override("font_size", 13)
	_speed_label.add_theme_color_override("font_color", Color.CYAN)
	_speed_label.custom_minimum_size.x = 50
	speed_row.add_child(_speed_label)

	var keys_hint := Label.new()
	keys_hint.text = "[ / ] = Speed"
	keys_hint.add_theme_font_size_override("font_size", 11)
	keys_hint.add_theme_color_override("font_color", Color(0.35, 0.4, 0.45))
	speed_row.add_child(keys_hint)


func _toggle_play() -> void:
	_replay_playing = not _replay_playing
	if _play_btn:
		_play_btn.text = "Pause" if _replay_playing else "Play"

	# If at end, restart
	if _replay_playing and _replay_index >= DataRecorder.samples.size() - 1:
		_replay_index = 0
		_replay_accumulator = 0.0
		_clear_heatmap()


func _on_scrubber_changed(value: float) -> void:
	var new_index := int(value)
	# Feed heatmap for all samples up to this point
	if new_index > _replay_index:
		_feed_heatmap_range(_replay_index, new_index)
	elif new_index < _replay_index:
		# Going backward: clear and rebuild
		_clear_heatmap()
		_feed_heatmap_range(0, new_index)
	_replay_index = new_index
	_update_scrubber_label()
	_update_replay_visuals()


func _populate_results() -> void:
	# Reset replay state
	_replay_playing = false
	_replay_index = 0
	_replay_accumulator = 0.0
	_speed_index = 1
	_replay_speed_multiplier = 1.0
	if _speed_label:
		_speed_label.text = "1.0x"

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

	# Find anomalies
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
		var time_s := 0.0
		if _replay_index < DataRecorder.samples.size():
			time_s = DataRecorder.samples[_replay_index].get("t", 0.0)
		_scrubber_label.text = "%d / %d  (%.1fs)" % [
			_replay_index + 1, DataRecorder.samples.size(), time_s]


func _on_export() -> void:
	var path := DataRecorder.export_session()
	if not path.is_empty():
		var base_text := _stats_label.text.split("\n\n--- Replay ---")[0]
		_stats_label.text = base_text + "\n\nExported to:\n%s" % path


func _get_main() -> Node:
	return get_tree().root.get_node_or_null("Main")
