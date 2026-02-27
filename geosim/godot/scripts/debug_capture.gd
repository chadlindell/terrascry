## Debug capture mode: auto-loads a scenario, auto-walks a zigzag path across
## the terrain, captures per-frame diagnostics and periodic screenshots,
## then writes a report.
##
## Activate via command line:
##   godot --path godot/ -- --debug-capture
## Options (after --):
##   --debug-capture          Enable debug capture mode
##   --debug-duration=30      Duration in seconds (default 20)
##   --debug-scenario=NAME    Scenario file base name (default "scattered-debris")
##   --debug-no-quit          Don't auto-quit after capture finishes
extends Node

## Configuration
var capture_duration := 20.0
var scenario_name := "scattered-debris"
var auto_quit := true

## State
var _active := false
var _elapsed := 0.0
var _frame_count := 0
var _screenshot_interval := 2.0
var _next_screenshot := 0.0
var _screenshot_count := 0
var _capture_dir := ""

## Auto-walk state
var _waypoints: Array[Vector3] = []
var _waypoint_idx := 0
var _walk_speed := 3.0  # m/s
var _terrain_x := Vector2(0, 20)
var _terrain_y := Vector2(0, 20)
var _surface_elev := 0.0

## Collected data
var _frame_log: Array[Dictionary] = []
var _screenshot_paths: Array[String] = []
var _start_time_msec := 0

## HUD label
var _debug_label: Label = null


func _ready() -> void:
	# Parse command-line args
	var args := OS.get_cmdline_user_args()
	if not _parse_args(args):
		# Not in debug mode — self-remove
		queue_free()
		return

	_active = true

	# Create output directory
	var timestamp := Time.get_datetime_string_from_system().replace(":", "-").replace("T", "_")
	_capture_dir = "user://debug_capture_%s" % timestamp
	DirAccess.make_dir_recursive_absolute(_capture_dir)
	print("[DebugCapture] Output dir: %s" % _capture_dir)
	print("[DebugCapture] Duration: %.0fs | Scenario: %s" % [capture_duration, scenario_name])

	# Create on-screen overlay
	_create_overlay()

	# Wait for main scene to finish _ready, then auto-load scenario and start survey
	await get_tree().process_frame
	await get_tree().process_frame
	_auto_start()


func _parse_args(args: Array) -> bool:
	var found := false
	for arg in args:
		var a: String = str(arg)
		if a == "--debug-capture":
			found = true
		elif a.begins_with("--debug-duration="):
			capture_duration = float(a.split("=")[1])
		elif a.begins_with("--debug-scenario="):
			scenario_name = a.split("=")[1]
		elif a == "--debug-no-quit":
			auto_quit = false
	return found


func _auto_start() -> void:
	print("[DebugCapture] Auto-loading scenario: %s" % scenario_name)

	# Resolve scenario path
	var scenario_path := "scenarios/%s.json" % scenario_name

	# Load scenario (skipping menu flow)
	var ok := await SurveyManager.load_scenario(scenario_path)
	if not ok:
		push_error("[DebugCapture] Failed to load scenario '%s'" % scenario_path)
		_finish_capture()
		return

	# Extract terrain bounds
	var terrain: Dictionary = SurveyManager.scenario_info.get("terrain", {})
	var x_ext: Array = terrain.get("x_extent", [0, 20])
	var y_ext: Array = terrain.get("y_extent", [0, 20])
	_terrain_x = Vector2(x_ext[0], x_ext[1])
	_terrain_y = Vector2(y_ext[0], y_ext[1])
	_surface_elev = terrain.get("surface_elevation", 0.0)

	# Set a default survey plan
	SurveyManager.survey_plan = {
		"origin": [x_ext[0], y_ext[0]],
		"line_spacing": 1.0,
		"line_length": x_ext[1] - x_ext[0],
		"num_lines": int((y_ext[1] - y_ext[0]) / 1.0),
		"direction": "east",
	}

	# Build zigzag waypoints across the terrain
	_build_walk_path()

	# Give terrain/vegetation a frame to build
	await get_tree().process_frame
	await get_tree().process_frame

	# Activate ground operator (normally done via operator_switched signal)
	var op: Node3D = get_parent().get_node_or_null("Operator")
	if op:
		op.visible = true
		op.set_process(true)
		op.set_physics_process(true)
		op.set_process_unhandled_input(true)
		var cam: Camera3D = op.get_node_or_null("Camera3D")
		if cam:
			cam.current = true
		SurveyManager.active_operator = op

	# Transition directly to SURVEYING (bypasses menus)
	SurveyManager.transition(SurveyManager.State.SURVEYING)

	# Wait one frame for instrument rig to be created by survey_started signal
	await get_tree().process_frame

	_start_time_msec = Time.get_ticks_msec()
	_next_screenshot = 0.5  # First screenshot at 0.5s (after initial render settles)
	print("[DebugCapture] Survey started — capture running for %.0f seconds" % capture_duration)
	print("[DebugCapture] Auto-walk: %d waypoints across terrain" % _waypoints.size())


func _build_walk_path() -> void:
	## Create a zigzag path that covers the terrain with margin.
	## Pattern: walk east, step north, walk west, step north, repeat.
	var margin := 2.0
	var x_min: float = _terrain_x.x + margin
	var x_max: float = _terrain_x.y - margin
	var z_min: float = _terrain_y.x + margin
	var z_max: float = _terrain_y.y - margin
	var y: float = _surface_elev + 2.0

	var line_spacing := 5.0  # Meters between zigzag lines
	var z := z_min
	var going_east := true

	_waypoints.clear()
	while z <= z_max:
		if going_east:
			_waypoints.append(Vector3(x_min, y, z))
			_waypoints.append(Vector3(x_max, y, z))
		else:
			_waypoints.append(Vector3(x_max, y, z))
			_waypoints.append(Vector3(x_min, y, z))
		going_east = not going_east
		z += line_spacing

	# Add a diagonal pass back through the center for variety
	var cx: float = (x_min + x_max) / 2.0
	var cz: float = (z_min + z_max) / 2.0
	_waypoints.append(Vector3(cx, y, cz))
	_waypoints.append(Vector3(x_min, y, z_min))

	_waypoint_idx = 0


func _process(delta: float) -> void:
	if not _active:
		return
	if SurveyManager.current_state != SurveyManager.State.SURVEYING:
		return

	_elapsed += delta
	_frame_count += 1

	# Force-hide instrument rig, markers, and pin camera every frame
	_enforce_debug_view()

	# Auto-walk the operator
	_auto_walk(delta)

	# Collect per-frame data
	var entry := _collect_frame_data()
	_frame_log.append(entry)

	# Periodic screenshots
	if _elapsed >= _next_screenshot:
		_take_screenshot()
		_next_screenshot += _screenshot_interval

	# Update overlay
	_update_overlay(entry)

	# Check duration
	if _elapsed >= capture_duration:
		_finish_capture()


func _enforce_debug_view() -> void:
	# Hide instrument rig every frame (it gets recreated on instrument switch)
	var op: Node3D = SurveyManager.active_operator
	if op:
		var inst_rig: Node3D = op.get_node_or_null("InstrumentRig")
		if inst_rig and inst_rig.visible:
			inst_rig.visible = false

	# Hide artifact overlay markers
	var markers: Node3D = get_parent().get_node_or_null("ObjectMarkers")
	if markers and markers.visible:
		markers.visible = false


func _auto_walk(delta: float) -> void:
	var op: CharacterBody3D = SurveyManager.active_operator as CharacterBody3D
	if not op or _waypoints.is_empty():
		return
	if _waypoint_idx >= _waypoints.size():
		_waypoint_idx = 0  # Loop

	var target := _waypoints[_waypoint_idx]
	var pos := op.global_position
	var to_target := target - pos
	to_target.y = 0  # Flatten for horizontal distance

	var dist := to_target.length()
	if dist < 0.5:
		# Reached waypoint — advance
		_waypoint_idx += 1
		if _waypoint_idx >= _waypoints.size():
			_waypoint_idx = 0
		return

	# Face toward target
	var dir := to_target.normalized()
	var target_yaw := atan2(dir.x, dir.z)
	op.rotation.y = lerp_angle(op.rotation.y, target_yaw, delta * 3.0)

	# Look steeply down at ground to capture terrain surface detail
	var cam: Camera3D = op.get_node_or_null("Camera3D")
	if cam:
		cam.rotation.x = deg_to_rad(-50.0)

	# Set velocity to walk toward target
	var walk_dir := Vector3(dir.x, 0, dir.z).normalized()
	op.velocity.x = walk_dir.x * _walk_speed
	op.velocity.z = walk_dir.z * _walk_speed


func _collect_frame_data() -> Dictionary:
	var fps: float = Engine.get_frames_per_second()
	var frame_time: float = 1000.0 / max(fps, 1.0)

	# Player position
	var pos := Vector3.ZERO
	var look_dir := Vector3.FORWARD
	if SurveyManager.active_operator:
		pos = SurveyManager.active_operator.global_position
		var cam: Camera3D = SurveyManager.active_operator.get_node_or_null("Camera3D")
		if cam:
			look_dir = -cam.global_transform.basis.z

	# Rendering stats from Performance singleton
	var draw_calls := Performance.get_monitor(Performance.RENDER_TOTAL_DRAW_CALLS_IN_FRAME)
	var objects_in_frame := Performance.get_monitor(Performance.RENDER_TOTAL_OBJECTS_IN_FRAME)
	var primitives := Performance.get_monitor(Performance.RENDER_TOTAL_PRIMITIVES_IN_FRAME)
	var video_mem := Performance.get_monitor(Performance.RENDER_VIDEO_MEM_USED)
	var texture_mem := Performance.get_monitor(Performance.RENDER_TEXTURE_MEM_USED)
	var buffer_mem := Performance.get_monitor(Performance.RENDER_BUFFER_MEM_USED)

	# Physics stats
	var physics_objects := Performance.get_monitor(Performance.PHYSICS_3D_ACTIVE_OBJECTS)
	var collision_pairs := Performance.get_monitor(Performance.PHYSICS_3D_COLLISION_PAIRS)

	# Memory
	var static_mem := Performance.get_monitor(Performance.MEMORY_STATIC)

	return {
		"t": snappedf(_elapsed, 0.001),
		"fps": fps,
		"frame_ms": snappedf(frame_time, 0.01),
		"pos": [snappedf(pos.x, 0.01), snappedf(pos.y, 0.01), snappedf(pos.z, 0.01)],
		"look": [snappedf(look_dir.x, 0.001), snappedf(look_dir.y, 0.001), snappedf(look_dir.z, 0.001)],
		"draw_calls": int(draw_calls),
		"objects": int(objects_in_frame),
		"primitives": int(primitives),
		"video_mem_mb": snappedf(video_mem / 1048576.0, 0.1),
		"texture_mem_mb": snappedf(texture_mem / 1048576.0, 0.1),
		"buffer_mem_mb": snappedf(buffer_mem / 1048576.0, 0.1),
		"static_mem_mb": snappedf(static_mem / 1048576.0, 0.1),
		"physics_objects": int(physics_objects),
		"collision_pairs": int(collision_pairs),
	}


func _take_screenshot() -> void:
	var img := get_viewport().get_texture().get_image()
	if not img:
		return
	var filename := "screenshot_%02d_%.1fs.png" % [_screenshot_count, _elapsed]
	var path := "%s/%s" % [_capture_dir, filename]
	img.save_png(path)
	_screenshot_paths.append(path)
	_screenshot_count += 1
	print("[DebugCapture] Screenshot %d at %.1fs" % [_screenshot_count, _elapsed])


func _finish_capture() -> void:
	_active = false
	print("[DebugCapture] Capture finished — %d frames, %d screenshots" % [
		_frame_count, _screenshot_count])

	# Take final screenshot
	_take_screenshot()

	# Compute summary stats
	var summary := _compute_summary()

	# Write frame log as JSON
	var frame_log_path := "%s/frame_log.json" % _capture_dir
	var f := FileAccess.open(frame_log_path, FileAccess.WRITE)
	if f:
		f.store_string(JSON.stringify(_frame_log, "  "))
		f.close()
		print("[DebugCapture] Frame log: %s (%d entries)" % [frame_log_path, _frame_log.size()])

	# Write summary report
	var report_path := "%s/report.txt" % _capture_dir
	var rf := FileAccess.open(report_path, FileAccess.WRITE)
	if rf:
		rf.store_string(_format_report(summary))
		rf.close()
		print("[DebugCapture] Report: %s" % report_path)

	# Print summary to console
	print("\n=== DEBUG CAPTURE SUMMARY ===")
	print("Scenario: %s" % scenario_name)
	print("Duration: %.1fs | Frames: %d" % [_elapsed, _frame_count])
	print("FPS: avg=%.1f  min=%.1f  max=%.1f  p1=%.1f  p5=%.1f" % [
		summary.fps_avg, summary.fps_min, summary.fps_max, summary.fps_p1, summary.fps_p5])
	print("Frame time: avg=%.2fms  max=%.2fms  p99=%.2fms" % [
		summary.frame_ms_avg, summary.frame_ms_max, summary.frame_ms_p99])
	print("Draw calls: avg=%d  max=%d" % [summary.draw_calls_avg, summary.draw_calls_max])
	print("Primitives: avg=%dk  max=%dk" % [summary.primitives_avg / 1000, summary.primitives_max / 1000])
	print("VRAM: %.1f MB (tex: %.1f MB, buf: %.1f MB)" % [
		summary.video_mem_mb, summary.texture_mem_mb, summary.buffer_mem_mb])
	print("Screenshots: %d saved to %s" % [_screenshot_count, _capture_dir])
	print("Output: %s" % ProjectSettings.globalize_path(_capture_dir))
	print("=============================\n")

	if auto_quit:
		# Brief delay so prints flush
		await get_tree().create_timer(0.5).timeout
		get_tree().quit()


func _compute_summary() -> Dictionary:
	if _frame_log.is_empty():
		return {}

	var fps_vals: Array[float] = []
	var frame_ms_vals: Array[float] = []
	var draw_call_vals: Array[int] = []
	var prim_vals: Array[int] = []
	var last_entry: Dictionary = _frame_log.back()

	for entry in _frame_log:
		fps_vals.append(float(entry.fps))
		frame_ms_vals.append(float(entry.frame_ms))
		draw_call_vals.append(int(entry.draw_calls))
		prim_vals.append(int(entry.primitives))

	fps_vals.sort()
	frame_ms_vals.sort()
	draw_call_vals.sort()
	prim_vals.sort()

	var n := fps_vals.size()

	return {
		"fps_avg": _arr_avg_f(fps_vals),
		"fps_min": fps_vals[0],
		"fps_max": fps_vals[n - 1],
		"fps_p1": fps_vals[max(int(n * 0.01), 0)],
		"fps_p5": fps_vals[max(int(n * 0.05), 0)],
		"frame_ms_avg": _arr_avg_f(frame_ms_vals),
		"frame_ms_max": frame_ms_vals[n - 1],
		"frame_ms_p99": frame_ms_vals[min(int(n * 0.99), n - 1)],
		"draw_calls_avg": int(_arr_avg_i(draw_call_vals)),
		"draw_calls_max": draw_call_vals[n - 1],
		"primitives_avg": int(_arr_avg_i(prim_vals)),
		"primitives_max": prim_vals[n - 1],
		"video_mem_mb": float(last_entry.video_mem_mb),
		"texture_mem_mb": float(last_entry.texture_mem_mb),
		"buffer_mem_mb": float(last_entry.buffer_mem_mb),
		"static_mem_mb": float(last_entry.static_mem_mb),
	}


func _arr_avg_f(arr: Array[float]) -> float:
	if arr.is_empty():
		return 0.0
	var s := 0.0
	for v in arr:
		s += v
	return s / float(arr.size())


func _arr_avg_i(arr: Array[int]) -> float:
	if arr.is_empty():
		return 0.0
	var s := 0
	for v in arr:
		s += v
	return float(s) / float(arr.size())


func _format_report(summary: Dictionary) -> String:
	var lines: PackedStringArray = []
	lines.append("GeoSim Debug Capture Report")
	lines.append("===========================")
	lines.append("Timestamp: %s" % Time.get_datetime_string_from_system())
	lines.append("Scenario: %s" % scenario_name)
	lines.append("Duration: %.1f seconds" % _elapsed)
	lines.append("Total frames: %d" % _frame_count)
	lines.append("")
	lines.append("--- Performance ---")
	lines.append("FPS:  avg=%.1f  min=%.1f  max=%.1f  1%%ile=%.1f  5%%ile=%.1f" % [
		summary.get("fps_avg", 0), summary.get("fps_min", 0), summary.get("fps_max", 0),
		summary.get("fps_p1", 0), summary.get("fps_p5", 0)])
	lines.append("Frame time:  avg=%.2fms  max=%.2fms  99%%ile=%.2fms" % [
		summary.get("frame_ms_avg", 0), summary.get("frame_ms_max", 0),
		summary.get("frame_ms_p99", 0)])
	lines.append("")
	lines.append("--- Rendering ---")
	lines.append("Draw calls:  avg=%d  max=%d" % [
		summary.get("draw_calls_avg", 0), summary.get("draw_calls_max", 0)])
	lines.append("Primitives:  avg=%d  max=%d" % [
		summary.get("primitives_avg", 0), summary.get("primitives_max", 0)])
	lines.append("")
	lines.append("--- Memory ---")
	lines.append("Video memory: %.1f MB" % summary.get("video_mem_mb", 0))
	lines.append("  Textures: %.1f MB" % summary.get("texture_mem_mb", 0))
	lines.append("  Buffers: %.1f MB" % summary.get("buffer_mem_mb", 0))
	lines.append("Static memory: %.1f MB" % summary.get("static_mem_mb", 0))
	lines.append("")
	lines.append("--- Screenshots ---")
	for path in _screenshot_paths:
		lines.append("  %s" % path)
	lines.append("")
	lines.append("--- Frame Log ---")
	lines.append("See frame_log.json for per-frame data (%d entries)" % _frame_log.size())
	return "\n".join(lines)


func _create_overlay() -> void:
	# Add a debug info label to the UI layer
	var canvas := CanvasLayer.new()
	canvas.name = "DebugCaptureOverlay"
	canvas.layer = 100  # On top of everything
	add_child(canvas)

	_debug_label = Label.new()
	_debug_label.name = "DebugLabel"
	_debug_label.position = Vector2(10, 10)
	_debug_label.size = Vector2(500, 300)
	_debug_label.add_theme_font_size_override("font_size", 14)
	_debug_label.add_theme_color_override("font_color", Color.YELLOW)
	# Drop shadow for readability
	_debug_label.add_theme_color_override("font_shadow_color", Color(0, 0, 0, 0.8))
	_debug_label.add_theme_constant_override("shadow_offset_x", 1)
	_debug_label.add_theme_constant_override("shadow_offset_y", 1)
	canvas.add_child(_debug_label)

	# Countdown bar
	var progress := ProgressBar.new()
	progress.name = "CaptureProgress"
	progress.position = Vector2(10, 4)
	progress.size = Vector2(300, 6)
	progress.min_value = 0.0
	progress.max_value = capture_duration
	progress.show_percentage = false
	progress.modulate = Color(1.0, 0.8, 0.0, 0.7)
	canvas.add_child(progress)


func _update_overlay(entry: Dictionary) -> void:
	if not _debug_label:
		return

	var remaining: float = max(capture_duration - _elapsed, 0.0)
	var pos: Array = entry.get("pos", [0, 0, 0])
	var wp_info := "wp %d/%d" % [_waypoint_idx, _waypoints.size()]

	_debug_label.text = (
		"[DEBUG CAPTURE] %.0fs remaining  %s\n" % [remaining, wp_info] +
		"FPS: %d  (%.1fms)\n" % [entry.fps, entry.frame_ms] +
		"Pos: (%.1f, %.1f, %.1f)\n" % [pos[0], pos[1], pos[2]] +
		"Draw calls: %d  Objects: %d\n" % [entry.draw_calls, entry.objects] +
		"Primitives: %d\n" % entry.primitives +
		"VRAM: %.0f MB  Tex: %.0f MB\n" % [entry.video_mem_mb, entry.texture_mem_mb] +
		"Screenshots: %d" % _screenshot_count
	)

	var progress: ProgressBar = _debug_label.get_parent().get_node_or_null("CaptureProgress")
	if progress:
		progress.value = _elapsed
