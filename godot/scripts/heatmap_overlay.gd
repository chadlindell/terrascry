## Real-time heatmap overlay projected onto terrain surface.
##
## Maintains a CPU-side data grid Image at configurable resolution (default 0.25m).
## Each pixel stores: R=normalized reading, G=coverage flag.
## A ShaderMaterial on a quad mesh maps this to a color ramp.
extends MeshInstance3D

## Grid resolution in meters per pixel
@export var resolution := 0.25

## Heatmap opacity
@export var opacity := 0.7

## Data state
var _grid_width := 0
var _grid_height := 0
var _grid_origin := Vector2.ZERO  # World-space origin (min corner)
var _data_image: Image
var _data_texture: ImageTexture
var _terrain_extent_x := Vector2(0, 20)
var _terrain_extent_y := Vector2(0, 20)

## Auto-ranging
var _min_reading := INF
var _max_reading := -INF
var _reading_count := 0
var _total_pixels := 0
var _visited_pixels := 0

## Raw reading storage for re-normalization (pixel index -> raw reading)
var _raw_readings: Dictionary = {}

## Color ramp texture
var _ramp_texture: GradientTexture1D


func _ready() -> void:
	DataRecorder.sample_recorded.connect(_on_sample_recorded)
	SurveyManager.scenario_loaded.connect(_on_scenario_loaded)
	SurveyManager.state_changed.connect(_on_state_changed)
	visible = false


func _on_state_changed(new_state: SurveyManager.State) -> void:
	if new_state == SurveyManager.State.SURVEYING:
		visible = true
	elif new_state == SurveyManager.State.MAIN_MENU:
		visible = false
		_reset()


func _on_scenario_loaded(info: Dictionary) -> void:
	var terrain: Dictionary = info.get("terrain", {})
	var x_ext: Array = terrain.get("x_extent", [0, 20])
	var y_ext: Array = terrain.get("y_extent", [0, 20])
	_terrain_extent_x = Vector2(x_ext[0], x_ext[1])
	_terrain_extent_y = Vector2(y_ext[0], y_ext[1])
	var surface_elev: float = terrain.get("surface_elevation", 0.0)
	_setup_grid(surface_elev)


func _setup_grid(surface_elev: float) -> void:
	var width := _terrain_extent_x.y - _terrain_extent_x.x
	var height := _terrain_extent_y.y - _terrain_extent_y.x

	_grid_width = int(ceil(width / resolution))
	_grid_height = int(ceil(height / resolution))
	_grid_origin = Vector2(_terrain_extent_x.x, _terrain_extent_y.x)
	_total_pixels = _grid_width * _grid_height
	_visited_pixels = 0

	# Create data image (RG format: R=value, G=coverage)
	_data_image = Image.create(_grid_width, _grid_height, false, Image.FORMAT_RGF)
	_data_image.fill(Color(0, 0, 0, 0))
	_data_texture = ImageTexture.create_from_image(_data_image)

	# Create color ramp (blue -> green -> yellow -> red)
	var gradient := Gradient.new()
	gradient.colors = PackedColorArray([
		Color(0.1, 0.2, 0.8),   # Blue (low)
		Color(0.2, 0.7, 0.3),   # Green
		Color(0.9, 0.8, 0.2),   # Yellow
		Color(0.9, 0.2, 0.1),   # Red (high)
	])
	gradient.offsets = PackedFloat32Array([0.0, 0.33, 0.66, 1.0])
	_ramp_texture = GradientTexture1D.new()
	_ramp_texture.gradient = gradient

	# Create quad mesh covering terrain
	var quad := PlaneMesh.new()
	quad.size = Vector2(width, height)
	mesh = quad

	# Position slightly above terrain
	position = Vector3(
		(_terrain_extent_x.x + _terrain_extent_x.y) / 2.0,
		surface_elev + 0.02,
		(_terrain_extent_y.x + _terrain_extent_y.y) / 2.0,
	)

	# Shader material
	var shader := _create_shader()
	var mat := ShaderMaterial.new()
	mat.shader = shader
	mat.set_shader_parameter("data_texture", _data_texture)
	mat.set_shader_parameter("color_ramp", _ramp_texture)
	mat.set_shader_parameter("opacity", opacity)
	material_override = mat

	_reset_ranges()


func _create_shader() -> Shader:
	var shader := Shader.new()
	shader.code = """
shader_type spatial;
render_mode unshaded, cull_disabled;

uniform sampler2D data_texture : filter_nearest;
uniform sampler2D color_ramp : filter_linear;
uniform float opacity : hint_range(0.0, 1.0) = 0.7;

void fragment() {
    vec2 data = texture(data_texture, UV).rg;
    float value = data.r;
    float coverage = data.g;

    if (coverage < 0.5) {
        discard;
    }

    vec3 ramp_color = texture(color_ramp, clamp(value, 0.0, 1.0)).rgb;
    ALBEDO = ramp_color;
    ALPHA = coverage * opacity;
}
"""
	return shader


func _on_sample_recorded(sample_index: int) -> void:
	if not visible or _data_image == null:
		return

	if sample_index < 1 or sample_index > DataRecorder.samples.size():
		return

	var sample: Dictionary = DataRecorder.samples[sample_index - 1]
	var x_e: float = sample.get("x_e", 0.0)
	var y_n: float = sample.get("y_n", 0.0)
	var reading: float = sample.get("reading", 0.0)

	# Convert GeoSim position to grid pixel
	var px := int((x_e - _grid_origin.x) / resolution)
	var py := int((y_n - _grid_origin.y) / resolution)

	if px < 0 or px >= _grid_width or py < 0 or py >= _grid_height:
		return

	# Update auto-ranging
	_reading_count += 1
	_min_reading = min(_min_reading, reading)
	_max_reading = max(_max_reading, reading)

	# Normalize reading
	var normalized := 0.5
	var range_val := _max_reading - _min_reading
	if range_val > 1e-15:
		normalized = clampf((reading - _min_reading) / range_val, 0.0, 1.0)

	# Store raw reading for renormalization
	var pixel_key := px + py * _grid_width
	_raw_readings[pixel_key] = reading

	# Check if this is a new pixel
	var old_coverage := _data_image.get_pixel(px, py).g
	if old_coverage < 0.5:
		_visited_pixels += 1

	# Set pixel: R=normalized value, G=coverage flag
	_data_image.set_pixel(px, py, Color(normalized, 1.0, 0.0))

	# Update texture (batch every few samples for performance)
	if _reading_count % 3 == 0 or _reading_count < 10:
		_data_texture.update(_data_image)

	# Re-normalize periodically for best color spread
	if _reading_count % 50 == 0 and range_val > 1e-15:
		_renormalize_grid()

	# Update coverage percentage in HUD
	if _total_pixels > 0:
		var coverage_pct := (float(_visited_pixels) / float(_total_pixels)) * 100.0
		var hud := _get_hud()
		if hud and hud.has_method("update_coverage"):
			hud.update_coverage(coverage_pct)


func _renormalize_grid() -> void:
	## Re-normalize all visited pixels based on current min/max range.
	var range_val := _max_reading - _min_reading
	if range_val < 1e-15:
		return

	# Rebuild all visited pixels with updated normalization
	for pixel_key in _raw_readings:
		var raw: float = _raw_readings[pixel_key]
		var normalized := clampf((raw - _min_reading) / range_val, 0.0, 1.0)
		@warning_ignore("integer_division")
		var px: int = pixel_key % _grid_width
		@warning_ignore("integer_division")
		var py: int = pixel_key / _grid_width
		_data_image.set_pixel(px, py, Color(normalized, 1.0, 0.0))

	_data_texture.update(_data_image)


func _reset_ranges() -> void:
	_min_reading = INF
	_max_reading = -INF
	_reading_count = 0
	_visited_pixels = 0
	_raw_readings.clear()


func _reset() -> void:
	_reset_ranges()
	if _data_image:
		_data_image.fill(Color(0, 0, 0, 0))
		if _data_texture:
			_data_texture.update(_data_image)


func _get_hud() -> Control:
	var main := get_tree().root.get_node_or_null("Main")
	if main:
		return main.get_node_or_null("UI/HUD")
	return null


## Get coverage percentage.
func get_coverage_percent() -> float:
	if _total_pixels <= 0:
		return 0.0
	return (float(_visited_pixels) / float(_total_pixels)) * 100.0
