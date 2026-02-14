## Real-time heatmap overlay draped onto terrain surface.
##
## Maintains a CPU-side data grid Image at configurable resolution (default 0.25m).
## Each pixel stores: R=normalized reading, G=coverage flag.
## A ShaderMaterial on a terrain-conforming ArrayMesh maps this to a color ramp.
## Uses P5/P95 percentile normalization to resist outlier color-pumping.
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

## Terrain reference for height queries
var _terrain: Node = null

## Auto-ranging (percentile-based)
var _reading_count := 0
var _total_pixels := 0
var _visited_pixels := 0
var _p5_reading := 0.0
var _p95_reading := 0.0

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
	elif new_state == SurveyManager.State.POST_SURVEY:
		visible = true  # Keep visible for replay
	elif new_state == SurveyManager.State.MAIN_MENU:
		visible = false
		_reset()


func _on_scenario_loaded(info: Dictionary) -> void:
	var terrain_data: Dictionary = info.get("terrain", {})
	var x_ext: Array = terrain_data.get("x_extent", [0, 20])
	var y_ext: Array = terrain_data.get("y_extent", [0, 20])
	_terrain_extent_x = Vector2(x_ext[0], x_ext[1])
	_terrain_extent_y = Vector2(y_ext[0], y_ext[1])
	var surface_elev: float = terrain_data.get("surface_elevation", 0.0)

	# Get terrain reference for height queries
	_terrain = get_parent().get_node_or_null("Terrain")

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

	# Build terrain-draped ArrayMesh instead of flat PlaneMesh
	mesh = _build_draped_mesh(surface_elev)

	# Position at world origin (mesh vertices are in world coords)
	position = Vector3.ZERO

	# Shader material
	var shader := _create_shader()
	var mat := ShaderMaterial.new()
	mat.shader = shader
	mat.set_shader_parameter("data_texture", _data_texture)
	mat.set_shader_parameter("color_ramp", _ramp_texture)
	mat.set_shader_parameter("opacity", opacity)
	material_override = mat

	_reset_ranges()


func _build_draped_mesh(surface_elev: float) -> ArrayMesh:
	## Build an ArrayMesh grid that conforms to terrain height at each vertex.
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)

	var cols := _grid_width
	var rows := _grid_height

	# Generate vertices at terrain height
	for row in range(rows + 1):
		for col in range(cols + 1):
			var world_x := _grid_origin.x + col * resolution
			var world_z := _grid_origin.y + row * resolution
			var h := _get_terrain_height(world_x, world_z, surface_elev) + 0.03

			var u := float(col) / float(cols)
			var v := float(row) / float(rows)

			st.set_uv(Vector2(u, v))
			st.add_vertex(Vector3(world_x, h, world_z))

	# Generate triangle indices
	for row in range(rows):
		for col in range(cols):
			var i00 := row * (cols + 1) + col
			var i10 := i00 + 1
			var i01 := i00 + (cols + 1)
			var i11 := i01 + 1

			# Triangle 1
			st.add_index(i00)
			st.add_index(i10)
			st.add_index(i01)

			# Triangle 2
			st.add_index(i10)
			st.add_index(i11)
			st.add_index(i01)

	st.generate_normals()
	return st.commit()


func _get_terrain_height(world_x: float, world_z: float, fallback: float) -> float:
	if _terrain and _terrain.has_method("get_height_at"):
		return _terrain.get_height_at(world_x, world_z)
	return fallback


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

    vec3 ramp_color = texture(color_ramp, vec2(clamp(value, 0.0, 1.0), 0.5)).rgb;
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

	_reading_count += 1

	# Store raw reading for renormalization
	var pixel_key := px + py * _grid_width
	_raw_readings[pixel_key] = reading

	# Check if this is a new pixel
	var old_coverage := _data_image.get_pixel(px, py).g
	if old_coverage < 0.5:
		_visited_pixels += 1

	# Normalize using percentile bounds
	var normalized := _normalize_reading(reading)

	# Set pixel: R=normalized value, G=coverage flag
	_data_image.set_pixel(px, py, Color(normalized, 1.0, 0.0))

	# Bilinear smoothing — blend into neighboring cells with 25% weight
	_smooth_neighbors(px, py, normalized)

	# Update texture (batch every few samples for performance)
	if _reading_count % 3 == 0 or _reading_count < 10:
		_data_texture.update(_data_image)

	# Recompute percentiles and re-normalize periodically
	if _reading_count % 50 == 0 and _raw_readings.size() >= 5:
		_update_percentiles()
		_renormalize_grid()

	# Update coverage percentage in HUD
	if _total_pixels > 0:
		var coverage_pct := (float(_visited_pixels) / float(_total_pixels)) * 100.0
		var hud := _get_hud()
		if hud and hud.has_method("update_coverage"):
			hud.update_coverage(coverage_pct)


func _normalize_reading(reading: float) -> float:
	var range_val := _p95_reading - _p5_reading
	if range_val < 1e-15:
		return 0.5
	return clampf((reading - _p5_reading) / range_val, 0.0, 1.0)


func _smooth_neighbors(px: int, py: int, value: float) -> void:
	## Blend into neighboring cells with 25% weight to reduce blockiness.
	var blend := 0.25
	for dy in range(-1, 2):
		for dx in range(-1, 2):
			if dx == 0 and dy == 0:
				continue
			var nx := px + dx
			var ny := py + dy
			if nx < 0 or nx >= _grid_width or ny < 0 or ny >= _grid_height:
				continue
			var existing := _data_image.get_pixel(nx, ny)
			if existing.g > 0.5:
				# Only blend if neighbor already has data
				var blended := existing.r * (1.0 - blend) + value * blend
				_data_image.set_pixel(nx, ny, Color(blended, 1.0, 0.0))


func _update_percentiles() -> void:
	## Compute P5 and P95 from raw readings for robust normalization.
	var values: Array[float] = []
	for key in _raw_readings:
		values.append(_raw_readings[key])
	values.sort()

	var n := values.size()
	if n < 2:
		if n == 1:
			_p5_reading = values[0]
			_p95_reading = values[0]
		return

	_p5_reading = values[int(n * 0.05)]
	_p95_reading = values[int(min(n * 0.95, n - 1))]


func _renormalize_grid() -> void:
	## Re-normalize all visited pixels based on current P5/P95 bounds.
	var range_val := _p95_reading - _p5_reading
	if range_val < 1e-15:
		return

	for pixel_key in _raw_readings:
		var raw: float = _raw_readings[pixel_key]
		var normalized := clampf((raw - _p5_reading) / range_val, 0.0, 1.0)
		@warning_ignore("integer_division")
		var px: int = pixel_key % _grid_width
		@warning_ignore("integer_division")
		var py: int = pixel_key / _grid_width
		_data_image.set_pixel(px, py, Color(normalized, 1.0, 0.0))

	_data_texture.update(_data_image)


func _reset_ranges() -> void:
	_p5_reading = 0.0
	_p95_reading = 0.0
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


## Clear the heatmap for replay mode (keeps grid setup, clears data).
func clear_for_replay() -> void:
	_reset_ranges()
	if _data_image:
		_data_image.fill(Color(0, 0, 0, 0))
		if _data_texture:
			_data_texture.update(_data_image)
	visible = true


## Add a single sample during replay (same logic as live recording).
func add_replay_sample(sample: Dictionary) -> void:
	if _data_image == null:
		return

	var x_e: float = sample.get("x_e", 0.0)
	var y_n: float = sample.get("y_n", 0.0)
	var reading: float = sample.get("reading", 0.0)

	var px := int((x_e - _grid_origin.x) / resolution)
	var py := int((y_n - _grid_origin.y) / resolution)

	if px < 0 or px >= _grid_width or py < 0 or py >= _grid_height:
		return

	_reading_count += 1

	var pixel_key := px + py * _grid_width
	_raw_readings[pixel_key] = reading

	var old_coverage := _data_image.get_pixel(px, py).g
	if old_coverage < 0.5:
		_visited_pixels += 1

	var normalized := _normalize_reading(reading)
	_data_image.set_pixel(px, py, Color(normalized, 1.0, 0.0))
	_smooth_neighbors(px, py, normalized)

	# Update texture periodically for performance
	if _reading_count % 5 == 0 or _reading_count < 10:
		_data_texture.update(_data_image)

	# Recompute percentiles periodically
	if _reading_count % 50 == 0 and _raw_readings.size() >= 5:
		_update_percentiles()
		_renormalize_grid()
