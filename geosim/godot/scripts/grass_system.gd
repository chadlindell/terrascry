## Procedural grass system using MultiMeshInstance3D with wind shader.
##
## Scatters grass blade quads across the terrain, with density fading near
## survey lines and color variation matching the terrain shader.
extends MultiMeshInstance3D

## Grass parameters
@export var grass_count := 2000
@export var blade_width := 0.08
@export var blade_height_min := 0.12
@export var blade_height_max := 0.25
@export var density_near_lines := 0.3  # Reduced density near survey paths

## Terrain reference
var _terrain: Node = null
var _x_extent := Vector2(0.0, 20.0)
var _y_extent := Vector2(0.0, 20.0)
var _surface_elevation := 0.0

## RNG
var _rng := RandomNumberGenerator.new()

## Cached shader material for player push uniform
var _grass_shader_mat: ShaderMaterial
var _grass_base_color := Color(0.25, 0.40, 0.15)
var _grass_tip_color := Color(0.45, 0.58, 0.25)
var _wind_strength := 0.4
var _wind_speed := 1.5
var _wind_turbulence := 0.3
var _alpha_cutoff := 0.3


func _ready() -> void:
	SurveyManager.scenario_loaded.connect(_on_scenario_loaded)


func _process(_delta: float) -> void:
	if _grass_shader_mat and SurveyManager.active_operator:
		var pos: Vector3 = SurveyManager.active_operator.global_position
		_grass_shader_mat.set_shader_parameter("player_world_pos", pos)


func _on_scenario_loaded(info: Dictionary) -> void:
	var terrain_data: Dictionary = info.get("terrain", {})
	var x_ext: Array = terrain_data.get("x_extent", [0, 20])
	var y_ext: Array = terrain_data.get("y_extent", [0, 20])
	_x_extent = Vector2(x_ext[0], x_ext[1])
	_y_extent = Vector2(y_ext[0], y_ext[1])
	_surface_elevation = terrain_data.get("surface_elevation", 0.0)

	_terrain = get_parent().get_node_or_null("Terrain")

	var scenario_name: String = info.get("name", "default")
	_rng.seed = hash(scenario_name) + 7  # Different seed from vegetation

	_apply_scenario_profile(info)

	# Defer generation so terrain has time to rebuild with crater features
	await get_tree().process_frame
	_generate_grass()


func _generate_grass() -> void:
	# Create grass blade mesh (billboard quad)
	var blade_mesh := _create_blade_mesh()

	# Load wind shader
	var shader := load("res://shaders/grass_wind.gdshader") as Shader
	if shader:
		var shader_mat := ShaderMaterial.new()
		shader_mat.shader = shader
		shader_mat.set_shader_parameter("grass_color_base", Vector3(
			_grass_base_color.r, _grass_base_color.g, _grass_base_color.b
		))
		shader_mat.set_shader_parameter("grass_color_tip", Vector3(
			_grass_tip_color.r, _grass_tip_color.g, _grass_tip_color.b
		))
		shader_mat.set_shader_parameter("wind_strength", _wind_strength)
		shader_mat.set_shader_parameter("wind_speed", _wind_speed)
		shader_mat.set_shader_parameter("wind_turbulence", _wind_turbulence)
		shader_mat.set_shader_parameter("alpha_cutoff", _alpha_cutoff)
		# Load blade texture if available
		var blade_tex = load("res://assets/textures/grass/grass_blade.png")
		if blade_tex:
			shader_mat.set_shader_parameter("blade_texture", blade_tex)
			shader_mat.set_shader_parameter("use_texture", true)
		material_override = shader_mat
		_grass_shader_mat = shader_mat
	else:
		var mat := StandardMaterial3D.new()
		mat.albedo_color = Color(0.3, 0.45, 0.18)
		mat.cull_mode = BaseMaterial3D.CULL_DISABLED
		material_override = mat

	# Create MultiMesh
	var mm := MultiMesh.new()
	mm.transform_format = MultiMesh.TRANSFORM_3D
	mm.mesh = blade_mesh
	mm.instance_count = grass_count

	# Place grass blades
	var margin := 2.0  # Extend slightly beyond survey bounds
	var x_min := _x_extent.x - margin
	var x_max := _x_extent.y + margin
	var z_min := _y_extent.x - margin
	var z_max := _y_extent.y + margin

	for i in range(grass_count):
		var px := _rng.randf_range(x_min, x_max)
		var pz := _rng.randf_range(z_min, z_max)
		var height := _get_height(px, pz)

		# Random scale variation
		var scale_y := _rng.randf_range(blade_height_min, blade_height_max) / blade_height_max
		var scale_x := _rng.randf_range(0.7, 1.3)

		var t := Transform3D()
		# Random Y rotation for variety
		t = t.rotated(Vector3.UP, _rng.randf() * TAU)
		# Slight random tilt
		t = t.rotated(Vector3.RIGHT, _rng.randf_range(-0.15, 0.15))
		t = t.scaled(Vector3(scale_x, scale_y, scale_x))
		t.origin = Vector3(px, height, pz)

		mm.set_instance_transform(i, t)

	multimesh = mm
	cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF

	print("[Grass] Placed %d grass blades" % grass_count)


func _create_blade_mesh() -> ArrayMesh:
	## Create 3 intersecting quads in a star pattern (0/60/120 degrees).
	## Each quad tapers at the top (30% width at tip).
	## No explicit back faces — shader uses cull_disabled for double-sided rendering.
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)

	var hw := blade_width / 2.0
	var tip_hw := hw * 0.3  # Taper to 30% at tip
	var h := blade_height_max

	# 3 rotations: 0, 60, 120 degrees for star pattern
	var angles := [0.0, PI / 3.0, 2.0 * PI / 3.0]

	for angle in angles:
		var ca := cos(angle)
		var sa := sin(angle)

		# Bottom-left, bottom-right, top-left, top-right
		var bl := Vector3(-hw * ca, 0.0, -hw * sa)
		var br := Vector3(hw * ca, 0.0, hw * sa)
		var tl := Vector3(-tip_hw * ca, h, -tip_hw * sa)
		var tr := Vector3(tip_hw * ca, h, tip_hw * sa)

		# Single face per quad (cull_disabled renders both sides)
		st.set_uv(Vector2(0, 0))
		st.add_vertex(bl)
		st.set_uv(Vector2(1, 0))
		st.add_vertex(br)
		st.set_uv(Vector2(1, 1))
		st.add_vertex(tr)

		st.set_uv(Vector2(0, 0))
		st.add_vertex(bl)
		st.set_uv(Vector2(1, 1))
		st.add_vertex(tr)
		st.set_uv(Vector2(0, 1))
		st.add_vertex(tl)

	st.generate_normals()
	return st.commit()


func _get_height(x: float, z: float) -> float:
	if _terrain and _terrain.has_method("get_height_at"):
		return _terrain.get_height_at(x, z)
	return _surface_elevation


func _apply_scenario_profile(info: Dictionary) -> void:
	var terrain_data: Dictionary = info.get("terrain", {})
	var layers: Array = terrain_data.get("layers", [])
	var metadata: Dictionary = info.get("metadata", {})
	var env_profile: Dictionary = info.get("environment_profile", {})
	var name := str(info.get("name", "")).to_lower()
	var desc := str(info.get("description", "")).to_lower()
	var category := str(metadata.get("category", "")).to_lower()

	var mean_cond := 0.05
	if not layers.is_empty():
		var s := 0.0
		for layer in layers:
			s += float(layer.get("conductivity", 0.05))
		mean_cond = s / float(layers.size())

	var wetness := clampf(mean_cond / 0.25, 0.0, 1.0)
	if metadata.has("water_table_depth"):
		var wt_depth := float(metadata.get("water_table_depth", 2.0))
		wetness = clampf(wetness + clampf((1.0 - wt_depth) / 1.0, 0.0, 0.5), 0.0, 1.0)
	if name.contains("swamp") or name.contains("marsh") or desc.contains("waterlogged"):
		wetness = clampf(wetness + 0.35, 0.0, 1.0)
	if env_profile.has("wetness"):
		wetness = clampf(float(env_profile.get("wetness", wetness)), 0.0, 1.0)

	# Density and geometry profile by moisture/category.
	var vegetation_density := clampf(float(env_profile.get("vegetation_density", 0.7)), 0.1, 1.0)
	grass_count = int(lerpf(7000.0, 4000.0, wetness) * vegetation_density)
	blade_height_min = lerpf(0.11, 0.16, wetness)
	blade_height_max = lerpf(0.24, 0.35, wetness)
	blade_width = lerpf(0.07, 0.09, wetness)
	density_near_lines = lerpf(0.28, 0.4, wetness)
	if category.contains("uxo") or name.contains("crater"):
		grass_count = int(float(grass_count) * 0.65)
		blade_height_max *= 0.85
	if category.contains("forensic"):
		grass_count = int(float(grass_count) * 0.78)
		blade_height_max *= 0.9

	var top_color := Color(0.28, 0.44, 0.18)
	if not layers.is_empty():
		top_color = _color_from_hex(str(layers[0].get("color", "")), top_color)
	_grass_base_color = top_color.darkened(0.2)
	_grass_tip_color = top_color.lightened(0.24)
	var wind_intensity := clampf(float(env_profile.get("wind_intensity", 0.5)), 0.0, 1.0)
	_wind_strength = lerpf(0.24, 0.62, max(wetness, wind_intensity))
	_wind_speed = lerpf(1.1, 2.1, wetness)
	_wind_turbulence = lerpf(0.22, 0.55, wetness)
	_alpha_cutoff = lerpf(0.34, 0.26, wetness)


func _color_from_hex(hex: String, fallback: Color) -> Color:
	if hex.is_empty():
		return fallback
	return Color.from_string(hex, fallback)
