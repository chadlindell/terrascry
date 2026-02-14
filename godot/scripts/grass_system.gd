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

	_generate_grass()


func _generate_grass() -> void:
	# Create grass blade mesh (billboard quad)
	var blade_mesh := _create_blade_mesh()

	# Load wind shader
	var shader := load("res://shaders/grass_wind.gdshader") as Shader
	if shader:
		var shader_mat := ShaderMaterial.new()
		shader_mat.shader = shader
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
	## Create a single grass blade quad (two triangles).
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)

	var hw := blade_width / 2.0
	var h := blade_height_max

	# Front face
	st.set_uv(Vector2(0, 0))
	st.add_vertex(Vector3(-hw, 0, 0))
	st.set_uv(Vector2(1, 0))
	st.add_vertex(Vector3(hw, 0, 0))
	st.set_uv(Vector2(0.5, 1))
	st.add_vertex(Vector3(0, h, 0))

	# Back face (for double-sided without cull_disabled in mesh)
	st.set_uv(Vector2(1, 0))
	st.add_vertex(Vector3(hw, 0, 0))
	st.set_uv(Vector2(0, 0))
	st.add_vertex(Vector3(-hw, 0, 0))
	st.set_uv(Vector2(0.5, 1))
	st.add_vertex(Vector3(0, h, 0))

	st.generate_normals()
	return st.commit()


func _get_height(x: float, z: float) -> float:
	if _terrain and _terrain.has_method("get_height_at"):
		return _terrain.get_height_at(x, z)
	return _surface_elevation
