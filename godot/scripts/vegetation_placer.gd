## Scatters procedural vegetation (trees, bushes, rocks) around the survey area.
##
## Uses MultiMeshInstance3D for efficient rendering. Placement follows density
## zones: sparse inside survey area, medium in buffer, dense tree line at perimeter.
## All meshes are procedurally generated with organic vertex displacement.
extends Node3D

## Density controls (instances per square meter)
@export var survey_density := 0.01  # Very sparse inside survey area
@export var buffer_density := 0.08  # Medium density in buffer zone
@export var perimeter_density := 0.15  # Dense tree line at perimeter

## Zone distances from survey boundary
@export var buffer_distance := 3.0  # Buffer zone width (m)
@export var perimeter_distance := 5.0  # Where tree line starts (m beyond survey)
@export var perimeter_width := 4.0  # Tree line depth (m)

## Terrain reference
var _terrain: Node = null
var _x_extent := Vector2(0.0, 20.0)
var _y_extent := Vector2(0.0, 20.0)
var _surface_elevation := 0.0

## MultiMesh nodes
var _tree_multimesh: MultiMeshInstance3D
var _bush_multimesh: MultiMeshInstance3D
var _rock_multimesh: MultiMeshInstance3D

## RNG seeded from scenario
var _rng := RandomNumberGenerator.new()

## Vegetation shader
var _vegetation_shader: Shader

## Cached vegetation ShaderMaterials for player push uniform updates
var _vegetation_materials: Array[ShaderMaterial] = []


func _ready() -> void:
	_vegetation_shader = load("res://shaders/vegetation.gdshader") as Shader
	SurveyManager.scenario_loaded.connect(_on_scenario_loaded)


func _process(_delta: float) -> void:
	if SurveyManager.active_operator and not _vegetation_materials.is_empty():
		var pos: Vector3 = SurveyManager.active_operator.global_position
		for mat in _vegetation_materials:
			mat.set_shader_parameter("player_world_pos", pos)


func _on_scenario_loaded(info: Dictionary) -> void:
	var terrain_data: Dictionary = info.get("terrain", {})
	var x_ext: Array = terrain_data.get("x_extent", [0, 20])
	var y_ext: Array = terrain_data.get("y_extent", [0, 20])
	_x_extent = Vector2(x_ext[0], x_ext[1])
	_y_extent = Vector2(y_ext[0], y_ext[1])
	_surface_elevation = terrain_data.get("surface_elevation", 0.0)

	# Get terrain reference for height queries
	_terrain = get_parent().get_node_or_null("Terrain")

	# Seed RNG from scenario name
	var scenario_name: String = info.get("name", "default")
	_rng.seed = hash(scenario_name)

	_generate_vegetation()


func _generate_vegetation() -> void:
	# Clear existing
	for child in get_children():
		child.queue_free()
	_vegetation_materials.clear()

	var tree_transforms: Array[Transform3D] = []
	var tree_custom: Array[Color] = []
	var bush_transforms: Array[Transform3D] = []
	var bush_custom: Array[Color] = []
	var rock_transforms: Array[Transform3D] = []
	var rock_custom: Array[Color] = []

	# Generate placement points using stratified random sampling
	var start_x := _x_extent.x - perimeter_distance - perimeter_width
	var start_z := _y_extent.x - perimeter_distance - perimeter_width
	var end_x := _x_extent.y + perimeter_distance + perimeter_width
	var end_z := _y_extent.y + perimeter_distance + perimeter_width

	# Sample points with density based on zone
	var cell_size := 1.0  # 1m grid for candidate generation
	var x := start_x
	while x < end_x:
		var z := start_z
		while z < end_z:
			var px := x + _rng.randf() * cell_size
			var pz := z + _rng.randf() * cell_size

			var zone := _get_zone(px, pz)
			var density := _zone_density(zone)

			if _rng.randf() < density:
				var height := _get_height(px, pz)
				var t := Transform3D()

				# Random Y rotation
				t = t.rotated(Vector3.UP, _rng.randf() * TAU)

				# Per-instance custom data: hue_shift, wind_phase, ao_factor
				var hue_shift := _rng.randf_range(-1.0, 1.0)
				var wind_phase := _rng.randf()
				var ao_factor := _rng.randf_range(0.6, 1.0)
				var custom := Color(hue_shift, wind_phase, ao_factor, 0.0)

				match zone:
					2:  # Perimeter — trees
						var scale_factor := _rng.randf_range(0.7, 1.3)
						t = t.scaled(Vector3.ONE * scale_factor)
						t.origin = Vector3(px, height, pz)
						tree_transforms.append(t)
						tree_custom.append(custom)
					1:  # Buffer — bushes and rocks
						if _rng.randf() < 0.6:
							var scale_factor := _rng.randf_range(0.5, 1.0)
							t = t.scaled(Vector3.ONE * scale_factor)
							t.origin = Vector3(px, height, pz)
							bush_transforms.append(t)
							bush_custom.append(custom)
						else:
							var scale_factor := _rng.randf_range(0.3, 0.8)
							t = t.scaled(Vector3.ONE * scale_factor)
							t.origin = Vector3(px, height, pz)
							rock_transforms.append(t)
							rock_custom.append(custom)
					0:  # Survey area — occasional small rock or low bush
						if _rng.randf() < 0.4:
							var scale_factor := _rng.randf_range(0.2, 0.5)
							t = t.scaled(Vector3.ONE * scale_factor)
							t.origin = Vector3(px, height, pz)
							rock_transforms.append(t)
							rock_custom.append(custom)
						else:
							var scale_factor := _rng.randf_range(0.3, 0.6)
							t = t.scaled(Vector3.ONE * scale_factor)
							t.origin = Vector3(px, height, pz)
							bush_transforms.append(t)
							bush_custom.append(custom)
			z += cell_size
		x += cell_size

	# Create MultiMesh instances
	if tree_transforms.size() > 0:
		var tree_mat := _create_vegetation_material(
			Color(0.18, 0.35, 0.12), Color(0.30, 0.50, 0.20), 3.5, 0.15)
		_tree_multimesh = _create_multimesh(
			_create_tree_mesh(), tree_transforms, tree_custom, tree_mat)
		add_child(_tree_multimesh)
		if tree_mat is ShaderMaterial:
			_vegetation_materials.append(tree_mat)

	if bush_transforms.size() > 0:
		var bush_mat := _create_vegetation_material(
			Color(0.22, 0.38, 0.15), Color(0.35, 0.52, 0.22), 0.5, 0.08)
		_bush_multimesh = _create_multimesh(
			_create_bush_mesh(), bush_transforms, bush_custom, bush_mat)
		add_child(_bush_multimesh)
		if bush_mat is ShaderMaterial:
			_vegetation_materials.append(bush_mat)

	if rock_transforms.size() > 0:
		_rock_multimesh = _create_multimesh(
			_create_rock_mesh(), rock_transforms, rock_custom,
			_create_rock_material())
		add_child(_rock_multimesh)

	print("[Vegetation] Placed %d trees, %d bushes, %d rocks" % [
		tree_transforms.size(), bush_transforms.size(), rock_transforms.size()])


## Zone classification: 0=survey, 1=buffer, 2=perimeter
func _get_zone(x: float, z: float) -> int:
	var dx := _distance_to_boundary(x, _x_extent)
	var dz := _distance_to_boundary(z, _y_extent)
	var dist := minf(dx, dz)

	if dist >= 0.0:
		return 0  # Inside survey area
	elif absf(dist) <= buffer_distance:
		return 1  # Buffer zone
	elif absf(dist) <= perimeter_distance + perimeter_width:
		return 2  # Perimeter (tree line)
	return -1  # Beyond placement range


func _distance_to_boundary(val: float, extent: Vector2) -> float:
	## Positive = inside, negative = outside
	return minf(val - extent.x, extent.y - val)


func _zone_density(zone: int) -> float:
	match zone:
		0: return survey_density
		1: return buffer_density
		2: return perimeter_density
		_: return 0.0


func _get_height(x: float, z: float) -> float:
	if _terrain and _terrain.has_method("get_height_at"):
		return _terrain.get_height_at(x, z)
	return _surface_elevation


func _create_multimesh(base_mesh: Mesh, transforms: Array[Transform3D],
		custom_data: Array[Color], mat: Material) -> MultiMeshInstance3D:
	var mm := MultiMesh.new()
	mm.transform_format = MultiMesh.TRANSFORM_3D
	mm.use_custom_data = true
	mm.mesh = base_mesh
	mm.instance_count = transforms.size()

	for i in range(transforms.size()):
		mm.set_instance_transform(i, transforms[i])
		mm.set_instance_custom_data(i, custom_data[i])

	var mmi := MultiMeshInstance3D.new()
	mmi.multimesh = mm
	mmi.material_override = mat
	mmi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_ON
	return mmi


## Simple seeded noise for vertex displacement
func _noise_displace(pos: Vector3, seed_val: int, amplitude: float) -> Vector3:
	# Hash-based displacement along normal direction
	var h := (sin(pos.x * 12.9898 + pos.y * 78.233 + pos.z * 45.164 + float(seed_val)) * 43758.5453)
	h = h - floor(h)
	return pos.normalized() * (h - 0.5) * 2.0 * amplitude


## Procedural tree mesh — organic trunk + multi-layer crown
func _create_tree_mesh() -> ArrayMesh:
	var st := SurfaceTool.new()
	var noise_rng := RandomNumberGenerator.new()
	noise_rng.seed = 7731

	# Trunk (cylinder with organic displacement, 8 sides)
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var trunk_radius := 0.12
	var trunk_height := 1.8
	var trunk_sides := 8
	var trunk_rings := 4

	# Build trunk vertices with rings for organic displacement
	var trunk_verts: Array[Vector3] = []
	for ring in range(trunk_rings + 1):
		var y := float(ring) / float(trunk_rings) * trunk_height
		var ring_radius := trunk_radius * (1.0 - float(ring) / float(trunk_rings) * 0.15)
		for seg in range(trunk_sides):
			var angle := float(seg) / float(trunk_sides) * TAU
			var v := Vector3(cos(angle) * ring_radius, y, sin(angle) * ring_radius)
			# Organic displacement
			var disp := noise_rng.randf_range(-0.02, 0.02)
			v.x += disp
			v.z += noise_rng.randf_range(-0.02, 0.02)
			trunk_verts.append(v)

	# Triangulate trunk
	for ring in range(trunk_rings):
		for seg in range(trunk_sides):
			var i00 := ring * trunk_sides + seg
			var i10 := ring * trunk_sides + (seg + 1) % trunk_sides
			var i01 := (ring + 1) * trunk_sides + seg
			var i11 := (ring + 1) * trunk_sides + (seg + 1) % trunk_sides

			# UV: use height for Y, angle for X
			var u0 := float(seg) / float(trunk_sides)
			var u1 := float(seg + 1) / float(trunk_sides)
			var v0 := float(ring) / float(trunk_rings)
			var v1 := float(ring + 1) / float(trunk_rings)

			st.set_uv(Vector2(u0, v0))
			st.add_vertex(trunk_verts[i00])
			st.set_uv(Vector2(u1, v0))
			st.add_vertex(trunk_verts[i10])
			st.set_uv(Vector2(u0, v1))
			st.add_vertex(trunk_verts[i01])

			st.set_uv(Vector2(u1, v0))
			st.add_vertex(trunk_verts[i10])
			st.set_uv(Vector2(u1, v1))
			st.add_vertex(trunk_verts[i11])
			st.set_uv(Vector2(u0, v1))
			st.add_vertex(trunk_verts[i01])

	# Multi-layer crown: 3 offset cones at different heights/radii
	var crown_layers := [
		{"base": trunk_height * 0.5, "top": trunk_height + 0.8, "radius": 1.1, "offset": Vector3(0.1, 0, 0.05)},
		{"base": trunk_height * 0.65, "top": trunk_height + 1.3, "radius": 0.85, "offset": Vector3(-0.05, 0, -0.08)},
		{"base": trunk_height * 0.8, "top": trunk_height + 1.7, "radius": 0.6, "offset": Vector3(0.0, 0, 0.03)},
	]
	var crown_sides := 8

	for layer in crown_layers:
		var cr: float = layer["radius"]
		var cb: float = layer["base"]
		var ct: float = layer["top"]
		var co: Vector3 = layer["offset"]

		for i in range(crown_sides):
			var angle0 := float(i) / float(crown_sides) * TAU
			var angle1 := float(i + 1) / float(crown_sides) * TAU
			# Organic radius variation per vertex
			var r0 := cr * (1.0 + noise_rng.randf_range(-0.15, 0.15))
			var r1 := cr * (1.0 + noise_rng.randf_range(-0.15, 0.15))
			var cx0 := cos(angle0) * r0 + co.x
			var cz0 := sin(angle0) * r0 + co.z
			var cx1 := cos(angle1) * r1 + co.x
			var cz1 := sin(angle1) * r1 + co.z

			# Crown UV: map to 0-1 range for gradient
			var crown_uv_base := (cb - trunk_height * 0.5) / (trunk_height + 1.7 - trunk_height * 0.5)
			var crown_uv_top := 1.0

			st.set_uv(Vector2(0, crown_uv_base))
			st.add_vertex(Vector3(cx0, cb, cz0))
			st.set_uv(Vector2(1, crown_uv_base))
			st.add_vertex(Vector3(cx1, cb, cz1))
			st.set_uv(Vector2(0.5, crown_uv_top))
			st.add_vertex(Vector3(co.x, ct, co.z))

	st.generate_normals()
	return st.commit()


## Procedural bush mesh — low sphere with organic displacement
func _create_bush_mesh() -> ArrayMesh:
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var noise_rng := RandomNumberGenerator.new()
	noise_rng.seed = 8842

	var radius := 0.5
	var height := 0.4
	var rings := 4
	var segments := 10

	var verts: Array[Vector3] = []
	var uvs: Array[Vector2] = []
	for ring in range(rings + 1):
		var v0 := float(ring) / float(rings)
		var y := -height * 0.3 + v0 * height
		var r := radius * sin(v0 * PI) * 1.2
		r = maxf(r, 0.05)
		for seg in range(segments):
			var a := float(seg) / float(segments) * TAU
			var v := Vector3(cos(a) * r, y, sin(a) * r)
			# Organic displacement
			v.x += noise_rng.randf_range(-0.04, 0.04)
			v.y += noise_rng.randf_range(-0.02, 0.02)
			v.z += noise_rng.randf_range(-0.04, 0.04)
			verts.append(v)
			uvs.append(Vector2(float(seg) / float(segments), v0))

	for ring in range(rings):
		for seg in range(segments):
			var i00 := ring * segments + seg
			var i10 := ring * segments + (seg + 1) % segments
			var i01 := (ring + 1) * segments + seg
			var i11 := (ring + 1) * segments + (seg + 1) % segments

			if i01 < verts.size() and i11 < verts.size():
				st.set_uv(uvs[i00])
				st.add_vertex(verts[i00])
				st.set_uv(uvs[i10])
				st.add_vertex(verts[i10])
				st.set_uv(uvs[i01])
				st.add_vertex(verts[i01])

				st.set_uv(uvs[i10])
				st.add_vertex(verts[i10])
				st.set_uv(uvs[i11])
				st.add_vertex(verts[i11])
				st.set_uv(uvs[i01])
				st.add_vertex(verts[i01])

	st.generate_normals()
	return st.commit()


## Procedural rock mesh — irregular low-poly sphere with variable squash
func _create_rock_mesh() -> ArrayMesh:
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)

	var radius := 0.3
	var segments := 8
	var rings := 5

	# Use deterministic seed for consistent rock shape
	var rock_rng := RandomNumberGenerator.new()
	rock_rng.seed = 42

	var verts: Array[Vector3] = []
	var y_squash := rock_rng.randf_range(0.4, 0.7)

	for ring in range(rings + 1):
		var phi := float(ring) / float(rings) * PI
		for seg in range(segments):
			var theta := float(seg) / float(segments) * TAU
			var r := radius * (0.7 + rock_rng.randf() * 0.6)
			var v := Vector3(
				cos(theta) * sin(phi) * r,
				cos(phi) * r * y_squash,
				sin(theta) * sin(phi) * r
			)
			# Additional jagged displacement along normal
			var disp := rock_rng.randf_range(-0.03, 0.03)
			v += v.normalized() * disp
			verts.append(v)

	for ring in range(rings):
		for seg in range(segments):
			var i00 := ring * segments + seg
			var i10 := ring * segments + (seg + 1) % segments
			var i01 := (ring + 1) * segments + seg
			var i11 := (ring + 1) * segments + (seg + 1) % segments

			if i01 < verts.size() and i11 < verts.size():
				st.add_vertex(verts[i00])
				st.add_vertex(verts[i10])
				st.add_vertex(verts[i01])

				st.add_vertex(verts[i10])
				st.add_vertex(verts[i11])
				st.add_vertex(verts[i01])

	st.generate_normals()
	return st.commit()


func _create_vegetation_material(base_col: Color, tip_col: Color,
		grad_height: float, wind_str: float) -> Material:
	if not _vegetation_shader:
		# Fallback to StandardMaterial3D if shader not found
		var mat := StandardMaterial3D.new()
		mat.albedo_color = base_col
		mat.roughness = 0.85
		mat.cull_mode = BaseMaterial3D.CULL_DISABLED
		return mat

	var mat := ShaderMaterial.new()
	mat.shader = _vegetation_shader
	mat.set_shader_parameter("base_color", Vector3(base_col.r, base_col.g, base_col.b))
	mat.set_shader_parameter("tip_color", Vector3(tip_col.r, tip_col.g, tip_col.b))
	mat.set_shader_parameter("gradient_height", grad_height)
	mat.set_shader_parameter("wind_strength", wind_str)
	mat.set_shader_parameter("wind_speed", 1.0)
	mat.set_shader_parameter("ao_strength", 0.5)
	return mat


func _create_rock_material() -> StandardMaterial3D:
	var mat := StandardMaterial3D.new()
	mat.albedo_color = Color(0.45, 0.42, 0.38)
	mat.roughness = 0.95
	return mat
