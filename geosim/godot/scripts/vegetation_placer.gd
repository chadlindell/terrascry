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

## Loaded meshes (if available) — used for MultiMesh instancing
var _tree_meshes: Array[Mesh] = []
var _bush_meshes: Array[Mesh] = []
var _rock_meshes: Array[Mesh] = []

## Loaded scenes for high-quality models (FBX with embedded textures)
## These are instanced individually to preserve their materials.
var _tree_scenes: Array[PackedScene] = []
var _bush_scenes: Array[PackedScene] = []

## Cached vegetation ShaderMaterials for player push uniform updates
var _vegetation_materials: Array[ShaderMaterial] = []
var _tree_base_color := Color(0.18, 0.35, 0.12)
var _tree_tip_color := Color(0.30, 0.50, 0.20)
var _bush_base_color := Color(0.22, 0.38, 0.15)
var _bush_tip_color := Color(0.35, 0.52, 0.22)
var _tree_wind_strength := 0.15
var _bush_wind_strength := 0.08
var _rock_color := Color(0.45, 0.42, 0.38)


func _ready() -> void:
	_vegetation_shader = load("res://shaders/vegetation.gdshader") as Shader
	_load_vegetation_meshes()
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

	_apply_scenario_profile(info)

	# Defer generation so terrain has time to rebuild with crater features
	await get_tree().process_frame
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

	# Place trees — use scene instancing for FBX models (preserves materials),
	# MultiMesh for extracted meshes, or procedural fallback
	if tree_transforms.size() > 0:
		if not _tree_scenes.is_empty():
			# Scene instancing: preserves Poly Haven PBR materials
			_place_as_scene_instances(_tree_scenes, tree_transforms)
		elif not _tree_meshes.is_empty():
			var tree_mat := _create_vegetation_material(
				_tree_base_color, _tree_tip_color, 3.5, _tree_wind_strength, true)
			_create_variant_multimeshes(_tree_meshes, tree_transforms, tree_custom, tree_mat)
			if tree_mat is ShaderMaterial:
				_vegetation_materials.append(tree_mat)
		else:
			var tree_mat := _create_vegetation_material(
				_tree_base_color, _tree_tip_color, 3.5, _tree_wind_strength, false)
			_tree_multimesh = _create_multimesh(
				_create_tree_mesh(), tree_transforms, tree_custom, tree_mat)
			add_child(_tree_multimesh)
			if tree_mat is ShaderMaterial:
				_vegetation_materials.append(tree_mat)

	# Place bushes — scene instancing for FBX, MultiMesh for extracted meshes
	if bush_transforms.size() > 0:
		if not _bush_scenes.is_empty():
			_place_as_scene_instances(_bush_scenes, bush_transforms)
		elif not _bush_meshes.is_empty():
			var bush_mat := _create_vegetation_material(
				_bush_base_color, _bush_tip_color, 0.5, _bush_wind_strength, true)
			_create_variant_multimeshes(_bush_meshes, bush_transforms, bush_custom, bush_mat)
			if bush_mat is ShaderMaterial:
				_vegetation_materials.append(bush_mat)
		else:
			var bush_mat := _create_vegetation_material(
				_bush_base_color, _bush_tip_color, 0.5, _bush_wind_strength, false)
			_bush_multimesh = _create_multimesh(
				_create_bush_mesh(), bush_transforms, bush_custom, bush_mat)
			add_child(_bush_multimesh)
			if bush_mat is ShaderMaterial:
				_vegetation_materials.append(bush_mat)

	# Place rocks — MultiMesh or procedural
	if rock_transforms.size() > 0:
		if not _rock_meshes.is_empty():
			var rock_mat := _create_vegetation_material(
				_rock_color, _rock_color.lightened(0.1), 0.3, 0.0, true)
			_create_variant_multimeshes(_rock_meshes, rock_transforms, rock_custom, rock_mat)
		else:
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
	# LOD: smooth fade-out at distance
	mmi.visibility_range_end = 60.0
	mmi.visibility_range_end_margin = 10.0
	mmi.visibility_range_fade_mode = GeometryInstance3D.VISIBILITY_RANGE_FADE_SELF
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

	# Multi-layer crown: 5 offset cones at different heights/radii for full canopy
	var crown_layers := [
		{"base": trunk_height * 0.45, "top": trunk_height + 0.7, "radius": 1.2, "offset": Vector3(0.12, 0, 0.08)},
		{"base": trunk_height * 0.55, "top": trunk_height + 1.0, "radius": 1.0, "offset": Vector3(-0.1, 0, 0.12)},
		{"base": trunk_height * 0.65, "top": trunk_height + 1.3, "radius": 0.85, "offset": Vector3(-0.05, 0, -0.08)},
		{"base": trunk_height * 0.75, "top": trunk_height + 1.5, "radius": 0.7, "offset": Vector3(0.08, 0, -0.1)},
		{"base": trunk_height * 0.85, "top": trunk_height + 1.7, "radius": 0.5, "offset": Vector3(0.0, 0, 0.03)},
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


## Procedural bush mesh — multi-cluster organic shape
func _create_bush_mesh() -> ArrayMesh:
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var noise_rng := RandomNumberGenerator.new()
	noise_rng.seed = 8842

	# Generate 4 overlapping sphere clusters at offset positions
	var clusters := [
		{"offset": Vector3(0.0, 0.0, 0.0), "radius": 0.45, "height": 0.38},
		{"offset": Vector3(0.18, 0.05, 0.1), "radius": 0.35, "height": 0.32},
		{"offset": Vector3(-0.15, 0.03, -0.12), "radius": 0.32, "height": 0.30},
		{"offset": Vector3(0.05, -0.02, -0.18), "radius": 0.28, "height": 0.28},
	]

	var rings := 4
	var segments := 8

	for cluster in clusters:
		var c_offset: Vector3 = cluster["offset"]
		var c_radius: float = cluster["radius"]
		var c_height: float = cluster["height"]

		var verts: Array[Vector3] = []
		var uvs: Array[Vector2] = []
		for ring in range(rings + 1):
			var v0 := float(ring) / float(rings)
			var y := -c_height * 0.3 + v0 * c_height
			var r := c_radius * sin(v0 * PI) * 1.2
			r = maxf(r, 0.05)
			for seg in range(segments):
				var a := float(seg) / float(segments) * TAU
				var v := Vector3(cos(a) * r, y, sin(a) * r) + c_offset
				# Organic displacement
				v.x += noise_rng.randf_range(-0.05, 0.05)
				v.y += noise_rng.randf_range(-0.03, 0.03)
				v.z += noise_rng.randf_range(-0.05, 0.05)
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


## Procedural rock mesh — irregular low-poly sphere with moss/lichen on top faces
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

	# Moss/lichen colors
	var moss_color := Color(0.25, 0.38, 0.18)
	var lichen_color := Color(0.55, 0.58, 0.42)

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
				# Vertex color based on upward-facing direction (moss on top)
				for vi in [i00, i10, i01, i10, i11, i01]:
					var vn := verts[vi].normalized()
					var up_factor := clampf((vn.y - 0.2) / 0.6, 0.0, 1.0)
					# Mix rock base -> moss/lichen on top faces
					var vert_color := _rock_color
					if up_factor > 0.1:
						var moss_noise := rock_rng.randf()
						if moss_noise < 0.5:
							vert_color = _rock_color.lerp(moss_color, up_factor * 0.7)
						else:
							vert_color = _rock_color.lerp(lichen_color, up_factor * 0.5)
					st.set_color(vert_color)
					st.add_vertex(verts[vi])

	st.generate_normals()
	return st.commit()


func _place_as_scene_instances(scenes: Array[PackedScene], transforms: Array[Transform3D]) -> void:
	## Place vegetation as individual scene instances (preserves model materials).
	## Computes AABB to offset models so their base sits on the ground.
	# Pre-compute Y offsets per scene variant (AABB min Y)
	var y_offsets: Array[float] = []
	for scene in scenes:
		y_offsets.append(_compute_scene_base_offset(scene))

	for i in range(transforms.size()):
		var scene_idx := i % scenes.size()
		var inst := scenes[scene_idx].instantiate() as Node3D
		if inst:
			var t := transforms[i]
			# Shift origin down so the model's lowest point sits on the ground
			t.origin.y -= y_offsets[scene_idx]
			inst.transform = t
			# Disable shadow casting for performance on many instances
			_set_shadow_mode_recursive(inst, GeometryInstance3D.SHADOW_CASTING_SETTING_OFF)
			add_child(inst)


func _set_shadow_mode_recursive(node: Node, mode: GeometryInstance3D.ShadowCastingSetting) -> void:
	if node is GeometryInstance3D:
		(node as GeometryInstance3D).cast_shadow = mode
	for child in node.get_children():
		_set_shadow_mode_recursive(child, mode)


func _compute_scene_base_offset(scene: PackedScene) -> float:
	## Compute the Y offset needed to place a model's base on the ground.
	## Returns the minimum Y coordinate of all MeshInstance3D AABBs in the scene.
	var inst := scene.instantiate() as Node3D
	if not inst:
		return 0.0
	var min_y := 999.0
	var found := false
	var nodes := [inst]
	while not nodes.is_empty():
		var n: Node = nodes.pop_back()
		if n is MeshInstance3D:
			var mi := n as MeshInstance3D
			if mi.mesh:
				var aabb := mi.mesh.get_aabb()
				# Account for the node's local transform
				var local_min_y: float = n.position.y + aabb.position.y
				if local_min_y < min_y:
					min_y = local_min_y
					found = true
		for child in n.get_children():
			nodes.append(child)
	inst.queue_free()
	return min_y if found else 0.0


func _create_variant_multimeshes(meshes: Array[Mesh], transforms: Array[Transform3D],
		custom_data: Array[Color], mat: Material) -> void:
	# Split transforms among mesh variants for visual diversity
	var variant_count := meshes.size()
	var variant_transforms: Array = []
	var variant_custom: Array = []
	for _i in range(variant_count):
		variant_transforms.append([])
		variant_custom.append([])

	for i in range(transforms.size()):
		var vi := i % variant_count
		variant_transforms[vi].append(transforms[i])
		variant_custom[vi].append(custom_data[i])

	for vi in range(variant_count):
		var vt: Array[Transform3D] = []
		var vc: Array[Color] = []
		for t in variant_transforms[vi]:
			vt.append(t)
		for c in variant_custom[vi]:
			vc.append(c)
		if vt.size() > 0:
			var mmi := _create_multimesh(meshes[vi], vt, vc, mat)
			add_child(mmi)


func _create_vegetation_material(base_col: Color, tip_col: Color,
		grad_height: float, wind_str: float, use_vert_height: bool = false) -> Material:
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
	mat.set_shader_parameter("use_vertex_height", use_vert_height)
	return mat


func _create_rock_material() -> StandardMaterial3D:
	var mat := StandardMaterial3D.new()
	mat.albedo_color = Color.WHITE
	mat.vertex_color_use_as_albedo = true
	mat.roughness = 0.95
	return mat


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
		wetness = clampf(wetness + 0.3, 0.0, 1.0)
	if env_profile.has("wetness"):
		wetness = clampf(float(env_profile.get("wetness", wetness)), 0.0, 1.0)

	# Density profile by environment:
	# - wetter sites => denser perimeter vegetation
	# - UXO/crater sites => sparser and rockier
	var vegetation_density := clampf(float(env_profile.get("vegetation_density", 0.7)), 0.1, 1.0)
	survey_density = lerpf(0.012, 0.004, wetness) * vegetation_density
	buffer_density = lerpf(0.10, 0.06, wetness) * vegetation_density
	perimeter_density = lerpf(0.12, 0.22, wetness) * vegetation_density
	if category.contains("uxo") or name.contains("crater"):
		survey_density *= 0.8
		buffer_density *= 0.85
		perimeter_density *= 0.75
	if category.contains("forensic"):
		survey_density *= 0.7
		buffer_density *= 0.8

	var top_color := Color(0.30, 0.46, 0.22)
	if not layers.is_empty():
		top_color = _color_from_hex(str(layers[0].get("color", "")), top_color)
	var stem_color := top_color.darkened(0.25)
	var tip_color := top_color.lightened(0.2)

	_tree_base_color = stem_color.darkened(0.08)
	_tree_tip_color = tip_color
	_bush_base_color = stem_color
	_bush_tip_color = tip_color.lightened(0.08)
	var wind_intensity := clampf(float(env_profile.get("wind_intensity", 0.5)), 0.0, 1.0)
	_tree_wind_strength = lerpf(0.10, 0.24, max(wetness, wind_intensity))
	_bush_wind_strength = lerpf(0.07, 0.17, max(wetness, wind_intensity))
	_rock_color = top_color.darkened(0.3).lerp(Color(0.48, 0.45, 0.42), 0.5)


func _load_packed_scene(path: String) -> PackedScene:
	## Try to load a PackedScene from any model file (FBX, GLB, etc).
	if not ResourceLoader.exists(path):
		return null
	var scene := load(path) as PackedScene
	return scene


func _load_mesh_from_scene(path: String) -> Mesh:
	## Load a mesh from any scene file (FBX, GLB, GLTF, .tscn).
	## Finds the first MeshInstance3D and extracts its mesh.
	if not ResourceLoader.exists(path):
		return null
	var scene := load(path) as PackedScene
	if not scene:
		return null
	var instance := scene.instantiate()
	if not instance:
		return null
	var result: Mesh = null
	var nodes := [instance]
	while not nodes.is_empty():
		var node: Node = nodes.pop_back()
		if node is MeshInstance3D and (node as MeshInstance3D).mesh:
			result = (node as MeshInstance3D).mesh
			break
		for child in node.get_children():
			nodes.append(child)
	instance.queue_free()
	return result


func _load_vegetation_meshes() -> void:
	# Try loading FBX models as full scenes (preserves materials/textures).
	# Fall back to GLB mesh extraction for MultiMesh instancing.
	var tree_fbx_paths := [
		"res://assets/models/vegetation/trees/tree_01.fbx",
		"res://assets/models/vegetation/trees/tree_02.fbx",
	]
	for p in tree_fbx_paths:
		var scene := _load_packed_scene(p)
		if scene:
			_tree_scenes.append(scene)

	# If no FBX scenes loaded, try GLB mesh extraction for MultiMesh
	if _tree_scenes.is_empty():
		var tree_glb_paths := [
			"res://assets/models/vegetation/trees/tree_01.glb",
			"res://assets/models/vegetation/trees/tree_02.glb",
			"res://assets/models/vegetation/trees/tree_03.glb",
			"res://assets/models/vegetation/trees/tree_04.glb",
		]
		for p in tree_glb_paths:
			var m := _load_mesh_from_scene(p)
			if m and _tree_meshes.size() < 4:
				_tree_meshes.append(m)

	# Bushes — FBX scenes first, then GLB mesh extraction
	var bush_fbx_paths := [
		"res://assets/models/vegetation/bushes/bush_01.fbx",
		"res://assets/models/vegetation/bushes/bush_02.fbx",
		"res://assets/models/vegetation/bushes/bush_03.fbx",
		"res://assets/models/vegetation/bushes/bush_04.fbx",
	]
	for p in bush_fbx_paths:
		var scene := _load_packed_scene(p)
		if scene:
			_bush_scenes.append(scene)

	if _bush_scenes.is_empty():
		var bush_glb_paths := [
			"res://assets/models/vegetation/bushes/bush_01.glb",
			"res://assets/models/vegetation/bushes/bush_02.glb",
			"res://assets/models/vegetation/bushes/bush_03.glb",
		]
		for p in bush_glb_paths:
			var m := _load_mesh_from_scene(p)
			if m and _bush_meshes.size() < 4:
				_bush_meshes.append(m)

	# Rocks — always use mesh extraction for MultiMesh (many small instances)
	var rock_paths := [
		"res://assets/models/vegetation/rocks/rock_01.fbx",
		"res://assets/models/vegetation/rocks/rock_02.fbx",
		"res://assets/models/vegetation/rocks/rock_01.glb",
		"res://assets/models/vegetation/rocks/rock_02.glb",
		"res://assets/models/vegetation/rocks/rock_03.glb",
	]
	for p in rock_paths:
		var m := _load_mesh_from_scene(p)
		if m and _rock_meshes.size() < 3:
			_rock_meshes.append(m)

	var total := _tree_scenes.size() + _tree_meshes.size() + _bush_scenes.size() \
		+ _bush_meshes.size() + _rock_meshes.size()
	if total > 0:
		print("[Vegetation] Loaded %d tree scenes, %d tree meshes, %d bush scenes, %d bush meshes, %d rock meshes" % [
			_tree_scenes.size(), _tree_meshes.size(),
			_bush_scenes.size(), _bush_meshes.size(), _rock_meshes.size()])


func _color_from_hex(hex: String, fallback: Color) -> Color:
	if hex.is_empty():
		return fallback
	return Color.from_string(hex, fallback)
