## Generates terrain mesh with noise-based heightmap from scenario extents.
##
## Creates a subdivided mesh using SurfaceTool with FastNoiseLite displacement,
## trimesh collision for accurate ground contact, and a ground shader that blends
## grass/dirt/rock based on slope. Provides get_height_at() for other systems.
## Features: dual-frequency noise, erosion-like ridge channels, rock formations.
extends MeshInstance3D

## Terrain dimensions from scenario
@export var x_extent := Vector2(0.0, 20.0)
@export var y_extent := Vector2(0.0, 20.0)
@export var surface_elevation := 0.0

## Grid line spacing in meters. Set to 0 to disable grid.
@export var grid_spacing := 5.0

## Heightmap parameters
@export var height_amplitude := 0.4  # meters of vertical displacement (primary)
@export var noise_frequency := 0.15
@export var noise_octaves := 3
@export var terrain_margin := 5.0  # extra meters beyond scenario bounds

## Secondary noise layer (broad hills)
@export var hill_amplitude := 0.25  # meters for broad undulations
@export var hill_frequency := 0.04  # low frequency for large features

## Ridge noise for erosion-like channels
@export var ridge_amplitude := 0.12
@export var ridge_frequency := 0.08

## Rock formation parameters
@export var rock_density := 0.02  # probability of rock at any grid point
@export var rock_height_min := 0.08
@export var rock_height_max := 0.2

## Mesh subdivision density (vertices per meter)
@export var subdivisions_per_meter := 3.0

## Internal references
var _collision_body: StaticBody3D
var _grid_mesh: MeshInstance3D
var _noise: FastNoiseLite
var _hill_noise: FastNoiseLite
var _ridge_noise: FastNoiseLite
var _rock_noise: FastNoiseLite
var _center_x: float
var _center_z: float
var _crater_features: Array[Dictionary] = []
var _shader_overrides: Dictionary = {}


func _ready() -> void:
	_create_terrain()


## Get the terrain height at a world-space (x, z) position.
## Returns surface_elevation + combined noise displacement.
func get_height_at(world_x: float, world_z: float) -> float:
	if not _noise:
		return surface_elevation
	return surface_elevation + _compute_height(world_x, world_z)


func _compute_height(world_x: float, world_z: float) -> float:
	## Compute combined height from all noise layers at a world position.
	var h := 0.0

	# Primary noise (mid-frequency detail)
	h += _noise.get_noise_2d(world_x, world_z) * height_amplitude

	# Broad hills (low-frequency undulations)
	if _hill_noise:
		h += _hill_noise.get_noise_2d(world_x, world_z) * hill_amplitude

	# Ridge/erosion channels: abs(noise) creates valley patterns
	if _ridge_noise:
		var ridge_val := _ridge_noise.get_noise_2d(world_x, world_z)
		h -= absf(ridge_val) * ridge_amplitude  # Valleys where ridge noise is non-zero

	# Rock formations (sparse elevated bumps)
	if _rock_noise:
		var rock_val := _rock_noise.get_noise_2d(world_x, world_z)
		if rock_val > (1.0 - rock_density * 10.0):
			# Sharp bump for rock outcrop
			var rock_strength := (rock_val - (1.0 - rock_density * 10.0)) / (rock_density * 10.0)
			h += rock_strength * rock_strength * lerpf(rock_height_min, rock_height_max, rock_val)

	# Structured terrain features from scenario metadata (e.g., crater bowl + rim).
	h += _compute_structured_features(world_x, world_z)

	return h


func _compute_structured_features(world_x: float, world_z: float) -> float:
	if _crater_features.is_empty():
		return 0.0

	var feature_h := 0.0
	var p := Vector2(world_x, world_z)
	for crater in _crater_features:
		var center: Vector2 = crater.get("center", Vector2.ZERO)
		var radius: float = max(crater.get("radius", 1.0), 0.2)
		var depth: float = max(crater.get("depth", 0.1), 0.0)
		var rim_gain: float = crater.get("rim_gain", 0.18)

		var dist := p.distance_to(center)
		var n := dist / radius

		# Crater bowl: smooth parabolic depression.
		if n < 1.0:
			var bowl := pow(1.0 - n * n, 2.0)
			feature_h -= depth * bowl

		# Raised rim around crater edge.
		var rim_n := absf(n - 1.0)
		if rim_n < 0.35:
			var rim := exp(-pow(rim_n / 0.16, 2.0))
			feature_h += depth * rim_gain * rim

	return feature_h


func _create_terrain() -> void:
	var base_width := x_extent.y - x_extent.x
	var base_depth := y_extent.y - y_extent.x
	var width := base_width + terrain_margin * 2.0
	var depth := base_depth + terrain_margin * 2.0

	_center_x = (x_extent.x + x_extent.y) / 2.0
	_center_z = (y_extent.x + y_extent.y) / 2.0

	# Set up primary noise
	_noise = FastNoiseLite.new()
	_noise.noise_type = FastNoiseLite.TYPE_SIMPLEX
	_noise.frequency = noise_frequency
	_noise.fractal_octaves = noise_octaves
	_noise.seed = hash(Vector2(x_extent.x, y_extent.x))

	# Set up hill noise (broad, low-frequency)
	_hill_noise = FastNoiseLite.new()
	_hill_noise.noise_type = FastNoiseLite.TYPE_SIMPLEX
	_hill_noise.frequency = hill_frequency
	_hill_noise.fractal_octaves = 2
	_hill_noise.seed = hash(Vector2(x_extent.x, y_extent.x)) + 1000

	# Set up ridge noise (for erosion channels)
	_ridge_noise = FastNoiseLite.new()
	_ridge_noise.noise_type = FastNoiseLite.TYPE_SIMPLEX
	_ridge_noise.frequency = ridge_frequency
	_ridge_noise.fractal_octaves = 2
	_ridge_noise.seed = hash(Vector2(x_extent.x, y_extent.x)) + 2000

	# Set up rock formation noise (high-frequency, sparse)
	_rock_noise = FastNoiseLite.new()
	_rock_noise.noise_type = FastNoiseLite.TYPE_CELLULAR
	_rock_noise.frequency = 0.3
	_rock_noise.seed = hash(Vector2(x_extent.x, y_extent.x)) + 3000

	# Build mesh with SurfaceTool
	var subdiv_x := int(width * subdivisions_per_meter)
	var subdiv_z := int(depth * subdivisions_per_meter)
	subdiv_x = clampi(subdiv_x, 10, 300)
	subdiv_z = clampi(subdiv_z, 10, 300)

	var half_w := width / 2.0
	var half_d := depth / 2.0

	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)

	# Generate vertex grid
	var vertices: Array[Vector3] = []
	var uvs: Array[Vector2] = []
	for iz in range(subdiv_z + 1):
		for ix in range(subdiv_x + 1):
			var u := float(ix) / float(subdiv_x)
			var v := float(iz) / float(subdiv_z)
			var local_x := -half_w + u * width
			var local_z := -half_d + v * depth
			var world_x := _center_x + local_x
			var world_z := _center_z + local_z

			# Combined noise displacement
			var h := _compute_height(world_x, world_z)

			# Fade height at edges beyond scenario bounds for smooth blending
			var edge_fade := _compute_edge_fade(world_x, world_z)
			h *= edge_fade

			vertices.append(Vector3(local_x, h, local_z))
			uvs.append(Vector2(u, v))

	# Generate triangle indices
	for iz in range(subdiv_z):
		for ix in range(subdiv_x):
			var i00 := iz * (subdiv_x + 1) + ix
			var i10 := i00 + 1
			var i01 := i00 + (subdiv_x + 1)
			var i11 := i01 + 1

			# Triangle 1
			st.set_uv(uvs[i00])
			st.add_vertex(vertices[i00])
			st.set_uv(uvs[i10])
			st.add_vertex(vertices[i10])
			st.set_uv(uvs[i01])
			st.add_vertex(vertices[i01])

			# Triangle 2
			st.set_uv(uvs[i10])
			st.add_vertex(vertices[i10])
			st.set_uv(uvs[i11])
			st.add_vertex(vertices[i11])
			st.set_uv(uvs[i01])
			st.add_vertex(vertices[i01])

	st.generate_normals()
	mesh = st.commit()

	# Position at center of extents, at surface elevation
	position = Vector3(_center_x, surface_elevation, _center_z)

	# Ground shader material
	var shader := load("res://shaders/terrain_ground.gdshader") as Shader
	if shader:
		var shader_mat := ShaderMaterial.new()
		shader_mat.shader = shader
		_apply_shader_overrides(shader_mat)
		_apply_pbr_textures(shader_mat)
		material_override = shader_mat
	else:
		# Fallback: plain green-brown material
		var mat := StandardMaterial3D.new()
		mat.albedo_color = Color(0.32, 0.42, 0.22)
		mat.roughness = 0.85
		material_override = mat

	# Trimesh collision for accurate ground contact with heightmap
	_setup_collision()

	# Grid lines for spatial reference
	if grid_spacing > 0:
		_create_grid_lines(width, depth)


func _compute_edge_fade(world_x: float, world_z: float) -> float:
	## Fade displacement to zero at the outer edges of the margin.
	var dx_min := world_x - (x_extent.x - terrain_margin)
	var dx_max := (x_extent.y + terrain_margin) - world_x
	var dz_min := world_z - (y_extent.x - terrain_margin)
	var dz_max := (y_extent.y + terrain_margin) - world_z

	var fade_dist := terrain_margin * 0.6
	var fx := clampf(minf(dx_min, dx_max) / fade_dist, 0.0, 1.0)
	var fz := clampf(minf(dz_min, dz_max) / fade_dist, 0.0, 1.0)
	return fx * fz


func _setup_collision() -> void:
	if _collision_body:
		_collision_body.queue_free()
		_collision_body = null

	_collision_body = StaticBody3D.new()
	_collision_body.name = "GroundBody"

	# Create trimesh collision from the displaced mesh
	if mesh:
		var trimesh_shape := mesh.create_trimesh_shape()
		var collision_shape := CollisionShape3D.new()
		collision_shape.shape = trimesh_shape
		_collision_body.add_child(collision_shape)

	add_child(_collision_body)


func _create_grid_lines(width: float, depth: float) -> void:
	if _grid_mesh:
		_grid_mesh.queue_free()

	var im := ImmediateMesh.new()
	_grid_mesh = MeshInstance3D.new()
	_grid_mesh.name = "GridLines"

	# Only draw grid within the scenario bounds (not the margin)
	var grid_half_w := (x_extent.y - x_extent.x) / 2.0
	var grid_half_d := (y_extent.y - y_extent.x) / 2.0

	im.surface_begin(Mesh.PRIMITIVE_LINES)

	# Lines parallel to X axis (varying Z position)
	var z := -grid_half_d
	while z <= grid_half_d + 0.01:
		var world_z := _center_z + z
		# Sample multiple points along the line for terrain following
		var steps := int(grid_half_w * 2.0 * subdivisions_per_meter)
		steps = maxi(steps, 2)
		for s in range(steps):
			var t0 := float(s) / float(steps)
			var t1 := float(s + 1) / float(steps)
			var x0 := -grid_half_w + t0 * grid_half_w * 2.0
			var x1 := -grid_half_w + t1 * grid_half_w * 2.0
			var y0 := _compute_height(_center_x + x0, world_z) + 0.02
			var y1 := _compute_height(_center_x + x1, world_z) + 0.02
			im.surface_add_vertex(Vector3(x0, y0, z))
			im.surface_add_vertex(Vector3(x1, y1, z))
		z += grid_spacing

	# Lines parallel to Z axis (varying X position)
	var x := -grid_half_w
	while x <= grid_half_w + 0.01:
		var world_x := _center_x + x
		var steps := int(grid_half_d * 2.0 * subdivisions_per_meter)
		steps = maxi(steps, 2)
		for s in range(steps):
			var t0 := float(s) / float(steps)
			var t1 := float(s + 1) / float(steps)
			var z0 := -grid_half_d + t0 * grid_half_d * 2.0
			var z1 := -grid_half_d + t1 * grid_half_d * 2.0
			var y0 := _compute_height(world_x, _center_z + z0) + 0.02
			var y1 := _compute_height(world_x, _center_z + z1) + 0.02
			im.surface_add_vertex(Vector3(x, y0, z0))
			im.surface_add_vertex(Vector3(x, y1, z1))
		x += grid_spacing

	im.surface_end()
	_grid_mesh.mesh = im

	var line_mat := StandardMaterial3D.new()
	line_mat.albedo_color = Color(0.4, 0.4, 0.35, 0.4)
	line_mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	line_mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	_grid_mesh.material_override = line_mat

	add_child(_grid_mesh)


## Update terrain from scenario info received from physics server.
func update_from_scenario(info: Dictionary) -> void:
	var terrain := info.get("terrain", {}) as Dictionary
	if terrain.has("x_extent"):
		x_extent = Vector2(terrain["x_extent"][0], terrain["x_extent"][1])
	if terrain.has("y_extent"):
		y_extent = Vector2(terrain["y_extent"][0], terrain["y_extent"][1])
	if terrain.has("surface_elevation"):
		surface_elevation = terrain["surface_elevation"]

	_apply_scenario_visual_profile(info, terrain)
	_create_terrain()


func _apply_scenario_visual_profile(info: Dictionary, terrain: Dictionary) -> void:
	_crater_features.clear()
	_shader_overrides.clear()

	var name := str(info.get("name", "")).to_lower()
	var desc := str(info.get("description", "")).to_lower()
	var metadata: Dictionary = info.get("metadata", {})
	var layers: Array = terrain.get("layers", [])
	var env_profile: Dictionary = info.get("environment_profile", {})

	# Wetness estimate from layer conductivity + optional water table metadata.
	var mean_conductivity := 0.05
	if not layers.is_empty():
		var sum_cond := 0.0
		for layer in layers:
			sum_cond += float(layer.get("conductivity", 0.05))
		mean_conductivity = sum_cond / float(layers.size())

	var wetness := clampf(mean_conductivity / 0.25, 0.0, 1.0)
	if metadata.has("water_table_depth"):
		var wt_depth := float(metadata.get("water_table_depth", 2.0))
		wetness = clampf(wetness + clampf((1.0 - wt_depth) / 1.0, 0.0, 0.6), 0.0, 1.0)
	if name.contains("swamp") or name.contains("marsh") or desc.contains("waterlogged"):
		wetness = clampf(wetness + 0.35, 0.0, 1.0)
	if env_profile.has("wetness"):
		wetness = clampf(float(env_profile.get("wetness", wetness)), 0.0, 1.0)

	var is_crater := name.contains("crater") or desc.contains("crater")
	var is_forensic := name.contains("burial") or desc.contains("grave") or str(metadata.get("category", "")).to_lower().contains("forensic")

	# Terrain roughness and rockiness by environment type.
	height_amplitude = lerpf(0.45, 0.18, wetness)
	hill_amplitude = lerpf(0.30, 0.10, wetness)
	ridge_amplitude = lerpf(0.16, 0.05, wetness)
	rock_density = lerpf(0.028, 0.008, wetness)
	noise_frequency = lerpf(0.17, 0.10, wetness)
	hill_frequency = lerpf(0.05, 0.025, wetness)

	if is_crater:
		height_amplitude += 0.08
		ridge_amplitude += 0.05
		rock_density += 0.01
	if is_forensic:
		height_amplitude *= 0.75
		ridge_amplitude *= 0.7
		rock_density *= 0.7

	# Add explicit crater from metadata when present.
	if metadata.has("crater"):
		var crater: Dictionary = metadata.get("crater", {})
		var c_arr: Array = crater.get("center", [])
		var cx := (x_extent.x + x_extent.y) * 0.5
		var cz := (y_extent.x + y_extent.y) * 0.5
		if c_arr.size() >= 2:
			cx = float(c_arr[0])
			cz = float(c_arr[1])
		var diameter := float(crater.get("diameter", 6.0))
		var depth := float(crater.get("depth", 1.2))
		_crater_features.append({
			"center": Vector2(cx, cz),
			"radius": max(diameter * 0.5, 0.5),
			"depth": clampf(depth * 0.35, 0.15, 2.0),
			"rim_gain": 0.20,
		})

	# Secondary crater-like anomalies inferred from anomaly zones.
	var anomalies: Array = info.get("anomaly_zones", [])
	for az in anomalies:
		var az_name := str(az.get("name", "")).to_lower()
		if not az_name.contains("crater"):
			continue
		var center_arr: Array = az.get("center", [])
		if center_arr.size() < 2:
			continue
		var dims: Dictionary = az.get("dimensions", {})
		var length := float(dims.get("length", 4.0))
		var width := float(dims.get("width", 4.0))
		var depth := float(dims.get("depth", 1.0))
		_crater_features.append({
			"center": Vector2(float(center_arr[0]), float(center_arr[1])),
			"radius": max(max(length, width) * 0.5, 0.4),
			"depth": clampf(depth * 0.22, 0.08, 1.2),
			"rim_gain": 0.16,
		})

	# Layer-driven color palette for terrain shader.
	var top_color := Color(0.32, 0.44, 0.24)
	var sub_color := Color(0.45, 0.36, 0.28)
	if not layers.is_empty():
		top_color = _color_from_hex(str(layers[0].get("color", "")), top_color)
	if layers.size() > 1:
		sub_color = _color_from_hex(str(layers[1].get("color", "")), sub_color)

	var dirt_color := top_color.darkened(0.28).lerp(sub_color, 0.55)
	var grass_a := top_color.lightened(0.07)
	var grass_b := top_color.lightened(0.16)
	var grass_c := top_color.darkened(0.1)
	var grass_d := top_color.darkened(0.2)

	# Compute extra material weights from scenario profile
	var mud_w := 0.0
	var gravel_w := 0.0
	var sand_w := 0.0

	# Swamp/marsh scenarios: heavy mud
	if name.contains("swamp") or name.contains("marsh") or desc.contains("waterlogged"):
		mud_w = clampf(0.5 + wetness * 0.3, 0.0, 0.85)
	elif wetness > 0.5:
		mud_w = clampf((wetness - 0.5) * 1.2, 0.0, 0.6)

	# Crater/UXO scenarios: gravel and rock debris
	if is_crater or name.contains("debris") or desc.contains("scattered"):
		gravel_w = clampf(0.35 + 0.2 * (1.0 - wetness), 0.0, 0.6)
	elif metadata.has("crater"):
		gravel_w = clampf(0.25, 0.0, 0.5)

	# Sandy/coastal scenarios
	if name.contains("sand") or desc.contains("sandy") or desc.contains("beach"):
		sand_w = clampf(0.5 + 0.2 * (1.0 - wetness), 0.0, 0.8)

	# Explicit overrides from environment_profile
	if env_profile.has("mud_weight"):
		mud_w = clampf(float(env_profile.get("mud_weight", mud_w)), 0.0, 1.0)
	if env_profile.has("gravel_weight"):
		gravel_w = clampf(float(env_profile.get("gravel_weight", gravel_w)), 0.0, 1.0)
	if env_profile.has("sand_weight"):
		sand_w = clampf(float(env_profile.get("sand_weight", sand_w)), 0.0, 1.0)

	_shader_overrides = {
		"grass_color_1": grass_a,
		"grass_color_2": grass_b,
		"grass_color_3": grass_c,
		"grass_color_4": grass_d,
		"dirt_color": dirt_color,
		"dirt_color_alt": dirt_color.darkened(0.12),
		"dirt_color_tan": dirt_color.lightened(0.14),
		"moisture_strength": clampf(0.25 + wetness * 0.6, 0.2, 0.9),
		"moisture_height_bias": lerpf(0.7, -0.6, wetness),
		"grass_coverage": lerpf(0.58, 0.42, wetness),
		"noise_scale": lerpf(3.4, 2.1, wetness),
		"mud_weight": mud_w,
		"gravel_weight": gravel_w,
		"sand_weight": sand_w,
	}


func _color_from_hex(hex: String, fallback: Color) -> Color:
	if hex.is_empty():
		return fallback
	return Color.from_string(hex, fallback)


func _apply_shader_overrides(shader_mat: ShaderMaterial) -> void:
	for key in _shader_overrides.keys():
		shader_mat.set_shader_parameter(str(key), _shader_overrides[key])


func _create_white_texture() -> ImageTexture:
	var img := Image.create(4, 4, false, Image.FORMAT_RGBA8)
	img.fill(Color.WHITE)
	return ImageTexture.create_from_image(img)


func _create_flat_normal_texture() -> ImageTexture:
	var img := Image.create(4, 4, false, Image.FORMAT_RGBA8)
	img.fill(Color(0.5, 0.5, 1.0, 1.0))  # Flat normal pointing up
	return ImageTexture.create_from_image(img)


## Generate procedural grass albedo (256x256 green noise with color variation).
func _create_procedural_grass_albedo() -> ImageTexture:
	var size := 256
	var img := Image.create(size, size, false, Image.FORMAT_RGBA8)
	var rng := RandomNumberGenerator.new()
	rng.seed = 42
	var n := FastNoiseLite.new()
	n.noise_type = FastNoiseLite.TYPE_SIMPLEX
	n.frequency = 0.02
	n.seed = 100
	var detail := FastNoiseLite.new()
	detail.noise_type = FastNoiseLite.TYPE_SIMPLEX
	detail.frequency = 0.08
	detail.seed = 200
	for y in range(size):
		for x in range(size):
			var nv := n.get_noise_2d(x, y) * 0.5 + 0.5
			var dv := detail.get_noise_2d(x, y) * 0.15
			var g := 0.35 + nv * 0.25 + dv
			var r := 0.18 + nv * 0.12 + dv * 0.5
			var b := 0.08 + nv * 0.06
			img.set_pixel(x, y, Color(clampf(r, 0, 1), clampf(g, 0, 1), clampf(b, 0, 1)))
	return ImageTexture.create_from_image(img)


## Generate procedural grass normal map.
func _create_procedural_grass_normal() -> ImageTexture:
	var size := 256
	var img := Image.create(size, size, false, Image.FORMAT_RGBA8)
	var n := FastNoiseLite.new()
	n.noise_type = FastNoiseLite.TYPE_SIMPLEX
	n.frequency = 0.04
	n.seed = 300
	for y in range(size):
		for x in range(size):
			var dx := (n.get_noise_2d(x + 1, y) - n.get_noise_2d(x - 1, y)) * 2.0
			var dy := (n.get_noise_2d(x, y + 1) - n.get_noise_2d(x, y - 1)) * 2.0
			img.set_pixel(x, y, Color(0.5 + dx * 0.4, 0.5 + dy * 0.4, 1.0))
	return ImageTexture.create_from_image(img)


## Generate procedural dirt albedo (brown noise with gravel detail).
func _create_procedural_dirt_albedo() -> ImageTexture:
	var size := 256
	var img := Image.create(size, size, false, Image.FORMAT_RGBA8)
	var n := FastNoiseLite.new()
	n.noise_type = FastNoiseLite.TYPE_SIMPLEX
	n.frequency = 0.03
	n.seed = 400
	var gravel := FastNoiseLite.new()
	gravel.noise_type = FastNoiseLite.TYPE_CELLULAR
	gravel.frequency = 0.1
	gravel.seed = 500
	for y in range(size):
		for x in range(size):
			var nv := n.get_noise_2d(x, y) * 0.5 + 0.5
			var gv := gravel.get_noise_2d(x, y) * 0.1
			var r := 0.38 + nv * 0.18 + gv
			var g := 0.30 + nv * 0.14 + gv
			var b := 0.20 + nv * 0.08 + gv
			img.set_pixel(x, y, Color(clampf(r, 0, 1), clampf(g, 0, 1), clampf(b, 0, 1)))
	return ImageTexture.create_from_image(img)


## Generate procedural rock albedo (gray cellular noise with cracks).
func _create_procedural_rock_albedo() -> ImageTexture:
	var size := 256
	var img := Image.create(size, size, false, Image.FORMAT_RGBA8)
	var n := FastNoiseLite.new()
	n.noise_type = FastNoiseLite.TYPE_CELLULAR
	n.frequency = 0.03
	n.seed = 600
	var crack := FastNoiseLite.new()
	crack.noise_type = FastNoiseLite.TYPE_SIMPLEX
	crack.frequency = 0.06
	crack.seed = 700
	for y in range(size):
		for x in range(size):
			var nv := n.get_noise_2d(x, y) * 0.5 + 0.5
			var cv := absf(crack.get_noise_2d(x, y))
			var base := 0.42 + nv * 0.18
			# Darken cracks
			if cv < 0.1:
				base *= 0.7
			img.set_pixel(x, y, Color(base, base * 0.97, base * 0.92))
	return ImageTexture.create_from_image(img)


## Generate procedural rock normal map with strong detail.
func _create_procedural_rock_normal() -> ImageTexture:
	var size := 256
	var img := Image.create(size, size, false, Image.FORMAT_RGBA8)
	var n := FastNoiseLite.new()
	n.noise_type = FastNoiseLite.TYPE_CELLULAR
	n.frequency = 0.04
	n.seed = 800
	for y in range(size):
		for x in range(size):
			var dx := (n.get_noise_2d(x + 1, y) - n.get_noise_2d(x - 1, y)) * 3.0
			var dy := (n.get_noise_2d(x, y + 1) - n.get_noise_2d(x, y - 1)) * 3.0
			img.set_pixel(x, y, Color(
				clampf(0.5 + dx * 0.4, 0, 1),
				clampf(0.5 + dy * 0.4, 0, 1), 1.0))
	return ImageTexture.create_from_image(img)


## Generate procedural AO texture (grayscale crevice darkening).
func _create_procedural_ao(seed_val: int) -> ImageTexture:
	var size := 256
	var img := Image.create(size, size, false, Image.FORMAT_RGBA8)
	var n := FastNoiseLite.new()
	n.noise_type = FastNoiseLite.TYPE_SIMPLEX
	n.frequency = 0.05
	n.seed = seed_val
	var detail := FastNoiseLite.new()
	detail.noise_type = FastNoiseLite.TYPE_CELLULAR
	detail.frequency = 0.08
	detail.seed = seed_val + 50
	for y in range(size):
		for x in range(size):
			# Base AO mostly white (lit) with darkened crevices
			var nv := n.get_noise_2d(x, y) * 0.15
			var dv := detail.get_noise_2d(x, y) * 0.1
			var ao := clampf(0.85 + nv + dv, 0.4, 1.0)
			img.set_pixel(x, y, Color(ao, ao, ao))
	return ImageTexture.create_from_image(img)


## Generate procedural height map (grayscale surface displacement).
func _create_procedural_height(seed_val: int, variation: float = 0.5) -> ImageTexture:
	var size := 256
	var img := Image.create(size, size, false, Image.FORMAT_RGBA8)
	var n := FastNoiseLite.new()
	n.noise_type = FastNoiseLite.TYPE_SIMPLEX
	n.frequency = 0.04
	n.fractal_octaves = 3
	n.seed = seed_val
	for y in range(size):
		for x in range(size):
			var nv := n.get_noise_2d(x, y) * 0.5 + 0.5
			var h := clampf(0.5 + (nv - 0.5) * variation * 2.0, 0.0, 1.0)
			img.set_pixel(x, y, Color(h, h, h))
	return ImageTexture.create_from_image(img)


## Generate procedural mud albedo (dark wet brown with organic debris).
func _create_procedural_mud_albedo() -> ImageTexture:
	var size := 256
	var img := Image.create(size, size, false, Image.FORMAT_RGBA8)
	var n := FastNoiseLite.new()
	n.noise_type = FastNoiseLite.TYPE_SIMPLEX
	n.frequency = 0.025
	n.seed = 1400
	var detail := FastNoiseLite.new()
	detail.noise_type = FastNoiseLite.TYPE_CELLULAR
	detail.frequency = 0.06
	detail.seed = 1500
	for y in range(size):
		for x in range(size):
			var nv := n.get_noise_2d(x, y) * 0.5 + 0.5
			var dv := detail.get_noise_2d(x, y) * 0.08
			var r := 0.26 + nv * 0.12 + dv
			var g := 0.19 + nv * 0.08 + dv
			var b := 0.12 + nv * 0.05 + dv
			img.set_pixel(x, y, Color(clampf(r, 0, 1), clampf(g, 0, 1), clampf(b, 0, 1)))
	return ImageTexture.create_from_image(img)


## Generate procedural mud normal map.
func _create_procedural_mud_normal() -> ImageTexture:
	var size := 256
	var img := Image.create(size, size, false, Image.FORMAT_RGBA8)
	var n := FastNoiseLite.new()
	n.noise_type = FastNoiseLite.TYPE_SIMPLEX
	n.frequency = 0.03
	n.seed = 1600
	for y in range(size):
		for x in range(size):
			var dx := (n.get_noise_2d(x + 1, y) - n.get_noise_2d(x - 1, y)) * 1.5
			var dy := (n.get_noise_2d(x, y + 1) - n.get_noise_2d(x, y - 1)) * 1.5
			img.set_pixel(x, y, Color(
				clampf(0.5 + dx * 0.35, 0, 1),
				clampf(0.5 + dy * 0.35, 0, 1), 1.0))
	return ImageTexture.create_from_image(img)


## Generate procedural gravel albedo (mixed pebble colors with cellular detail).
func _create_procedural_gravel_albedo() -> ImageTexture:
	var size := 256
	var img := Image.create(size, size, false, Image.FORMAT_RGBA8)
	var n := FastNoiseLite.new()
	n.noise_type = FastNoiseLite.TYPE_CELLULAR
	n.frequency = 0.06
	n.seed = 1700
	var color_noise := FastNoiseLite.new()
	color_noise.noise_type = FastNoiseLite.TYPE_SIMPLEX
	color_noise.frequency = 0.04
	color_noise.seed = 1800
	for y in range(size):
		for x in range(size):
			var nv := n.get_noise_2d(x, y) * 0.5 + 0.5
			var cv := color_noise.get_noise_2d(x, y) * 0.12
			var base := 0.40 + nv * 0.16 + cv
			# Slight warm tint
			var r := base * 1.03
			var g := base * 1.0
			var b := base * 0.94
			img.set_pixel(x, y, Color(clampf(r, 0, 1), clampf(g, 0, 1), clampf(b, 0, 1)))
	return ImageTexture.create_from_image(img)


## Generate procedural gravel normal map (high-frequency pebble bumps).
func _create_procedural_gravel_normal() -> ImageTexture:
	var size := 256
	var img := Image.create(size, size, false, Image.FORMAT_RGBA8)
	var n := FastNoiseLite.new()
	n.noise_type = FastNoiseLite.TYPE_CELLULAR
	n.frequency = 0.07
	n.seed = 1900
	for y in range(size):
		for x in range(size):
			var dx := (n.get_noise_2d(x + 1, y) - n.get_noise_2d(x - 1, y)) * 3.5
			var dy := (n.get_noise_2d(x, y + 1) - n.get_noise_2d(x, y - 1)) * 3.5
			img.set_pixel(x, y, Color(
				clampf(0.5 + dx * 0.4, 0, 1),
				clampf(0.5 + dy * 0.4, 0, 1), 1.0))
	return ImageTexture.create_from_image(img)


## Generate procedural sand albedo (warm beige with fine grain).
func _create_procedural_sand_albedo() -> ImageTexture:
	var size := 256
	var img := Image.create(size, size, false, Image.FORMAT_RGBA8)
	var n := FastNoiseLite.new()
	n.noise_type = FastNoiseLite.TYPE_SIMPLEX
	n.frequency = 0.04
	n.seed = 2000
	var grain := FastNoiseLite.new()
	grain.noise_type = FastNoiseLite.TYPE_SIMPLEX
	grain.frequency = 0.15
	grain.seed = 2100
	for y in range(size):
		for x in range(size):
			var nv := n.get_noise_2d(x, y) * 0.5 + 0.5
			var gv := grain.get_noise_2d(x, y) * 0.06
			var r := 0.68 + nv * 0.12 + gv
			var g := 0.60 + nv * 0.10 + gv
			var b := 0.44 + nv * 0.08 + gv
			img.set_pixel(x, y, Color(clampf(r, 0, 1), clampf(g, 0, 1), clampf(b, 0, 1)))
	return ImageTexture.create_from_image(img)


## Generate procedural sand normal map (fine ripple pattern).
func _create_procedural_sand_normal() -> ImageTexture:
	var size := 256
	var img := Image.create(size, size, false, Image.FORMAT_RGBA8)
	var n := FastNoiseLite.new()
	n.noise_type = FastNoiseLite.TYPE_SIMPLEX
	n.frequency = 0.05
	n.seed = 2200
	for y in range(size):
		for x in range(size):
			var dx := (n.get_noise_2d(x + 1, y) - n.get_noise_2d(x - 1, y)) * 1.2
			var dy := (n.get_noise_2d(x, y + 1) - n.get_noise_2d(x, y - 1)) * 1.2
			img.set_pixel(x, y, Color(
				clampf(0.5 + dx * 0.3, 0, 1),
				clampf(0.5 + dy * 0.3, 0, 1), 1.0))
	return ImageTexture.create_from_image(img)


## Generate procedural detail normal map (high-frequency micro-surface).
func _create_procedural_detail_normal() -> ImageTexture:
	var size := 256
	var img := Image.create(size, size, false, Image.FORMAT_RGBA8)
	var n := FastNoiseLite.new()
	n.noise_type = FastNoiseLite.TYPE_SIMPLEX
	n.frequency = 0.08
	n.fractal_octaves = 3
	n.seed = 5000
	for y in range(size):
		for x in range(size):
			var dx := (n.get_noise_2d(x + 1, y) - n.get_noise_2d(x - 1, y)) * 1.5
			var dy := (n.get_noise_2d(x, y + 1) - n.get_noise_2d(x, y - 1)) * 1.5
			img.set_pixel(x, y, Color(
				clampf(0.5 + dx * 0.4, 0, 1),
				clampf(0.5 + dy * 0.4, 0, 1), 1.0))
	return ImageTexture.create_from_image(img)


## Generate procedural roughness texture.
func _create_procedural_roughness(base: float, variation: float, seed_val: int) -> ImageTexture:
	var size := 256
	var img := Image.create(size, size, false, Image.FORMAT_RGBA8)
	var n := FastNoiseLite.new()
	n.noise_type = FastNoiseLite.TYPE_SIMPLEX
	n.frequency = 0.04
	n.seed = seed_val
	for y in range(size):
		for x in range(size):
			var v := clampf(base + n.get_noise_2d(x, y) * variation, 0.0, 1.0)
			img.set_pixel(x, y, Color(v, v, v))
	return ImageTexture.create_from_image(img)


func _apply_pbr_textures(shader_mat: ShaderMaterial) -> void:
	# Set procedural fallback textures (noise-based, much better than plain white)
	# --- Grass ---
	shader_mat.set_shader_parameter("grass_albedo", _create_procedural_grass_albedo())
	shader_mat.set_shader_parameter("grass_normal", _create_procedural_grass_normal())
	shader_mat.set_shader_parameter("grass_roughness_tex", _create_procedural_roughness(0.8, 0.15, 901))
	shader_mat.set_shader_parameter("grass_ao", _create_procedural_ao(1001))
	shader_mat.set_shader_parameter("grass_height", _create_procedural_height(1101, 0.6))
	shader_mat.set_shader_parameter("grass_displacement", _create_procedural_height(1151, 0.4))
	# --- Dirt ---
	shader_mat.set_shader_parameter("dirt_albedo", _create_procedural_dirt_albedo())
	shader_mat.set_shader_parameter("dirt_normal", _create_procedural_grass_normal())
	shader_mat.set_shader_parameter("dirt_roughness_tex", _create_procedural_roughness(0.9, 0.1, 902))
	shader_mat.set_shader_parameter("dirt_ao", _create_procedural_ao(1002))
	shader_mat.set_shader_parameter("dirt_height", _create_procedural_height(1102, 0.4))
	shader_mat.set_shader_parameter("dirt_displacement", _create_procedural_height(1152, 0.3))
	# --- Rock ---
	shader_mat.set_shader_parameter("rock_albedo", _create_procedural_rock_albedo())
	shader_mat.set_shader_parameter("rock_normal", _create_procedural_rock_normal())
	shader_mat.set_shader_parameter("rock_roughness_tex", _create_procedural_roughness(0.7, 0.2, 903))
	shader_mat.set_shader_parameter("rock_ao", _create_procedural_ao(1003))
	shader_mat.set_shader_parameter("rock_height", _create_procedural_height(1103, 0.5))
	# --- Mud ---
	shader_mat.set_shader_parameter("mud_albedo", _create_procedural_mud_albedo())
	shader_mat.set_shader_parameter("mud_normal", _create_procedural_mud_normal())
	shader_mat.set_shader_parameter("mud_roughness_tex", _create_procedural_roughness(0.4, 0.2, 904))
	shader_mat.set_shader_parameter("mud_ao", _create_procedural_ao(1004))
	shader_mat.set_shader_parameter("mud_height", _create_procedural_height(1104, 0.3))
	# --- Gravel ---
	shader_mat.set_shader_parameter("gravel_albedo", _create_procedural_gravel_albedo())
	shader_mat.set_shader_parameter("gravel_normal", _create_procedural_gravel_normal())
	shader_mat.set_shader_parameter("gravel_roughness_tex", _create_procedural_roughness(0.85, 0.15, 905))
	shader_mat.set_shader_parameter("gravel_ao", _create_procedural_ao(1005))
	shader_mat.set_shader_parameter("gravel_height", _create_procedural_height(1105, 0.7))
	# --- Sand ---
	shader_mat.set_shader_parameter("sand_albedo", _create_procedural_sand_albedo())
	shader_mat.set_shader_parameter("sand_normal", _create_procedural_sand_normal())
	shader_mat.set_shader_parameter("sand_roughness_tex", _create_procedural_roughness(0.7, 0.1, 906))
	shader_mat.set_shader_parameter("sand_ao", _create_procedural_ao(1006))
	shader_mat.set_shader_parameter("sand_height", _create_procedural_height(1106, 0.3))
	# --- Detail ---
	shader_mat.set_shader_parameter("detail_normal", _create_procedural_detail_normal())

	# Try loading real PBR textures (null-safe — fallbacks remain if missing)
	var texture_paths := {
		"grass_albedo": "res://assets/textures/terrain/grass/Color.jpg",
		"grass_normal": "res://assets/textures/terrain/grass/NormalGL.jpg",
		"grass_roughness_tex": "res://assets/textures/terrain/grass/Roughness.jpg",
		"grass_ao": "res://assets/textures/terrain/grass/AmbientOcclusion.jpg",
		"grass_height": "res://assets/textures/terrain/grass/Displacement.jpg",
		"grass_displacement": "res://assets/textures/terrain/grass/Displacement.jpg",
		"dirt_albedo": "res://assets/textures/terrain/dirt/Color.jpg",
		"dirt_normal": "res://assets/textures/terrain/dirt/NormalGL.jpg",
		"dirt_roughness_tex": "res://assets/textures/terrain/dirt/Roughness.jpg",
		"dirt_ao": "res://assets/textures/terrain/dirt/AmbientOcclusion.jpg",
		"dirt_height": "res://assets/textures/terrain/dirt/Displacement.jpg",
		"dirt_displacement": "res://assets/textures/terrain/dirt/Displacement.jpg",
		"rock_albedo": "res://assets/textures/terrain/rock/Color.jpg",
		"rock_normal": "res://assets/textures/terrain/rock/NormalGL.jpg",
		"rock_roughness_tex": "res://assets/textures/terrain/rock/Roughness.jpg",
		"rock_ao": "res://assets/textures/terrain/rock/AmbientOcclusion.jpg",
		"rock_height": "res://assets/textures/terrain/rock/Displacement.jpg",
		"mud_albedo": "res://assets/textures/terrain/mud/Color.jpg",
		"mud_normal": "res://assets/textures/terrain/mud/NormalGL.jpg",
		"mud_roughness_tex": "res://assets/textures/terrain/mud/Roughness.jpg",
		"mud_ao": "res://assets/textures/terrain/mud/AmbientOcclusion.jpg",
		"mud_height": "res://assets/textures/terrain/mud/Displacement.jpg",
		"gravel_albedo": "res://assets/textures/terrain/gravel/Color.jpg",
		"gravel_normal": "res://assets/textures/terrain/gravel/NormalGL.jpg",
		"gravel_roughness_tex": "res://assets/textures/terrain/gravel/Roughness.jpg",
		"gravel_ao": "res://assets/textures/terrain/gravel/AmbientOcclusion.jpg",
		"gravel_height": "res://assets/textures/terrain/gravel/Displacement.jpg",
		"sand_albedo": "res://assets/textures/terrain/sand/Color.jpg",
		"sand_normal": "res://assets/textures/terrain/sand/NormalGL.jpg",
		"sand_roughness_tex": "res://assets/textures/terrain/sand/Roughness.jpg",
		"sand_ao": "res://assets/textures/terrain/sand/AmbientOcclusion.jpg",
		"sand_height": "res://assets/textures/terrain/sand/Displacement.jpg",
		"detail_normal": "res://assets/textures/terrain/detail_normal.jpg",
	}

	# Also try .png variants for each texture
	for param_name in texture_paths:
		var path: String = texture_paths[param_name]
		if ResourceLoader.exists(path):
			var tex = load(path)
			if tex:
				shader_mat.set_shader_parameter(param_name, tex)
		else:
			# Try .png extension
			var png_path := path.get_basename() + ".png"
			if ResourceLoader.exists(png_path):
				var tex = load(png_path)
				if tex:
					shader_mat.set_shader_parameter(param_name, tex)
