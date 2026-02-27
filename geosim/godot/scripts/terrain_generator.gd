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

	return h


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
	_create_terrain()
