## Generates terrain mesh with collision from scenario extents.
##
## Creates a subdivided PlaneMesh with a StaticBody3D for ground collision
## and optional grid lines for spatial reference during surveys.
extends MeshInstance3D

## Terrain dimensions from scenario
@export var x_extent := Vector2(0.0, 20.0)
@export var y_extent := Vector2(0.0, 20.0)
@export var surface_elevation := 0.0

## Grid line spacing in meters. Set to 0 to disable grid.
@export var grid_spacing := 5.0

## Internal references
var _material: StandardMaterial3D
var _collision_body: StaticBody3D
var _grid_mesh: MeshInstance3D


func _ready() -> void:
	_create_terrain()


func _create_terrain() -> void:
	var width := x_extent.y - x_extent.x
	var depth := y_extent.y - y_extent.x

	# Terrain surface mesh
	var plane := PlaneMesh.new()
	plane.size = Vector2(width, depth)
	plane.subdivide_width = 50
	plane.subdivide_depth = 50
	mesh = plane

	# Position at center of extents
	position = Vector3(
		(x_extent.x + x_extent.y) / 2.0,
		surface_elevation,
		(y_extent.x + y_extent.y) / 2.0,
	)

	# Brown ground material
	_material = StandardMaterial3D.new()
	_material.albedo_color = Color(0.55, 0.45, 0.33)
	_material.roughness = 0.9
	material_override = _material

	# Ground collision so CharacterBody3D can stand on terrain
	_setup_collision(width, depth)

	# Grid lines for spatial reference
	if grid_spacing > 0:
		_create_grid_lines(width, depth)


func _setup_collision(width: float, depth: float) -> void:
	# Remove old collision body if rebuilding
	if _collision_body:
		_collision_body.queue_free()

	_collision_body = StaticBody3D.new()
	_collision_body.name = "GroundBody"
	var collision_shape := CollisionShape3D.new()
	var box := BoxShape3D.new()
	box.size = Vector3(width, 0.1, depth)
	collision_shape.shape = box
	collision_shape.position.y = -0.05  # Top of box flush with plane surface
	_collision_body.add_child(collision_shape)
	add_child(_collision_body)


func _create_grid_lines(width: float, depth: float) -> void:
	if _grid_mesh:
		_grid_mesh.queue_free()

	var im := ImmediateMesh.new()
	_grid_mesh = MeshInstance3D.new()
	_grid_mesh.name = "GridLines"
	_grid_mesh.position.y = 0.005  # Slightly above terrain to avoid z-fighting

	var half_w := width / 2.0
	var half_d := depth / 2.0

	im.surface_begin(Mesh.PRIMITIVE_LINES)

	# Lines parallel to X axis (varying Z position)
	var z := -half_d
	while z <= half_d + 0.01:
		im.surface_add_vertex(Vector3(-half_w, 0, z))
		im.surface_add_vertex(Vector3(half_w, 0, z))
		z += grid_spacing

	# Lines parallel to Z axis (varying X position)
	var x := -half_w
	while x <= half_w + 0.01:
		im.surface_add_vertex(Vector3(x, 0, -half_d))
		im.surface_add_vertex(Vector3(x, 0, half_d))
		x += grid_spacing

	im.surface_end()
	_grid_mesh.mesh = im

	var line_mat := StandardMaterial3D.new()
	line_mat.albedo_color = Color(0.3, 0.3, 0.3, 0.5)
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
