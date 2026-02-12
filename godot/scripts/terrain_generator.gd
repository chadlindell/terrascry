## Generates a flat terrain mesh from scenario extents.
##
## Creates a PlaneMesh representing the survey area ground surface.
## In Phase 3, this will support height maps and textured soil layers.
extends MeshInstance3D

## Terrain dimensions from scenario
@export var x_extent := Vector2(0.0, 20.0)
@export var y_extent := Vector2(0.0, 20.0)
@export var surface_elevation := 0.0

## Material
var _material: StandardMaterial3D


func _ready() -> void:
	_create_terrain()


func _create_terrain() -> void:
	var width := x_extent.y - x_extent.x
	var depth := y_extent.y - y_extent.x

	var plane := PlaneMesh.new()
	plane.size = Vector2(width, depth)
	plane.subdivide_width = 10
	plane.subdivide_depth = 10
	mesh = plane

	# Position at center of extents
	position = Vector3(
		(x_extent.x + x_extent.y) / 2.0,
		surface_elevation,
		(y_extent.x + y_extent.y) / 2.0,
	)

	# Brown ground material
	_material = StandardMaterial3D.new()
	_material.albedo_color = Color(0.55, 0.45, 0.33)  # Brown
	_material.roughness = 0.9
	material_override = _material


## Update terrain from scenario info received from physics server.
func update_from_scenario(info: Dictionary) -> void:
	var terrain := info.get("terrain", {}) as Dictionary
	if terrain.has("x_extent"):
		x_extent = Vector2(terrain["x_extent"][0], terrain["x_extent"][1])
	if terrain.has("y_extent"):
		y_extent = Vector2(terrain["y_extent"][0], terrain["y_extent"][1])
	_create_terrain()
