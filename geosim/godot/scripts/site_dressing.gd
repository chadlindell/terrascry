## Archaeological site dressing — visual props for survey realism.
##
## Generates grid stakes, boundary tape, datum marker, equipment staging area,
## and pin flags along survey lines using MultiMeshInstance3D for performance.
## Attach as a child of Main; call setup() after scenario load.
extends Node3D


## Terrain reference for height queries
var _terrain: Node = null

## Survey area bounds (GeoSim coords, used as Godot X/Z)
var _x_extent := Vector2(0.0, 20.0)
var _y_extent := Vector2(0.0, 20.0)
var _surface_elevation := 0.0

## Containers
var _stakes_node: Node3D
var _tape_node: Node3D
var _equipment_node: Node3D
var _pin_flags_node: Node3D


func _ready() -> void:
	name = "SiteDressing"

	_stakes_node = Node3D.new()
	_stakes_node.name = "GridStakes"
	add_child(_stakes_node)

	_tape_node = Node3D.new()
	_tape_node.name = "BoundaryTape"
	add_child(_tape_node)

	_equipment_node = Node3D.new()
	_equipment_node.name = "Equipment"
	add_child(_equipment_node)

	_pin_flags_node = Node3D.new()
	_pin_flags_node.name = "PinFlags"
	add_child(_pin_flags_node)


## Set up all site dressing from scenario info.
func setup(terrain: Node, x_ext: Vector2, y_ext: Vector2, surface_elev: float) -> void:
	_terrain = terrain
	_x_extent = x_ext
	_y_extent = y_ext
	_surface_elevation = surface_elev

	_create_grid_stakes()
	_create_boundary_tape()
	_create_datum_marker()
	_create_equipment_area()


## Add pin flags along planned survey lines.
func add_survey_line_flags(lines: Array) -> void:
	_clear_children(_pin_flags_node)
	if lines.is_empty():
		return
	_create_pin_flags(lines)


## Get terrain height at world position, with fallback.
func _get_height(world_x: float, world_z: float) -> float:
	if _terrain and _terrain.has_method("get_height_at"):
		return _terrain.get_height_at(world_x, world_z)
	return _surface_elevation


func _clear_children(node: Node3D) -> void:
	for child in node.get_children():
		child.queue_free()


# ---------- Z1: Grid Intersection Stakes ----------

func _create_grid_stakes() -> void:
	_clear_children(_stakes_node)

	var spacing := 5.0
	var positions: Array[Vector3] = []
	var colors: Array[Color] = []

	# Datum / origin point
	var origin_x := _x_extent.x
	var origin_z := _y_extent.x

	var x := _x_extent.x
	while x <= _x_extent.y + 0.01:
		var z := _y_extent.x
		while z <= _y_extent.y + 0.01:
			var h := _get_height(x, z)
			positions.append(Vector3(x, h, z))

			# Color: pink at origin, white at corners, orange elsewhere
			var is_origin := absf(x - origin_x) < 0.1 and absf(z - _y_extent.x) < 0.1
			var is_corner := (absf(x - _x_extent.x) < 0.1 or absf(x - _x_extent.y) < 0.1) and \
				(absf(z - _y_extent.x) < 0.1 or absf(z - _y_extent.y) < 0.1)

			if is_origin:
				colors.append(Color(1.0, 0.4, 0.6))  # Pink
			elif is_corner:
				colors.append(Color(0.95, 0.95, 0.95))  # White
			else:
				colors.append(Color(1.0, 0.6, 0.1))  # Orange
			z += spacing
		x += spacing

	if positions.is_empty():
		return

	# Stake posts via MultiMeshInstance3D
	var stake_multi := MultiMeshInstance3D.new()
	stake_multi.name = "StakePosts"
	var stake_mm := MultiMesh.new()
	stake_mm.transform_format = MultiMesh.TRANSFORM_3D
	stake_mm.use_colors = true
	stake_mm.instance_count = positions.size()

	var stake_mesh := CylinderMesh.new()
	stake_mesh.top_radius = 0.010
	stake_mesh.bottom_radius = 0.012
	stake_mesh.height = 0.4
	stake_mm.mesh = stake_mesh

	var stake_mat := StandardMaterial3D.new()
	stake_mat.albedo_color = Color(0.45, 0.3, 0.15)
	stake_mat.roughness = 0.8
	stake_mat.vertex_color_use_as_albedo = false
	stake_multi.material_override = stake_mat

	for i in range(positions.size()):
		var t := Transform3D.IDENTITY
		t.origin = positions[i] + Vector3(0, 0.2, 0)  # Half height above ground
		stake_mm.set_instance_transform(i, t)

	stake_multi.multimesh = stake_mm
	_stakes_node.add_child(stake_multi)

	# Flag tabs via MultiMeshInstance3D
	var flag_multi := MultiMeshInstance3D.new()
	flag_multi.name = "StakeFlags"
	var flag_mm := MultiMesh.new()
	flag_mm.transform_format = MultiMesh.TRANSFORM_3D
	flag_mm.use_colors = true
	flag_mm.instance_count = positions.size()

	var flag_mesh := BoxMesh.new()
	flag_mesh.size = Vector3(0.08, 0.05, 0.002)
	flag_mm.mesh = flag_mesh

	# Unshaded so flags are always visible
	var flag_mat := StandardMaterial3D.new()
	flag_mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	flag_mat.vertex_color_use_as_albedo = true
	flag_multi.material_override = flag_mat

	for i in range(positions.size()):
		var t := Transform3D.IDENTITY
		t.origin = positions[i] + Vector3(0.04, 0.4, 0)  # At top of stake, offset right
		flag_mm.set_instance_transform(i, t)
		flag_mm.set_instance_color(i, colors[i])

	flag_multi.multimesh = flag_mm
	_stakes_node.add_child(flag_multi)


# ---------- Z2: Survey Boundary Tape ----------

func _create_boundary_tape() -> void:
	_clear_children(_tape_node)

	var tape_height := 0.3
	var corners: Array[Vector2] = [
		Vector2(_x_extent.x, _y_extent.x),
		Vector2(_x_extent.y, _y_extent.x),
		Vector2(_x_extent.y, _y_extent.y),
		Vector2(_x_extent.x, _y_extent.y),
	]

	var im := ImmediateMesh.new()
	var mesh_inst := MeshInstance3D.new()
	mesh_inst.name = "BoundaryRibbon"

	im.surface_begin(Mesh.PRIMITIVE_TRIANGLE_STRIP)

	var tape_width := 0.03
	var subdivs_per_meter := 2.0

	for edge in range(4):
		var c0 := corners[edge]
		var c1 := corners[(edge + 1) % 4]
		var length := c0.distance_to(c1)
		var steps := int(ceil(length * subdivs_per_meter))
		steps = maxi(steps, 1)

		for s in range(steps + 1):
			var t := float(s) / float(steps)
			var pt := c0.lerp(c1, t)
			var h := _get_height(pt.x, pt.y) + tape_height

			# Upper and lower edge of tape ribbon
			im.surface_add_vertex(Vector3(pt.x, h + tape_width * 0.5, pt.y))
			im.surface_add_vertex(Vector3(pt.x, h - tape_width * 0.5, pt.y))

	im.surface_end()

	mesh_inst.mesh = im

	var tape_mat := StandardMaterial3D.new()
	tape_mat.albedo_color = Color(1.0, 0.65, 0.0)
	tape_mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	tape_mat.cull_mode = BaseMaterial3D.CULL_DISABLED
	mesh_inst.material_override = tape_mat

	_tape_node.add_child(mesh_inst)


# ---------- Z3: Datum Marker ----------

func _create_datum_marker() -> void:
	var datum_x := _x_extent.x
	var datum_z := _y_extent.x
	var datum_h := _get_height(datum_x, datum_z)

	# Larger stake
	var stake := MeshInstance3D.new()
	stake.name = "DatumStake"
	var stake_mesh := CylinderMesh.new()
	stake_mesh.top_radius = 0.015
	stake_mesh.bottom_radius = 0.018
	stake_mesh.height = 0.5
	stake.mesh = stake_mesh
	stake.position = Vector3(datum_x, datum_h + 0.25, datum_z)

	var stake_mat := StandardMaterial3D.new()
	stake_mat.albedo_color = Color(0.6, 0.45, 0.2)
	stake_mat.roughness = 0.5
	stake_mat.metallic = 0.3
	stake.material_override = stake_mat
	_stakes_node.add_child(stake)

	# Brass sphere on top
	var sphere := MeshInstance3D.new()
	var sphere_mesh := SphereMesh.new()
	sphere_mesh.radius = 0.02
	sphere_mesh.height = 0.04
	sphere.mesh = sphere_mesh
	sphere.position = Vector3(datum_x, datum_h + 0.52, datum_z)

	var brass_mat := StandardMaterial3D.new()
	brass_mat.albedo_color = Color(0.75, 0.6, 0.2)
	brass_mat.roughness = 0.3
	brass_mat.metallic = 0.7
	sphere.material_override = brass_mat
	_stakes_node.add_child(sphere)

	# DATUM label
	var label := Label3D.new()
	label.text = "DATUM"
	label.font_size = 32
	label.pixel_size = 0.002
	label.position = Vector3(datum_x, datum_h + 0.6, datum_z)
	label.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	label.modulate = Color(0.9, 0.75, 0.3)
	label.outline_size = 4
	label.outline_modulate = Color(0.1, 0.1, 0.1)
	_stakes_node.add_child(label)


# ---------- Z4: Equipment Staging Area ----------

func _create_equipment_area() -> void:
	_clear_children(_equipment_node)

	# Place at SE corner (max X, min Z)
	var eq_x := _x_extent.y + 1.5
	var eq_z := _y_extent.x - 1.5
	var eq_h := _get_height(eq_x, eq_z)

	# Equipment case materials
	var case_dark_mat := StandardMaterial3D.new()
	case_dark_mat.albedo_color = Color(0.15, 0.15, 0.18)
	case_dark_mat.roughness = 0.6

	var case_orange_mat := StandardMaterial3D.new()
	case_orange_mat.albedo_color = Color(0.85, 0.4, 0.05)
	case_orange_mat.roughness = 0.5

	var metal_mat := StandardMaterial3D.new()
	metal_mat.albedo_color = Color(0.5, 0.5, 0.52)
	metal_mat.roughness = 0.3
	metal_mat.metallic = 0.6

	# Case 1 (dark, large)
	var case1 := MeshInstance3D.new()
	var case1_mesh := BoxMesh.new()
	case1_mesh.size = Vector3(0.5, 0.2, 0.3)
	case1.mesh = case1_mesh
	case1.position = Vector3(eq_x, eq_h + 0.1, eq_z)
	case1.material_override = case_dark_mat
	_equipment_node.add_child(case1)

	# Case 2 (orange, stacked)
	var case2 := MeshInstance3D.new()
	var case2_mesh := BoxMesh.new()
	case2_mesh.size = Vector3(0.45, 0.18, 0.28)
	case2.mesh = case2_mesh
	case2.position = Vector3(eq_x + 0.05, eq_h + 0.29, eq_z)
	case2.rotation.y = deg_to_rad(8)
	case2.material_override = case_orange_mat
	_equipment_node.add_child(case2)

	# Case 3 (small, beside)
	var case3 := MeshInstance3D.new()
	var case3_mesh := BoxMesh.new()
	case3_mesh.size = Vector3(0.3, 0.15, 0.2)
	case3.mesh = case3_mesh
	case3.position = Vector3(eq_x + 0.6, eq_h + 0.075, eq_z + 0.1)
	case3.rotation.y = deg_to_rad(-12)
	case3.material_override = case_dark_mat
	_equipment_node.add_child(case3)

	# Tripod (GPS base station)
	var tripod_mat := metal_mat
	var tripod_height := 1.2
	var tripod_spread := 0.35
	var tripod_center := Vector3(eq_x - 0.8, eq_h, eq_z + 0.3)

	for angle in [0.0, 2.094, 4.189]:  # 0, 120, 240 degrees
		var leg := MeshInstance3D.new()
		var leg_mesh := CylinderMesh.new()
		leg_mesh.top_radius = 0.006
		leg_mesh.bottom_radius = 0.006
		leg_mesh.height = tripod_height
		leg.mesh = leg_mesh

		var foot_offset := Vector3(sin(angle) * tripod_spread, 0, cos(angle) * tripod_spread)
		var leg_center := tripod_center + Vector3(foot_offset.x * 0.5, tripod_height * 0.5, foot_offset.z * 0.5)
		leg.position = leg_center

		# Tilt leg outward
		var tilt_angle := atan2(tripod_spread, tripod_height)
		leg.rotation.x = -cos(angle) * tilt_angle
		leg.rotation.z = sin(angle) * tilt_angle

		leg.material_override = tripod_mat
		_equipment_node.add_child(leg)

	# Tripod hub at top
	var hub := MeshInstance3D.new()
	var hub_mesh := SphereMesh.new()
	hub_mesh.radius = 0.02
	hub_mesh.height = 0.04
	hub.mesh = hub_mesh
	hub.position = tripod_center + Vector3(0, tripod_height, 0)
	hub.material_override = tripod_mat
	_equipment_node.add_child(hub)

	# Folding table with data logger
	var table_x := eq_x + 0.5
	var table_z := eq_z - 0.5
	var table_h := _get_height(table_x, table_z)

	var table_top := MeshInstance3D.new()
	var table_mesh := BoxMesh.new()
	table_mesh.size = Vector3(0.6, 0.02, 0.4)
	table_top.mesh = table_mesh
	table_top.position = Vector3(table_x, table_h + 0.7, table_z)

	var table_mat := StandardMaterial3D.new()
	table_mat.albedo_color = Color(0.3, 0.3, 0.32)
	table_mat.roughness = 0.6
	table_top.material_override = table_mat
	_equipment_node.add_child(table_top)

	# Table legs
	for offset in [Vector3(-0.25, 0, -0.15), Vector3(0.25, 0, -0.15),
			Vector3(-0.25, 0, 0.15), Vector3(0.25, 0, 0.15)]:
		var t_leg := MeshInstance3D.new()
		var t_leg_mesh := CylinderMesh.new()
		t_leg_mesh.top_radius = 0.008
		t_leg_mesh.bottom_radius = 0.008
		t_leg_mesh.height = 0.69
		t_leg.mesh = t_leg_mesh
		t_leg.position = Vector3(table_x + offset.x, table_h + 0.345, table_z + offset.z)
		t_leg.material_override = metal_mat
		_equipment_node.add_child(t_leg)

	# Data logger on table
	var logger := MeshInstance3D.new()
	var logger_mesh := BoxMesh.new()
	logger_mesh.size = Vector3(0.2, 0.06, 0.12)
	logger.mesh = logger_mesh
	logger.position = Vector3(table_x, table_h + 0.74, table_z)
	logger.material_override = case_dark_mat
	_equipment_node.add_child(logger)

	# Tool bag
	var bag := MeshInstance3D.new()
	var bag_mesh := BoxMesh.new()
	bag_mesh.size = Vector3(0.35, 0.2, 0.2)
	bag.mesh = bag_mesh
	bag.position = Vector3(eq_x + 0.3, eq_h + 0.1, eq_z + 0.5)
	bag.rotation.y = deg_to_rad(25)

	var bag_mat := StandardMaterial3D.new()
	bag_mat.albedo_color = Color(0.25, 0.35, 0.2)
	bag_mat.roughness = 0.75
	bag.material_override = bag_mat
	_equipment_node.add_child(bag)


# ---------- Z5: Pin Flags Along Survey Lines ----------

func _create_pin_flags(lines: Array) -> void:
	var flag_interval := 5.0  # meters between flags
	var positions: Array[Vector3] = []

	for line in lines:
		if line.size() < 2:
			continue
		var accumulated := 0.0
		positions.append(Vector3(line[0].x, _get_height(line[0].x, line[0].z), line[0].z))

		for i in range(1, line.size()):
			var prev: Vector3 = line[i - 1]
			var curr: Vector3 = line[i]
			var seg_len := Vector2(curr.x - prev.x, curr.z - prev.z).length()
			accumulated += seg_len

			while accumulated >= flag_interval:
				accumulated -= flag_interval
				var t := 1.0 - (accumulated / seg_len) if seg_len > 0 else 1.0
				var pt := prev.lerp(curr, t)
				positions.append(Vector3(pt.x, _get_height(pt.x, pt.z), pt.z))

	if positions.is_empty():
		return

	# Wire stakes via MultiMesh
	var wire_multi := MultiMeshInstance3D.new()
	wire_multi.name = "PinFlagWires"
	var wire_mm := MultiMesh.new()
	wire_mm.transform_format = MultiMesh.TRANSFORM_3D
	wire_mm.instance_count = positions.size()

	var wire_mesh := CylinderMesh.new()
	wire_mesh.top_radius = 0.001
	wire_mesh.bottom_radius = 0.001
	wire_mesh.height = 0.35
	wire_mm.mesh = wire_mesh

	var wire_mat := StandardMaterial3D.new()
	wire_mat.albedo_color = Color(0.6, 0.6, 0.6)
	wire_mat.metallic = 0.5
	wire_mat.roughness = 0.3
	wire_multi.material_override = wire_mat

	for i in range(positions.size()):
		var t := Transform3D.IDENTITY
		t.origin = positions[i] + Vector3(0, 0.175, 0)
		wire_mm.set_instance_transform(i, t)

	wire_multi.multimesh = wire_mm
	_pin_flags_node.add_child(wire_multi)

	# Flag tabs via MultiMesh
	var tab_multi := MultiMeshInstance3D.new()
	tab_multi.name = "PinFlagTabs"
	var tab_mm := MultiMesh.new()
	tab_mm.transform_format = MultiMesh.TRANSFORM_3D
	tab_mm.instance_count = positions.size()

	var tab_mesh := BoxMesh.new()
	tab_mesh.size = Vector3(0.04, 0.03, 0.001)
	tab_mm.mesh = tab_mesh

	var tab_mat := StandardMaterial3D.new()
	tab_mat.albedo_color = Color(1.0, 0.45, 0.2)
	tab_mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	tab_multi.material_override = tab_mat

	for i in range(positions.size()):
		var t := Transform3D.IDENTITY
		t.origin = positions[i] + Vector3(0.02, 0.35, 0)
		tab_mm.set_instance_transform(i, t)

	tab_multi.multimesh = tab_mm
	_pin_flags_node.add_child(tab_multi)
