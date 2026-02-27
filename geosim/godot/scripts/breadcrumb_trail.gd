## QC-colored breadcrumb trail of measurement points.
##
## Uses MultiMeshInstance3D for performance with thousands of small disc markers.
## Color-coded: green = good (on-line, correct speed), yellow = too fast,
## red = dropout or off-line.
extends MultiMeshInstance3D

## Disc appearance
@export var disc_radius := 0.03
@export var max_breadcrumbs := 10000

## Color scheme
const COLOR_GOOD := Color(0.2, 0.8, 0.2, 0.8)       # quality=1
const COLOR_FAST := Color(0.9, 0.8, 0.2, 0.8)       # quality=2
const COLOR_BAD := Color(0.9, 0.2, 0.2, 0.8)        # quality=0 or 3

var _count := 0
var _surface_elevation := 0.0
var _terrain: Node = null


func _ready() -> void:
	_setup_multimesh()
	DataRecorder.sample_recorded.connect(_on_sample_recorded)
	SurveyManager.state_changed.connect(_on_state_changed)
	SurveyManager.scenario_loaded.connect(_on_scenario_loaded)


func _on_scenario_loaded(info: Dictionary) -> void:
	var terrain_data: Dictionary = info.get("terrain", {})
	_surface_elevation = terrain_data.get("surface_elevation", 0.0)
	_terrain = get_parent().get_node_or_null("Terrain")


func _on_state_changed(new_state: SurveyManager.State) -> void:
	if new_state == SurveyManager.State.SURVEYING:
		_count = 0
		visible = true
	elif new_state == SurveyManager.State.MAIN_MENU:
		_count = 0
		visible = false


func _setup_multimesh() -> void:
	var mm := MultiMesh.new()
	mm.transform_format = MultiMesh.TRANSFORM_3D
	mm.use_colors = true
	mm.instance_count = max_breadcrumbs
	mm.visible_instance_count = 0

	# Small disc mesh
	var disc := CylinderMesh.new()
	disc.top_radius = disc_radius
	disc.bottom_radius = disc_radius
	disc.height = 0.005
	mm.mesh = disc

	multimesh = mm

	# Unshaded material
	var mat := StandardMaterial3D.new()
	mat.vertex_color_use_as_albedo = true
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	material_override = mat


func _on_sample_recorded(sample_index: int) -> void:
	if _count >= max_breadcrumbs:
		return

	if sample_index < 1 or sample_index > DataRecorder.samples.size():
		return

	var sample: Dictionary = DataRecorder.samples[sample_index - 1]
	var quality: int = sample.get("quality", 1)

	# Position: convert from GeoSim to Godot
	var gs_pos := Vector3(sample.get("x_e", 0.0), sample.get("y_n", 0.0), sample.get("z_up", 0.0))
	var godot_pos := CoordUtil.to_godot(gs_pos)
	# Place just above actual terrain surface (follows heightmap + craters)
	if _terrain and _terrain.has_method("get_height_at"):
		godot_pos.y = _terrain.get_height_at(godot_pos.x, godot_pos.z) + 0.01
	else:
		godot_pos.y = _surface_elevation + 0.01

	var t := Transform3D.IDENTITY
	t.origin = godot_pos
	multimesh.set_instance_transform(_count, t)

	# Color based on quality
	var color: Color
	match quality:
		0: color = COLOR_BAD
		1: color = COLOR_GOOD
		2: color = COLOR_FAST
		3: color = COLOR_BAD
		_: color = COLOR_GOOD

	multimesh.set_instance_color(_count, color)
	_count += 1
	multimesh.visible_instance_count = _count
