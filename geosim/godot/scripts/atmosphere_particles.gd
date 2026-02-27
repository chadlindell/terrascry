## Atmospheric particle system for environmental ambience.
##
## Spawns floating dust motes, pollen, insects, and ground mist around the
## active operator. Particle density and types are driven by scenario metadata
## (wetness, vegetation density, time of day).
extends Node3D


## Particle containers
var _dust_particles: GPUParticles3D
var _pollen_particles: GPUParticles3D
var _insect_particles: GPUParticles3D
var _mist_particles: GPUParticles3D

## Scenario-driven parameters
var _wetness := 0.3
var _vegetation_density := 0.7
var _surface_elevation := 0.0


func _ready() -> void:
	SurveyManager.scenario_loaded.connect(_on_scenario_loaded)


func _process(_delta: float) -> void:
	if not SurveyManager.active_operator:
		return
	var pos: Vector3 = SurveyManager.active_operator.global_position
	# Dust and pollen follow the player
	if _dust_particles:
		_dust_particles.global_position = pos + Vector3(0, 2.0, 0)
	if _pollen_particles:
		_pollen_particles.global_position = pos + Vector3(0, 1.5, 0)
	if _insect_particles:
		_insect_particles.global_position = pos + Vector3(0, 0.8, 0)


func _on_scenario_loaded(info: Dictionary) -> void:
	var terrain_data: Dictionary = info.get("terrain", {})
	var metadata: Dictionary = info.get("metadata", {})
	var env_profile: Dictionary = info.get("environment_profile", {})
	var layers: Array = terrain_data.get("layers", [])
	_surface_elevation = terrain_data.get("surface_elevation", 0.0)

	# Compute wetness
	var mean_cond := 0.05
	if not layers.is_empty():
		var s := 0.0
		for layer in layers:
			s += float(layer.get("conductivity", 0.05))
		mean_cond = s / float(layers.size())
	_wetness = clampf(mean_cond / 0.25, 0.0, 1.0)
	if metadata.has("water_table_depth"):
		var wt_depth := float(metadata.get("water_table_depth", 2.0))
		_wetness = clampf(_wetness + clampf((1.0 - wt_depth) / 1.0, 0.0, 0.5), 0.0, 1.0)
	var name_lower := str(info.get("name", "")).to_lower()
	if name_lower.contains("swamp") or name_lower.contains("marsh"):
		_wetness = clampf(_wetness + 0.3, 0.0, 1.0)
	if env_profile.has("wetness"):
		_wetness = clampf(float(env_profile.get("wetness", _wetness)), 0.0, 1.0)
	_vegetation_density = clampf(float(env_profile.get("vegetation_density", 0.7)), 0.1, 1.0)

	# Clear existing particles
	for child in get_children():
		child.queue_free()

	_create_dust_system()
	_create_pollen_system()
	_create_insect_system()
	_create_mist_system()

	print("[Atmosphere] Particles spawned (wetness=%.2f, veg=%.2f)" % [_wetness, _vegetation_density])


func _create_dust_system() -> void:
	_dust_particles = GPUParticles3D.new()
	_dust_particles.name = "DustMotes"
	_dust_particles.emitting = true
	_dust_particles.amount = int(lerpf(80.0, 200.0, 1.0 - _wetness))
	_dust_particles.lifetime = 6.0
	_dust_particles.preprocess = 3.0

	var mat := ParticleProcessMaterial.new()
	mat.direction = Vector3(0.2, 0.1, 0.1)
	mat.spread = 180.0
	mat.initial_velocity_min = 0.02
	mat.initial_velocity_max = 0.08
	mat.gravity = Vector3(0, -0.01, 0)
	mat.damping_min = 0.5
	mat.damping_max = 1.0
	mat.scale_min = 0.005
	mat.scale_max = 0.015
	mat.emission_shape = ParticleProcessMaterial.EMISSION_SHAPE_BOX
	mat.emission_box_extents = Vector3(8.0, 3.0, 8.0)

	# Subtle warm specks
	var color_ramp := Gradient.new()
	color_ramp.set_color(0, Color(0.9, 0.85, 0.7, 0.0))
	color_ramp.add_point(0.15, Color(0.95, 0.9, 0.8, 0.4))
	color_ramp.add_point(0.85, Color(0.9, 0.85, 0.75, 0.3))
	color_ramp.set_color(1, Color(0.85, 0.8, 0.7, 0.0))
	var color_tex := GradientTexture1D.new()
	color_tex.gradient = color_ramp
	mat.color_ramp = color_tex

	_dust_particles.process_material = mat

	var mesh := QuadMesh.new()
	mesh.size = Vector2(0.02, 0.02)
	_dust_particles.draw_pass_1 = mesh
	_dust_particles.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF

	add_child(_dust_particles)


func _create_pollen_system() -> void:
	if _vegetation_density < 0.3:
		return

	_pollen_particles = GPUParticles3D.new()
	_pollen_particles.name = "Pollen"
	_pollen_particles.emitting = true
	_pollen_particles.amount = int(lerpf(20.0, 60.0, _vegetation_density))
	_pollen_particles.lifetime = 8.0
	_pollen_particles.preprocess = 4.0

	var mat := ParticleProcessMaterial.new()
	mat.direction = Vector3(0.3, 0.3, 0.1)
	mat.spread = 120.0
	mat.initial_velocity_min = 0.01
	mat.initial_velocity_max = 0.05
	mat.gravity = Vector3(0, 0.02, 0)  # Slight upward drift
	mat.damping_min = 0.3
	mat.damping_max = 0.8
	mat.scale_min = 0.003
	mat.scale_max = 0.008
	mat.emission_shape = ParticleProcessMaterial.EMISSION_SHAPE_BOX
	mat.emission_box_extents = Vector3(6.0, 2.0, 6.0)

	var color_ramp := Gradient.new()
	color_ramp.set_color(0, Color(1.0, 1.0, 0.8, 0.0))
	color_ramp.add_point(0.2, Color(1.0, 1.0, 0.85, 0.6))
	color_ramp.add_point(0.8, Color(1.0, 0.98, 0.8, 0.5))
	color_ramp.set_color(1, Color(1.0, 1.0, 0.85, 0.0))
	var color_tex := GradientTexture1D.new()
	color_tex.gradient = color_ramp
	mat.color_ramp = color_tex

	_pollen_particles.process_material = mat

	var mesh := QuadMesh.new()
	mesh.size = Vector2(0.01, 0.01)
	_pollen_particles.draw_pass_1 = mesh
	_pollen_particles.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF

	add_child(_pollen_particles)


func _create_insect_system() -> void:
	if _wetness < 0.2:
		return

	_insect_particles = GPUParticles3D.new()
	_insect_particles.name = "Insects"
	_insect_particles.emitting = true
	_insect_particles.amount = int(lerpf(8.0, 30.0, _wetness))
	_insect_particles.lifetime = 4.0
	_insect_particles.preprocess = 2.0
	_insect_particles.randomness = 0.8

	var mat := ParticleProcessMaterial.new()
	mat.direction = Vector3(0.0, 0.0, 0.0)
	mat.spread = 180.0
	mat.initial_velocity_min = 0.1
	mat.initial_velocity_max = 0.4
	mat.gravity = Vector3(0, 0.0, 0)
	mat.damping_min = 2.0
	mat.damping_max = 4.0
	mat.scale_min = 0.004
	mat.scale_max = 0.010
	mat.emission_shape = ParticleProcessMaterial.EMISSION_SHAPE_SPHERE
	mat.emission_sphere_radius = 4.0

	# Turbulence for erratic flight
	mat.turbulence_enabled = true
	mat.turbulence_noise_strength = 3.0
	mat.turbulence_noise_speed_random = 0.5
	mat.turbulence_noise_speed = Vector3(0.5, 0.3, 0.5)
	mat.turbulence_influence_min = 0.3
	mat.turbulence_influence_max = 0.8

	var color_ramp := Gradient.new()
	color_ramp.set_color(0, Color(0.15, 0.12, 0.08, 0.0))
	color_ramp.add_point(0.1, Color(0.2, 0.18, 0.12, 0.8))
	color_ramp.add_point(0.9, Color(0.18, 0.15, 0.1, 0.7))
	color_ramp.set_color(1, Color(0.15, 0.12, 0.08, 0.0))
	var color_tex := GradientTexture1D.new()
	color_tex.gradient = color_ramp
	mat.color_ramp = color_tex

	_insect_particles.process_material = mat

	var mesh := QuadMesh.new()
	mesh.size = Vector2(0.008, 0.008)
	_insect_particles.draw_pass_1 = mesh
	_insect_particles.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF

	add_child(_insect_particles)


func _create_mist_system() -> void:
	if _wetness < 0.3:
		return

	_mist_particles = GPUParticles3D.new()
	_mist_particles.name = "GroundMist"
	_mist_particles.emitting = true
	_mist_particles.amount = int(lerpf(12.0, 40.0, _wetness))
	_mist_particles.lifetime = 10.0
	_mist_particles.preprocess = 5.0

	var mat := ParticleProcessMaterial.new()
	mat.direction = Vector3(0.1, 0.0, 0.05)
	mat.spread = 60.0
	mat.initial_velocity_min = 0.02
	mat.initial_velocity_max = 0.06
	mat.gravity = Vector3(0, 0.0, 0)
	mat.damping_min = 0.1
	mat.damping_max = 0.3

	# Scale up over lifetime for expanding mist wisps
	var scale_curve := CurveTexture.new()
	var curve := Curve.new()
	curve.add_point(Vector2(0.0, 0.3))
	curve.add_point(Vector2(0.3, 0.8))
	curve.add_point(Vector2(1.0, 1.0))
	scale_curve.curve = curve
	mat.scale_curve = scale_curve
	mat.scale_min = 0.5
	mat.scale_max = 1.5

	mat.emission_shape = ParticleProcessMaterial.EMISSION_SHAPE_BOX
	mat.emission_box_extents = Vector3(12.0, 0.2, 12.0)

	var color_ramp := Gradient.new()
	color_ramp.set_color(0, Color(0.7, 0.75, 0.78, 0.0))
	color_ramp.add_point(0.2, Color(0.75, 0.78, 0.82, 0.12))
	color_ramp.add_point(0.7, Color(0.72, 0.76, 0.80, 0.10))
	color_ramp.set_color(1, Color(0.7, 0.75, 0.78, 0.0))
	var color_tex := GradientTexture1D.new()
	color_tex.gradient = color_ramp
	mat.color_ramp = color_tex

	_mist_particles.process_material = mat

	# Larger soft quad for mist wisps
	var mesh := QuadMesh.new()
	mesh.size = Vector2(1.0, 1.0)
	_mist_particles.draw_pass_1 = mesh
	_mist_particles.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF

	# Position at ground level
	_mist_particles.position = Vector3(0, _surface_elevation + 0.2, 0)

	var billboard_mat := StandardMaterial3D.new()
	billboard_mat.billboard_mode = BaseMaterial3D.BILLBOARD_PARTICLES
	billboard_mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	billboard_mat.albedo_color = Color(1, 1, 1, 1)
	billboard_mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	_mist_particles.material_override = billboard_mat

	add_child(_mist_particles)
