## Audio manager autoload — handles ambient, footstep, equipment, and UI sounds.
##
## Uses AudioStreamPlayer3D for spatial footsteps and bird chirps.
## Wind volume modulated by operator speed. Footstep interval scales with velocity.
## Procedural synthesis for all sounds: layered footsteps, FM bird chirps,
## filtered wind ambient, insect chirps, and exponential-envelope beeps.
extends Node

## Volume controls (linear 0.0 - 1.0)
var ambient_volume := 0.7
var sfx_volume := 0.8
var ui_volume := 0.6

## VCO (Voltage-Controlled Oscillator) sonification
var vco_enabled := false
var _vco_player: AudioStreamPlayer
var _vco_generator: AudioStreamGenerator
var _vco_playback: AudioStreamGeneratorPlayback
var _vco_phase := 0.0
var _vco_sample_rate := 44100.0
var _vco_current_freq := 220.0
var _vco_current_volume := 0.0
var _vco_reference_scale := 1e-7  # Reading value for full-scale (instrument-dependent)
var _vco_noise_floor := 1e-9  # Below this, VCO is quiet baseline
var _vco_baseline_value := 0.0  # Subtracted before scaling (e.g. earth field for total-field)

## Audio players (flat — for non-spatial sounds)
var _ambient_wind: AudioStreamPlayer
var _ambient_birds: AudioStreamPlayer
var _beep_player: AudioStreamPlayer
var _alert_player: AudioStreamPlayer
var _ui_click_player: AudioStreamPlayer

## Spatial audio players (3D positioned)
var _footstep_player_3d: AudioStreamPlayer3D
var _bird_player_3d: AudioStreamPlayer3D
var _insect_player_3d: AudioStreamPlayer3D

## Footstep tracking
var _footstep_variants: Array[AudioStream] = []
var _footstep_rng := RandomNumberGenerator.new()
var _last_footstep_time := 0.0
var _min_footstep_interval := 0.25  # seconds (dynamic, updated each frame)

## Bird chirp scheduling
var _bird_timer := 0.0
var _bird_next_interval := 3.0
var _bird_variants: Array[AudioStream] = []
var _bird_rng := RandomNumberGenerator.new()
var _bird_interval_min := 2.0
var _bird_interval_max := 8.0

## Insect chirp scheduling
var _insect_timer := 0.0
var _insect_next_interval := 7.0
var _insect_variants: Array[AudioStream] = []
var _insect_rng := RandomNumberGenerator.new()
var _insect_interval_min := 5.0
var _insect_interval_max := 15.0

## Terrain bounds for spatial placement
var _terrain_extent_x := Vector2(0, 20)
var _terrain_extent_y := Vector2(0, 20)

## Wind base volume (for speed modulation)
var _wind_base_db := -17.0
var _birds_base_offset_db := 8.0

## State
var _ambient_active := false
var _sample_rate := 22050
var _spatial_setup_done := false


func _ready() -> void:
	_footstep_rng.seed = 12345
	_bird_rng.seed = 54321
	_insect_rng.seed = 77889

	# Ensure audio buses exist
	_setup_audio_buses()

	# Create flat audio players (ambient wind/birds stay flat — they're environmental)
	_ambient_wind = _create_player("AmbientWind", -22.0, "Ambience")
	_ambient_birds = _create_player("AmbientBirds", -15.0, "Ambience")
	_beep_player = _create_player("Beep", -10.0, "SFX")
	_alert_player = _create_player("Alert", -8.0, "SFX")
	_ui_click_player = _create_player("UIClick", -10.0, "UI")

	# VCO tone generator
	_setup_vco()

	# Load audio assets (graceful — system works without them)
	_load_ambient_audio()
	_load_footstep_audio()
	_load_equipment_audio()
	_load_ui_audio()
	_generate_bird_variants()
	_generate_insect_variants()

	# Connect to state changes
	SurveyManager.state_changed.connect(_on_state_changed)
	SurveyManager.scenario_loaded.connect(_on_scenario_loaded)
	SurveyManager.instrument_changed.connect(_on_instrument_changed)
	_update_vco_scaling(SurveyManager.current_instrument)


func _process(delta: float) -> void:
	# Deferred spatial player setup (needs active operator in scene tree)
	if not _spatial_setup_done and SurveyManager.active_operator:
		_setup_spatial_players()

	# VCO tone — fill audio buffer with current frequency
	if vco_enabled and _vco_playback and _ambient_active:
		_fill_vco_buffer()

	if not _ambient_active:
		return

	# Schedule bird chirps at random intervals during surveying
	_bird_timer += delta
	if _bird_timer >= _bird_next_interval:
		_bird_timer = 0.0
		_bird_next_interval = _bird_rng.randf_range(_bird_interval_min, _bird_interval_max)
		_play_bird_chirp()

	# Schedule insect chirps at random intervals
	_insect_timer += delta
	if _insect_timer >= _insect_next_interval:
		_insect_timer = 0.0
		_insect_next_interval = _insect_rng.randf_range(_insect_interval_min, _insect_interval_max)
		_play_insect_chirp()

	# Modulate wind volume by operator speed
	if SurveyManager.active_operator and _ambient_wind.playing:
		var speed_val = SurveyManager.active_operator.get("current_speed")
		if speed_val != null:
			var speed: float = speed_val
			var ref_val = SurveyManager.active_operator.get("walk_speed")
			if ref_val == null:
				ref_val = SurveyManager.active_operator.get("max_speed")
			var ref_speed: float = ref_val if ref_val != null else 1.5
			var speed_factor := clampf(speed / maxf(ref_speed, 0.1), 0.0, 1.0)
			_ambient_wind.volume_db = lerpf(-28.0, -18.0, speed_factor)

	# Update dynamic footstep interval based on speed (ground mode only)
	var is_drone: bool = (SurveyManager.current_operator_mode == SurveyManager.OperatorMode.DRONE)
	if not is_drone and SurveyManager.active_operator:
		var speed: float = SurveyManager.active_operator.get("current_speed")
		if speed != null:
			_min_footstep_interval = clampf(0.5 / maxf(speed, 0.1), 0.2, 0.6)


func _setup_audio_buses() -> void:
	## Create Ambience, SFX, and UI buses if they don't already exist.
	var bus_names := ["Ambience", "SFX", "UI"]
	for bus_name in bus_names:
		if AudioServer.get_bus_index(bus_name) == -1:
			var idx := AudioServer.bus_count
			AudioServer.add_bus(idx)
			AudioServer.set_bus_name(idx, bus_name)
			AudioServer.set_bus_send(idx, "Master")


func _setup_spatial_players() -> void:
	## Create AudioStreamPlayer3D nodes attached to the operator for spatial audio.
	var operator := SurveyManager.active_operator
	if not operator:
		return

	# Spatial footstep player — positioned at operator feet (ground mode only)
	var is_drone: bool = (SurveyManager.current_operator_mode == SurveyManager.OperatorMode.DRONE)
	if not is_drone and not _footstep_player_3d:
		_footstep_player_3d = AudioStreamPlayer3D.new()
		_footstep_player_3d.name = "SpatialFootstep"
		_footstep_player_3d.bus = "SFX"
		_footstep_player_3d.position = Vector3(0, 0.05, 0)  # At feet level
		_footstep_player_3d.max_distance = 20.0
		_footstep_player_3d.unit_size = 3.0
		operator.add_child(_footstep_player_3d)

	# Spatial bird player — positioned at random world locations
	if not _bird_player_3d:
		var scene_root := operator.get_parent()
		if scene_root:
			_bird_player_3d = AudioStreamPlayer3D.new()
			_bird_player_3d.name = "SpatialBird"
			_bird_player_3d.bus = "Ambience"
			_bird_player_3d.max_distance = 40.0
			_bird_player_3d.unit_size = 5.0
			scene_root.add_child(_bird_player_3d)

	# Spatial insect player — positioned at ground level nearby
	if not _insect_player_3d:
		var scene_root := operator.get_parent()
		if scene_root:
			_insect_player_3d = AudioStreamPlayer3D.new()
			_insect_player_3d.name = "SpatialInsect"
			_insect_player_3d.bus = "Ambience"
			_insect_player_3d.max_distance = 15.0
			_insect_player_3d.unit_size = 2.0
			scene_root.add_child(_insect_player_3d)

	_spatial_setup_done = true


func _on_scenario_loaded(info: Dictionary) -> void:
	var terrain_data: Dictionary = info.get("terrain", {})
	var x_ext: Array = terrain_data.get("x_extent", [0, 20])
	var y_ext: Array = terrain_data.get("y_extent", [0, 20])
	_terrain_extent_x = Vector2(x_ext[0], x_ext[1])
	_terrain_extent_y = Vector2(y_ext[0], y_ext[1])

	var layers: Array = terrain_data.get("layers", [])
	var metadata: Dictionary = info.get("metadata", {})
	var env_profile: Dictionary = info.get("environment_profile", {})
	var name := str(info.get("name", "")).to_lower()
	var desc := str(info.get("description", "")).to_lower()
	var category := str(metadata.get("category", "")).to_lower()

	var mean_cond := 0.05
	if not layers.is_empty():
		var sum_cond := 0.0
		for layer in layers:
			sum_cond += float(layer.get("conductivity", 0.05))
		mean_cond = sum_cond / float(layers.size())

	var wetness := clampf(mean_cond / 0.25, 0.0, 1.0)
	if metadata.has("water_table_depth"):
		var wt_depth := float(metadata.get("water_table_depth", 2.0))
		wetness = clampf(wetness + clampf((1.0 - wt_depth) / 1.0, 0.0, 0.6), 0.0, 1.0)
	if name.contains("swamp") or name.contains("marsh") or desc.contains("waterlogged"):
		wetness = clampf(wetness + 0.35, 0.0, 1.0)
	if env_profile.has("wetness"):
		wetness = clampf(float(env_profile.get("wetness", wetness)), 0.0, 1.0)
	var wind_intensity := clampf(float(env_profile.get("wind_intensity", wetness)), 0.0, 1.0)

	_wind_base_db = lerpf(-26.0, -19.0, wind_intensity)
	_birds_base_offset_db = lerpf(7.0, 10.0, wetness)
	_bird_interval_min = lerpf(2.2, 4.8, wetness)
	_bird_interval_max = lerpf(7.0, 12.0, wetness)
	_insect_interval_min = lerpf(6.0, 2.8, wetness)
	_insect_interval_max = lerpf(14.0, 8.0, wetness)
	if category.contains("uxo") or name.contains("crater"):
		_bird_interval_min += 2.0
		_bird_interval_max += 4.0
		_insect_interval_min += 1.0
		_insect_interval_max += 2.0

	if _ambient_birds:
		_ambient_birds.volume_db = linear_to_db(ambient_volume) - _birds_base_offset_db


func _on_instrument_changed(instrument: SurveyManager.Instrument) -> void:
	_update_vco_scaling(instrument)
	# Metal detector: auto-enable VCO — audio IS the primary instrument output
	if instrument == SurveyManager.Instrument.METAL_DETECTOR:
		if _ambient_active and not vco_enabled:
			set_vco_enabled(true)


func _update_vco_scaling(instrument: SurveyManager.Instrument) -> void:
	## Set VCO reference scale and noise floor appropriate to the current instrument.
	## Each instrument produces readings in different units and magnitudes.
	match instrument:
		SurveyManager.Instrument.MAG_GRADIOMETER:
			# Gradient in T/m: anomalies ~1e-9 to 1e-6 T/m
			_vco_reference_scale = 5e-8
			_vco_noise_floor = 5e-10
			_vco_baseline_value = 0.0
		SurveyManager.Instrument.METAL_DETECTOR:
			# ΔT in T: ground-balanced anomaly, already zero-centered
			# Typical anomalies: 1-1000 nT near targets
			_vco_reference_scale = 1e-7  # 100 nT full scale
			_vco_noise_floor = 1e-10  # sub-nT noise
			_vco_baseline_value = 0.0  # ΔT is already zero-centered
		SurveyManager.Instrument.EM_FDEM:
			# EM response (real part): values ~1e-5 to 1e-3
			_vco_reference_scale = 5e-4
			_vco_noise_floor = 1e-5
			_vco_baseline_value = 0.0
		SurveyManager.Instrument.RESISTIVITY:
			# Apparent resistivity in ohm-m: background ~100, anomaly drops to ~0-50
			# Sonify deviation from background (100 - reading)
			_vco_reference_scale = 50.0
			_vco_noise_floor = 2.0
			_vco_baseline_value = 0.0  # Special handling in update_vco_reading


func _create_player(player_name: String, volume_db: float, bus_name: String = "Master") -> AudioStreamPlayer:
	var player := AudioStreamPlayer.new()
	player.name = player_name
	player.volume_db = volume_db
	player.bus = bus_name
	add_child(player)
	return player


func _load_ambient_audio() -> void:
	var wind_path := "res://assets/audio/ambient/wind_loop.ogg"
	var birds_path := "res://assets/audio/ambient/birds_loop.ogg"

	if ResourceLoader.exists(wind_path):
		_ambient_wind.stream = load(wind_path)
	else:
		_ambient_wind.stream = _generate_wind_ambient()

	if ResourceLoader.exists(birds_path):
		_ambient_birds.stream = load(birds_path)


func _load_footstep_audio() -> void:
	for i in range(1, 5):
		var path := "res://assets/audio/footsteps/step_grass_%02d.ogg" % i
		if ResourceLoader.exists(path):
			_footstep_variants.append(load(path))

	# If no audio files found, create layered procedural footsteps (8 variants)
	if _footstep_variants.is_empty():
		var gen_rng := RandomNumberGenerator.new()
		gen_rng.seed = 99887
		for i in range(8):
			_footstep_variants.append(_generate_layered_footstep(gen_rng))


func _load_equipment_audio() -> void:
	var beep_path := "res://assets/audio/equipment/beep.ogg"
	var alert_path := "res://assets/audio/equipment/alert.ogg"

	if ResourceLoader.exists(beep_path):
		_beep_player.stream = load(beep_path)
	else:
		_beep_player.stream = _generate_beep_sound(800.0, 0.05)

	if ResourceLoader.exists(alert_path):
		_alert_player.stream = load(alert_path)
	else:
		_alert_player.stream = _generate_beep_sound(1200.0, 0.15)


func _load_ui_audio() -> void:
	var click_path := "res://assets/audio/ui/click.ogg"
	if ResourceLoader.exists(click_path):
		_ui_click_player.stream = load(click_path)
	else:
		_ui_click_player.stream = _generate_click_sound()


## Start ambient audio loops (called on survey start).
func start_ambient() -> void:
	if _ambient_active:
		return
	_ambient_active = true

	if _ambient_wind.stream:
		_ambient_wind.volume_db = _wind_base_db
		_ambient_wind.play()
	if _ambient_birds.stream:
		_ambient_birds.volume_db = linear_to_db(ambient_volume) - _birds_base_offset_db
		_ambient_birds.play()


## Stop ambient audio (called on pause/menu).
func stop_ambient() -> void:
	_ambient_active = false
	# Fade out via tween
	var tween := create_tween()
	if _ambient_wind.playing:
		tween.tween_property(_ambient_wind, "volume_db", -40.0, 0.5)
	if _ambient_birds.playing:
		tween.tween_property(_ambient_birds, "volume_db", -40.0, 0.5)
	tween.tween_callback(_stop_ambient_players)


func _stop_ambient_players() -> void:
	_ambient_wind.stop()
	_ambient_birds.stop()


## Play a randomized footstep sound (spatial if available, flat fallback).
func play_footstep() -> void:
	var now := Time.get_ticks_msec() / 1000.0
	if now - _last_footstep_time < _min_footstep_interval:
		return
	_last_footstep_time = now

	if _footstep_variants.is_empty():
		return

	var variant := _footstep_variants[_footstep_rng.randi() % _footstep_variants.size()]

	# Use spatial 3D player if available
	if _footstep_player_3d and is_instance_valid(_footstep_player_3d):
		_footstep_player_3d.stream = variant
		_footstep_player_3d.pitch_scale = _footstep_rng.randf_range(0.9, 1.1)
		_footstep_player_3d.volume_db = linear_to_db(sfx_volume) - 3.0
		_footstep_player_3d.play()


## Play equipment beep (sample tick).
func play_beep() -> void:
	if _beep_player.stream:
		_beep_player.volume_db = linear_to_db(sfx_volume) - 8.0
		_beep_player.play()


## Play anomaly alert tone.
func play_alert() -> void:
	if _alert_player.stream:
		_alert_player.volume_db = linear_to_db(sfx_volume)
		_alert_player.play()


## Play UI click sound.
func play_ui_click() -> void:
	if _ui_click_player.stream:
		_ui_click_player.volume_db = linear_to_db(ui_volume)
		_ui_click_player.play()


func _play_bird_chirp() -> void:
	if _bird_variants.is_empty():
		return

	var variant := _bird_variants[_bird_rng.randi() % _bird_variants.size()]

	# Use spatial 3D player if available — position at random world location
	if _bird_player_3d and is_instance_valid(_bird_player_3d):
		var rand_x := _bird_rng.randf_range(_terrain_extent_x.x - 3.0, _terrain_extent_x.y + 3.0)
		var rand_z := _bird_rng.randf_range(_terrain_extent_y.x - 3.0, _terrain_extent_y.y + 3.0)
		var rand_y := _bird_rng.randf_range(2.0, 5.0)  # Birds are up in trees
		_bird_player_3d.global_position = Vector3(rand_x, rand_y, rand_z)
		_bird_player_3d.stream = variant
		_bird_player_3d.pitch_scale = _bird_rng.randf_range(0.85, 1.15)
		_bird_player_3d.volume_db = linear_to_db(ambient_volume) - _bird_rng.randf_range(3.0, 8.0)
		_bird_player_3d.play()


func _play_insect_chirp() -> void:
	if _insect_variants.is_empty():
		return

	var variant := _insect_variants[_insect_rng.randi() % _insect_variants.size()]

	if _insect_player_3d and is_instance_valid(_insect_player_3d):
		# Position at ground level, randomly near the operator
		var op := SurveyManager.active_operator
		if op:
			var offset_x := _insect_rng.randf_range(-5.0, 5.0)
			var offset_z := _insect_rng.randf_range(-5.0, 5.0)
			_insect_player_3d.global_position = Vector3(
				op.global_position.x + offset_x,
				op.global_position.y - 1.0,  # Ground level
				op.global_position.z + offset_z,
			)
		_insect_player_3d.stream = variant
		_insect_player_3d.pitch_scale = _insect_rng.randf_range(0.9, 1.2)
		_insect_player_3d.volume_db = linear_to_db(ambient_volume) - _insect_rng.randf_range(5.0, 12.0)
		_insect_player_3d.play()


## Set up the VCO AudioStreamGenerator for continuous tone output.
func _setup_vco() -> void:
	_vco_generator = AudioStreamGenerator.new()
	_vco_generator.mix_rate = _vco_sample_rate
	_vco_generator.buffer_length = 0.1  # 100ms buffer

	_vco_player = AudioStreamPlayer.new()
	_vco_player.name = "VCO"
	_vco_player.stream = _vco_generator
	_vco_player.bus = "SFX"
	_vco_player.volume_db = -80.0  # Start silent
	add_child(_vco_player)


## Toggle VCO sonification on/off.
func set_vco_enabled(enabled: bool) -> void:
	vco_enabled = enabled
	if enabled:
		if not _vco_player.playing:
			_vco_player.play()
			_vco_playback = _vco_player.get_stream_playback()
	else:
		_vco_player.volume_db = -80.0
		_vco_phase = 0.0


## VCO modulation state for metal detector warble
var _vco_warble_phase := 0.0
var _vco_signal_level := 0.0  # 0-1 normalized signal strength


## Update VCO frequency and volume from a sensor reading.
## The reading is in instrument-native units; scaling is set by _update_vco_scaling().
func update_vco_reading(reading: float) -> void:
	if not vco_enabled:
		return

	# Subtract baseline (e.g. earth field for total-field metal detector)
	var anomaly := absf(reading - _vco_baseline_value)

	# For resistivity: sonify deviation from background (~100 ohm-m)
	if SurveyManager.current_instrument == SurveyManager.Instrument.RESISTIVITY:
		anomaly = absf(100.0 - reading)

	_vco_signal_level = clampf(anomaly / _vco_reference_scale, 0.0, 1.0)

	# Volume envelope: quiet baseline hum, ramps up with signal strength
	if anomaly < _vco_noise_floor:
		_vco_current_volume = -28.0  # Quiet baseline hum
	else:
		_vco_current_volume = lerpf(-22.0, -6.0, _vco_signal_level)

	# Metal detector: narrower range (80-440 Hz), other instruments use wider range
	if SurveyManager.current_instrument == SurveyManager.Instrument.METAL_DETECTOR:
		# 80 Hz baseline hum → 440 Hz near target (~2.5 octaves)
		_vco_current_freq = 80.0 * pow(2.0, 2.46 * _vco_signal_level)
	else:
		# Other instruments: 2-octave range (150-600 Hz)
		var normalized := clampf(anomaly / _vco_reference_scale, 0.0, 1.0)
		_vco_current_freq = 150.0 * pow(2.0, 2.0 * normalized)

	_vco_player.volume_db = _vco_current_volume


## Fill the VCO audio buffer with instrument-appropriate waveform.
func _fill_vco_buffer() -> void:
	if not _vco_playback:
		return

	var frames_available := _vco_playback.get_frames_available()
	if frames_available <= 0:
		return

	var increment := _vco_current_freq / _vco_sample_rate
	var is_metal_det: bool = (SurveyManager.current_instrument == SurveyManager.Instrument.METAL_DETECTOR)

	# Warble rate increases with signal: 3-12 Hz modulation
	var warble_rate := lerpf(3.0, 12.0, _vco_signal_level)
	var warble_inc := warble_rate / _vco_sample_rate
	# Warble depth: subtle at low signal, pronounced near target
	var warble_depth := lerpf(0.02, 0.15, _vco_signal_level)

	for i in range(frames_available):
		var sample: float
		if is_metal_det:
			# Metal detector: sawtooth-ish waveform (fundamental + harmonics)
			# gives the characteristic "buzzy" tone of real VLF detectors
			var phase_val := _vco_phase * TAU
			# Fundamental + odd harmonics (band-limited to avoid aliasing)
			sample = sin(phase_val) * 0.45
			sample += sin(phase_val * 2.0) * 0.18  # 2nd harmonic
			sample += sin(phase_val * 3.0) * 0.12  # 3rd harmonic
			sample += sin(phase_val * 5.0) * 0.05  # 5th harmonic

			# Amplitude warble (tremolo) — the characteristic "wah-wah"
			var warble := 1.0 - warble_depth * (0.5 + 0.5 * sin(_vco_warble_phase * TAU))
			sample *= warble
		else:
			# Other instruments: cleaner sine tone
			sample = sin(_vco_phase * TAU) * 0.5

		_vco_playback.push_frame(Vector2(sample, sample))
		_vco_phase = fmod(_vco_phase + increment, 1.0)
		_vco_warble_phase = fmod(_vco_warble_phase + warble_inc, 1.0)


func _on_state_changed(new_state: SurveyManager.State) -> void:
	match new_state:
		SurveyManager.State.SURVEYING, SurveyManager.State.HIRT_SURVEY:
			start_ambient()
			# Auto-enable VCO for metal detector — audio IS the instrument
			if SurveyManager.current_instrument == SurveyManager.Instrument.METAL_DETECTOR:
				set_vco_enabled(true)
		SurveyManager.State.PAUSED:
			stop_ambient()
			# Pause VCO but don't disable (resume will restart)
			if _vco_player:
				_vco_player.volume_db = -80.0
		SurveyManager.State.MAIN_MENU, SurveyManager.State.POST_SURVEY:
			stop_ambient()
			set_vco_enabled(false)
			# Clean up spatial players on return to menu
			if _footstep_player_3d and is_instance_valid(_footstep_player_3d):
				_footstep_player_3d.queue_free()
				_footstep_player_3d = null
			if _bird_player_3d and is_instance_valid(_bird_player_3d):
				_bird_player_3d.queue_free()
				_bird_player_3d = null
			if _insect_player_3d and is_instance_valid(_insect_player_3d):
				_insect_player_3d.queue_free()
				_insect_player_3d = null
			_spatial_setup_done = false


# ── One-pole low-pass filter utility ──

func _one_pole_coeff(cutoff: float) -> float:
	## Returns filter coefficient 'a' for: y = a * y_prev + (1-a) * x
	return exp(-TAU * cutoff / float(_sample_rate))


# ── Layered footstep synthesis ──

func _generate_layered_footstep(rng: RandomNumberGenerator) -> AudioStreamWAV:
	## 3-layer footstep: thud + crunch + rustle with reverb tail.
	## Improved: lower thud, wider crunch band with micro-impacts,
	## longer rustle tail, and delayed reverb copy.
	var duration := 0.35  # Longer for reverb tail
	var num_samples := int(_sample_rate * duration)
	var data := PackedByteArray()
	data.resize(num_samples * 2)

	# Randomized parameters per variant
	var heel_freq := rng.randf_range(40.0, 60.0)
	var thud_decay := rng.randf_range(20.0, 30.0)
	var crunch_gain := rng.randf_range(0.2, 0.35)
	var crunch_duration := rng.randf_range(0.008, 0.015)
	var rustle_gain := rng.randf_range(0.12, 0.22)
	var rustle_duration := rng.randf_range(0.10, 0.15)
	var pitch_variation := rng.randf_range(0.85, 1.15)

	# Micro-impact timing offsets (2-3 sub-impacts for crunch)
	var n_micro := rng.randi_range(2, 3)
	var micro_offsets: Array[float] = [0.0]
	for _mi in range(n_micro - 1):
		micro_offsets.append(rng.randf_range(0.002, 0.006))

	var rustle_lp_a := _one_pole_coeff(3500.0)
	var rustle_hp_a := _one_pole_coeff(600.0)
	var lp_prev := 0.0
	var hp_prev := 0.0

	# Generate dry signal first
	var dry_buffer: PackedFloat32Array = []
	dry_buffer.resize(num_samples)

	for i in range(num_samples):
		var t := float(i) / float(_sample_rate)
		var attack := minf(1.0, t * 300.0)

		# Layer 1: Thud (Low Sine, deeper and longer)
		var thud_env := exp(-t * thud_decay) * attack
		var thud := sin(t * heel_freq * pitch_variation * TAU) * 0.55 * thud_env

		# Layer 2: Crunch (Filtered Noise Burst with micro-impacts)
		var crunch := 0.0
		for offset in micro_offsets:
			var ct := t - offset
			if ct >= 0:
				var crunch_env := exp(-ct * (50.0 + rng.randf_range(0.0, 30.0))) * minf(1.0, ct * 500.0)
				crunch_env *= 1.0 if ct < crunch_duration else exp(-(ct - crunch_duration) * 80.0)
				var noise := (rng.randf() * 2.0 - 1.0)
				crunch += noise * crunch_env * crunch_gain / float(n_micro)

		# Layer 3: Grass Rustle (Band-passed noise, longer tail)
		var rustle_env_t := t / rustle_duration
		var rustle_env := 0.0
		if rustle_env_t < 1.0:
			rustle_env = pow(maxf(0.0, 1.0 - rustle_env_t), 1.5) * attack
		else:
			# Longer tail beyond primary duration
			rustle_env = exp(-(t - rustle_duration) * 12.0) * 0.3

		var noise := (rng.randf() * 2.0 - 1.0)
		lp_prev = rustle_lp_a * lp_prev + (1.0 - rustle_lp_a) * noise
		hp_prev = rustle_hp_a * hp_prev + (1.0 - rustle_hp_a) * lp_prev
		var rustle := (lp_prev - hp_prev) * rustle_gain * rustle_env

		dry_buffer[i] = thud + crunch + rustle

	# Add subtle reverb tail (delayed, attenuated copy at ~25ms)
	var reverb_delay := int(_sample_rate * 0.025)
	var reverb_gain := 0.15
	for i in range(num_samples):
		var reverb := 0.0
		if i >= reverb_delay:
			reverb = dry_buffer[i - reverb_delay] * reverb_gain
		var mixed := dry_buffer[i] + reverb
		var s16 := int(clampf(mixed, -1.0, 1.0) * 32000.0)
		data[i * 2] = s16 & 0xFF
		data[i * 2 + 1] = (s16 >> 8) & 0xFF

	var stream := AudioStreamWAV.new()
	stream.format = AudioStreamWAV.FORMAT_16_BITS
	stream.mix_rate = _sample_rate
	stream.data = data
	return stream


# ── Procedural wind ambient (15s, 3-band) ──

func _generate_wind_ambient() -> AudioStreamWAV:
	## 15-second looping wind with 3-band approach:
	##   Low bed: slow pink noise (< 200 Hz)
	##   Mid gusts: intermittent amplitude swells (3-8s period)
	##   High hiss: speed-responsive high-frequency content
	## Multiple LFO frequencies with incommensurate periods avoid repetition.
	var duration := 15.0
	var num_samples := int(_sample_rate * duration)
	var crossfade_samples := int(_sample_rate * 1.5)  # Longer crossfade for smoother loop
	var data := PackedByteArray()
	data.resize(num_samples * 2)

	var wind_rng := RandomNumberGenerator.new()
	wind_rng.seed = 77442

	# Pink noise filter states (Paul Kellet's optimization) — band 1 (low bed)
	var b0 := 0.0; var b1 := 0.0; var b2 := 0.0; var b3 := 0.0
	var b4 := 0.0; var b5 := 0.0; var b6 := 0.0

	# Secondary pink noise for mid band
	var m0 := 0.0; var m1 := 0.0; var m2 := 0.0; var m3 := 0.0
	var m4 := 0.0; var m5 := 0.0; var m6 := 0.0

	# Low-pass state for low bed
	var low_lp_prev := 0.0
	var low_lp_coeff := exp(-TAU * 200.0 / float(_sample_rate))

	# Band-pass states for mid
	var mid_lp_prev := 0.0
	var mid_hp_prev := 0.0
	var mid_lp_coeff := exp(-TAU * 800.0 / float(_sample_rate))
	var mid_hp_coeff := exp(-TAU * 150.0 / float(_sample_rate))

	for i in range(num_samples):
		var t := float(i) / float(_sample_rate)

		# === Band 1: Low bed (slow pink noise, < 200 Hz) ===
		var white1 := wind_rng.randf() * 2.0 - 1.0
		b0 = 0.99886 * b0 + white1 * 0.0555179
		b1 = 0.99332 * b1 + white1 * 0.0750759
		b2 = 0.96900 * b2 + white1 * 0.1538520
		b3 = 0.86650 * b3 + white1 * 0.3104856
		b4 = 0.55000 * b4 + white1 * 0.5329522
		b5 = -0.7616 * b5 - white1 * 0.0168980
		var pink1 := b0 + b1 + b2 + b3 + b4 + b5 + b6 + white1 * 0.5362
		b6 = white1 * 0.115926
		pink1 *= 0.11

		# Low-pass to keep only sub-200Hz
		low_lp_prev = low_lp_coeff * low_lp_prev + (1.0 - low_lp_coeff) * pink1
		var low_bed := low_lp_prev * 0.6

		# === Band 2: Mid gusts (intermittent swells, 150-800 Hz) ===
		var white2 := wind_rng.randf() * 2.0 - 1.0
		m0 = 0.99886 * m0 + white2 * 0.0555179
		m1 = 0.99332 * m1 + white2 * 0.0750759
		m2 = 0.96900 * m2 + white2 * 0.1538520
		m3 = 0.86650 * m3 + white2 * 0.3104856
		m4 = 0.55000 * m4 + white2 * 0.5329522
		m5 = -0.7616 * m5 - white2 * 0.0168980
		var pink2 := m0 + m1 + m2 + m3 + m4 + m5 + m6 + white2 * 0.5362
		m6 = white2 * 0.115926
		pink2 *= 0.11

		# Band-pass: low-pass at 800, high-pass at 150
		mid_lp_prev = mid_lp_coeff * mid_lp_prev + (1.0 - mid_lp_coeff) * pink2
		mid_hp_prev = mid_hp_coeff * mid_hp_prev + (1.0 - mid_hp_coeff) * mid_lp_prev
		var mid_band := (mid_lp_prev - mid_hp_prev)

		# Gust envelope: multiple LFOs with incommensurate periods
		# Periods: ~3.7s, ~5.3s, ~7.1s — no common factor, never repeats in 15s
		var gust_env := 0.15
		gust_env += 0.25 * maxf(0.0, sin(t * TAU / 3.7))
		gust_env += 0.2 * maxf(0.0, sin(t * TAU / 5.3 + 1.7))
		gust_env += 0.15 * maxf(0.0, sin(t * TAU / 7.1 + 3.2))
		# Sub-modulation: slow wobble on gust intensity
		gust_env *= 0.7 + 0.3 * sin(t * TAU / 11.3 + 0.5)

		var mid_gusts := mid_band * gust_env * 0.5

		# === Band 3: High hiss (> 800 Hz, very subtle) ===
		var white3 := wind_rng.randf() * 2.0 - 1.0
		var high_hiss := white3 * 0.02  # Greatly reduced from 0.08

		# Combined — lower overall gain
		var sample := (low_bed + mid_gusts + high_hiss) * 0.45

		var s16 := int(clampf(sample, -1.0, 1.0) * 32000.0)
		data[i * 2] = s16 & 0xFF
		data[i * 2 + 1] = (s16 >> 8) & 0xFF

	# Crossfade loop seam (1.5s)
	for i in range(crossfade_samples):
		var blend := float(i) / float(crossfade_samples)
		# Equal-power crossfade
		var fade_out := cos(blend * PI * 0.5)
		var fade_in := sin(blend * PI * 0.5)
		var end_idx := num_samples - crossfade_samples + i
		var start_idx := i

		var end_lo := data[end_idx * 2]
		var end_hi := data[end_idx * 2 + 1]
		var end_s := (end_lo | (end_hi << 8))
		if end_s >= 32768: end_s -= 65536

		var start_lo := data[start_idx * 2]
		var start_hi := data[start_idx * 2 + 1]
		var start_s := (start_lo | (start_hi << 8))
		if start_s >= 32768: start_s -= 65536

		var mixed := int(float(end_s) * fade_out + float(start_s) * fade_in)
		data[end_idx * 2] = mixed & 0xFF
		data[end_idx * 2 + 1] = (mixed >> 8) & 0xFF

	var stream := AudioStreamWAV.new()
	stream.format = AudioStreamWAV.FORMAT_16_BITS
	stream.mix_rate = _sample_rate
	stream.loop_mode = AudioStreamWAV.LOOP_FORWARD
	stream.loop_end = num_samples
	stream.data = data
	return stream


# ── FM bird chirps ──

func _generate_bird_variants() -> void:
	## Generate 4 distinct FM-synthesis bird call patterns.
	var chirp_rng := RandomNumberGenerator.new()
	chirp_rng.seed = 33221

	for i in range(4):
		_bird_variants.append(_generate_bird_chirp(chirp_rng))


func _generate_bird_chirp(rng: RandomNumberGenerator) -> AudioStreamWAV:
	## Short FM synthesis burst — carrier 1500-3000 Hz, modulator 5-15 Hz.
	var duration := rng.randf_range(0.05, 0.15)
	var carrier_freq := rng.randf_range(1500.0, 3000.0)
	var mod_freq := rng.randf_range(5.0, 15.0)
	var mod_depth := rng.randf_range(200.0, 800.0)
	var num_samples := int(_sample_rate * duration)

	# Some chirps have a second note
	var two_notes := rng.randf() < 0.5
	var gap_samples := int(_sample_rate * 0.04) if two_notes else 0
	var note2_samples := int(_sample_rate * rng.randf_range(0.03, 0.08)) if two_notes else 0
	var total_samples := num_samples + gap_samples + note2_samples

	var data := PackedByteArray()
	data.resize(total_samples * 2)

	for i in range(total_samples):
		var t := float(i) / float(_sample_rate)
		var sample := 0.0

		if i < num_samples:
			# First note
			var note_t := float(i) / float(num_samples)
			var env := pow(maxf(0.0, 1.0 - note_t), 1.5) * minf(1.0, note_t * _sample_rate * 0.01)
			var mod := sin(t * mod_freq * TAU) * mod_depth
			sample = sin(t * (carrier_freq + mod) * TAU) * 0.2 * env
		elif two_notes and i >= num_samples + gap_samples:
			# Second note (slightly higher pitch)
			var note_i := i - num_samples - gap_samples
			var note_t := float(note_i) / float(note2_samples)
			var env := pow(maxf(0.0, 1.0 - note_t), 1.5) * minf(1.0, note_t * _sample_rate * 0.01)
			var carrier2 := carrier_freq * rng.randf_range(1.1, 1.4)
			var mod2 := sin(t * mod_freq * 1.3 * TAU) * mod_depth * 0.8
			sample = sin(t * (carrier2 + mod2) * TAU) * 0.15 * env

		var s16 := int(clampf(sample, -1.0, 1.0) * 32000.0)
		data[i * 2] = s16 & 0xFF
		data[i * 2 + 1] = (s16 >> 8) & 0xFF

	var stream := AudioStreamWAV.new()
	stream.format = AudioStreamWAV.FORMAT_16_BITS
	stream.mix_rate = _sample_rate
	stream.data = data
	return stream


# ── Insect chirps (FM synthesis, high-frequency) ──

func _generate_insect_variants() -> void:
	## Generate 6 distinct insect chirp patterns using FM synthesis.
	var chirp_rng := RandomNumberGenerator.new()
	chirp_rng.seed = 44556

	for i in range(6):
		_insect_variants.append(_generate_insect_chirp(chirp_rng))


func _generate_insect_chirp(rng: RandomNumberGenerator) -> AudioStreamWAV:
	## FM synthesis insect chirp — carrier 4000-6000 Hz, modulator 80-120 Hz.
	## Produces cricket-like chirp bursts.
	var n_bursts := rng.randi_range(1, 4)
	var burst_duration := rng.randf_range(0.02, 0.06)
	var burst_gap := rng.randf_range(0.03, 0.08)
	var total_duration := n_bursts * burst_duration + (n_bursts - 1) * burst_gap + 0.02
	var num_samples := int(_sample_rate * total_duration)

	var carrier_freq := rng.randf_range(4000.0, 6000.0)
	var mod_freq := rng.randf_range(80.0, 120.0)
	var mod_depth := rng.randf_range(300.0, 800.0)

	var data := PackedByteArray()
	data.resize(num_samples * 2)

	for i in range(num_samples):
		var t := float(i) / float(_sample_rate)
		var sample := 0.0

		# Determine which burst we're in
		var burst_period := burst_duration + burst_gap
		var burst_t := fmod(t, burst_period)
		var burst_index := int(t / burst_period)

		if burst_index < n_bursts and burst_t < burst_duration:
			# Inside a chirp burst
			var local_t := burst_t / burst_duration
			# Sharp attack, quick decay envelope
			var env := minf(1.0, local_t * _sample_rate * 0.005) * pow(maxf(0.0, 1.0 - local_t), 0.8)
			var mod := sin(t * mod_freq * TAU) * mod_depth
			sample = sin(t * (carrier_freq + mod) * TAU) * 0.12 * env

		var s16 := int(clampf(sample, -1.0, 1.0) * 32000.0)
		data[i * 2] = s16 & 0xFF
		data[i * 2 + 1] = (s16 >> 8) & 0xFF

	var stream := AudioStreamWAV.new()
	stream.format = AudioStreamWAV.FORMAT_16_BITS
	stream.mix_rate = _sample_rate
	stream.data = data
	return stream


# ── Equipment sounds (exponential envelope) ──

func _generate_beep_sound(freq: float, duration: float) -> AudioStreamWAV:
	## Sine beep with exponential decay envelope.
	var num_samples := int(_sample_rate * duration)
	var data := PackedByteArray()
	data.resize(num_samples * 2)

	for i in range(num_samples):
		var t := float(i) / float(_sample_rate)
		# Exponential envelope replaces linear fadeout
		var envelope := pow(maxf(0.0, 1.0 - t / duration), 1.5)
		# Attack ramp to prevent click
		envelope *= minf(1.0, t * 500.0)
		var sample := sin(t * freq * TAU) * 0.3 * envelope
		var s16 := int(clampf(sample, -1.0, 1.0) * 32000.0)
		data[i * 2] = s16 & 0xFF
		data[i * 2 + 1] = (s16 >> 8) & 0xFF

	var stream := AudioStreamWAV.new()
	stream.format = AudioStreamWAV.FORMAT_16_BITS
	stream.mix_rate = _sample_rate
	stream.data = data
	return stream


func _generate_click_sound() -> AudioStreamWAV:
	## Short click with exponential decay.
	var duration := 0.03
	var num_samples := int(_sample_rate * duration)
	var data := PackedByteArray()
	data.resize(num_samples * 2)

	for i in range(num_samples):
		var t := float(i) / float(_sample_rate)
		var envelope := exp(-t * 100.0) * minf(1.0, t * 500.0)
		var sample := sin(t * 2000.0 * TAU) * 0.25 * envelope
		var s16 := int(clampf(sample, -1.0, 1.0) * 32000.0)
		data[i * 2] = s16 & 0xFF
		data[i * 2 + 1] = (s16 >> 8) & 0xFF

	var stream := AudioStreamWAV.new()
	stream.format = AudioStreamWAV.FORMAT_16_BITS
	stream.mix_rate = _sample_rate
	stream.data = data
	return stream
